"""Calibrate + benchmark a teensy-v7 GRU on the real-world protocol.

    .venv/bin/python scripts/eval_gru.py --model models/teensy-v7-gru96.npz \
        --calibrate          # sweep thr on AMI dev → save into meta
    .venv/bin/python scripts/eval_gru.py --model models/teensy-v7-gru96.npz \
        --eval --out models/comparison_v7_gru96.json

Same data, same frame grid, same scoring as compare_all.py: TEN VAD public
set (AUC + best-F1 sweep) and 8 held-out AMI SDM meetings (F1/AUC at
AMI-dev-calibrated thresholds), plus streaming speed per 20 ms chunk.
The GRU is evaluated CAUSALLY with persistent state — exactly how the
StreamingVAD runtime would run it in production.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensyvad.audio import float_to_pcm16  # noqa: E402
from teensyvad.rnn import TinyGRU  # noqa: E402
from scripts.eval_realworld import (SR, centers_for, energy_decisions_on,  # noqa: E402
                                    frame_truth, load_any_rate,
                                    parse_ami_segments, parse_ten_labels)
from scripts.distill_label import load_teacher, teacher_probs  # noqa: E402

CALIB_MEETINGS = ["ES2002a", "IS1000a", "TS3003a"]
def eval_meetings(wav_dir: Path):
    """Every AMI Array1-01 wav except the calibration trio — same set
    selection as compare_all.py's eval_ami_models."""
    out = []
    for w in sorted(wav_dir.glob("*.Array1-01.wav")):
        meeting = w.name.split(".")[0]
        if meeting in CALIB_MEETINGS:
            continue
        out.append(meeting)
    return out


def gru_probs_on(m: TinyGRU, lm, x8k: np.ndarray) -> np.ndarray:
    """Causal streaming probabilities, one 10 ms frame at a time."""
    F = lm(x8k)
    F = ((F - m.in_mean) / m.in_std).astype(np.float32)
    m.reset_state(1)
    out = np.empty(len(F), dtype=np.float64)
    for i in range(len(F)):
        out[i] = np.asarray(m.step(F[i])).reshape(-1)[0]
    return out


def gru_stream_speed(m: TinyGRU, lm, x8k: np.ndarray) -> float:
    """Median µs per 20 ms telephony chunk (frame + GRU step)."""
    F = ((lm(x8k) - m.in_mean) / m.in_std).astype(np.float32)
    m.reset_state(1)
    ts = []
    for i in range(len(F)):
        t0 = time.perf_counter()
        m.step(F[i])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 2 * 1e6)      # per-frame → per 20 ms (2 frames)


def ami_split(m: TinyGRU, lm, wav_dir: Path, manual_dir: Path,
              meetings, verbose=False):
    """→ (P, Y, speed_us) over the given meetings with causal streaming."""
    P, Y = [], []
    speed = None
    w0 = wav_dir / f"{meetings[0]}.Array1-01.wav"
    x0 = load_any_rate(w0)
    speed = gru_stream_speed(m, lm, x0)
    for meeting in meetings:
        w = wav_dir / f"{meeting}.Array1-01.wav"
        spans = parse_ami_segments(meeting, manual_dir)
        try:
            x = load_any_rate(w)
        except Exception as e:
            print(f"  !! {meeting}: unreadable wav ({e}) — skipping", flush=True)
            continue
        F = lm(x)
        centers = centers_for(len(F), lm)
        truth = frame_truth(centers, spans)
        probs = gru_probs_on(m, lm, x)
        P.append(probs)
        Y.append(truth)
        if verbose:
            print(f"  {meeting}: {len(F)/6000:.1f} min, "
                  f"{truth.mean()*100:.0f}% speech", flush=True)
    return np.concatenate(P), np.concatenate(Y).astype(bool), speed


def sweep_best(P: np.ndarray, Y: np.ndarray):
    best = (0.5, -1.0)
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = P >= thr
        tp = float((pred & Y).sum()); fp = float((pred & ~Y).sum())
        fn = float((~pred & Y).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        if f1 > best[1]:
            best = (float(thr), f1)
    return best


def score(P: np.ndarray, Y: np.ndarray, thr: float | None):
    from teensyvad.model import auc
    out = {}
    pred = P >= thr if thr is not None else P > 0.5
    tp = float((pred & Y).sum()); fp = float((pred & ~Y).sum())
    fn = float((~pred & Y).sum())
    out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1.0)
    out["far"] = fp / max((~Y).sum(), 1.0)
    out["mr"] = fn / max(Y.sum(), 1.0)
    out["auc"] = auc(P, Y.astype(np.float32))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, default=Path("data/raw/ami/wav"))
    ap.add_argument("--manual", type=Path, default=Path("data/raw/ami/manual"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="v7-gru")
    args = ap.parse_args()

    m = TinyGRU.load(args.model)
    lm = None                                     # lazy: built in gru_probs_on

    if args.calibrate:
        from teensyvad.features import LogMel
        lm = LogMel(sr=SR)
        P, Y, _ = ami_split(m, lm, args.wav_dir, args.manual,
                            CALIB_MEETINGS, verbose=True)
        best = sweep_best(P, Y)
        print(f"best frame F1 on AMI dev: {best[1]:.4f} @ thr_hi {best[0]:.2f}")
        m.meta["thr_hi"] = round(best[0], 3)
        m.meta["thr_lo"] = round(0.6 * best[0], 3)
        m.meta["calibrated_on"] = "ami-dev:" + ",".join(CALIB_MEETINGS)
        m.save(args.model)
        print(f"saved thresholds → {args.model}")
        return

    if not args.eval:
        ap.error("choose --calibrate and/or --eval")

    from teensyvad.features import LogMel
    lm = LogMel(sr=SR)
    rows = {"model": f"{args.tag}", "params": m.n_params(),
            "kb": round(args.model.stat().st_size / 1024, 1)}

    if args.eval:
        tdir = Path("data/raw/ten-vad/testset")
        P, Y = [], []
        for w in sorted(tdir.glob("testset-audio-*.wav")):
            lab = parse_ten_labels(w.with_suffix(".scv"))
            x = load_any_rate(w)
            F = lm(x)
            centers = centers_for(len(F), lm)
            truth = frame_truth(centers, lab)
            P.append(gru_probs_on(m, lm, x))
            Y.append(truth)
        Pc = np.concatenate(P); Yc = np.concatenate(Y).astype(bool)
        from teensyvad.model import auc
        thr_hi = float(m.meta.get("thr_hi", 0.5))
        at = score(Pc, Yc, thr_hi)
        best = sweep_best(Pc, Yc)
        rows["ten_f1"] = round(best[1], 4)
        rows["ten_auc"] = round(at["auc"], 4)
        print(f"TEN: F1* {rows['ten_f1']} (best thr {best[0]:.2f}), "
              f"AUC {rows['ten_auc']} @ calibrated thr {thr_hi}")

        meetings = eval_meetings(args.wav_dir)
        print(f"AMI eval meetings ({len(meetings)}):", ", ".join(meetings))
        P, Y, speed = ami_split(m, lm, args.wav_dir, args.manual,
                                meetings, verbose=True)
        thr_hi = float(m.meta.get("thr_hi", 0.5))
        at = score(P, Y, thr_hi)
        from teensyvad.model import auc as _auc
        rows["ami_f1"] = round(at["f1"], 4)
        rows["ami_auc"] = round(_auc(P, Y.astype(np.float32)), 4)
        rows["us_per_20ms"] = round(speed, 1)
        print(f"AMI: F1 {rows['ami_f1']} AUC {rows['ami_auc']} @ {thr_hi} | "
              f"{rows['us_per_20ms']} µs/20ms")

    if args.out:
        args.out.write_text(json.dumps({"rows": [rows]}, indent=2))
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
