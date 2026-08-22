"""Real-world evaluation: human-labelled recordings, not our synthetic mixtures.

    .venv/bin/python scripts/eval_realworld.py                  # TEN set, all models
    .venv/bin/python scripts/eval_realworld.py --ami            # + AMI meetings

Sets
----
TEN VAD public set  30 short real-world recordings (16 kHz) with manual
                    speech/non-speech labels — the SAME set FlashVAD
                    published its numbers on (their card: AUC 0.882,
                    F1 0.889, FAR 26.3 %, MR 13.0 %), so we can compare
                    directly.  Scored at 8 kHz (telephony) and also after
                    a G.711 µ-law round-trip (PSTN realism).
AMI SDM             12 real multi-party meetings, single DISTANT mic
                    (Array1-01), manual transcription segments unioned
                    into VAD truth — rooms, crosstalk, keyboards, chairs.

Everything is scored on the same 10 ms frame grid: a frame is speech iff
its centre lies inside a labelled speech segment.
"""

from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import float_to_pcm16, read_wav, resample_fft, telephony_roundtrip  # noqa: E402
from teensyvad.energy_vad import EnergyVAD  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import MLP, auc, prf  # noqa: E402
from teensyvad.quant import QuantizedMLP, load_any  # noqa: E402
from scripts_utils import context_windows  # noqa: E402
from scripts.distill_label import load_teacher, teacher_probs  # noqa: E402

SR = 8000
HOP_S = 0.010


# ---------------------------------------------------------------- labels

def parse_ten_labels(path: Path):
    """testset-audio-01.scv → np.array of (start, end, class) triplets."""
    parts = path.read_text().strip().split(",")
    triplets = []
    for i in range(1, len(parts) - 2, 3):
        triplets.append((float(parts[i]), float(parts[i + 1]), int(float(parts[i + 2]))))
    return np.array(triplets)


def parse_ami_segments(meeting: str, manual_dir: Path):
    """Union of all speakers' transcription segments → speech intervals."""
    seg_dir = manual_dir / "segments"
    spans = []
    for xml in sorted(seg_dir.glob(f"{meeting}.*.segments.xml")):
        text = xml.read_text(errors="ignore")
        for m in re.finditer(r'transcriber_start="([\d.]+)"\s+transcriber_end="([\d.]+)"', text):
            spans.append((float(m.group(1)), float(m.group(2))))
    spans.sort()
    # merge overlaps
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged) if merged else np.zeros((0, 2))


def frame_truth(centers: np.ndarray, spans: np.ndarray, label_col: int | None = None):
    """Frame centre inside any 1-labelled segment (TEN) or any span (AMI)."""
    if len(spans) == 0:
        return np.zeros(len(centers), dtype=bool)
    if spans.shape[1] == 3:                      # TEN: (start, end, class)
        mask = spans[:, 2] > 0.5
        spans = spans[mask][:, :2]
    starts, ends = spans[:, 0], spans[:, 1]
    # vectorised: for each center, any span covering it
    inside = (centers[:, None] >= starts[None, :]) & (centers[:, None] < ends[None, :])
    return inside.any(axis=1)


# ---------------------------------------------------------------- audio paths

def load_any_rate(path: Path, sr: int = SR) -> np.ndarray:
    x, sr_in = read_wav(path)
    return resample_fft(x, sr_in, sr) if sr_in != sr else x


def model_probs_on(m, lm: LogMel, x8k: np.ndarray):
    F = lm(x8k)
    K = int(m.meta["context"])
    X, _ = context_windows(F, F[:, 0], K)
    return m.probs(X), F, K


def centers_for(n_frames: int, lm: LogMel):
    return (lm.frame_len / 2 + np.arange(n_frames) * lm.hop_len) / lm.sr


def silero_probs_on(teacher, torch, x8k: np.ndarray):
    tp, tt = teacher_probs(teacher, torch, x8k)
    return tp, tt


def energy_decisions_on(x8k: np.ndarray):
    """EnergyVAD in_speech trajectory per 10 ms frame."""
    ev = EnergyVAD(sr=SR)
    pcm = float_to_pcm16(x8k)
    states = []
    for i in range(0, len(pcm), 160):            # 10 ms chunks
        ev.feed(pcm[i:i + 160])
        states.append(ev.in_speech)
    return np.array(states, dtype=float)


# ---------------------------------------------------------------- scoring

def score(probs_or_decisions, truth_bool, thr: float | None):
    """Returns dict(F1, AUC, FAR, MR) — probs if thr given else 0/1 decisions."""
    y = truth_bool.astype(np.float32)
    if thr is not None:
        p = np.asarray(probs_or_decisions, dtype=np.float64)
        pred = p >= thr
        f1 = prf(p, y, thr)[2]
        a = auc(p, y)
    else:
        pred = np.asarray(probs_or_decisions) > 0.5
        f1 = prf(pred.astype(float), y, 0.5)[2]
        a = auc(pred.astype(float), y)
    tp = int(np.sum(pred & truth_bool))
    fp = int(np.sum(pred & ~truth_bool))
    fn = int(np.sum(~pred & truth_bool))
    tn = int(np.sum(~pred & ~truth_bool))
    return dict(f1=f1, auc=a, far=fp / max(fp + tn, 1), mr=fn / max(fn + tp, 1))


def sweep_best(P: np.ndarray, Y: np.ndarray):
    """Best-F1 threshold on this data (upper bound; NOT a deployment number)."""
    from teensyvad.model import best_threshold
    thr = best_threshold(P, Y.astype(np.float32))
    s = score(P, Y.astype(bool), thr)
    s["thr"] = thr
    return s


def eval_ten(models: dict, lm: LogMel, ulaw: bool, verbose=False):
    tdir = Path("data/raw/ten-vad/testset")
    wavs = sorted(tdir.glob("testset-audio-*.wav"))
    results = {name: dict(P=[], y=[]) for name in models}
    results["_silero"] = dict(P=[], y=[])
    results["_energy"] = dict(P=[], y=[])

    teacher, torch = load_teacher()
    for w in wavs:
        lab = parse_ten_labels(w.with_suffix(".scv"))
        x = load_any_rate(w)
        if ulaw:
            x = telephony_roundtrip(x)
        F = lm(x)
        centers = centers_for(len(F), lm)
        truth = frame_truth(centers, lab)
        K = int(list(models.values())[0].meta["context"])
        X, _ = context_windows(F, F[:, 0], K)
        cc = centers[K - 1:]
        yy = truth[K - 1:]
        for name, m in models.items():
            results[name]["P"].append(m.probs(X))
            results[name]["y"].append(yy)
        tp, tt = silero_probs_on(teacher, torch, x)
        results["_silero"]["P"].append(np.interp(cc, tt, tp))
        results["_silero"]["y"].append(yy)
        # energy decisions are per-frame → drop the first K-1 to align with yy
        e_states = energy_decisions_on(x)
        results["_energy"]["P"].append(e_states[K - 1:][:len(cc)])
        results["_energy"]["y"].append(yy)
        if verbose:
            print(f"  {w.stem}: {len(cc)} frames, {yy.mean()*100:.0f}% speech")

    out = {}
    swept = {}
    for name, r in results.items():
        P = np.concatenate(r["P"]); Y = np.concatenate(r["y"]).astype(bool)
        if name.startswith("_silero"):
            out[name] = score(P, Y, None)        # decisions already
            swept[name] = sweep_best(P, Y)
        elif name.startswith("_energy"):
            out[name] = score(P, Y, None)
        else:
            thr = float(models[name].meta.get("thr_hi", 0.5))
            out[name] = score(P, Y, thr)
            swept[name] = sweep_best(P, Y)
    return out, swept


def eval_ami(models: dict, lm: LogMel, verbose=False,
             exclude=("ES2002a", "IS1000a", "TS3003a")):
    """AMI SDM meetings, manual labels.  `exclude` = the calibration
    meetings (used by calibrate_realworld.py) — never scored here."""
    wav_dir = Path("data/raw/ami/wav")
    manual = Path("data/raw/ami/manual")
    out = {name: dict(P=[], y=[]) for name in models}
    out["_silero"] = dict(P=[], y=[])
    out["_energy"] = dict(P=[], y=[])
    teacher, torch = load_teacher()
    for w in sorted(wav_dir.glob("*.Array1-01.wav")):
        meeting = w.name.split(".")[0]
        if meeting in exclude:
            continue
        spans = parse_ami_segments(meeting, manual)
        if len(spans) == 0:
            continue
        x = load_any_rate(w)
        F = lm(x)
        centers = centers_for(len(F), lm)
        truth = frame_truth(centers, spans)
        K = int(list(models.values())[0].meta["context"])
        X, _ = context_windows(F, F[:, 0], K)
        # chunk windows to keep memory sane (~200k frames at a time)
        cc = centers[K - 1:]
        yy = truth[K - 1:]
        step = 200_000
        for name in list(models) + ["_silero"]:
            out[name]["P"].append(np.zeros(0))
        for s in range(0, len(X), step):
            Xs = X[s:s + step]
            for name, m in models.items():
                out[name]["P"][-1] = np.concatenate([out[name]["P"][-1], m.probs(Xs)])
            out["_silero"]["P"][-1] = np.concatenate([out["_silero"]["P"][-1],
                                                      np.zeros(len(Xs))])
        # silero over full clip (stateful, must run sequentially)
        tp, tt = silero_probs_on(teacher, torch, x)
        out["_silero"]["P"][-1] = np.interp(cc, tt, tp)
        e_states = energy_decisions_on(x)
        out["_energy"]["P"].append(e_states[K - 1:][:len(cc)])
        for name in list(models) + ["_silero", "_energy"]:
            out[name]["y"].append(yy)
        if verbose:
            print(f"  {meeting}: {len(cc)/6000:.1f} min, {yy.mean()*100:.0f}% speech")
    scores = {}
    for name, r in out.items():
        P = np.concatenate(r["P"]); Y = np.concatenate(r["y"]).astype(bool)
        if name.startswith("_"):
            scores[name] = score(P, Y, None)
        else:
            thr = float(models[name].meta.get("thr_hi", 0.5))
            scores[name] = score(P, Y, thr)
    return scores


def print_table(title, scores, swept=None):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'model':<16}{'F1':>8}{'AUC':>8}{'FAR':>8}{'MR':>8}")
    for name, s in scores.items():
        disp = {"_silero": "silero (teacher)", "_energy": "energy baseline"}.get(name, name)
        print(f"  {disp:<16}{s['f1']:>8.4f}{s['auc']:>8.4f}"
              f"{s['far']*100:>7.1f}%{s['mr']*100:>7.1f}%")
    if swept:
        print(f"  {'— at best-F1 threshold on this set (upper bound):':<46}")
        for name, s in swept.items():
            disp = {"_silero": "silero (teacher)"}.get(name, name)
            print(f"  {disp:<16}{s['f1']:>8.4f}  (thr {s['thr']:.2f}, "
                  f"FAR {s['far']*100:.1f}%, MR {s['mr']*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=[
        "models/teensy-v1.npz", "models/teensy-v2.npz", "models/teensy-v2-qat.npz"])
    ap.add_argument("--ami", action="store_true", help="also evaluate AMI SDM meetings")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    lm = LogMel(sr=SR)
    models = {}
    for p in args.models:
        if not Path(p).exists():
            print(f"(skipping missing {p})")
            continue
        m = load_any(p)
        models[Path(p).stem.replace("teensy-", "")] = m
    assert models, "no model files found"

    scores, swept = eval_ten(models, lm, ulaw=False, verbose=args.verbose)
    print_table("TEN VAD public set (30 real recordings, 8 kHz, native)", scores, swept)

    scores_u, swept_u = eval_ten(models, lm, ulaw=True, verbose=False)
    print_table("TEN VAD public set — after G.711 µ-law round-trip (PSTN realism)",
                scores_u, swept_u)

    print("\nreference (FlashVAD card, same TEN set): F1 0.889  AUC 0.882  FAR 26.3%  MR 13.0%")

    if args.ami:
        scores_a = eval_ami(models, lm, verbose=args.verbose)
        print_table("AMI SDM meetings (real rooms, distant mic, manual labels)", scores_a)


if __name__ == "__main__":
    main()
