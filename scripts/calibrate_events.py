"""Calibrate StreamingVAD's event thresholds on the validation set.

Frame-level F1 (used by train.py) picks a threshold for *frames*, but the
product is *events*.  This script sweeps (thr_hi, thr_lo) pairs at the
EVENT level — segments matched against construction-truth spans with the
same matcher as evaluate.py — and writes the winner back into the model
file's metadata.

    .venv/bin/python scripts/calibrate_events.py --model models/teensy-v2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import read_wav  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import load_model  # noqa: E402
from teensyvad.streaming import Hysteresis, hysteresis_events  # noqa: E402
from scripts.evaluate import events_from_labels, match_events  # noqa: E402
from scripts_utils import context_windows  # noqa: E402

HOP_S = 0.01


def clip_probs(model, lm: LogMel, x: np.ndarray) -> np.ndarray:
    """Offline probability trajectory for one clip (matches streaming)."""
    F = lm(x)
    X, _ = context_windows(F, F[:, 0], int(model.meta["context"]))
    return model.probs(X)


def segs_from_probs(probs, thr_hi, thr_lo, on_frames, off_frames):
    """Segments (s) from a prob trajectory, including end-of-stream flush."""
    _, ev = hysteresis_events(probs, thr_hi, thr_lo, on_frames, off_frames)
    segs, start = [], None
    for kind, i in ev:
        if kind == "speech_start":
            start = i * HOP_S
        else:
            segs.append((start, i * HOP_S))
            start = None
    if start is not None:                       # flush open segment at EOF
        segs.append((start, len(probs) * HOP_S))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--hangover-ms", type=float, default=250.0)
    args = ap.parse_args()

    model = load_model(args.model)
    lm = LogMel(sr=8000)
    off_frames = max(1, int(round(args.hangover_ms / 1000 / HOP_S)))
    on_frames = int(model.meta.get("on_frames", 3))

    z = np.load(args.data / f"{args.split}.npz")
    cl = z["clip_len"]
    wavs = sorted((args.data / "audio").glob(f"{args.split}_*.wav"))
    assert len(wavs) == len(cl), f"{len(wavs)} wavs vs {len(cl)} clips"

    # precompute probs + truth spans once
    clips = []
    pos = 0
    for w in wavs:
        n = int(cl[len(clips)])
        y = z["y"][pos:pos + n]
        pos += n
        x, _ = read_wav(w)
        # NOTE: context windows shift labels; probs[t] corresponds to frame
        # t+K-1 of the clip.  Truth spans must use the same alignment.
        probs = clip_probs(model, lm, x)
        yw = y[int(model.meta["context"]) - 1:]
        clips.append((probs, events_from_labels(yw, HOP_S)))
    print(f"calibrating on {len(clips)} {args.split} clips "
          f"(hangover {args.hangover_ms:.0f} ms, on_frames {on_frames})")

    best = None
    grid_hi = np.round(np.arange(0.40, 0.86, 0.05), 3)
    for thr_hi in grid_hi:
        for thr_lo in np.round(np.arange(0.20, thr_hi, 0.05), 3):
            tp = np_ = 0
            nt = 0
            for probs, truth in clips:
                pred = segs_from_probs(probs, float(thr_hi), float(thr_lo),
                                       on_frames, off_frames)
                a, b, c = match_events(pred, truth)
                tp += a; np_ += b; nt += c
            prec = tp / max(np_, 1); rec = tp / max(nt, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if best is None or f1 > best["f1"]:
                best = dict(f1=f1, prec=prec, rec=rec,
                            thr_hi=float(thr_hi), thr_lo=float(thr_lo),
                            n_pred=np_, n_true=nt)

    print(f"best: thr_hi={best['thr_hi']:.2f} thr_lo={best['thr_lo']:.2f} → "
          f"event P={best['prec']:.3f} R={best['rec']:.3f} F1={best['f1']:.3f} "
          f"({best['n_pred']} pred / {best['n_true']} true)")

    # write back into the model file
    meta = dict(model.meta)
    meta["thr_hi"], meta["thr_lo"] = best["thr_hi"], best["thr_lo"]
    meta["hangover_ms"] = args.hangover_ms
    meta["event_f1_val"] = round(best["f1"], 4)
    model.meta = meta
    model.save(args.model)
    print(f"saved thresholds → {args.model}")


if __name__ == "__main__":
    main()
