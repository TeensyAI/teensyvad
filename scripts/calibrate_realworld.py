"""Calibrate VAD thresholds on REAL audio (AMI dev meetings).

    .venv/bin/python scripts/calibrate_realworld.py --model models/teensy-v3.npz

The v2 lesson: thresholds tuned on synthetic mixtures (0.85) were far too
conservative for real recordings — MR 31 % on the TEN set while AUC was
fine.  Ranking transfers across domains; operating points don't.

This sweeps (thr_hi, thr_lo) at frame level on 3 AMI dev meetings
(manual labels, real rooms) and writes the winner into the model file.
The other 8 AMI meetings are NEVER used here — they stay a clean eval
set for scripts/eval_realworld.py --ami.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.quant import load_any  # noqa: E402
from eval_realworld import (SR, centers_for, frame_truth, load_any_rate,  # noqa: E402
                            parse_ami_segments)
from scripts_utils import context_windows  # noqa: E402

# dev meetings for calibration (eval set in eval_realworld stays untouched)
CALIB_MEETINGS = ["ES2002a", "IS1000a", "TS3003a"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, default=Path("data/raw/ami/wav"))
    ap.add_argument("--manual", type=Path, default=Path("data/raw/ami/manual"))
    args = ap.parse_args()

    m = load_any(args.model)
    lm = LogMel(sr=SR)
    K = int(m.meta["context"])

    # accumulate probs + truth on dev meetings
    P, Y = [], []
    for meeting in CALIB_MEETINGS:
        w = args.wav_dir / f"{meeting}.Array1-01.wav"
        spans = parse_ami_segments(meeting, args.manual)
        x = load_any_rate(w)
        F = lm(x)
        centers = centers_for(len(F), lm)
        truth = frame_truth(centers, spans)
        X, _ = context_windows(F, F[:, 0], K)
        # chunked inference (long meetings)
        for s in range(0, len(X), 200_000):
            P.append(m.probs(X[s:s + 200_000]))
        Y.append(truth[K - 1:][:len(X)])
        print(f"  {meeting}: {len(X)/6000:.1f} min, {truth.mean()*100:.0f}% speech")
    P = np.concatenate(P)
    Y = np.concatenate(Y).astype(np.float32)
    print(f"calibration pool: {len(P):,} frames, {Y.mean()*100:.1f}% speech")

    # sweep frame-level F1 over (thr_hi, thr_lo) — lo only matters with
    # hysteresis, so sweep hi finely and set lo = 0.6*hi (v1 heuristic ratio)
    from teensyvad.model import prf
    best = None
    for thr in np.arange(0.10, 0.91, 0.02):
        f1 = prf(P, Y, float(thr))[2]
        if best is None or f1 > best[1]:
            best = (float(thr), f1)
    thr_hi = best[0]
    print(f"best frame F1 on AMI dev: {best[1]:.4f} @ thr_hi {thr_hi:.2f}")

    meta = dict(m.meta)
    meta["thr_hi"] = round(thr_hi, 3)
    meta["thr_lo"] = round(0.6 * thr_hi, 3)
    meta["calibrated_on"] = "ami-dev:" + ",".join(CALIB_MEETINGS)
    m.meta = meta
    m.save(args.model)
    print(f"saved thresholds → {args.model}")


if __name__ == "__main__":
    main()
