"""Train the teensy VAD model on the prepared features.

    .venv/bin/python scripts/train.py            # → models/teensy-v1.npz

Design choices, briefly:
* context 10 frames (100 ms) — VAD decisions need a little temporal shape;
* model 400→48→24→1  (~20k params, ~82 KB float32);
* frame labels come from the mixture construction (see prepare_data.py);
* early stopping on validation F1, then the operating threshold is chosen
  on validation by an F1 sweep — the *threshold* is part of the model
  artifact (saved into its metadata, loaded by StreamingVAD).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import MLP, auc, prf, train  # noqa: E402
from scripts_utils import context_windows  # noqa: E402


def load_split(path: Path, K: int, ycol: str = "y"):
    """(F, y) → context-stacked (X, y). Window t covers frames [t .. t+K-1],
    labelled by its NEWEST frame — same convention as StreamingVAD."""
    z = np.load(path)
    F, y = z["F"], z[ycol]
    X, yw = context_windows(F, y, K)
    return X, yw.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--out", type=Path, default=Path("models/teensy-v1.npz"))
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--hidden", type=int, nargs=2, default=[48, 24])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data-suffix", default="", help="e.g. '.distill' for teacher labels")
    ap.add_argument("--ycol", default="y", choices=["y", "ysoft"],
                    help="y = hard 0/1 labels, ysoft = teacher probabilities")
    args = ap.parse_args()

    K = args.context
    Xtr, ytr = load_split(args.data / f"train{args.data_suffix}.npz", K, args.ycol)
    Xva, yva = load_split(args.data / f"val{args.data_suffix}.npz", K, args.ycol)
    print(f"train: {Xtr.shape}  speech {ytr.mean()*100:.1f}%   "
          f"val: {Xva.shape}  speech {yva.mean()*100:.1f}%")

    feat = LogMel(sr=8000)
    model = MLP(sizes=(Xtr.shape[1], *args.hidden, 1), seed=11)
    print(f"model: {model.sizes}  {model.n_params():,} params  "
          f"({model.n_params()*4/1024:.0f} KB float32)")

    t0 = time.time()
    train(model, Xtr, ytr, Xva, yva, epochs=args.epochs, bs=args.batch,
          lr=args.lr, patience=10)
    print(f"trained in {time.time()-t0:.1f}s")

    pv = model.probs(Xva)
    from teensyvad.model import best_threshold
    thr = best_threshold(pv, yva)
    prec, rec, f1, _ = prf(pv, yva, thr)
    print(f"val @thr={thr:.3f}:  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
          f"AUC={auc(pv, yva):.4f}")

    meta = dict(
        sr=feat.sr, n_mels=feat.n_mels, win_ms=feat.win_ms, hop_ms=feat.hop_ms,
        n_fft=feat.n_fft, fmin=float(feat.fmin), fmax=float(feat.fmax),
        deltas=True, context=K,
        thr_hi=round(float(thr), 4), thr_lo=round(float(0.6 * thr), 4),
        hangover_ms=250.0, on_frames=3,
        arch="mlp", hidden=args.hidden,
        val_f1=round(float(f1), 4), val_auc=round(float(auc(pv, yva)), 4),
        train_frames=len(ytr), trained_with="scripts/train.py",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out, extra_meta=meta)
    print(f"saved → {args.out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
