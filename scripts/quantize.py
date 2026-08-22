"""Quantize a trained model — full or selective — and measure everything.

    .venv/bin/python scripts/quantize.py --model models/teensy-v2.npz

Does, in order:
1. **Sensitivity analysis**: quantize one layer at a time, measure the
   frame-level AUC/F1 cost on the validation split.  This is what makes
   quantization *selective*: layers whose solo cost is tiny get int8,
   sensitive layers stay float32.
2. Builds two candidates: **full-int8** (all layers) and **selective**
   (layers under the tolerance), saves both as .npz (drop-in for
   StreamingVAD).
3. Reports accuracy (vs float32, vs teacher), model size, and measured
   numpy speed — honestly (see quant.py: numpy float32 BLAS usually
   beats int8-in-numpy on an M-series Mac; the win is size + ONNX export).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.model import auc, load_model, prf  # noqa: E402
from teensyvad.quant import QuantizedMLP, int8_bytes  # noqa: E402
from scripts_utils import context_windows  # noqa: E402

LAYERS = ["W1", "W2", "W3"]


def eval_model(m, X, y, thr):
    p = m.probs(X)
    _, _, f1, _ = prf(p, y, thr)
    return f1, auc(p, y), p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v2.npz"))
    ap.add_argument("--data", type=Path, default=Path("data/prepared/val.distill.npz"))
    ap.add_argument("--out-full", type=Path, default=None)
    ap.add_argument("--out-selective", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.003,
                    help="max acceptable AUC drop for a layer to be quantized")
    args = ap.parse_args()
    out_full = args.out_full or args.model.with_name(args.model.stem + "-int8.npz")
    out_sel = args.out_selective or args.model.with_name(args.model.stem + "-int8sel.npz")

    base = load_model(args.model)
    K = int(base.meta["context"])
    thr = float(base.meta.get("thr_hi", 0.5))
    z = np.load(args.data)
    X, _ = context_windows(z["F"], z["F"][:, 0], K)
    y = z["y"][K - 1:]           # teacher-hard labels
    print(f"model  : {args.model.name}  ({base.n_params():,} params)")
    print(f"eval on: {args.data.name}  {len(y):,} frames, thr={thr:.2f}")

    f1_0, auc_0, p0 = eval_model(base, X, y, thr)
    print(f"float32: F1 {f1_0:.4f}  AUC {auc_0:.4f}")

    # ---- 1) per-layer sensitivity ----------------------------------------
    print("\nlayer sensitivity (quantize one layer at a time):")
    print(f"  {'layer':<6}{'params':>8}  {'AUC':>7}  {'ΔAUC':>8}  {'ΔF1':>8}  decision")
    drops = {}
    for k in LAYERS:
        qm = QuantizedMLP(list(base.sizes), {kk: kk == k for kk in LAYERS}).quantize_from(base)
        f1q, aucq, _ = eval_model(qm, X, y, thr)
        d_auc, d_f1 = auc_0 - aucq, f1_0 - f1q
        drops[k] = d_auc
        ok = d_auc <= args.tol
        print(f"  {k:<6}{base.p[k].size:>8,}  {aucq:.4f}  {d_auc:+8.4f}  {d_f1:+8.4f}  "
              f"{'quantize' if ok else 'KEEP float32'}")

    # ---- 2) build candidates ----------------------------------------------
    results = {}
    for name, qmask, out in [
        ("full-int8 ", {k: True for k in LAYERS}, out_full),
        ("selective ", {k: drops[k] <= args.tol for k in LAYERS}, out_sel),
    ]:
        qm = QuantizedMLP(list(base.sizes), qmask).quantize_from(base)
        f1q, aucq, pq = eval_model(qm, X, y, thr)
        agree = float(np.mean((pq >= thr) == (p0 >= thr)))
        nbytes = int8_bytes(qm) + sum(qm.p[k].nbytes for k in LAYERS if not qmask[k])
        fp_bytes = sum(base.p[k].nbytes for k in LAYERS)
        results[name.strip()] = dict(path=str(out), f1=f1q, auc=aucq, agree=agree,
                                     bytes=nbytes, qmask=qmask)
        print(f"\n{name}: qmask={qmask}")
        print(f"  F1 {f1q:.4f} (Δ{f1q-f1_0:+.4f})   AUC {aucq:.4f} (Δ{aucq-auc_0:+.4f})   "
              f"decision agreement with float32 {agree*100:.2f}%")
        print(f"  weights: {nbytes/1024:.1f} KB int8-path vs {fp_bytes/1024:.1f} KB float32 "
              f"({fp_bytes/nbytes:.2f}× smaller)")
        meta_extra = {"quant_src": str(args.model)}
        qm.save(out, extra_meta=meta_extra)
        print(f"  saved → {out}")

    # ---- 3) honest numpy speed check --------------------------------------
    print("\nnumpy speed (single-frame, median of 2000):")
    import time
    X1 = X[:1]
    for tag, m in [("float32", base),
                   ("full-int8", QuantizedMLP(list(base.sizes), {k: True for k in LAYERS}).quantize_from(base)),
                   ("selective", QuantizedMLP(list(base.sizes), results["selective"]["qmask"]).quantize_from(base))]:
        for _ in range(200): m.probs(X1)
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter(); m.probs(X1); ts.append(time.perf_counter()-t0)
        print(f"  {tag:<9}: {np.median(ts)*1e6:6.2f} µs/frame")
    print("\n(note: pure numpy has no int8 BLAS — float32 gemm usually wins on speed here;")
    print(" the int8 win is model size + drop-in ONNX Runtime export.)")
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
