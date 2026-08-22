"""PTQ vs QAT vs "train wide, quantize down" — the quantization bake-off.

    .venv/bin/python scripts/qat_bakeoff.py

Contenders (all 8 kHz, all same data — distilled teacher labels):
  v2-float    20k params, float32 (accuracy reference)
  v2-ptq      trained float32, quantized after    (train high → round)
  v2-qat      float32 warm-start, fine-tuned under int8 simulation
  wide-float  2× wider MLP (44k params), float32
  wide-ptq    wide trained float32, quantized after
  wide-qat    wide fine-tuned under int8 simulation

Prints accuracy (vs Silero labels, test split), size, and measured numpy
speed — then writes the winner notes.  Models saved as models/*.npz.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.model import MLP, auc, load_model, prf, train  # noqa: E402
from teensyvad.quant import QuantizedMLP, qat_train  # noqa: E402
from scripts_utils import context_windows  # noqa: E402

ALL_Q = {"W1": True, "W2": True, "W3": True}
WIDE = [96, 48]


def load_xy(path: Path, K: int, ycol: str = "y"):
    z = np.load(path)
    X, _ = context_windows(z["F"], z["F"][:, 0], K)
    return X.astype(np.float32), z[ycol][K - 1:].astype(np.float32)


def bench(fn, n=1500, warmup=150):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qat-epochs", type=int, default=12)
    ap.add_argument("--wide-epochs", type=int, default=40)
    args = ap.parse_args()

    models_dir = Path("models")
    data = Path("data/prepared")

    base_f = load_model(models_dir / "teensy-v2.npz")
    K = int(base_f.meta["context"])
    thr = float(base_f.meta.get("thr_hi", 0.5))
    Xtr, ytr = load_xy(data / "train.distill.npz", K)
    Xva, yva = load_xy(data / "val.distill.npz", K)
    Xte, yte = load_xy(data / "test.distill.npz", K)
    print(f"data: train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}  thr {thr:.2f}")

    cands: list[tuple[str, object]] = [("v2-float", base_f)]

    # ---- v2 PTQ --------------------------------------------------------
    ptq = QuantizedMLP(list(base_f.sizes), ALL_Q).quantize_from(base_f)
    ptq.save(models_dir / "teensy-v2-int8.npz")
    cands.append(("v2-ptq", ptq))

    # ---- v2 QAT (warm-start from the float model) ------------------------
    q = models_dir / "teensy-v2-qat.npz"
    if q.exists():
        cands.append(("v2-qat", load_model(q)))
    else:
        m = MLP(list(base_f.sizes))
        m.p = {k: v.copy() for k, v in base_f.p.items()}
        m.in_mean, m.in_std = base_f.in_mean.copy(), base_f.in_std.copy()
        m.meta = dict(base_f.meta)
        print("\n[v2-qat] fine-tuning under int8 simulation …")
        qat_train(m, Xtr, ytr, ALL_Q, Xva, yva,
                  epochs=args.qat_epochs, lr=1e-3)
        mq = QuantizedMLP(list(m.sizes), ALL_Q).quantize_from(m)
        mq.save(q)
        cands.append(("v2-qat", mq))

    # ---- wide model (train high capacity, then quantize) -----------------
    wf = models_dir / "teensy-v2-wide.npz"
    if wf.exists():
        wide = load_model(wf)
    else:
        print(f"\n[wide] training float32 {Xtr.shape[1]}→{WIDE[0]}→{WIDE[1]}→1 …")
        wide = MLP(sizes=(Xtr.shape[1], *WIDE, 1), seed=13)
        train(wide, Xtr, ytr, Xva, yva, epochs=args.wide_epochs,
              bs=1024, lr=2e-3, patience=10)
        wide.meta.update(base_f.meta)          # inherit feature/threshold config
        wide.meta["hidden"] = WIDE
        wide.save(wf)
    cands.append(("wide-float", wide))

    wptq = QuantizedMLP(list(wide.sizes), ALL_Q).quantize_from(wide)
    wptq.save(models_dir / "teensy-v2-wide-int8.npz")
    cands.append(("wide-ptq", wptq))

    wq = models_dir / "teensy-v2-wide-qat.npz"
    if wq.exists():
        cands.append(("wide-qat", load_model(wq)))
    else:
        m = MLP(list(wide.sizes))
        m.p = {k: v.copy() for k, v in wide.p.items()}
        m.in_mean, m.in_std = wide.in_mean.copy(), wide.in_std.copy()
        m.meta = dict(wide.meta)
        print("\n[wide-qat] fine-tuning under int8 simulation …")
        qat_train(m, Xtr, ytr, ALL_Q, Xva, yva,
                  epochs=args.qat_epochs, lr=1e-3)
        mwq = QuantizedMLP(list(m.sizes), ALL_Q).quantize_from(m)
        mwq.save(wq)
        cands.append(("wide-qat", mwq))

    # ---- measure ----------------------------------------------------------
    p_ref = base_f.probs(Xte)
    X1 = Xte[:1]
    Xb = Xte[:20000]

    print("\n" + "=" * 96)
    print(f"{'model':<11}{'params':>8}{'KB':>7}{'F1':>8}{'AUC':>8}"
          f"{'agree%':>8}{'µs single':>11}{'µs/1k batch':>13}")
    print("=" * 96)
    rows = {}
    for name, m in cands:
        p = m.probs(Xte)
        f1 = prf(p, yte, thr)[2]
        a = auc(p, yte)
        agree = float(np.mean((p >= thr) == (p_ref >= thr))) * 100
        kb = _kb_of(m, name, models_dir)
        t1 = bench(lambda: m.probs(X1)) * 1e6
        tb = bench(lambda: m.probs(Xb), n=25, warmup=3) / len(Xb) * 1e6
        rows[name] = (f1, a, kb)
        print(f"{name:<11}{m.n_params():>8,}{kb:>7.1f}{f1:>8.4f}{a:>8.4f}"
              f"{agree:>8.2f}{t1:>11.2f}{tb:>13.3f}")

    print("=" * 96)
    print("(F1/AUC vs Silero teacher labels, test split; agree% = decision")
    print(" match with v2-float; speed = pure numpy, this machine)")
    best_acc = max(rows, key=lambda k: rows[k][0])
    best_small = min(rows, key=lambda k: rows[k][2])
    print(f"\nmost accurate : {best_acc}  (F1 {rows[best_acc][0]:.4f})")
    print(f"smallest      : {best_small}  ({rows[best_small][2]:.1f} KB)")


def _kb_of(m, name: str, models_dir: Path) -> float:
    file = {
        "v2-float": "teensy-v2.npz", "v2-ptq": "teensy-v2-int8.npz",
        "v2-qat": "teensy-v2-qat.npz", "wide-float": "teensy-v2-wide.npz",
        "wide-ptq": "teensy-v2-wide-int8.npz", "wide-qat": "teensy-v2-wide-qat.npz",
    }[name]
    p = models_dir / file
    return p.stat().st_size / 1024 if p.exists() else float("nan")


if __name__ == "__main__":
    main()
