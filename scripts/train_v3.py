"""Train teensy-v3 on the scaled set — lazy windowing, float → PTQ → QAT.

    .venv/bin/python scripts/train_v3.py                # full run
    .venv/bin/python scripts/train_v3.py --stage qat    # resume at QAT

Why a custom trainer: data/prepared_v3 train has ~5–6M frames; the
context-stacked design matrix would be 6M×400 floats ≈ 9.6 GB — too big.
Instead windows are stacked per minibatch (F stays ~1 GB, each batch is
1.6 MB).  Float32 training, PTQ export, and QAT fine-tuning all share
the same lazy-window batch feeder.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensyvad.model import MLP, Adam, auc, bce_with_logits, prf  # noqa: E402
from teensyvad.quant import QuantizedMLP, qat_loss_and_grads  # noqa: E402

ALL_Q = {"W1": True, "W2": True, "W3": True}
HIDDEN = [48, 24]
K = 10


class LazyWindows:
    """On-demand context windows over a big feature array.

    Uses sliding_window_view (a zero-copy view) + row gathering, so a
    batch materialises only its own windows — never the full matrix.
    """

    def __init__(self, F: np.ndarray, y: np.ndarray, K: int,
                 *, memmap_cache_frames: int = 0):
        # Keep large training sets memory-mapped by default.  This avoids
        # duplicating a 2.8 GB float16 F.npy in RAM on developer machines.
        # Batch() materialises only its own K×40 window block as float32.
        # A small sequential warm cache can be requested by callers, but it
        # is deliberately opt-in instead of forcing an OOM-prone full copy.
        self.F = F if isinstance(F, np.memmap) else np.ascontiguousarray(F, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.K = K
        self._sw = np.lib.stride_tricks.sliding_window_view(self.F, K, axis=0)

    def __len__(self):
        return len(self.F) - self.K + 1

    def batch(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gather windows for arbitrary window indices; label = NEWEST frame
        (same convention as scripts_utils.context_windows)."""
        idx = np.asarray(idx)
        sel = self._sw[idx]                          # (B, dim, K) gather
        X = np.ascontiguousarray(sel.transpose(0, 2, 1)) \
            .reshape(len(idx), self.K * self.F.shape[1])
        if X.dtype != np.float32:                    # float16 store → fp32 compute
            X = X.astype(np.float32)
        return X, self.y[idx + self.K - 1]


def eval_lazy(m, data: LazyWindows, thr: float, bs: int = 65536):
    """Frame F1/AUC + mean prob — chunked to keep memory sane."""
    P = np.empty(len(data), dtype=np.float32)
    for s in range(0, len(data), bs):
        idx = np.arange(s, min(s + bs, len(data)))
        X, _ = data.batch(idx)
        P[idx] = m.probs(X)
    y = data.y[data.K - 1:]
    return dict(f1=prf(P, y, thr)[2], auc=auc(P, y), P=P, y=y)


def train_float(model, tr: LazyWindows, va: LazyWindows, *, epochs=30, bs=2048,
                lr=2e-3, patience=6, thr=0.5, seed=0, tag="float"):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    best = {"f1": -1, "auc": -1, "ep": -1,
            "params": {k: v.copy() for k, v in model.p.items()}}
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(tr))
        losses = []
        for s in range(0, len(order), bs):
            Xb, yb = tr.batch(order[s:s + bs])
            loss, grads = model.loss_and_grads(Xb, yb)
            opt.step(model.p, grads)
            losses.append(loss)
        ev = eval_lazy(model, va, thr, bs=131072)
        print(f"[{tag}] epoch {ep:3d}  loss {np.mean(losses):.4f}  "
              f"val_f1 {ev['f1']:.4f}  val_auc {ev['auc']:.4f}", flush=True)
        if ev["f1"] > best["f1"]:
            best = {"f1": ev["f1"], "auc": ev["auc"], "ep": ep,
                    "params": {k: v.copy() for k, v in model.p.items()}}
        elif ep - best["ep"] >= patience:
            print(f"[{tag}] early stop (best {best['f1']:.4f} @ {best['ep']})")
            break
    for k, v in best["params"].items():
        model.p[k] = v
    return model, best


def qat_finetune(model, tr: LazyWindows, va: LazyWindows, *, epochs=10, bs=2048,
                 lr=1e-3, patience=10, thr=0.5, seed=1, tag="qat"):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    best = {"f1": -1, "params": {k: v.copy() for k, v in model.p.items()}, "ep": -1}
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(tr))
        losses = []
        for s in range(0, len(order), bs):
            Xb, yb = tr.batch(order[s:s + bs])
            loss, grads = qat_loss_and_grads(model, Xb, yb, ALL_Q)
            opt.step(model.p, grads)
            losses.append(loss)
        qm = QuantizedMLP(list(model.sizes), ALL_Q).quantize_from(model)
        ev = eval_lazy(qm, va, thr, bs=131072)
        print(f"[{tag}] epoch {ep:3d}  loss {np.mean(losses):.4f}  "
              f"int8-val F1 {ev['f1']:.4f}", flush=True)
        if ev["f1"] > best["f1"]:
            best = {"f1": ev["f1"], "ep": ep,
                    "params": {k: v.copy() for k, v in model.p.items()}}
        elif ep - best["ep"] >= patience:
            print(f"[{tag}] early stop (best {best['f1']:.4f} @ {best['ep']})")
            break
    for k, v in best["params"].items():
        model.p[k] = v
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared_v3"))
    ap.add_argument("--out", type=Path, default=Path("models/teensy-v3.npz"))
    ap.add_argument("--out-qat", type=Path, default=Path("models/teensy-v3-qat.npz"))
    ap.add_argument("--stage", choices=["all", "float", "qat"], default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, nargs=2, default=HIDDEN)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="training batch size; reduce when memory constrained")
    args = ap.parse_args()

    t0 = time.time()
    # train from the scaled set; val/test stay on the v2 (dev-clean) splits
    # so model selection is comparable across versions and untouched by
    # train-clean-100 speakers.
    if (args.data / "F.npy").exists() and (args.data / "yteach.npy").exists():
        # v4 memmap mode: float16 features + teacher labels on disk
        Ftr = np.load(args.data / "F.npy", mmap_mode="r")
        ytr = np.asarray(np.load(args.data / "yteach.npy"), dtype=np.float32)
        tr = LazyWindows(Ftr, ytr, K)
        print(f"(memmap mode: F.npy {Ftr.shape} float16)")
    else:
        ztr = np.load(args.data / "train.distill.npz")
        tr = LazyWindows(ztr["F"], ztr["y"], K)
    zva = np.load(Path("data/prepared/val.distill.npz"))
    va = LazyWindows(zva["F"], zva["y"], K)
    print(f"train {len(tr):,} windows  val {len(va):,}  "
          f"({time.time()-t0:.0f}s to load)")

    # operating threshold from v2's calibrated point; recalibrated after
    thr = 0.5

    if args.stage in ("all", "float"):
        model = MLP(sizes=(K * tr.F.shape[1], *args.hidden, 1), seed=args.seed)
        # input stats from a 1M-frame chunk, float64 accumulation (works
        # for both float32 arrays and float16 memmaps)
        chunk = np.asarray(tr.F[:1_000_000], dtype=np.float64)
        mean40 = chunk.mean(0)
        std40 = np.maximum(chunk.std(0), 1e-3)
        del chunk
        model.in_mean = np.tile(mean40, K).astype(np.float32)
        model.in_std = np.tile(std40, K).astype(np.float32)
        # lazy training doesn't auto-fit stats (train() does); we just did.
        model, best = train_float(model, tr, va, epochs=args.epochs, bs=args.batch_size,
                                  thr=thr, seed=args.seed)
        meta = dict(sr=8000, n_mels=20, win_ms=25.0, hop_ms=10.0, n_fft=256,
                    fmin=80.0, fmax=3800.0, deltas=True, context=K,
                    thr_hi=0.5, thr_lo=0.3, hangover_ms=250.0, on_frames=3,
                    arch="mlp", hidden=args.hidden,
                    trained_with="scripts/train_v3.py", data=args.data.name,
                    val_f1=round(float(best["f1"]), 4),
                    val_auc=round(float(best["auc"]), 4))
        model.save(args.out, extra_meta=meta)
        print(f"saved → {args.out}  ({time.time()-t0:.0f}s)")

    if args.stage in ("all", "qat"):
        from teensyvad.quant import load_any
        src = args.out if args.out.exists() else None
        assert src is not None, "run float stage first"
        m = load_any(src)
        m = qat_finetune(m, tr, va, epochs=args.qat_epochs, bs=args.batch_size,
                         thr=thr, seed=24)
        qm = QuantizedMLP(list(m.sizes), ALL_Q).quantize_from(m)
        meta = dict(m.meta)
        qm.save(args.out_qat, extra_meta={**meta, "qat": True})
        print(f"saved → {args.out_qat}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
