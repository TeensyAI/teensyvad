"""Post-training quantization for the teensyvad MLP — 8-bit, measured.

Why: the float32 model is 87 KB and scores a frame in ~14 µs.  Int8
weights shrink it ~4× (relevant for embedding it somewhere tiny) and,
with *real* int8 kernels (ONNX Runtime), run faster.  Two honesty notes
baked into this module's design:

* **numpy `int8 @ int8` silently overflows** (accumulates in int8 and
  wraps!) — so the quantized matmul here forces int32 accumulation.
* **numpy has no int8 BLAS path** — float32 gemm via Accelerate/OpenBLAS
  is *faster* than our int8 path in pure numpy.  We therefore report
  measured numbers, never promised ones.

Scheme: dynamic activation quantization + per-output-channel weight
scales (the standard "dynamic quantization" used by ONNX Runtime):

    W_q = int8(W / s_w)        s_w[j] = max|W[:,j]| / 127      (per column)
    x_q = int8(x / s_x)        s_x[i] = max|x[i,:]| / 127      (per row)
    x @ W ≈ (x_q @ W_q) · outer(s_x, s_w)

"Selective" quantization quantizes only the layers whose accuracy cost
is small, leaving sensitive layers in float32 — see scripts/quantize.py,
which measures per-layer sensitivity and decides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import MLP, _sigmoid

Q_MIN, Q_MAX = -127, 127


def quantize_weight(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 (in, out) → (int8 weights, float32 per-column scales)."""
    W = np.asarray(W, dtype=np.float32)
    s = np.abs(W).max(axis=0) / Q_MAX                    # (out,)
    s = np.maximum(s, 1e-12).astype(np.float32)
    q = np.clip(np.rint(W / s), Q_MIN, Q_MAX).astype(np.int8)
    return q, s


def quantize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic per-row activation quantization → (int8, scales)."""
    x = np.asarray(x, dtype=np.float32)
    s = np.abs(x).max(axis=1, keepdims=True) / Q_MAX
    s = np.maximum(s, 1e-12).astype(np.float32)
    q = np.clip(np.rint(x / s), Q_MIN, Q_MAX).astype(np.int8)
    return q, np.ravel(s)


def qmatmul(x_q: np.ndarray, s_x: np.ndarray, W_q: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    """int8 @ int8 with FORCED int32 accumulation, rescaled to float32."""
    acc = np.einsum("ij,jk->ik", x_q.astype(np.int32), W_q.astype(np.int32))
    return (acc.astype(np.float32) * np.outer(s_x, s_w))


class QuantizedMLP(MLP):
    """An MLP whose dense layers run from int8 weights (mixed allowed).

    `qmask` names which weight matrices are quantized, e.g.
    ``{"W1": True, "W2": True, "W3": False}`` — W3 stays float32.
    Everything else (input normalisation, thresholds, metadata, the
    streaming stack above it) is untouched: a quantized .npz is a
    drop-in replacement for the float one.
    """

    def __init__(self, sizes, qmask: dict[str, bool], seed=None):
        super().__init__(sizes=sizes, seed=seed)
        self.qmask = dict(qmask)
        self.Wq: dict[str, np.ndarray] = {}
        self.sq: dict[str, np.ndarray] = {}

    # -- quantize / dequantize -------------------------------------------

    def quantize_from(self, m: MLP) -> "QuantizedMLP":
        """Adopt weights from a float32 model, quantizing masked layers."""
        self.p = {k: v.copy() for k, v in m.p.items()}
        self.in_mean, self.in_std = m.in_mean.copy(), m.in_std.copy()
        self.meta = dict(m.meta)
        self.meta["quantized"] = True
        self.meta["qmask"] = self.qmask
        for k, on in self.qmask.items():
            if on:
                q, s = quantize_weight(self.p[k])
                self.Wq[k], self.sq[k] = q, s
        return self

    def effective_weight(self, k: str) -> np.ndarray:
        """Dequantized float view of layer k (reconstructs p[k] on load)."""
        if k in self.Wq:
            return (self.Wq[k].astype(np.float32) * self.sq[k]).astype(np.float32)
        return self.p[k]

    # -- forward ------------------------------------------------------------

    def logits(self, x, cache=None):
        x = self._norm(np.asarray(x, dtype=np.float32))
        h = x
        for i in (1, 2, 3):
            Wk, bk = f"W{i}", f"b{i}"
            if Wk in self.Wq:
                hq, s = quantize_rows(h)
                h = qmatmul(hq, s, self.Wq[Wk], self.sq[Wk]) + self.p[bk]
            else:
                h = h @ self.p[Wk] + self.p[bk]
            if i < 3:
                h = np.maximum(h, 0.0)
        return h

    # -- persistence ----------------------------------------------------------

    def save(self, path, extra_meta=None):
        """Write the model; quantized layers are stored ONLY as int8+scales
        (float copies are dropped — that's the entire point of the file)."""
        meta = dict(self.meta)
        if extra_meta:
            meta.update(extra_meta)
        arrays = {f"p/{k}": v for k, v in self.p.items()
                  if not self.qmask.get(k, False)}
        for k, q in self.Wq.items():
            arrays[f"q/{k}"] = q
            arrays[f"q/{k}.scale"] = self.sq[k]
        arrays["in_mean"] = self.in_mean
        arrays["in_std"] = self.in_std
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path) -> "QuantizedMLP":
        with np.load(str(path)) as z:
            meta = json.loads(str(z["meta"]))
            in_dim = len(z["in_mean"])
            hidden = [int(h) for h in meta.get("hidden", [48, 24])]
            qmask = meta.get("qmask", {})
            m = cls([in_dim] + hidden + [1], qmask=qmask)
            m.in_mean = z["in_mean"].astype(np.float32)
            m.in_std = z["in_std"].astype(np.float32)
            for k in list(m.p):
                if qmask.get(k, False):
                    m.Wq[k] = z[f"q/{k}"]
                    m.sq[k] = z[f"q/{k}.scale"].astype(np.float32)
                    m.p[k] = m.effective_weight(k)   # dequantized float view
                else:
                    m.p[k] = z[f"p/{k}"].astype(np.float32)
            m.meta = meta
        return m


def load_any(path: str | Path) -> MLP:
    """Load either a float32 or quantized model file transparently."""
    with np.load(str(path)) as z:
        quantized = "quantized" in str(z["meta"])
    return QuantizedMLP.load(path) if quantized else MLP.load(path)


def int8_bytes(m: QuantizedMLP) -> int:
    return int(sum(q.nbytes for q in m.Wq.values()))


# --------------------------------------------------------------------------
# Quantization-aware training (QAT)
# --------------------------------------------------------------------------
# PTQ trains in float32 and rounds afterwards; QAT makes the *training*
# forward pass pretend to be int8 (fake-quantize weights AND activations
# with freshly estimated scales each step).  Gradients flow through the
# quantizer via the straight-through estimator: the backward pass uses the
# quantized values as if quantization were the identity.  The float master
# weights keep being updated by Adam, but every gradient is computed at a
# point the deployed int8 model can actually represent — so nothing is
# "surprised" by rounding at export time.

def fake_quant_weight(W: np.ndarray) -> np.ndarray:
    """W → round-to-grid → back to float (trainable simulation)."""
    q, s = quantize_weight(W)
    return q.astype(np.float32) * s


def fake_quant_rows(x: np.ndarray) -> np.ndarray:
    """Dynamic per-row activation simulation → float."""
    q, s = quantize_rows(x)
    return q.astype(np.float32) * s[:, None]


def qat_forward(m: MLP, x: np.ndarray, qmask: dict[str, bool]):
    """Forward that mirrors QuantizedMLP.logits exactly (fake-quant).
    Returns (logits, cache) where cache holds the tensors the backward
    pass needs — already at their quantized operating points."""
    x = m._norm(np.asarray(x, dtype=np.float32))
    cache = {}
    h = x
    for i in (1, 2, 3):
        Wk, bk = f"W{i}", f"b{i}"
        W = fake_quant_weight(m.p[Wk]) if qmask.get(Wk) else m.p[Wk]
        hin = fake_quant_rows(h) if qmask.get(Wk) else h
        pre = hin @ W + m.p[bk]
        act = np.maximum(pre, 0.0) if i < 3 else pre
        cache[f"h{i-1}_in"] = hin          # input actually used
        cache[f"W{i}"] = W                  # weight actually used
        cache[f"h{i}"] = act                # post-activation
        h = act
    return h, cache


def qat_loss_and_grads(m: MLP, x: np.ndarray, y: np.ndarray, qmask: dict[str, bool]):
    """BCE loss + STE gradients for one batch (mirrors loss_and_grads)."""
    from .model import bce_with_logits, _sigmoid

    z, c = qat_forward(m, x, qmask)
    p = _sigmoid(z)
    n = len(y)
    dz = (p - y.reshape(-1, 1).astype(np.float32)) / n     # dL/dlogit

    grads = {}
    dh = dz
    for i in (3, 2, 1):
        hin = c[f"h{i-1}_in"]
        W = c[f"W{i}"]
        grads[f"W{i}"] = hin.T @ dh                         # at quantised point
        grads[f"b{i}"] = dh.sum(axis=0)
        if i > 1:
            dh_prev = dh @ W.T                              # STE: through identity
            dh_prev[c[f"h{i-1}"] <= 0] = 0.0                # ReLU gate
            dh = dh_prev
    return bce_with_logits(z, y), grads


def qat_train(m: MLP, X: np.ndarray, y: np.ndarray, qmask: dict[str, bool],
              Xval=None, yval=None, *, epochs: int = 20, bs: int = 1024,
              lr: float = 1e-3, seed: int = 0, verbose: bool = True):
    """Fine-tune `m` (usually a converged float model) under int8 simulation."""
    from .model import Adam, prf

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(seed)
    opt = Adam(m.p, lr=lr)
    best = {"f1": -1.0, "params": {k: v.copy() for k, v in m.p.items()}}
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        losses = []
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            loss, grads = qat_loss_and_grads(m, X[b], y[b], qmask)
            opt.step(m.p, grads)
            losses.append(loss)
        msg = f"qat epoch {ep:3d}  loss {float(np.mean(losses)):.4f}"
        if Xval is not None:
            pv = QuantizedMLP(list(m.sizes), qmask).quantize_from(m).probs(Xval)
            f1 = prf(pv, yval)[2]
            msg += f"  int8-val F1 {f1:.4f}"
            if f1 > best["f1"]:
                best = {"f1": f1, "params": {k: v.copy() for k, v in m.p.items()}}
        if verbose:
            print(msg, flush=True)
    if Xval is not None:                                    # restore best
        for k, v in best["params"].items():
            m.p[k] = v
    return m
