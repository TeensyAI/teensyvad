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
