"""A 3-layer MLP in plain numpy — with hand-written backpropagation.

No PyTorch, no TensorFlow.  The whole model is ~20k parameters (about
80 KB), trains in seconds on a laptop CPU, and scores one 10 ms frame in
a handful of microseconds.  If you can read this file, you understand
what the inference half of a production VAD actually computes.

Architecture::

    x (context-stacked log-mel, e.g. 400 dims)
      → Dense(48) + ReLU
      → Dense(24) + ReLU
      → Dense(1)            ← a logit; sigmoid gives P(speech)

Training uses binary cross-entropy on the logit (numerically stable
form) and Adam — both written out below on purpose, because "how does
the gradient flow" is half of understanding a neural net.

The weights live in a plain ``.npz`` (numpy zip) together with a JSON
metadata blob (feature config, context size, thresholds).  You can open
a trained model with ``numpy.load`` and inspect every number in it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

META_FORMAT = 1


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid (never exponentiates a large positive number).
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy given raw logits (stable form).

    Ravel first: a (n,1) logit column against an (n,) label vector would
    silently broadcast to (n,n) and quietly corrupt the loss.
    """
    z = np.asarray(z, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(loss.mean())


class MLP:
    """Fully-connected classifier trained with :func:`train` below."""

    def __init__(self, sizes=(400, 48, 24, 1), seed: int | None = None):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.p: dict[str, np.ndarray] = {}
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            # He initialisation for ReLU layers; small init for the head so
            # training starts near P=0.5 rather than saturated.
            scale = np.sqrt(2.0 / a) if i < len(self.sizes) - 2 else np.sqrt(1.0 / a)
            self.p[f"W{i+1}"] = rng.normal(0.0, scale, (a, b)).astype(np.float32)
            self.p[f"b{i+1}"] = np.zeros(b, dtype=np.float32)
        # Input standardisation stats (filled in by train()); applied at the
        # top of forward(), so saved models are self-contained.
        self.in_mean = np.zeros(self.sizes[0], dtype=np.float32)
        self.in_std = np.ones(self.sizes[0], dtype=np.float32)
        self.meta: dict = {"format": META_FORMAT, "hidden": self.sizes[1:-1]}

    # -- inference ---------------------------------------------------------

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.in_mean) / self.in_std

    def logits(self, x: np.ndarray, cache: tuple | None = None):
        x = self._norm(np.asarray(x, dtype=np.float32))
        h1 = np.maximum(x @ self.p["W1"] + self.p["b1"], 0.0)
        h2 = np.maximum(h1 @ self.p["W2"] + self.p["b2"], 0.0)
        z = h2 @ self.p["W3"] + self.p["b3"]
        if cache is not None:
            cache[0], cache[1], cache[2] = x, h1, h2
        return z

    def probs(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(x)).ravel()

    # -- learning ----------------------------------------------------------

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        """BCE loss + gradients for one batch (the backprop, written out)."""
        cache: list = [None, None, None]
        z = self.logits(x, cache=cache)
        xv, h1, h2 = cache
        p = _sigmoid(z)
        n = len(y)

        dz = (p - y.reshape(-1, 1).astype(np.float32)) / n   # dL/dlogit

        grads = {}
        grads["W3"] = h2.T @ dz
        grads["b3"] = dz.sum(axis=0)
        dh2 = dz @ self.p["W3"].T
        dh2[h2 <= 0] = 0.0                                   # ReLU gate
        grads["W2"] = h1.T @ dh2
        grads["b2"] = dh2.sum(axis=0)
        dh1 = dh2 @ self.p["W2"].T
        dh1[h1 <= 0] = 0.0
        grads["W1"] = xv.T @ dh1
        grads["b1"] = dh1.sum(axis=0)
        return bce_with_logits(z, y), grads

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path, extra_meta: dict | None = None) -> None:
        meta = dict(self.meta)
        if extra_meta:
            meta.update(extra_meta)
        arrays = {f"p/{k}": v for k, v in self.p.items()}
        arrays["in_mean"] = self.in_mean
        arrays["in_std"] = self.in_std
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        with np.load(str(path)) as z:
            meta = json.loads(str(z["meta"]))
            in_dim = len(z["in_mean"])
            hidden = [int(h) for h in meta.get("hidden", [48, 24])]
            m = cls([in_dim] + hidden + [1])
            m.in_mean = z["in_mean"].astype(np.float32)
            m.in_std = z["in_std"].astype(np.float32)
            for k in list(m.p):
                m.p[k] = z[f"p/{k}"].astype(np.float32)
            m.meta = meta
        return m


def load_model(path: str | Path) -> MLP:
    return MLP.load(path)


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) — Adaptive Moment Estimation.

    Keeps per-parameter running averages of the gradient (m) and its
    squared magnitude (v); the ratio nudges weights with an automatically
    scaled, sign-like step.  It's the default optimiser almost everywhere
    because it just works.
    """

    def __init__(self, params: dict[str, np.ndarray], lr: float = 2e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, self.betas, self.eps = lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray],
             grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.betas
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            m_hat = self.m[k] / (1 - b1 ** self.t)   # bias correction
            v_hat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------
# Training / evaluation helpers
# --------------------------------------------------------------------------

def prf(probs: np.ndarray, y: np.ndarray, thr: float = 0.5):
    """Precision / recall / F1 at a threshold."""
    pred = probs >= thr
    tp = int(np.sum(pred & (y > 0.5)))
    fp = int(np.sum(pred & (y < 0.5)))
    fn = int(np.sum(~pred & (y > 0.5)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, (tp, fp, fn)


def auc(probs: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC by the Mann–Whitney rank statistic (ties counted ½)."""
    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=np.float64)
    sp = np.asarray(probs)[order]
    i = 0
    while i < len(sp):  # average ranks for tied values
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    y = np.asarray(y, dtype=np.float64)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(probs: np.ndarray, y: np.ndarray, n: int = 201) -> float:
    """Sweep thresholds; return the one maximising F1."""
    lo, hi = np.percentile(probs, 1), np.percentile(probs, 99)
    grid = np.linspace(lo, hi, n)
    f1s = [prf(probs, y, t)[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def train(model: MLP, X: np.ndarray, y: np.ndarray,
          Xval=None, yval=None, *, epochs: int = 40, bs: int = 512,
          lr: float = 2e-3, patience: int = 8, verbose: bool = True,
          seed: int = 0):
    """Mini-batch Adam with early stopping on validation F1.

    Also fits the input standardisation stats (mean/std of each input
    dimension) from the training data and stores them on the model.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    model.in_mean = X.mean(axis=0).astype(np.float32)
    model.in_std = np.maximum(X.std(axis=0), 1e-3).astype(np.float32)

    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    best = {"f1": -1.0, "params": {k: v.copy() for k, v in model.p.items()},
            "epoch": -1}
    history = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        losses = []
        for s in range(0, len(idx), bs):
            b = idx[s: s + bs]
            loss, grads = model.loss_and_grads(X[b], y[b])
            opt.step(model.p, grads)
            losses.append(loss)
        tr_loss = float(np.mean(losses))
        entry = {"epoch": ep, "loss": tr_loss}
        if Xval is not None:
            pv = model.probs(Xval)
            prec, rec, f1, _ = prf(pv, yval)
            entry.update(val_loss=bce_with_logits(model.logits(Xval), yval),
                         val_f1=f1)
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": ep,
                        "params": {k: v.copy() for k, v in model.p.items()}}
            elif ep - best["epoch"] >= patience:
                if verbose:
                    print(f"early stop at epoch {ep} "
                          f"(best val F1 {best['f1']:.4f} @ {best['epoch']})")
                break
        history.append(entry)
        if verbose and (ep == 1 or ep % 5 == 0 or ep == len(history)):
            msg = f"epoch {ep:3d}  loss {tr_loss:.4f}"
            if Xval is not None:
                msg += f"  val_loss {entry['val_loss']:.4f}  val_f1 {f1:.4f}"
            print(msg)
    if Xval is not None:
        for k, v in best["params"].items():   # restore best checkpoint
            model.p[k] = v
    return history
