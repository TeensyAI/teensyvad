"""TinyGRU — numpy streaming runtime for the teensy-v7/v8 recurrent VADs.

    m = TinyGRU.load("teensy-v7-gru96.npz")
    m.reset_state(1)
    for frame in mel_frames:            # (40,) per 10 ms @ 8 kHz
        p = m.step(frame)               # P(speech) — state persists

Supports stacked layers (v8: GRU-192 × 3 ≈ 594k params). Training runs in
torch (scripts/train_rnn.py); this runtime is verified frame-exact against
the torch model (max |Δp| < 1e-4 over full streams).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


class TinyGRU:
    """Stacked GRU (1–N layers) + 2-layer head, frame-level VAD.

    Per layer li: gates z/r/n with inputs [x (li==0) or h_{li-1}] and
    h_{li-1}; head h1→relu→h2 reads the LAST layer's output.
    """

    def __init__(self, in_dim: int = 40, hidden: int = 96, head: int = 24,
                 seed: int = 7, layers: int = 1):
        self.in_dim, self.hidden, self.head = in_dim, hidden, head
        self.layers = max(1, int(layers))
        rng = np.random.default_rng(seed)

        def recurrent(std):
            return rng.uniform(-std, std, size=(hidden, hidden)).astype(np.float32)

        def input_(std):
            return rng.uniform(-std, std, size=(in_dim, hidden)).astype(np.float32)

        def bias():
            return np.concatenate([np.full(1, 1.0, dtype=np.float32),
                                   np.zeros(hidden - 1, dtype=np.float32)])

        ru = 1.0 / np.sqrt(hidden)
        self.layers_W, self.layers_b = [], []
        for li in range(self.layers):
            src = in_dim if li == 0 else hidden
            iu = 1.0 / np.sqrt(src)
            self.layers_W.append({
                "Wiz": input_(iu), "Wir": input_(iu), "Win": input_(iu),
                "Whz": recurrent(ru), "Whr": recurrent(ru), "Whn": recurrent(ru),
            })
            self.layers_b.append({"z": bias(), "r": np.zeros(hidden, np.float32),
                                  "n": np.zeros(hidden, np.float32)})
        self.Wh = {"h1": (rng.uniform(-ru, ru, size=(hidden, head))).astype(np.float32),
                   "h2": (rng.uniform(-1.0, 1.0, size=(head, 1))).astype(np.float32)}
        self.bh = {"h1": np.zeros(head, np.float32), "h2": np.zeros(1, np.float32)}
        self.in_mean = np.zeros(in_dim, np.float32)
        self.in_std = np.ones(in_dim, np.float32)
        self.meta: dict = {}
        self._h: np.ndarray | None = None        # streaming state (L, B, H)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.arrays().values()))

    def reset_state(self, batch: int = 1) -> None:
        self._h = np.zeros((self.layers, batch, self.hidden), dtype=np.float32)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.in_mean) / self.in_std

    def step(self, x: np.ndarray) -> np.ndarray:
        """One frame (in,) or (B,in) → P(speech). State persists across calls."""
        batch = 1 if x.ndim == 1 else x.shape[0]
        if self._h is None or self._h.shape[1] != batch:
            self.reset_state(batch=batch)
        h = np.ascontiguousarray(self._norm(x).reshape(batch, -1), dtype=np.float32)
        for li in range(self.layers):
            Wl, bl = self.layers_W[li], self.layers_b[li]
            hp = self._h[li]
            zg = SIGMOID(h @ Wl["Wiz"] + hp @ Wl["Whz"] + bl["z"])
            rg = SIGMOID(h @ Wl["Wir"] + hp @ Wl["Whr"] + bl["r"])
            ng = np.tanh(h @ Wl["Win"] + rg * (hp @ Wl["Whn"]) + bl["n"])
            h = (1.0 - zg) * ng + zg * hp
            self._h[li] = h
        hid = np.maximum(h @ self.Wh["h1"] + self.bh["h1"], 0.0)
        return SIGMOID((hid @ self.Wh["h2"] + self.bh["h2"]).ravel())

    def probs_seq(self, X: np.ndarray) -> np.ndarray:
        """Stateful pass over a sequence (N, in) → (N,) probabilities."""
        self.reset_state(batch=1)
        out = np.empty(len(X), dtype=np.float64)
        Xn = self._norm(np.asarray(X, dtype=np.float32))
        for i in range(len(Xn)):
            out[i] = self.step(Xn[i])
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        d = {}
        for li, (Wl, bl) in enumerate(zip(self.layers_W, self.layers_b)):
            for k, v in Wl.items():
                d[f"gru/L{li}/{k}"] = v
            for k, v in bl.items():
                d[f"gru/L{li}/b{k}"] = v
        d.update({f"Wh/{k}": v for k, v in self.Wh.items()})
        d.update({f"bh/{k}": v for k, v in self.bh.items()})
        return d

    def save(self, path: str | Path, extra_meta: dict | None = None) -> None:
        meta = dict(self.meta)
        if extra_meta:
            meta.update(extra_meta)
        arrays = self.arrays()
        arrays["in_mean"] = self.in_mean
        arrays["in_std"] = self.in_std
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "TinyGRU":
        with np.load(str(path), allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            L = int(meta.get("layers", 1))
            m = cls(in_dim=len(z["in_mean"]), hidden=int(meta["hidden"]),
                    head=int(meta.get("head", 24)), layers=L)
            for li in range(L):
                Wl, bl = m.layers_W[li], m.layers_b[li]
                for k in Wl:
                    Wl[k] = z[f"gru/L{li}/{k}"]
                for k in bl:
                    bl[k] = z[f"gru/L{li}/b{k}"]
            for k, v in m.Wh.items():
                m.Wh[k] = z[f"Wh/{k}"]
            for k, v in m.bh.items():
                m.bh[k] = z[f"bh/{k}"]
            m.in_mean = z["in_mean"]; m.in_std = z["in_std"]
            m.meta = meta
        return m
