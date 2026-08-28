"""TinyGRU — a minimal recurrent VAD core (the v7 experiment).

Design goal: give the family what the stateless MLP cannot have — memory.
A GRU carries state across the whole stream, replacing the fixed 100–250 ms
context window. Distilled from Silero like the rest of the family.

    x (B, T, 40)  →  GRU(H)  →  dense(H→24)  →  dense(24→1)  →  logit/frame

Everything is numpy (forward AND hand-written BPTT backward), consistent
with the teensyvad "no ML framework" ethos. Pure-python reference speed is
fine on Apple Accelerate sgemm for our scale.

Param count at H=96: 3 gates × (40·H + H·H + 2H) ≈ 39.7k + head 2.3k ≈ 42k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


class TinyGRU:
    """Single-layer GRU + 2-layer head, frame-level VAD."""

    def __init__(self, in_dim: int = 40, hidden: int = 96, head: int = 24,
                 seed: int = 7):
        self.in_dim, self.hidden, self.head = in_dim, hidden, head
        rng = np.random.default_rng(seed)

        def recurrent(std):                      # (H,H) recurrent blocks
            return rng.uniform(-std, std, size=(hidden, hidden)).astype(np.float32)

        def input_(std):                         # (in,H) input blocks
            return rng.uniform(-std, std, size=(in_dim, hidden)).astype(np.float32)

        def bias():                              # forget-gate bias trick on z
            return np.concatenate([np.full(1, 1.0, dtype=np.float32),
                                   np.zeros(hidden - 1, dtype=np.float32)])

        ru = 1.0 / np.sqrt(hidden)
        iu = 1.0 / np.sqrt(in_dim)
        self.W = {k: v for k, v in {
            "Wiz": input_(iu), "Wir": input_(iu), "Win": input_(iu),
            "Whz": recurrent(ru), "Whr": recurrent(ru), "Whn": recurrent(ru),
        }.items()}
        self.b = {"z": bias(), "r": np.zeros(hidden, np.float32),
                  "n": np.zeros(hidden, np.float32)}
        self.Wh = {"h1": (rng.uniform(-ru, ru, size=(hidden, head))).astype(np.float32),
                   "h2": (rng.uniform(-1.0, 1.0, size=(head, 1))).astype(np.float32)}
        self.bh = {"h1": np.zeros(head, np.float32), "h2": np.zeros(1, np.float32)}
        self.in_mean = np.zeros(in_dim, np.float32)
        self.in_std = np.ones(in_dim, np.float32)
        self.meta: dict = {}
        self._h: np.ndarray | None = None        # streaming state (B, H)

    # ------------------------------------------------------------ params --
    def arrays(self) -> dict[str, np.ndarray]:
        d = {f"W/{k}": v for k, v in self.W.items()}
        d.update({f"b/{k}": v for k, v in self.b.items()})
        d.update({f"Wh/{k}": v for k, v in self.Wh.items()})
        d.update({f"bh/{k}": v for k, v in self.bh.items()})
        return d

    def n_params(self) -> int:
        return int(sum(v.size for v in self.arrays().values()))

    # ---------------------------------------------------------- training --
    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.in_mean) / self.in_std

    def forward_chunk(self, X: np.ndarray, h0: np.ndarray):
        """X (B, T, in) → logits (B, T). Inference-only numpy forward;
        training happens in torch (scripts/train_rnn.py), which is the same
        convention as the family: numpy runtime, torch teacher/tooling."""
        B, T, _ = X.shape
        logits = np.empty((B, T), dtype=np.float32)
        h = h0
        for t in range(T):
            xt = X[:, t, :]
            zg = SIGMOID(xt @ self.W["Wiz"] + h @ self.W["Whz"] + self.b["z"])
            rg = SIGMOID(xt @ self.W["Wir"] + h @ self.W["Whr"] + self.b["r"])
            ng = np.tanh(xt @ self.W["Win"] + rg * (h @ self.W["Whn"]) + self.b["n"])
            h = (1.0 - zg) * ng + zg * h
            hid = np.maximum(h @ self.Wh["h1"] + self.bh["h1"], 0.0)
            logits[:, t] = (hid @ self.Wh["h2"] + self.bh["h2"]).ravel()
        return logits

    # --------------------------------------------------------- streaming --
    def reset_state(self, batch: int = 1) -> None:
        self._h = np.zeros((batch, self.hidden), dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """One frame (in,) or (B,in) → P(speech). State persists across calls."""
        batch = 1 if x.ndim == 1 else x.shape[0]
        if self._h is None or self._h.shape[0] != batch:
            self.reset_state(batch=batch)
        h = self._h
        zg = SIGMOID(x @ self.W["Wiz"] + h @ self.W["Whz"] + self.b["z"])
        rg = SIGMOID(x @ self.W["Wir"] + h @ self.W["Whr"] + self.b["r"])
        ng = np.tanh(x @ self.W["Win"] + rg * (h @ self.W["Whn"]) + self.b["n"])
        self._h = (1.0 - zg) * ng + zg * h
        hid = np.maximum(self._h @ self.Wh["h1"] + self.bh["h1"], 0.0)
        return SIGMOID((hid @ self.Wh["h2"] + self.bh["h2"]).ravel())

    def probs_seq(self, X: np.ndarray) -> np.ndarray:
        """Stateful pass over a sequence (N, 40) → (N,) probabilities."""
        self.reset_state(batch=1)
        out = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            out[i] = self.step(X[i])
        return out

    # -------------------------------------------------------- persistence --
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
            m = cls(in_dim=len(z["in_mean"]), hidden=int(meta["hidden"]),
                    head=int(meta.get("head", 24)))
            for k, v in m.W.items():
                m.W[k] = z[f"W/{k}"]
            for k, v in m.b.items():
                m.b[k] = z[f"b/{k}"]
            for k, v in m.Wh.items():
                m.Wh[k] = z[f"Wh/{k}"]
            for k, v in m.bh.items():
                m.bh[k] = z[f"bh/{k}"]
            m.in_mean = z["in_mean"]; m.in_std = z["in_std"]
            m.meta = meta
        return m
