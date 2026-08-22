"""Shared helpers for teensyvad scripts (kept dependency-free)."""

from __future__ import annotations

import numpy as np


def context_windows(F: np.ndarray, y: np.ndarray, K: int):
    """Stack K consecutive feature rows per sample, frame-major.

    Returns X (N-K+1, K·dim) and labels of each window's NEWEST frame.
    Row t of X is [frame t dims | frame t+1 dims | … | frame t+K-1 dims]
    — byte-for-byte the layout StreamingVAD feeds the MLP, so offline
    evaluation and live inference can never drift apart.
    """
    F = np.ascontiguousarray(F, dtype=np.float32)
    win = np.lib.stride_tricks.sliding_window_view(F, K, axis=0)  # (W, dim, K)
    X = np.ascontiguousarray(win.transpose(0, 2, 1)).reshape(len(F) - K + 1, K * F.shape[1])
    return X, np.asarray(y[K - 1:], dtype=np.float32)
