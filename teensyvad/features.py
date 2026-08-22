"""Log-mel feature extraction — the "cochlea" of teensyvad.

A raw waveform is ~8000 numbers per second and tells a neural net almost
nothing directly.  Every practical VAD first converts audio into a small
sequence of *frames*, each summarising "how much energy was in each
frequency band during the last 25 ms".  This module implements that in
three classic steps:

1. FRAME     chop the signal into overlapping windows
             (25 ms window, 10 ms hop → 100 frames/second)
2. SPECTRUM  FFT each window → energy per frequency bin
3. MEL BANK  squash bins into ~20 perceptually-spaced bands, take log

The result: a (T, n_mels) matrix — a tiny "fingerprint" per 10 ms.

Two normalisation tricks that matter for telephony:

* **log** compression → features shift *linearly* with gain, so loud and
  quiet speakers land near each other;
* **per-frame band-mean subtraction** (see `_normalise`) removes that
  linear shift entirely → the model sees only the *shape* of the spectrum,
  making it immune to line-level differences between calls.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# The mel scale
# --------------------------------------------------------------------------

def hz_to_mel(f):
    """Hz → mel.  Mel approximates human pitch perception: fine resolution
    at low frequencies, coarse at high ones — which is why VADs spend more
    bands below 1 kHz where speech lives."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 20,
                   fmin: float = 80.0, fmax: float | None = None):
    """Build a triangular filterbank matrix (n_mels, n_fft//2+1).

    Row i is a triangle that peaks at band i's centre frequency; multiplying
    the power spectrum by it sums the energy "under" that triangle.
    Each row is normalised to sum to 1 so bands are comparable.
    """
    if fmax is None:
        fmax = min(sr / 2 - 200.0, 8000.0)  # stay off the Nyquist edge
    if not 0 < fmin < fmax <= sr / 2:
        raise ValueError(f"bad band limits: {fmin=} {fmax=} for sr={sr}")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        left, cen, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - left) / max(cen - left, 1e-9)
        down = (right - fft_freqs) / max(right - cen, 1e-9)
        weights[i] = np.maximum(0.0, np.minimum(up, down))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32), hz_pts[1:-1].astype(np.float32)


# --------------------------------------------------------------------------
# Offline + streaming log-mel extraction
# --------------------------------------------------------------------------

class LogMel:
    """Turn mono float audio into normalised log-mel (+ delta) frames.

    Parameters default to telephony-friendly values at 8 kHz:
    20 mel bands over 80–3800 Hz, 25 ms window, 10 ms hop, n_fft 256.
    With `deltas=True` each frame is [mel shape (n_mels), Δmel (n_mels)].
    """

    def __init__(self, sr: int = 8000, n_mels: int = 20, win_ms: float = 25.0,
                 hop_ms: float = 10.0, n_fft: int = 256, fmin: float = 80.0,
                 fmax: float | None = None, deltas: bool = True,
                 floor: float = 1e-10):
        self.sr, self.n_mels, self.n_fft = sr, n_mels, n_fft
        self.win_ms, self.hop_ms = win_ms, hop_ms
        self.fmin, self.fmax, self.deltas, self.floor = fmin, fmax, deltas, floor
        if fmax is None:
            fmax = min(sr / 2 - 200.0, 8000.0)
            self.fmax = fmax

        self.frame_len = int(round(win_ms / 1000.0 * sr))
        self.hop_len = int(round(hop_ms / 1000.0 * sr))
        if self.frame_len > n_fft:
            raise ValueError("n_fft must be >= frame length")
        self.window = np.hanning(self.frame_len).astype(np.float32)
        self.filters, self.band_centers = mel_filterbank(
            sr, n_fft, n_mels, fmin, fmax)
        self.dim = n_mels * (2 if deltas else 1)

    # -- framing -----------------------------------------------------------

    def frame(self, x: np.ndarray) -> np.ndarray:
        """(N,) signal → (T, frame_len) overlapping frames (a *view*, cheap)."""
        x = np.asarray(x, dtype=np.float32)
        if len(x) < self.frame_len:
            return np.zeros((0, self.frame_len), dtype=np.float32)
        sw = np.lib.stride_tricks.sliding_window_view(x, self.frame_len)
        return sw[:: self.hop_len]

    # -- per-frame DSP -----------------------------------------------------

    def _frames_to_logmel(self, frames: np.ndarray) -> np.ndarray:
        """Windowed frames → per-frame normalised log-mel rows (T, n_mels)."""
        win = frames * self.window              # taper edges (Hann)
        spec = np.fft.rfft(win, n=self.n_fft, axis=1)   # zero-pad to n_fft
        power = (spec.real ** 2 + spec.imag ** 2)       # |X(f)|², per bin
        mel = power @ self.filters.T            # (T, n_mels) band energies
        v = np.log(mel + self.floor)            # log compression
        # Gain invariance: a louder line shifts every band by the same
        # constant in log space — subtracting the per-frame mean removes it.
        v -= v.mean(axis=1, keepdims=True)
        return v.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Whole clip at once → (T, dim) feature matrix."""
        frames = self.frame(x)
        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._frames_to_logmel(frames)
        return self._with_delta(v)

    def _with_delta(self, v: np.ndarray) -> np.ndarray:
        if not self.deltas:
            return v
        d = np.zeros_like(v)
        if len(v) > 1:
            d[1:] = v[1:] - v[:-1]              # first-order temporal delta
        return np.concatenate([v, d], axis=1)


class StreamingLogMel(LogMel):
    """Same features, one chunk at a time (for live audio).

    Keeps the last (frame_len - hop) samples between calls so frames stay
    perfectly aligned with the offline path — `feed()` output concatenated
    over chunks equals `LogMel.__call__` on the whole signal, exactly.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_v: np.ndarray | None = None

    def feed(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        self._buf = np.concatenate([self._buf, x]) if len(self._buf) else x.copy()
        rows: list[np.ndarray] = []
        while len(self._buf) >= self.frame_len:
            v = self._frames_to_logmel(self._buf[: self.frame_len][None, :])[0]
            if self._prev_v is None:
                d = np.zeros_like(v)
            else:
                d = v - self._prev_v
            rows.append(np.concatenate([v, d]) if self.deltas else v)
            self._prev_v = v
            self._buf = self._buf[self.hop_len:]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._prev_v = None
