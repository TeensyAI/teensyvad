"""Audio I/O and low-level DSP helpers — the "ear plumbing" of teensyvad.

All internal audio is **mono float32 in [-1, 1)** at the project sample rate
(8 kHz by default).  Why 8 kHz?  That is the native rate of Asterisk's
`slin` format and of the classic telephone band (300–3400 Hz): if it works
here, it drops straight into a phone system with zero resampling.

Glossary
--------
PCM16   16-bit signed little-endian raw samples — what Asterisk AudioSocket
        streams to you (320 bytes per 20 ms frame at 8 kHz).
µ-law   G.711 companded 8-bit telephony codec — what your actual phone line
        almost certainly carries.  We use it as a *training augmentation* so
        the model has heard "telephone quality" audio before.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

DEFAULT_SR = 8000  # Asterisk `slin` / telephone bandwidth


# --------------------------------------------------------------------------
# PCM16 <-> bytes  (the format Asterisk AudioSocket hands you)
# --------------------------------------------------------------------------

def pcm16_to_float(data: bytes) -> np.ndarray:
    """Bytes of little-endian int16 PCM → float32 in [-1, 1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    """Float [-1, 1) → bytes of little-endian int16 PCM (clipped, dither-free)."""
    x = np.clip(x, -1.0, 1.0 - 2.0 / 32768.0)
    return (x * 32768.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------
# WAV files (stdlib `wave` — only uncompressed PCM; we convert FLAC/other
# formats to wav during dataset preparation)
# --------------------------------------------------------------------------

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav file → (mono float32 [-1,1), sample_rate).

    Multi-channel files are averaged down to mono.
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:  # unsigned 8-bit
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:  # int32
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width} bytes")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(x), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int = DEFAULT_SR) -> None:
    """Write mono float32 [-1,1) → 16-bit PCM wav."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0 - 2.0 / 32768.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32768.0).astype("<i2").tobytes())


def load_wav(path: str | Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """Read a wav of any rate, downmix to mono, resample to `sr`."""
    x, sr_in = read_wav(path)
    return resample_fft(x, sr_in, sr)


# --------------------------------------------------------------------------
# G.711 µ-law — the classic telephony codec, as an augmentation
# --------------------------------------------------------------------------
# µ-law squashes 16 bits into 8 by quantizing logarithmically: fine steps
# near silence, coarse steps near full scale.  Feeding the model µ-law'd
# audio during training is a cheap simulation of "this came over a phone
# line", including its subtle distortion.

_MU = 255.0
_BIAS = 0x84  # 132 — the µ-law bias (shifts small values off the floor)


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """Float [-1,1) → uint8 µ-law bytes (true G.711 bit layout)."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0 - 2.0 / 32768.0)
    s = (x * 32768.0).astype(np.int32)
    sign = (s >> 8) & 0x80
    mag = np.abs(s) + _BIAS               # add bias
    mag = np.minimum(mag, 32635)          # clip
    seg = mag >> 7                        # 0..255
    exp = np.where(seg > 0, np.floor(np.log2(np.maximum(seg, 1))).astype(np.int32), 0)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant)) & 0xFF


def mulaw_decode(u: np.ndarray) -> np.ndarray:
    """uint8 µ-law bytes → float32 [-1,1)."""
    u = np.asarray(u, dtype=np.uint8)
    b = (~u) & 0xFF                        # un-invert
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = ((b >> 4) & 0x07).astype(np.int32)
    mant = (b & 0x0F).astype(np.int32)
    s = (((mant << 3) + _BIAS) << exp) - _BIAS
    return (sign * s).astype(np.float32) / 32768.0


def telephony_roundtrip(x: np.ndarray) -> np.ndarray:
    """Simulate a phone line: encode + decode µ-law (8-bit, G.711)."""
    return mulaw_decode(mulaw_encode(x))


# --------------------------------------------------------------------------
# Resampling (offline, FFT-based)
# --------------------------------------------------------------------------

def resample_fft(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited resampling of a whole clip via FFT.

    `np.fft.irfft(X, n=new_len)` re-synthesizes the signal at a new length:
    growing it interpolates (sinc), shrinking it drops the highest bins —
    which is exactly the anti-alias filtering a good resampler must do.

    Assumes the clip is roughly stationary at its edges (fine for our
    multi-second dataset clips; not for tiny buffers — use a proper
    polyphase filter if you ever need streaming resampling).
    """
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32).copy()
    n_in = len(x)
    n_out = max(1, int(round(n_in * sr_out / sr_in)))
    X = np.fft.rfft(x)
    return np.fft.irfft(X, n=n_out).astype(np.float32)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a buffer (linear scale)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def dbfs(x: np.ndarray) -> float:
    """Level in dB relative to full scale (silence clamps to -100 dB)."""
    r = rms(x)
    return max(-100.0, 20.0 * np.log10(max(r, 1e-10)))
