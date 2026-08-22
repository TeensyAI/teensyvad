"""StreamingVAD — chunk in, speech events out.

This is the class a telephony system actually talks to::

    vad = StreamingVAD("models/teensy-v1.npz")   # 8 kHz PCM16
    for chunk in audio_source:                    # e.g. 20 ms Asterisk frames
        for ev in vad.feed(chunk):                # bytes, any size
            print(f"{ev.t:8.3f}s  {ev.type}")

Pipeline per 10 ms frame::

    PCM → log-mel(+Δ) → stack last K frames (context) → MLP → P(speech)
        → hysteresis + hangover state machine → speech_start/speech_end

The last stage matters as much as the neural net: raw frame probabilities
flicker.  Hysteresis (separate enter/exit thresholds) plus hangover (keep
"speech" for a grace period after P drops) turn noisy per-frame scores
into clean, telephony-grade events — the same tricks WebRTC VAD and
commercial VADs use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import DEFAULT_SR, pcm16_to_float
from .features import LogMel, StreamingLogMel
from .model import MLP
from .quant import load_any as _load_any

load_model = _load_any   # accepts float32 AND quantized .npz files


@dataclass
class VADEvent:
    """A state transition, timestamped in seconds since stream start."""
    type: str   # "speech_start" | "speech_end"
    t: float

    def __repr__(self) -> str:
        return f"VADEvent({self.type}, t={self.t:.3f}s)"


def default_model_path() -> Path | None:
    """Where we look for a trained model: $TEENSYVAD_MODEL, then the repo's
    models/ directory, then one shipped inside the package."""
    env = os.environ.get("TEENSYVAD_MODEL")
    if env and Path(env).exists():
        return Path(env)
    here = Path(__file__).resolve()
    for cand in (here.parent.parent / "models",
                 here.parent / "models"):
        for name in ("teensy-v2.npz", "teensy-v1.npz"):
            p = cand / name
            if p.exists():
                return p
    return None


class Hysteresis:
    """Shared speech state machine (used by both VADs).

    * silence → speech: score ≥ thr_hi for on_frames consecutive frames;
    * speech → silence: score < thr_lo for off_frames consecutive frames
      ("hangover") — so brief dips inside an utterance don't chop it up.

    Events are reported at the *first* frame of the triggering streak,
    i.e. speech_start lands at the true onset, not on_frames later, and
    speech_end at the moment speech actually stopped, not after hangover.
    """

    def __init__(self, thr_hi: float, thr_lo: float, on_frames: int, off_frames: int):
        assert 0 <= thr_lo <= thr_hi <= 1
        self.thr_hi, self.thr_lo = thr_hi, thr_lo
        self.on_frames, self.off_frames = max(1, on_frames), max(1, off_frames)
        self.in_speech = False
        self._on_streak = 0
        self._on_start = 0
        self._off_streak = 0
        self._off_start = 0
        self._i = 0          # frames processed so far

    def push(self, score: float) -> list[tuple[str, int]]:
        """Feed one frame's score → zero or more (event, frame_index)."""
        events: list[tuple[str, int]] = []
        i = self._i
        if not self.in_speech:
            if score >= self.thr_hi:
                if self._on_streak == 0:
                    self._on_start = i
                self._on_streak += 1
                if self._on_streak >= self.on_frames:
                    self.in_speech = True
                    self._off_streak = 0
                    events.append(("speech_start", self._on_start))
            else:
                self._on_streak = 0
        else:
            if score < self.thr_lo:
                if self._off_streak == 0:
                    self._off_start = i
                self._off_streak += 1
                if self._off_streak >= self.off_frames:
                    self.in_speech = False
                    self._on_streak = 0
                    events.append(("speech_end", self._off_start))
            else:
                self._off_streak = 0
        self._i += 1
        return events

    def reset(self) -> None:
        self.in_speech = False
        self._on_streak = self._off_streak = 0
        self._i = 0


def hysteresis_events(scores, thr_hi, thr_lo, on_frames, off_frames):
    """Offline convenience wrapper around :class:`Hysteresis`."""
    h = Hysteresis(thr_hi, thr_lo, on_frames, off_frames)
    out: list[tuple[str, int]] = []
    for s in scores:
        out.extend(h.push(float(s)))   # push() already stamps true frame idx
    return h, out


class StreamingVAD:
    """Real-time VAD over raw PCM chunks (bytes = PCM16LE, or float arrays).

    All feature/model configuration is read from the model file's metadata,
    so a saved model is fully self-describing.
    """

    def __init__(self, model: MLP | str | Path | None = None, *,
                 thr_lo: float | None = None, thr_hi: float | None = None,
                 hangover_ms: float | None = None, on_frames: int = 3):
        if model is None:
            p = default_model_path()
            if p is None:
                raise RuntimeError(
                    "no trained model found — run `python scripts/train.py` "
                    "or pass a path: StreamingVAD('models/teensy-v1.npz')")
            model = p
        self.model = load_model(model) if not isinstance(model, MLP) else model
        m = self.model.meta

        self.sr = int(m.get("sr", DEFAULT_SR))
        self.context = int(m.get("context", 10))
        self.hop_s = float(m.get("hop_ms", 10.0)) / 1000.0
        feat_kw = dict(sr=self.sr, n_mels=int(m.get("n_mels", 20)),
                       win_ms=float(m.get("win_ms", 25.0)),
                       hop_ms=float(m.get("hop_ms", 10.0)),
                       n_fft=int(m.get("n_fft", 256)),
                       fmin=float(m.get("fmin", 80.0)),
                       fmax=float(m.get("fmax", 3800.0)) if m.get("fmax") else None,
                       deltas=bool(m.get("deltas", True)))
        self._feats = StreamingLogMel(**feat_kw)
        self._feat_dim = self._feats.dim

        # Context history starts as zeros = "assume silence before start".
        self._hist = np.zeros((self.context, self._feat_dim), dtype=np.float32)

        self.thr_lo = float(thr_lo if thr_lo is not None else m.get("thr_lo", 0.40))
        self.thr_hi = float(thr_hi if thr_hi is not None else m.get("thr_hi", 0.60))
        hangover_ms = hangover_ms if hangover_ms is not None else float(m.get("hangover_ms", 250.0))
        off_frames = max(1, int(round(hangover_ms / 1000.0 / self.hop_s)))
        self._hyst = Hysteresis(self.thr_hi, self.thr_lo, on_frames, off_frames)

        self.prob_history: list[float] = []
        self.last_prob = 0.0
        self._speech_seconds = 0.0
        self._last_speech_frame = None

    # ------------------------------------------------------------------

    def feed(self, chunk: bytes | bytearray | memoryview | np.ndarray) -> list[VADEvent]:
        """Feed one chunk of audio; returns events triggered by this chunk."""
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            x = pcm16_to_float(bytes(chunk))
        else:
            x = np.asarray(chunk, dtype=np.float32)

        frames = self._feats.feed(x)          # (n_new, feat_dim)
        events: list[VADEvent] = []
        if len(frames) == 0:
            return events

        # Stack context windows for all new frames in one vectorised pass:
        # rows of `hist` carry older frames; each row of X is one inference.
        joined = np.concatenate([self._hist, frames], axis=0)   # (K-1+n, dim)
        K = self.context
        X = np.lib.stride_tricks.sliding_window_view(joined, K, axis=0)  # (W, dim, K)
        X = X[joined.shape[0] - K - len(frames) + 1:]                    # last n windows
        # (n, dim, K) → (n, K, dim): flatten frame-major so row t is
        # [frame t-K+1 dims | ... | frame t dims], matching the offline path.
        X = np.ascontiguousarray(X.transpose(0, 2, 1)).reshape(len(frames), K * self._feat_dim)
        probs = self.model.probs(X)

        self._hist = joined[-K:].copy()
        for p, at in [(float(pv), i) for i, pv in enumerate(probs)]:
            self.last_prob = p
            self.prob_history.append(p)
            base = self._hyst._i                    # frame index of this score
            for kind, frame_idx in self._hyst.push(p):
                events.append(VADEvent(kind, frame_idx * self.hop_s))
            if self._hyst.in_speech:
                self._speech_seconds += self.hop_s
                self._last_speech_frame = base
        return events

    # ------------------------------------------------------------------

    def flush(self) -> list[VADEvent]:
        """Close any open speech segment at the current position.

        Call at end-of-stream (hangup, EOF).  Without it, speech that is
        still ongoing when audio ends is never reported as a segment.
        """
        if not self._hyst.in_speech:
            return []
        t = self._hyst._i * self.hop_s
        # force the state machine closed
        self._hyst.in_speech = False
        self._hyst._off_streak = 0
        return [VADEvent("speech_end", t)]

    @property
    def in_speech(self) -> bool:
        return self._hyst.in_speech

    @property
    def speech_seconds(self) -> float:
        """Total time spent in the speech state (roughly, time talking)."""
        return self._speech_seconds

    def reset(self) -> None:
        """Start over (new call)."""
        self._feats.reset()
        self._hist = np.zeros((self.context, self._feat_dim), dtype=np.float32)
        self._hyst.reset()
        self.prob_history.clear()
        self.last_prob = 0.0
        self._speech_seconds = 0.0
