"""The energy VAD — your grandpa's voice activity detector.

Rule: "speech is loud, silence is quiet."  On a clean recording this works
shockingly well.  On a noisy phone line (fan, street, café) the noise floor
rides far above true silence and the rule collapses — which is exactly the
failure mode that motivated statistical/neural VADs.

We keep it as a *baseline* with the same feed()/events interface as
:class:`teensyvad.streaming.StreamingVAD`, so the evaluation script can
compare them fairly.

How it works, per 10 ms frame:
  1. frame RMS → dBFS
  2. track a slow-adapting *noise floor* (only adapts while we believe it's
     quiet — otherwise speech would drag the floor up after itself)
  3. speech ⟺ level exceeds floor + margin, with hysteresis + hangover.
"""

from __future__ import annotations

import numpy as np

from .audio import pcm16_to_float
from .streaming import Hysteresis, VADEvent  # shared state machine


class EnergyVAD:
    def __init__(self, sr: int = 8000, *, win_ms: float = 25.0, hop_ms: float = 10.0,
                 margin_db: float = 9.0, hyst_db: float = 4.0,
                 hangover_ms: float = 300.0, on_frames: int = 3,
                 floor_db: float = -65.0, adapt: float = 0.02):
        self.sr = sr
        self.frame_len = int(round(win_ms / 1000 * sr))
        self.hop_len = int(round(hop_ms / 1000 * sr))
        self.margin_db = margin_db
        self.floor_db = floor_db
        self._floor0 = floor_db
        self._adapt = adapt
        self._buf = np.zeros(0, dtype=np.float32)
        self._frame_idx = 0
        self._warmup = 8          # frames of floor calibration before judging
        self.hop_s = self.hop_len / sr
        # Map "level above floor" into a score so we can reuse the same
        # hysteresis machine the neural VAD uses: score 1.0 = at margin.
        self._hyst = Hysteresis(thr_hi=1.0, thr_lo=hyst_db / margin_db,
                                on_frames=on_frames,
                                off_frames=max(1, int(round(hangover_ms / 1000 / self.hop_s))))
        self.last_db = -100.0

    # ------------------------------------------------------------------

    def feed(self, chunk: bytes | np.ndarray) -> list[VADEvent]:
        """Feed PCM16 bytes or float samples; returns speech events."""
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            x = pcm16_to_float(bytes(chunk))
        else:
            x = np.asarray(chunk, dtype=np.float32)
        self._buf = np.concatenate([self._buf, x]) if len(self._buf) else x.copy()

        events: list[VADEvent] = []
        while len(self._buf) >= self.frame_len:
            frame = self._buf[: self.frame_len]
            self._buf = self._buf[self.hop_len:]

            r = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            db = max(-100.0, 20.0 * np.log10(max(r, 1e-10)))
            self.last_db = db

            # Cold start: the first few frames calibrate the noise floor
            # (fast adaptation, no decisions).  Without this, a fixed
            # initial floor would instantly "detect" ordinary room noise.
            if self._warmup > 0:
                self._warmup -= 1
                rate = 0.5
                self.floor_db = (1 - rate) * self.floor_db + rate * db
                self.floor_db = float(np.clip(self.floor_db, -75.0, -20.0))
                self._frame_idx += 1
                continue

            score = (db - self.floor_db) / self.margin_db
            for kind, at in self._hyst.push(score):
                events.append(VADEvent(kind, at * self.hop_s))

            # Track the floor only on frames that look like noise (below
            # floor + margin) — loud frames are probably speech, and
            # letting them raise the floor would blind the detector.
            if db < self.floor_db + self.margin_db:
                self.floor_db = float(
                    np.clip((1 - self._adapt) * self.floor_db + self._adapt * db,
                            -75.0, -20.0))
            self._frame_idx += 1
        return events

    @property
    def in_speech(self) -> bool:
        return self._hyst.in_speech

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._hyst.reset()
        self._warmup = 8
        self.floor_db = self._floor0
