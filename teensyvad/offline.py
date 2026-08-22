"""OfflineVAD — whole-file VAD with fsmn-vad-style ergonomics.

    from teensyvad import OfflineVAD

    vad = OfflineVAD("Teensy/teensy-vad-v4")        # auto-downloads from HF
    segments = vad.segments("long_audio.wav")       # [[start_ms, end_ms], ...]

`segments()` mirrors funasr/fsmn-vad's return convention exactly (a list
of [start_ms, end_ms] pairs) so it drops into the same pipelines.  Under
the hood it is the same StreamingVAD as the telephony path — offline mode
simply feeds the whole file in 20 ms chunks and flushes at EOF, so
offline and streaming decisions are identical by construction.

Model resolution order:
  1. an existing local file path (``.npz``),
  2. a Hugging Face repo id (``Teensy/teensy-vad-v4``) — downloaded via
     :mod:`huggingface_hub` if installed (only optional dependency),
  3. local auto-discovery (``models/`` next to the package).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_wav
from .streaming import StreamingVAD, default_model_path

# default model file per HF repo id
DEFAULT_HF_FILES = {
    "Teensy/teensy-vad-1": "teensy-v1.npz",
    "Teensy/teensy-vad-2": "teensy-v2.npz",
    "Teensy/teensy-vad-3": "teensy-v3.npz",
    "Teensy/teensy-vad-v4": "teensy-v4.npz",
}
DEFAULT_MODEL = "Teensy/teensy-vad-v4"


def _resolve_model(model: str, model_file: str | None) -> str | Path:
    """Local path → HF repo id → local discovery."""
    p = Path(model)
    if p.exists():
        return p
    if "/" in model and not p.suffix:            # looks like an HF repo id
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                f"'{model}' looks like a Hugging Face repo but "
                "huggingface_hub is not installed — "
                "pip install huggingface_hub  (or pass a local .npz path)"
            ) from e
        fname = model_file or DEFAULT_HF_FILES.get(model, "teensy-v4.npz")
        return hf_hub_download(repo_id=model, filename=fname)
    local = default_model_path()
    if local is not None:
        return local
    raise RuntimeError(f"cannot resolve model '{model}' — pass a .npz path")


class OfflineVAD:
    """Whole-file voice activity detection.

    Parameters
    ----------
    model : str
        Local ``.npz`` path, HF repo id (default ``Teensy/teensy-vad-v4``),
        or "" to use local auto-discovery.
    model_file : str, optional
        Override the .npz filename inside an HF repo (e.g.
        ``"teensy-v4-80k.npz"``).
    """

    def __init__(self, model: str = DEFAULT_MODEL, model_file: str | None = None):
        path = _resolve_model(model, model_file)
        self._vad = StreamingVAD(path)
        self.sr = self._vad.sr

    # ------------------------------------------------------------------

    def _events(self, wav) -> list:
        x = load_wav(wav, sr=self.sr)             # any rate → 8 kHz mono
        chunk = self.sr // 50                     # 20 ms
        events = []
        for i in range(0, len(x), chunk):
            events += self._vad.feed(x[i:i + chunk])
        events += self._vad.flush()               # close segment open at EOF
        self._vad.reset()
        return events

    def segments(self, wav) -> list[list[int]]:
        """Speech segments as ``[[start_ms, end_ms], ...]`` — the same
        convention as funasr/fsmn-vad, ready for ASR pipelines."""
        segs, start = [], None
        for ev in self._events(wav):
            if ev.type == "speech_start":
                start = ev.t
            elif ev.type == "speech_end" and start is not None:
                segs.append([int(round(start * 1000)), int(round(ev.t * 1000))])
                start = None
        return segs

    def probabilities(self, wav) -> np.ndarray:
        """Per-10 ms P(speech) trajectory (for custom thresholding)."""
        x = load_wav(wav, sr=self.sr)
        chunk = self.sr // 50
        for i in range(0, len(x), chunk):
            self._vad.feed(x[i:i + chunk])
        p = np.array(self._vad.prob_history, dtype=np.float32)
        self._vad.reset()
        return p

    def slice(self, wav, start_ms: int, end_ms: int) -> np.ndarray:
        """Audio between two timestamps as float32 @ 8 kHz (for ASR input)."""
        x = load_wav(wav, sr=self.sr)
        return x[int(start_ms) * self.sr // 1000: int(end_ms) * self.sr // 1000]

    def speech_ratio(self, wav) -> float:
        """Fraction of the file detected as speech (0..1)."""
        segs = self.segments(wav)
        x = load_wav(wav, sr=self.sr)
        dur_ms = len(x) / self.sr * 1000
        speech_ms = sum(e - s for s, e in segs)
        return speech_ms / max(dur_ms, 1.0)
