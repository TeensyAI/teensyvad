"""teensyvad — a tiny VAD you can read in one sitting.

The point of this package is that nothing is hidden:

    PCM audio ─▶ log-mel features ─▶ tiny MLP ─▶ P(speech) ─▶ smoothing ─▶ events

    * :mod:`teensyvad.audio`    — bytes/PCM/µ-law/WAV plumbing (the "ear")
    * :mod:`teensyvad.features` — log-mel spectrogram frames (the "cochlea")
    * :mod:`teensyvad.model`    — a 3-layer MLP with hand-written backprop
    * :mod:`teensyvad.streaming`— real-time chunk-in / event-out VAD
    * :mod:`teensyvad.energy_vad`— grandpa's energy baseline, for comparison
"""

from .audio import DEFAULT_SR
from .energy_vad import EnergyVAD
from .model import MLP, load_model
from .streaming import StreamingVAD

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SR",
    "EnergyVAD",
    "MLP",
    "StreamingVAD",
    "load_model",
    "__version__",
]
