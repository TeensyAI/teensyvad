---
license: cc-by-nc-sa-4.0
pipeline_tag: voice-activity-detection
tags:
- voice-activity-detection
- vad
- telephony
- 8khz
- numpy
- onnx
- streaming
library_name: teensyvad
model-index:
- name: teensy-vad-1
  results:
  - task:
      type: voice-activity-detection
      name: Voice Activity Detection
    dataset:
      name: teensyvad synthetic test (unseen speakers + unseen noise types)
      type: synthetic-mixtures
    metrics:
    - type: f1
      value: 0.903
      name: frame F1
    - type: auc
      value: 0.889
      name: frame ROC-AUC
  - task:
      type: voice-activity-detection
    dataset:
      name: TEN VAD public set (30 real recordings)
      type: real-world
    metrics:
    - type: auc
      value: 0.848
      name: frame ROC-AUC
---

# teensy-vad-1 — a VAD small enough to understand

The first model of the **teensyvad** family: a complete voice activity
detector in **20,449 parameters (87 KB)**, written in pure numpy with
hand-written backprop — built to learn how VAD works and to run in an
Asterisk telephony stack.

**Telephony-native**: 8 kHz, 16-bit mono — exactly what a phone call is.
No resampling anywhere (PSTN / G.711 / Asterisk `slin` are all 8 kHz).

## Architecture

| | |
|---|---|
| Model | 3-layer MLP, 400 → 48 → 24 → 1 (ReLU, sigmoid head) |
| Parameters | 20,449 (87 KB float32) |
| Input | 10 frames × (20 log-mel + 20 Δ) = 400 dims (100 ms context) |
| Output | P(speech) per 10 ms frame |
| Framing | 25 ms window / 10 ms hop @ 8 kHz |
| Speed | ~14 µs/frame numpy single, 0.27 µs batched; 6.6 µs ONNX single |

### Exact feature specification (reimplementable)

1. Mono float32 8 kHz, frames of 200 samples (25 ms), hop 80 (10 ms).
2. Per frame: Hann window → zero-padded FFT to 256 → power spectrum.
3. 20 mel triangles over 80–3800 Hz (rows normalised to sum 1).
4. `log(mel + 1e-10)`, then **subtract per-frame mean over bands**
   (gain invariance — the model sees spectral *shape*, not level).
5. First-order delta (current − previous), concatenate → 40 dims/frame.
6. Stack 10 frames (oldest→newest, frame-major), standardise with the
   `in_mean` / `in_std` vectors stored in the `.npz`.

## What's in this repo

| file | what |
|---|---|
| `teensy-v1.npz` | float32 weights + norm stats + JSON metadata (thresholds, feature config) |
| `teensy-v1.onnx` | same model as ONNX (input normalisation folded in; raw 400-d features in, logit out) |
| `teensy-v1-int8.onnx` | dynamic int8 ONNX (22 KB) |

Minimal numpy inference:

```python
import numpy as np, json
z = np.load("teensy-v1.npz")
meta = json.loads(str(z["meta"]))            # thresholds + feature config
def score(x):                                 # x: standardised 400-dim window
    h1 = np.maximum(x @ z["p/W1"] + z["p/b1"], 0)
    h2 = np.maximum(h1 @ z["p/W2"] + z["p/b2"], 0)
    return 1 / (1 + np.exp(-(h2 @ z["p/W3"] + z["p/b3"])))   # P(speech)
```

For streaming (chunk-in → `speech_start`/`speech_end` events out, with
hysteresis + hangover) use the teensyvad source that accompanies this
model family.

## Training

* Speech: LibriSpeech `dev-clean` (1,200 utterances, speaker-disjoint splits)
* Noise: ESC-50 (fold-disjoint; human-vocal categories excluded)
* Mixing: SNR 0–20 dB, 40 % of clips round-tripped through G.711 µ-law
  (realistic telephone degradation)
* Labels: **construction ground truth** — a frame is speech iff the
  placed utterance covers its centre (utterance edges energy-trimmed)

## Measured quality

| benchmark | frame F1 | AUC |
|---|---|---|
| synthetic test (unseen speakers, unseen noise) | 0.903 | 0.889 |
| event-level streaming (120 clips) | 0.976 | — |
| TEN VAD public set (real recordings) | 0.850 | 0.848 |
| AMI SDM meetings (real rooms, manual labels) | 0.743 | 0.835 |

The energy-baseline comparison that motivated this model: on the same
streamed clips an adaptive energy VAD reaches event F1 0.535 (2.5×
over-triggering) vs 0.976 here.

## Limitations (honest)

* English read-speech + environmental noise only; music and multi-talker
  babble were **not** in training (see `teensy-vad-3` for those).
* Labels include intra-utterance pauses marked "speech" — boundaries are
  loose (onset ~430 ms early vs a Silero teacher). `teensy-vad-2` fixes
  this via distillation.
* Operating thresholds stored in metadata were tuned on synthetic
  validation; real-domain thresholds differ (see v3's profiles).

## License & data

Code: MIT. Weights: **CC BY-NC-SA 4.0** (training noise ESC-50 is
CC BY-NC-SA 3.0; speech LibriSpeech CC BY 4.0). Not for commercial
deployment without replacing the ESC-50-derived training data.
