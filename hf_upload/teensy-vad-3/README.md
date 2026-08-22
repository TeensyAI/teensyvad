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
- distillation
library_name: teensyvad
base_model:
- snakers4/silero-vad
model-index:
- name: teensy-vad-3
  results:
  - task:
      type: voice-activity-detection
      name: Voice Activity Detection
    dataset:
      name: teensyvad synthetic val (Silero-teacher labels)
      type: synthetic-mixtures
    metrics:
    - type: f1
      value: 0.911
      name: frame F1
    - type: auc
      value: 0.929
      name: frame AUC
  - task:
      type: voice-activity-detection
    dataset:
      name: TEN VAD public set (30 real recordings)
      type: real-world
    metrics:
    - type: auc
      value: 0.873
      name: frame ROC-AUC
  - task:
      type: voice-activity-detection
    dataset:
      name: AMI SDM meetings (manual labels)
      type: real-world
    metrics:
    - type: f1
      value: 0.886
      name: frame F1
    - type: auc
      value: 0.861
      name: frame ROC-AUC
---

# teensy-vad-3 — scaled data, real-world hardened

The best model of the **teensyvad** family: **the same 20,449-parameter
architecture** as [v1](https://huggingface.co/Teensy/teensy-vad-1) and
[v2](https://huggingface.co/Teensy/teensy-vad-2), trained on 10× more
and much harder data — and evaluated on human-labelled real recordings.

**Headline**: on both real-world test sets this 87 KB student
**out-ranks its own Silero teacher** (and every other model in the
family) by ROC-AUC.

| real-world benchmark | v1 | v2 | **v3** | Silero (teacher) |
|---|---|---|---|---|
| TEN VAD public set — AUC | 0.848 | 0.868 | **0.873** | 0.863 |
| AMI SDM meetings — AUC | 0.835 | 0.848 | **0.861** | 0.772 |
| AMI SDM meetings — F1 | 0.743 | 0.880 | **0.886** | 0.714 |

(FlashVAD v0.1, for reference, publishes F1 0.889 / AUC 0.882 on the
TEN set — threshold-tuned on that set, as is our 0.894 best-threshold
number: parity within caveats, at 2.3× fewer parameters.)

## Training (what changed vs v2)

* **Speech**: 8,000 utterances from LibriSpeech `train-clean-100`
  (~30 h of mixtures, 10.7M teacher-labelled frames — 10× v2)
* **Noise, three families**:
  * ESC-50 environmental (as before, fold-disjoint)
  * **synthetic 7-talker babble** — summed disjoint LibriSpeech
    speakers; the classic hard case for VAD (background conversations)
  * **real AMI room ambience** — non-speech stretches of distant-mic
    meeting audio (chairs, keyboards, HVAC), taken **only** from the 3
    meetings reserved for calibration (the 8 evaluation meetings were
    never seen)
* **SNR −5 … 20 dB** (down from 0 dB floor)
* Labels: Silero teacher (as in v2), µ-law augmentation kept (G.711
  round-trip shifts real-set numbers by < 0.5 %)
* Lazy context-window training (`LazyWindows` — the full design matrix
  would be 17 GB; windows materialise per batch)

## What's in this repo

| file | what |
|---|---|
| `teensy-v3.npz` | float32 model — default |
| `teensy-v3-qat.npz` | QAT int8 version (28 KB; int8 val F1 0.912) |
| `teensy-v3.onnx` | float32 ONNX export |
| `teensy-v3-int8.onnx` | dynamic int8 ONNX (22 KB) |

Architecture and the exact reimplementable feature spec are identical to
[v1's card](https://huggingface.co/Teensy/teensy-vad-1): 8 kHz, 25/10 ms
framing, 20 log-mel + Δ with per-frame band-mean subtraction, 10-frame
context, 400→48→24→1 MLP.

## Domain threshold profiles (important)

Rankings transfer across domains; **operating points do not**. Best
threshold measured: ~0.45 close-mic/telephony, 0.10 distant-room,
0.85 synthetic-events. The `.npz` metadata therefore ships **profiles**:

```json
{"profiles": {
   "close_mic":    {"thr_hi": 0.45, "thr_lo": 0.27},   // default (telephony)
   "distant_room": {"thr_hi": 0.10, "thr_lo": 0.06}}}  // AMI-calibrated
```

`distant_room` was calibrated on 3 held-out AMI meetings; the 8
evaluation meetings were never used for any tuning.

## Limitations (honest)

* At distant-room operating points everyone's false-alarm rate is high
  on overlapped meeting speech (labels mark foreground speech only).
* English speech; no music in training (a known gap — music reads as
  "activity").
* 100 ms context is short: unvoiced fricatives in noise remain the
  hardest frames; a GRU/TCN would help (deliberately out of scope —
  this family stays a readable MLP).
* µ-law/PSTN robustness verified; packet-loss concealment not modelled.

## License & data

Code: MIT. Weights: **CC BY-NC-SA 4.0** (ESC-50 CC BY-NC-SA 3.0 noise
in training; LibriSpeech CC BY 4.0; AMI CC BY 4.0; Silero teacher MIT).
Not for commercial deployment without replacing non-commercial training
data.
