# teensy-vad-3: A 20k-Parameter Distilled Voice Activity Detector Hardened with Real-Room Ambience and Multi-Talker Babble

## Abstract

We present teensy-vad-3, the third generation of the teensyvad family of
voice activity detectors (VADs). It keeps the family's readable 3-layer
MLP (20,449 parameters, 88 KB) while scaling training data 10× over v2
and hardening it against real-world conditions: three noise families —
ESC-50 environmental noise, synthetic 7-talker babble built from
disjoint LibriSpeech speakers, and real AMI room ambience — at SNR
−5…20 dB, with Silero-teacher labels and G.711 µ-law augmentation. On
human-labelled recordings it reaches TEN VAD set AUC 0.873 and AMI SDM
F1 0.886 (AUC 0.861) at AMI-dev-calibrated thresholds, beating WebRTC
VAD on every comparable metric while running in ~63 µs per 20 ms
telephony chunk in pure numpy. Silero VAD ranks best by AUC, but its
stock operating point misses 44 % of room speech; teensy-vad-3
operates best in rooms, per KB and per µs.

## 1. Introduction

Voice activity detection answers one question per 10 ms frame: is
somebody talking right now? Earlier teensyvad generations (v1
construction labels, v2 Silero-distilled) were trained on ~1M synthetic
frames and evaluated only on synthetic mixtures. teensy-vad-3 changes
two things: (i) **scaled, harder data** — ~30 h of mixtures / 10.7M
teacher-labelled frames (10× v2), with real-room ambience and 7-talker
babble added to the noise pool, and the SNR floor pushed to −5 dB; and
(ii) **evaluation on human-labelled real audio** (the TEN VAD public
set and AMI SDM meetings) rather than synthetic holdouts only.

## 2. Data

Speech is LibriSpeech `train-clean-100` (8,000 utterances, CC BY 4.0).
Noise combines three families: ESC-50 environmental noise
(CC BY-NC-SA 3.0, fold-disjoint), synthetic 7-talker babble (summed
disjoint LibriSpeech speakers), and real AMI room ambience
(non-speech stretches of distant-mic meeting audio — chairs,
keyboards, HVAC — taken only from the 3 meetings reserved for
calibration; the 8 evaluation meetings were never seen). Mixtures span
SNR −5…20 dB; 40 % of clips round-trip through G.711 µ-law (shifting
real-set numbers by < 0.5 %). Labels are Silero-teacher (hard).
We state honestly: **the ESC-50 share carries a non-commercial
license** (CC BY-NC-SA 3.0), which propagates to the published weights.

## 3. Method

Native 8 kHz mono, 25 ms frames / 10 ms hop. Features: 20 log-mel
filters over 80–3800 Hz with per-frame band-mean subtraction, plus
deltas (40 dims/frame); 10 frames (100 ms) of context are stacked into
a 400-dim vector. Classifier: a 400→48→24→1 MLP with ReLU (20,449
parameters), trained with hand-written backprop + Adam. Streaming uses
hysteresis (enter above a high threshold, exit below a lower one) and
hangover (~250 ms), with domain threshold profiles shipped in metadata
(`close_mic` thr_hi 0.45; `distant_room` thr_hi 0.10, AMI-calibrated).
Exports include QAT int8 (29 KB; int8 val F1 0.912) and ONNX
float32/int8. Capacity variants (40k/80k/100k) were trained on the same
frames; capacity pays only because data scaled: TEN AUC 0.873 (20k) →
0.877 (80k), saturating at 100k.

## 4. Results

Protocol: 10 ms frame grid on human-labelled audio; every detector's
operating point calibrated on AMI dev meetings; AUC on raw
probabilities; speed = median µs per 20 ms chunk, one Apple M2 Pro core.

| | teensy-v3 (20k) | teensy-v3-80k | Silero VAD | WebRTC VAD | Energy VAD |
|---|---|---|---|---|---|
| params | 20,449 | 80,373 | 1,774,000 | ~6k (C) | — |
| TEN VAD set — F1 (best thr*) | 0.894 | **0.894** | 0.938 | n/a | — |
| TEN VAD set — AUC | 0.873 | **0.877** | 0.952 | n/a | 0.670 |
| AMI SDM — F1 (calibrated) | **0.886** | 0.882 | 0.714 | 0.842 | 0.592 |
| AMI SDM — AUC | 0.861 | 0.861 | **0.894** | 0.760 | 0.658 |
| µs / 20 ms chunk | 63 | 64 | 89 | **2** | 7 |

\* tuned on that set — like-for-like with FlashVAD's published
F1 0.889 / AUC 0.882.

## 5. Limitations

English speech only; no music in training (music reads as "activity").
100 ms context is short — unvoiced fricatives in noise remain the
hardest frames (a GRU/TCN would help, deliberately out of scope). No
acoustic echo cancellation: far-end echo/crosstalk can double-trigger;
use an upstream echo canceller. At distant-room operating points all
systems show high false-alarm rates on overlapped meeting speech
(labels mark foreground speech only). Packet-loss concealment is not
modelled. Weights inherit ESC-50's NC license.

## 6. License

Code: MIT. Weights: **CC BY-NC-SA 4.0** as currently published, because
ESC-50 noise (CC BY-NC-SA 3.0) is in the training mix (LibriSpeech and
AMI are CC BY 4.0; Silero teacher MIT). Not for commercial deployment
without replacing the non-commercial training data; a commercial-safe
retrain on permissively licensed noise is planned.

## Reproducibility

All numbers reproduce from the project scripts: `prepare_data_v3.py`,
`noise_pools.py`, `train_v3.py`, `eval_realworld.py`,
`calibrate_realworld.py`, `compare_all.py`, `quantize.py`,
`export_onnx.py`. Model files are plain `.npz` inspectable with
`numpy.load`.
