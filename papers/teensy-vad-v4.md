# teensy-vad-v4: The 100-Hour Generation of a 20k-Parameter Voice Activity Detector

## Abstract

We present teensy-vad-v4, the fourth generation of the teensyvad
family: the same readable 3-layer MLP, now trained on the full ~100 h
of LibriSpeech `train-clean-100` (all 28,539 utterances — 3.5× the v3
dataset, 37.9M Silero-teacher-labelled frames), with the complete
20k/40k/80k/100k capacity sweep and improved quantization-aware
training. On human-labelled audio, v4-80k reaches TEN VAD set AUC 0.880
and AMI SDM AUC 0.862 (F1 0.880 at calibrated thresholds) — essentially
matching FlashVAD's published AUC (0.882) at 2.3× fewer parameters —
while the 20k default runs in ~63 µs per 20 ms telephony chunk in pure
numpy. Int8 QAT variants are the family's best per-KB artifacts: the
20k QAT holds TEN AUC 0.876 in a 28 KB file.

## 1. Introduction

The v3 experiment showed that capacity scales only with data: at 1M
training frames, growing parameters 20k→100k was flat; at 10.7M frames
the sweet spot moved to ~80k. teensy-vad-v4 completes the curve at
37.9M frames — and capacity keeps paying to the largest size tested
(val F1 0.9137 at 20k → 0.9205 at 100k). Two further contributions
distinguish v4: (i) **improved training and QAT** — memmap/float16
plumbing for the 60 GB-scale design matrix, and a full QAT variant
ladder fine-tuned with int8 simulated in the forward pass
(straight-through-estimator gradients); and (ii) **best-AUC variants**
— v4-80k is the family's real-world AUC champion, and several QAT
variants beat their float counterparts on real-set AUC.

## 2. Data

Speech: all of LibriSpeech `train-clean-100` (CC BY 4.0; 37,936,872
verified training windows). Noise families, unchanged from v3:
ESC-50 environmental noise, synthetic 7-talker babble (disjoint
LibriSpeech speakers), and real AMI room ambience from the calibration
meetings only; SNR −5…20 dB, with G.711 µ-law augmentation. Labels are
Silero-teacher (hard). As in v3 we state honestly: **the ESC-50 share
carries a non-commercial license** (CC BY-NC-SA 3.0), which propagates
to the published weights.

## 3. Method

Identical architecture to the whole family: 8 kHz mono, 25 ms frames /
10 ms hop; 20 log-mel filters over 80–3800 Hz with per-frame band-mean
subtraction, plus deltas (40 dims/frame); 10-frame (100 ms) context
stacked into a 400-dim vector; a 400→48→24→1 ReLU MLP (20,449
parameters at the default size; 39,609 / 80,373 / 99,593 for the 40k /
80k / 100k variants). Streaming uses hysteresis and hangover with
domain threshold profiles (`close_mic` thr_hi 0.45 default;
`distant_room` thr_hi 0.10, AMI-calibrated). Exports: float `.npz`,
QAT int8 `.npz` (20k/40k/80k/100k), and ONNX float32/int8.

## 4. Results

Shared protocol (human-labelled audio, AMI-dev-calibrated operating
points, AUC on raw probabilities):

| | v4 (20k) | v4-40k | **v4-80k** | v4-100k | v4-qat (int8) | v3-80k | Silero | WebRTC | Energy |
|---|---|---|---|---|---|---|---|---|---|
| params | 20,449 | 39,609 | 80,373 | 99,593 | 20,449 | 80,373 | 1,774,000 | ~6k | — |
| TEN VAD set — F1 (best thr*) | 0.892 | 0.892 | **0.896** | 0.892 | 0.894 | 0.894 | 0.938 | n/a | — |
| TEN VAD set — AUC | 0.871 | 0.875 | **0.880** | 0.875 | 0.876 | 0.877 | 0.952 | n/a | 0.670 |
| AMI SDM — F1 (calibrated) | **0.884** | 0.883 | 0.880 | 0.882 | 0.884 | 0.882 | 0.714 | 0.842 | 0.592 |
| AMI SDM — AUC | 0.861 | **0.862** | 0.862 | 0.861 | 0.862 | 0.861 | 0.894 | 0.760 | 0.658 |
| µs / 20 ms chunk | 63 | 65 | 66 | 64 | 92 | 66 | 89 | 2 | 7 |

\* tuned on that set — like-for-like with FlashVAD's published
F1 0.889 / AUC 0.882: v4-80k essentially matches FlashVAD's AUC
(0.880 vs 0.882) at 2.3× fewer parameters. The int8 QAT variant holds
TEN AUC 0.876 in a 28 KB file — within 0.005 of the 80k champion at
1/11th the size.

## 5. Limitations

English speech only; no music in training (music reads as "activity").
100 ms context: fricatives in noise are the hardest frames. No acoustic
echo cancellation — far-end echo/crosstalk is not modelled; use an
upstream echo canceller (e.g. Asterisk's). At distant-room operating
points, false-alarm rates are high on overlapped meeting speech for all
systems (labels mark foreground speech only). Weights inherit ESC-50's
NC license.

## 6. License

Code: MIT. Weights: **CC BY-NC-SA 4.0** as currently published, because
ESC-50 noise (CC BY-NC-SA 3.0) is in the training mix (LibriSpeech and
AMI are CC BY 4.0; Silero teacher MIT). Not for commercial deployment
without replacing the non-commercial training data; a commercial-safe
retrain on permissively licensed noise is planned.

## Reproducibility

All numbers reproduce from the project scripts: `prepare_data_v3.py
--utts 30000 --npy`, `distill_label.py`, `train_v3.py`,
`qat_bakeoff.py`, `calibrate_realworld.py`, `compare_all.py`,
`make_charts.py`, `quantize.py`, `export_onnx.py`. Model files are
plain `.npz` (weights + JSON metadata), inspectable with `numpy.load`.
