# teensy-vad-2: Boundary-Accurate Distillation from Silero VAD into a 20k-Parameter Student

**Abstract.** teensy-vad-2 keeps the teensyvad-family architecture (20,449 parameters, 87 KB, pure numpy) but replaces construction labels with frame-level labels produced by Silero VAD, a knowledge-distillation step that fixes the loose speech boundaries inherited from v1's placement-based labels.

## 1 Introduction

In the family timeline, teensy-vad-1 proved that a 20k-parameter MLP can decisively beat energy-based VAD, but its construction labels know *where the utterance was placed*, not where speech actually starts and stops: intra-utterance pauses, breathy tails and lead-in silence were labelled "speech", giving onsets ~430 ms early and offsets ~718 ms late. teensy-vad-2 solved this boundary problem by distillation: Silero VAD (~2M-parameter teacher) relabelled every frame of the training mixtures, and the same tiny student learned where speech *actually* is. On the test split the teacher disagreed with construction labels on 11.6 % of frames — 9.2 % of them pauses/tails mislabelled "speech" — silent label contamination that had capped v1. For telephony, where barge-in latency and silence-triggered stopping depend on boundary timing, boundaries are the product; the hard-label student produced tighter boundaries than the soft variant and is the default.

## 2 Data

- Same mixtures as v1: LibriSpeech `dev-clean` speech (CC BY 4.0) + ESC-50 noise (CC BY-NC-SA 3.0), SNR 0–20 dB, 40 % of clips round-tripped through G.711 µ-law; ~1M frames.
- Labels: Silero VAD (MIT) run at 8 kHz on every mixture frame — soft probabilities interpolated to the 10 ms grid. Two students were trained: hard 0/1 teacher labels (this model, the default) and soft probabilities (slightly higher validation AUC, 0.929 vs 0.922, but looser event boundaries).

## 3 Method

The architecture is identical to teensy-vad-1: a 3-layer MLP, 400 → 48 → 24 → 1 (ReLU, sigmoid head), trained with hand-written backprop + Adam. The feature frontend is likewise unchanged: 8 kHz native (no resampling anywhere), 25 ms windows / 10 ms hop, 20 mel bands over 80–3800 Hz, log-mel with per-frame band-mean subtraction, first-order deltas, and a 10-frame (100 ms) context stack — 400 input dimensions standardised with stored statistics. Streaming decisions use the family's hysteresis + hangover state machine. The distillation difference is supervisory only: the training targets are the teacher's frame labels rather than construction labels, and event thresholds were calibrated at event level on validation data.

## 4 Results

| benchmark | frame F1 | AUC |
|---|---|---|
| synthetic test (vs teacher labels) | 0.905 | 0.923 |
| TEN VAD public set (stored thr) | 0.869 | 0.868 |
| TEN VAD public set (best thr, upper bound) | 0.890 | — |
| AMI SDM meetings (AMI-calibrated thr) | 0.880 | 0.848 |

Boundary quality vs the Silero teacher (100 streamed clips): onset Δ improved from −432 ms (v1) to −172 ms; offset Δ from +718 ms to +478 ms. The student already out-ranks its teacher on AMI (F1 0.880 vs 0.714) while the teacher stays better calibrated close-mic (TEN F1 0.937 / AUC 0.952). On the shared protocol teensy-v2 attains TEN AUC 0.868 and AMI F1 0.880 / AUC 0.848; like-for-like with FlashVAD's threshold-tuned TEN numbers, F1 0.890 vs 0.889 with 2.3× fewer parameters. Quantization is essentially free (PTQ ΔAUC 0.0000, 99.7 % decision agreement; QAT adds +0.9 F1 points over PTQ at 28 KB). Full family comparison: see family benchmarks (BENCHMARKS.md appendix in the Hugging Face repos).

## 5 Limitations

- A student inherits its teacher's biases; Silero's conservatism on sung/tone-like audio transfers.
- English read speech + environmental noise only; no babble or room tone in training (addressed in teensy-vad-3).
- Operating thresholds are domain-specific: the real-set numbers above used AMI-calibrated thresholds (0.10); the stored default suits synthetic-event use.

## 6 License

Code: MIT. Weights: CC BY-NC-SA 4.0 (ESC-50 training noise is CC BY-NC-SA 3.0; speech LibriSpeech is CC BY 4.0; teacher Silero VAD is MIT). Not for commercial deployment without retraining on permissive noise data.

## Reproducibility

Pipeline stages (all in `scripts/`): `prepare_data.py --save-audio` (mixtures), `distill_label.py` (teacher labelling + disagreement report), `train.py --data-suffix .distill --ycol y` (hard-label student; `--ycol ysoft` for the soft variant), `calibrate_events.py` (event threshold calibration), `evaluate_distill.py` (agreement, boundary timing, speed), plus the shared `train_all.py`, `quantize.py`, `export_onnx.py`, `benchmark.py`, and `audiosocket_server.py`.
