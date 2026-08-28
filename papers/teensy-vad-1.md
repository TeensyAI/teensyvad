# teensy-vad-1: A 20k-Parameter Voice Activity Detector for Telephony Audio

**Abstract.** teensy-vad-1 is the first model of the teensyvad family: a complete voice activity detector of 20,449 parameters (87 KB) implemented in pure numpy with hand-written backprop, built to run natively in an 8 kHz telephony stack and to serve as a readable baseline for the later, distilled and data-scaled members of the family.

## 1 Introduction

Voice activity detection (VAD) answers one binary question per audio frame: is somebody talking right now? Classical energy-based detectors fail at phone-line noise levels because "loud" is not "speech"; existing neural VADs are orders of magnitude larger than a telephony task requires. teensy-vad-1 solved the founding problem of the teensyvad family timeline: demonstrating that a tiny, fully understandable classifier — trained on real mixed data — can decisively outperform the energy baseline (event F1 0.976 vs 0.535) while remaining small enough to read in one sitting and fast enough for Asterisk AudioSocket streaming. It was also the vehicle for a capacity sweep (40k/80k/100k variants) showing that accuracy is flat across sizes on a 1M-frame training set, which set up the data-scaling question addressed by teensy-vad-3.

## 2 Data

- Speech: LibriSpeech `dev-clean`, 1,200 utterances, speaker-disjoint train/val/test splits (CC BY 4.0).
- Noise: ESC-50, fold-disjoint, human-vocal categories excluded (CC BY-NC-SA 3.0).
- Mixing: utterances embedded in noise at SNR 0–20 dB; 40 % of clips round-tripped through G.711 µ-law to simulate telephone-line degradation. ~1M labelled frames.
- Labels: construction ground truth — a frame is labelled speech iff the placed utterance covers its centre, with utterance edges energy-trimmed.

## 3 Method

The model is a 3-layer MLP, 400 → 48 → 24 → 1 (ReLU hidden layers, sigmoid head), with a hand-written backprop and Adam implementation. Output is P(speech) per 10 ms frame.

Feature frontend (telephony-native, 8 kHz end-to-end, no resampling):
1. Frames of 200 samples (25 ms window) at a hop of 80 (10 ms).
2. Per frame: Hann window → zero-padded FFT to 256 → power spectrum.
3. 20 mel triangles spanning 80–3800 Hz (rows normalised to sum 1).
4. log(mel + 1e-10), then per-frame mean subtraction over bands (gain invariance — the model sees spectral shape, not level).
5. First-order delta (current − previous frame), concatenated → 40 dims/frame.
6. A stack of 10 frames (100 ms context, oldest→newest, frame-major) → 400 input dims, standardised with stored mean/std vectors.

Streaming decisions add hysteresis and hangover on top of per-frame probabilities.

## 4 Results

| benchmark | frame F1 | AUC |
|---|---|---|
| synthetic test (unseen speakers, unseen noise) | 0.903 | 0.889 |
| event-level streaming (120 clips) | 0.976 | — |
| TEN VAD public set (real recordings) | 0.850 | 0.848 |
| AMI SDM meetings (real rooms, manual labels) | 0.743 | 0.835 |

An adaptive energy VAD on the same streamed clips reaches event F1 0.535 (2.5× over-triggering). On the shared real-world protocol (AMI-dev-calibrated operating points), teensy-v1 attains AMI SDM F1 0.887 / AUC 0.835 vs Silero VAD 0.714 / 0.894, WebRTC VAD 0.842 / 0.760, and Energy VAD 0.592 / 0.658; Silero ranks best by AUC (87× larger) but its stock threshold misses 44 % of room speech. Full family comparison: see family benchmarks (BENCHMARKS.md appendix in the Hugging Face repos).

## 5 Limitations

- Trained on English read speech plus environmental noise only; music and multi-talker babble were not in training (addressed in teensy-vad-3).
- Construction labels mark intra-utterance pauses as speech, so boundaries are loose (onset ~430 ms early relative to a Silero teacher); teensy-vad-2 addresses this via distillation.
- Stored operating thresholds were tuned on synthetic validation; real-domain thresholds differ (domain profiles appear from v3 onward).

## 6 License

Code: MIT. Weights: CC BY-NC-SA 4.0 (training noise ESC-50 is CC BY-NC-SA 3.0; speech LibriSpeech is CC BY 4.0). Not for commercial deployment without replacing the ESC-50-derived training data.

## Reproducibility

Pipeline stages (all in `scripts/`): `prepare_data.py` (mixture construction), `train.py` (training + threshold calibration), `train_all.py` (resumable end-to-end), `evaluate.py` and `benchmark.py` (quality and speed), `quantize.py` and `export_onnx.py` (int8 / ONNX artifacts), `demo_file.py`, and `audiosocket_server.py` for Asterisk streaming. The library (`teensyvad`) is pure numpy; tests live in `tests/test_all.py`.
