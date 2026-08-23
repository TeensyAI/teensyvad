# TeensyVAD — a VAD small enough to understand

**TeensyVAD by VoxLogic** · [models on HuggingFace](https://huggingface.co/Teensy) ·
[paper page](https://metacritical-teensyresearch.static.hf.space/index.html) ·
contact: [pankaj@voxlogic.ai](mailto:pankaj@voxlogic.ai)

**Voice activity detection** answers one question per audio frame: *is
somebody talking right now?*  It's the piece that decides when a
voice assistant should start listening, when a voicemail system stops
recording, when a conference server mutes a noisy line, and how much of
a call recording is worth storing or transcribing.

teensyvad is a complete, working VAD you can read in one sitting:

* **~20k parameters, 87 KB** — a 3-layer MLP written in plain numpy,
  backprop and Adam included, no deep-learning framework anywhere;
* **fast**: ~63 µs of CPU per 20 ms telephony chunk end-to-end
  (**≈300× real time** on a laptop core; the model alone scores a frame
  in ~14 µs, 0.3 µs batched);
* **telephony-native**: 8 kHz, 16-bit mono — exactly what an Asterisk
  AudioSocket hands you;
* **trained on real data**: LibriSpeech speech + ESC-50 noise, mixed at
  0–20 dB SNR with G.711 µ-law "phone line" augmentation;
* **streaming**: chunk-in → `speech_start` / `speech_end` events out,
  with hysteresis and hangover like the big commercial VADs;
* **an honest baseline**: grandpa's energy VAD ships alongside, same
  API, so you can measure exactly why neural VADs exist.

```
PCM audio ─▶ log-mel features ─▶ tiny MLP ─▶ P(speech) ─▶ smoothing ─▶ events
  audio.py      features.py        model.py                  streaming.py
```

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest matplotlib
.venv/bin/python -m pytest tests/ -v          # 26 tests, <1 s

# the whole pipeline in one command (resumable, ~10 min download + ~12 min compute):
.venv/bin/python scripts/train_all.py

# individual steps still work:
.venv/bin/python scripts/prepare_data.py       # → data/prepared/*.npz
.venv/bin/python scripts/train.py              # ~30 s  → models/teensy-v1.npz
.venv/bin/python scripts/evaluate.py           # quality vs the energy VAD
.venv/bin/python scripts/benchmark.py          # speed

# try it on any wav
.venv/bin/python scripts/demo_file.py data/prepared/demo/test_00.wav --plot
```

The repo ships with a trained `models/teensy-v1.npz`, so the demos work
without the data steps.

## Quick Start

```python
from teensyvad import OfflineVAD

# Standalone VAD — auto-downloads from the Hugging Face Hub on first use
model = OfflineVAD("Teensy/teensy-vad-v4")
result = model.segments("long_audio.wav")
# Returns speech segments: [[start_ms, end_ms], [start_ms, end_ms], ...]
print(result)                    # e.g. [[90, 5150]] — the fsmn-vad convention
```

(`teensyvad` is the numpy-only package in this repo — `pip install numpy`
is the only dependency; `pip install huggingface_hub` enables the
auto-download above, or pass a local `.npz` path directly.  A specific
variant: `OfflineVAD("Teensy/teensy-vad-v4", model_file="teensy-v4-80k.npz")`.)

## Use as Part of an ASR Pipeline

```python
from teensyvad import OfflineVAD
from teensyvad.audio import write_wav

# VAD first: segment long audio so the ASR only ever sees speech
vad = OfflineVAD("Teensy/teensy-vad-v4")
segments = vad.segments("meeting_2hours.wav")     # [[start_ms, end_ms], ...]

for i, (start_ms, end_ms) in enumerate(segments):
    seg = vad.slice("meeting_2hours.wav", start_ms, end_ms)   # float32 @ 8 kHz
    write_wav(f"speech_{i}.wav", seg, 8000)                   # feed to your ASR
    # text = my_asr.transcribe(f"speech_{i}.wav")   # <- your ASR here
```

Notes: audio is native **8 kHz** (telephony band — PSTN/G.711/slin), so
for a 16 kHz ASR resample the slices (or load the original file at the
ASR's rate and slice by `start_ms`/`end_ms`).  Per-frame probabilities
for custom thresholds: `vad.probabilities(wav)`; talk-time ratio:
`vad.speech_ratio(wav)`.

## Streaming / telephony (Asterisk)

```python
from teensyvad import StreamingVAD

vad = StreamingVAD("models/teensy-v4.npz")        # or any HF-downloaded .npz
for frame in phone_frames:                        # 20 ms PCM16LE @ 8 kHz
    for ev in vad.feed(frame):                    # any chunk size, bytes or float
        print(f"{ev.t:7.2f}s  {ev.type}")         # speech_start / speech_end
events = vad.flush()                              # close at hangup/EOF
print(f"talk time: {vad.speech_seconds:.1f}s")
```

Offline and streaming decisions are identical by construction —
`OfflineVAD` is the streaming engine fed whole files.  Thresholds ship
inside each model file (domain profiles for v3/v4: `close_mic` default,
`distant_room` in metadata); ONNX exports live in each HF repo for
non-Python or throughput use.  A working Asterisk AudioSocket server:
`scripts/audiosocket_server.py`.

## Measured quality (test set: unseen speakers, unseen noise types)

| conditions | frame F1 | frame AUC |
|---|---|---|
| clean | 0.945 | 0.950 |
| 20 dB SNR | 0.946 | 0.933 |
| 15 dB SNR | 0.927 | 0.912 |
| 10 dB SNR | 0.921 | 0.886 |
| 5 dB SNR | 0.903 | 0.855 |
| **0 dB SNR** | **0.862** | **0.804** |
| overall | 0.903 | 0.889 |

On the event level (120 streamed clips, 20 ms chunks) the neural VAD finds
speech segments cleanly, while the energy baseline over-triggers by >2×
on the same audio:

| event-level, 120 clips | precision | recall | F1 | segments |
|---|---|---|---|---|
| **teensyvad (neural)** | **0.916** | **1.000** | **0.956** | 131 pred / 120 true |
| energy baseline | 0.375 | 0.933 | 0.535 | 299 pred / 120 true |

**That gap is the whole point of this project**: energy VAD only knows
"loud", so at phone-line noise levels it calls the noise floor speech.

## How VAD actually works — the 5-minute tour

Each module is one stage, and each is deliberately readable:

### 1. Framing (`features.py`)
Raw samples are nearly useless to a classifier. We chop audio into
**25 ms windows every 10 ms** (100 decisions/second) — short enough that
speech is stationary within a window, long enough to resolve the
harmonics that identify voice.

### 2. Log-mel features (`features.py`)
Each window is FFT'd, and band energy is summed through ~20 triangular
filters spaced on the **mel scale** — human hearing resolution: dense
below 1 kHz where speech lives, sparse above. Then a **log** makes
features shift linearly with loudness, and subtracting each frame's
band-mean removes that shift entirely: **the model sees spectral shape,
not level**. That's why it survives wildly different phone lines.

### 3. Temporal context (`streaming.py`)
One 25 ms frame can't tell /s/ from a fan. The model sees the last
**10 frames (100 ms)** flattened into one vector — enough to capture
onsets and the temporal texture of speech.

### 4. The classifier (`model.py`)
400 → 48 → 24 → 1 MLP with ReLU, trained with hand-written backprop +
Adam on ~1M labelled frames (binary cross-entropy). The output is
**P(speech)** for the newest frame. ~20k parameters is genuinely enough
for this — VAD is a 1-bit question.

### 5. Decision smoothing (`streaming.py`)
Raw probabilities flicker frame-to-frame. Two classic tricks turn them
into clean events:
* **hysteresis**: enter speech only above a high threshold, leave only
  below a lower one;
* **hangover**: once in speech, stay ~250 ms after P drops, so brief
  pauses inside words don't chop utterances apart.

The same state machine runs in WebRTC's VAD and in carrier-grade
telephony stacks.

### Labels (`scripts/prepare_data.py`)
Every training clip is *constructed*: a LibriSpeech utterance (silence-
trimmed by energy so its studio padding isn't mislabelled) is embedded
in ESC-50 environmental noise at a random SNR, 40% of clips round-trip
through **G.711 µ-law** to simulate a real phone line, and frame labels
come from where the utterance actually sits. Speakers are disjoint
between train/val/test, and noise types in test were never heard in
training.

## Asterisk integration

`scripts/audiosocket_server.py` speaks Asterisk's **AudioSocket**
protocol (Asterisk ≥ 18): per call, Asterisk opens a TCP connection and
streams 8 kHz slin (PCM16LE) in 20 ms frames — we run the VAD on them
and log events live.

```ini
; /etc/asterisk/extensions.conf
[teensyvad]
ext => 500,1,NoOp(teensyvad)
  same => n,Answer()
  same => n,AudioSocket(127.0.0.1:9092,call-${UNIQUEID})
  same => n,Hangup()
```

```bash
.venv/bin/python scripts/audiosocket_server.py --port 9092 &
asterisk -rx "dialplan reload"      # then dial 500 from any extension
```

Server console output during a call:

```
[+] connection from 127.0.0.1
    call 6f1c...: VAD armed (8000 Hz, thr 0.50/0.30)
    [    4.31s] speech_start  (stream t=2.98s, P=0.97)
    [   10.02s] speech_end    (stream t=8.69s, P=0.04)
[-] call ... ended:  12.4s audio,  5.7s speech (46%)
```

Practical uses once connected: stop recording after N seconds of
silence, gate transcription to speech segments, measure talk-time
ratios, or feed `speech_start` events to a barge-in detector. With
Asterisk ≥ 16 you can also keep audio out of the dialplan entirely and
let this server be the consumer; for media *manipulation* (vs analysis)
you'd add a `TRACE`/ARI path — ask and we'll build it.

## Train the full thing end to end — one command

```bash
.venv/bin/python scripts/train_all.py
```

Downloads LibriSpeech + ESC-50, builds 8 kHz mixtures, Silero-relabels
them, trains v1 (construction labels) and v2 (distilled, hard + soft),
calibrates event thresholds, evaluates, quantizes (int8 + selective +
ONNX) and runs the tests. **Every stage is resumable** — it skips work
whose outputs already exist, so you can re-run freely; `--only quantize`
or `--skip download extract` narrow it down. From an empty cache on an
M2: ~10 min download + ~12 min compute.

**8 kHz doctrine**: everything is native 8 kHz, end to end — no
resampling anywhere in training or serving. PSTN lines, G.711 µ-law and
A-law, and Asterisk's `slin` are all 8 kHz; real telephony is narrowband,
so a 16 kHz core (like some toolkits use) pays an upsample-per-hop tax
for bandwidth a phone call never carries.

## Quantization (post-training, int8)

```bash
.venv/bin/python scripts/quantize.py --model models/teensy-v2.npz   # numpy int8
.venv/bin/python scripts/export_onnx.py --model models/teensy-v2.npz # ONNX f32+int8
```

`quantize.py` first measures **per-layer sensitivity** (quantize one
layer at a time, check the AUC cost) — that's what makes quantization
*selective*: only layers under tolerance go int8, sensitive ones stay
float32. Measured on this model: **every layer quantizes for free**
(ΔAUC ±0.0000, decision agreement 99.72%), so selective chose
everything — but the analysis runs every time and decides from data.

Honest numbers (M2 Pro, measured, not promised):

| path | single frame | batched | size | note |
|---|---|---|---|---|
| numpy float32 | 14–15 µs | 0.27 µs | 86.6 KB | default; zero deps |
| numpy int8 | 44 µs | ~3 µs | **28.3 KB** | **no int8 BLAS in numpy** — slower but 3× smaller; at our scale still ~230× RT |
| ONNX float32 | 6.6 µs | 0.09 µs | 80.2 KB | matches numpy to 3e-6 |
| ONNX int8 | 7.5 µs | **0.067 µs** | **22.3 KB** | real int8 kernels |

(Trap documented in `quant.py`: numpy `int8 @ int8` **silently
overflows** — it accumulates in int8 and wraps; we force int32
accumulation. Never trust an untested quantized matmul.)

## Scaling up & real-world accuracy (v3)

Question: *does more data help, and how good is this on real recordings?*
`scripts/prepare_data_v3.py` + `train_v3.py` answer it.

**v3 = same 20,449-param architecture, but trained on:**
* 8,000 utterances from LibriSpeech **train-clean-100** (~30 h of mixtures,
  10.7M teacher-labelled frames — 10× v2)
* three noise families: ESC-50 + **synthetic 7-talker babble** (disjoint
  LibriSpeech speakers — the classic hard case) + **real AMI room
  ambience** (non-speech stretches of distant-mic meetings)
* SNR pushed to **−5 … 20 dB**
* lazy context-window training (`LazyWindows` — windows materialise per
  batch; the full 10.7M×400 matrix would be 17 GB)

**Real-world evaluation** (`scripts/eval_realworld.py`) — human-labelled
recordings, never seen in training. *AUC everywhere on raw probabilities*
(earlier drafts binarised Silero's output, understating its ranking —
corrected):

*TEN VAD public set — 30 real recordings (the set FlashVAD reports on):*

| model | F1 (stored thr) | AUC | F1 @ best thr |
|---|---|---|---|
| v1 | 0.877 | 0.848 | 0.877 |
| v2 | 0.890 | 0.868 | 0.890 |
| **v3** | 0.894 | 0.873 | **0.894** |
| Silero (teacher) | 0.938 | **0.952** | 0.938 |
| FlashVAD (published) | 0.889 | 0.882 | — |

*AMI SDM meetings — 8 real meeting rooms, distant mic, manual labels:*

| model | F1 | AUC | FAR | MR |
|---|---|---|---|---|
| v2 | 0.880 | 0.848 | 64% | 9% |
| **v3** | **0.886** | 0.861 | 80% | 4% |
| Silero (teacher) | 0.714 | **0.894** | 1% | 44% |
| WebRTC VAD (agg=0) | 0.842 | 0.760 | 25% | 23% |
| energy baseline | 0.592 | 0.658 | 12% | 57% |

Findings, stated honestly:

1. **Silero ranks best by AUC everywhere** (0.952 TEN / 0.894 AMI) — an
   87× larger model. What the tiny student wins is the *operating
   point*: Silero's stock threshold misses 44 % of speech in real rooms
   (AMI F1 0.714) while calibrated v3 catches 0.886. Rankings transfer;
   thresholds don't — calibrate whatever you deploy.
2. **v3 ≥ WebRTC VAD on every comparable metric**, in pure numpy.
2. **More data helped modestly, harder data helped more**: val AUC
   0.9235 → 0.9289 (synthetic), but the real-world AUC gains (+0.006
   TEN, +0.013 AMI over v2) come mostly from babble + ambience + low
   SNR exposure, not raw hours.
3. **Thresholds are domain-specific; rankings are not.** Best thr:
   ~0.45 close-mic, 0.10 distant-room, 0.85 synthetic-events. A single
   stored threshold cannot serve all domains — v3 ships **profiles** in
   its metadata: `close_mic` (default, for telephony) and
   `distant_room`. `calibrate_realworld.py` re-derives them from AMI dev
   meetings (3 meetings reserved for calibration; the 8 eval meetings
   are never used for any tuning).
4. **FAR/MR trade is a policy choice**: at the distant-room operating
   point everyone's FAR is high on overlapped meetings (labels mark
   *foreground* speech only); Silero's 44% MR on AMI shows the teacher
   is conservative in rooms — which is exactly the behaviour v3 was
   distilled *away* from, with real room-tone in training.
5. FlashVAD comparison: v3 matches its published TEN F1 (0.894 vs 0.889
   best-threshold) with a 2.3× smaller model — but both numbers are
   threshold-tuned against that same 30-clip set (their card admits it;
   ours is labelled "upper bound" in the table). Treat as parity, not
   victory.

G.711 µ-law round-trip changes every number by <0.5% (see
`eval_realworld.py` output) — the training-time telephony augmentation
earned its keep.

### Capacity sweep: 20k → 40k → 80k → 100k params

`scripts/compare_all.py` + `scripts/scale_sweep.sh` trained all three
families at ~40k / ~80k / ~100k parameters (hidden sizes 88/48, 164/88,
200/96). Result, in one line: **capacity scales only with data.**

| family (training frames) | 20k | 40k | 80k | 100k |
|---|---|---|---|---|
| v1 (1M, construction labels) | 0.911 | 0.909 | 0.910 | 0.910 |
| v2 (1M, teacher labels) | 0.910 | 0.912 | 0.912 | 0.911 |
| v3 (10.7M, teacher labels) | 0.911 | 0.916 | **0.917** | 0.916 |
| **v4 (37.9M = full 100 h)** | 0.914 | 0.918 | 0.920 | **0.921** |

(val F1.) Real-world TEN AUC peaks at **v4-80k: 0.880** — essentially
matching FlashVAD's published 0.882 at 2.3× fewer parameters, and the
best teensy number yet. The v4 family
(`scripts/prepare_data_v3.py --utts 30000 --npy`, memmap/float16
plumbing for the 60 GB-scale design matrix) is published as
[Teensy/teensy-vad-v4](https://huggingface.co/Teensy/teensy-vad-v4) with
all four sizes + ONNX exports. Full 22-row comparison table:
`comparison.json`, `hf_assets/*.png`, and the BENCHMARKS.md appendix in
the HF repos.



## PTQ vs QAT vs "train wide, quantize down" — the bake-off

`scripts/qat_bakeoff.py` (also the `qat` stage of `train_all.py`) trains
all three schools and measures them on the same test split (labels =
Silero teacher):

* **PTQ** — train float32, round to int8 afterwards ("train high, quantize later")
* **QAT** — fine-tune the float model with int8 *simulated in the forward
  pass* (fake-quant weights + activations, straight-through-estimator
  gradients — see `qat_forward`/`qat_train` in `teensyvad/quant.py`;
  a unit test pins simulation ≡ deployed inference)
* **wide** — 2× the hidden width (43k params), then quantized down

| model | params | KB | F1 | AUC | µs single | µs/1k batch |
|---|---|---|---|---|---|---|
| v2-float | 20,449 | 86.6 | 0.8442 | 0.9229 | **14.1** | **0.27** |
| v2-ptq | 20,449 | 28.2 | 0.8445 | 0.9229 | 44.0 | 4.57 |
| **v2-qat** | 20,449 | **28.2** | 0.8538 | **0.9235** | 43.6 | 4.60 |
| wide-float | 43,201 | 175.5 | 0.8511 | 0.9226 | 14.3 | 0.32 |
| wide-ptq | 43,201 | 50.9 | 0.8513 | 0.9226 | 49.1 | 7.69 |
| **wide-qat** | 43,201 | 50.9 | **0.8565** | 0.9233 | 49.1 | 7.58 |

Takeaways, honestly stated:

1. **QAT > PTQ at identical size** — +0.9 F1 points for v2, +0.5 for
   wide. Weights that trained *on* the int8 grid beat weights rounded
   onto it after the fact.
2. **v2-qat even edges v2-float on F1** — but AUC (threshold-free) is
   nearly identical (0.9235 vs 0.9229), so most of that F1 edge is
   probability sharpening at the operating threshold, plus the 12 extra
   fine-tuning epochs acting as regularization. The proven QAT win is
   over PTQ, not over float32.
3. **Capacity saturates** — wide buys little (the ceiling is data/labels,
   not parameters), though under quantization wide-qat is the accuracy
   winner (0.8565 at 50.9 KB — more accurate than v2-float in a smaller
   file).
4. Speed in numpy unchanged by QAT (same int8 path); float32 BLAS still
   wins single-frame latency; ONNX int8 remains the batched champion.

**Recommendations**: `teensy-v2-qat.npz` (28 KB) as the smallest accurate
artifact for embedding; `teensy-v2-wide-qat.npz` (51 KB) when accuracy is
paramount; ONNX int8 for throughput; float32 numpy for readability.




## Distillation: teensyvad 2.0 from Silero (teacher → student)

Construction labels know *where the utterance was placed*, but not where
speech actually starts and stops inside it — LibriSpeech utterances
contain internal pauses, breathy tails and lead-in silence that we were
labelling "speech". **Knowledge distillation** fixes that: a strong
teacher (Silero VAD) relabels our mixtures, and the same 20k-param
student learns from *its* frame probabilities.

```
.venv/bin/pip install silero-vad onnxruntime        # teacher, labeling only
.venv/bin/python scripts/prepare_data.py --save-audio  # mixtures + wavs
.venv/bin/python scripts/distill_label.py              # teacher → soft labels
.venv/bin/python scripts/train.py --data-suffix .distill --ycol ysoft \
    --out models/teensy-v2-soft.npz                   # soft-target student
.venv/bin/python scripts/train.py --data-suffix .distill --ycol y \
    --out models/teensy-v2.npz                        # hard-target student
.venv/bin/python scripts/calibrate_events.py --model models/teensy-v2.npz
```

**What the teacher revealed about our labels**: on the test split the
teacher disagreed with construction labels on 11.6% of frames — 9.2%
"construction says speech, teacher silent" (the pauses/tails we
suspected) and 2.4% the other way (teacher hears voice where we placed
'noise'). That 9.2% was silent label contamination capping v1.

**Results (test set, 100 streamed clips)**:

| | v1 (construction) | v2 (distilled, hard) | v2-soft |
|---|---|---|---|
| frame F1 vs teacher | 0.893 | **0.905** | 0.906 |
| onset Δ vs teacher | −432 ms | **−172 ms** | −300 ms |
| offset Δ vs teacher | +718 ms | **+478 ms** | +554 ms |
| events vs construction truth | 0.976 | 0.877 | 0.901 |

Read the last row carefully — it looks backwards but isn't: v1 scores
highest against construction truth *because it was trained on that very
truth*, padded pauses included (it fires 432 ms early and releases
718 ms late). The distilled students track where speech actually is,
per an independent strong model. For telephony — barge-in latency,
silence-triggered stop — **boundaries are the product**, so
`teensy-v2.npz` (hard targets, tightest boundaries) is the default.

Two honest caveats:
* **No speed win here**: torch-JIT Silero ≈1.15× faster than our numpy
  student on an M2. Distillation bought *knowledge*, not speed — both
  run ~300× real time, and a compiled (ONNX) student would flip the
  ratio if you ever care.
* A student cannot exceed its teacher's idea of speech; Silero's own
  biases (e.g. skeptical of sung/tone-like audio) are inherited.
  Distillation is also how FlashVAD was labelled — this section is a
  map of that decision's consequences.

New scripts: `distill_label.py` (teacher labeling + disagreement
report), `calibrate_events.py` (event-level threshold sweep, written
back into the model metadata), `evaluate_distill.py` (agreement,
boundary timing, speed ratio).



| path | what it teaches |
|---|---|
| `teensyvad/audio.py` | PCM16, WAV I/O, G.711 µ-law, resampling |
| `teensyvad/features.py` | framing, FFT, mel filterbank, log, deltas |
| `teensyvad/model.py` | MLP forward, backprop (gradient-checked), Adam |
| `teensyvad/streaming.py` | context stacking, hysteresis, hangover, events |
| `teensyvad/energy_vad.py` | the energy baseline, same API |
| `scripts/prepare_data.py` | mixture construction, SNR, µ-law augmentation |
| `scripts/train.py` | training loop + threshold calibration |
| `scripts/train_all.py` | the whole pipeline, one resumable command |
| `scripts/prepare_data_v3.py` | scaled v3 set: babble + room ambience, −5 dB |
| `scripts/noise_pools.py` | ESC-50 / babble / AMI-ambience noise builders |
| `scripts/train_v3.py` | scaled training with lazy context windows |
| `scripts/eval_realworld.py` | TEN VAD set + AMI meetings, human labels |
| `scripts/calibrate_realworld.py` | domain thresholds from real audio |
| `scripts/quantize.py` | int8 + selective quantization w/ sensitivity analysis |
| `scripts/export_onnx.py` | ONNX float32/int8 export + speed table |
| `scripts/evaluate.py` | frame + event metrics, baseline comparison |
| `scripts/benchmark.py` | µs/frame and real-time factor |
| `scripts/demo_file.py` | run/plot VAD on any wav |
| `scripts/audiosocket_server.py` | the telephony front door |
| `scripts/distill_label.py` | Silero teacher → soft labels (+ disagreement report) |
| `scripts/calibrate_events.py` | event-level threshold calibration |
| `scripts/evaluate_distill.py` | student vs teacher: agreement, boundaries, speed |
| `teensyvad/quant.py` | int8 dynamic quantization (per-layer, selective) |
| `tests/test_all.py` | 26 tests incl. streaming≡offline, gradcheck, quantized round-trip |

## Limits & next steps

An MLP on 100 ms of context is the *smallest thing that works well*.
Known limits, in the order I'd attack them:

1. **Unvoiced speech** (fricatives, stop gaps) looks like noise in a
   100 ms window → try a small GRU or a temporal conv stack.
2. **Music/TV in background** counts as "activity" — it is audio
   activity, just not speech; distinguishing needs more data variety.
3. **Echo/crosstalk** from the far end can double-trigger; barge-in
   needs echo-aware features.
4. If you need state of the art, Silero VAD is the standard small
   model — after reading this repo, its architecture will look familiar.

## Data & license notes

* Speech: [LibriSpeech](https://www.openslr.org/12) dev/test subsets (CC BY 4.0)
* Noise: [ESC-50](https://github.com/karolpiczak/ESC-50) (CC BY-NC-SA 3.0 —
  fine for personal/evaluation use; swap Musan/AudioSet if you ship commercial)
* Code: MIT
* Dataset conversion uses macOS `afconvert` (Linux: swap in `sox` or `ffmpeg`)
