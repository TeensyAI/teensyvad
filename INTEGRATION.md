# teensyvad — Integration Brief

> Self-contained brief for developers and AI agents integrating teensyvad
> (a tiny, fast, 8 kHz voice activity detector) into an application.
> Everything below has been executed against the published artifacts.

## 1. What it is

Voice activity detection for **8 kHz mono telephony audio** (PSTN /
G.711 / Asterisk `slin` native — no resampling). A 3-layer MLP
(20k–100k params, 87–400 KB) scoring one P(speech) every 10 ms, with
fsmn-vad-style `[[start_ms, end_ms], ...]` segment output and a
streaming chunk-in/event-out engine for real-time use.

- Pure numpy runtime (ONNX exports for other languages/runtimes)
- ~65 µs per 20 ms chunk on one modern core (≈300× real time)
- Verified on human-labelled real audio: TEN VAD public set (AUC up to
  0.880) and AMI SDM meetings (F1 0.886 at calibrated thresholds)

## 2. Links

| resource | url |
|---|---|
| Default model (v4, 20k) | https://huggingface.co/Teensy/teensy-vad-v4 |
| 80k variant (best real-world AUC) | https://huggingface.co/Teensy/teensy-vad-v4/blob/main/teensy-v4-80k.npz |
| Model card (v4) | https://huggingface.co/Teensy/teensy-vad-v4/raw/main/README.md |
| This brief | https://huggingface.co/Teensy/teensy-vad-v4/raw/main/INTEGRATION.md |
| Family v1 | https://huggingface.co/Teensy/teensy-vad-1 |
| Family v2 (Silero-distilled) | https://huggingface.co/Teensy/teensy-vad-2 |
| Family v3 (30h data) | https://huggingface.co/Teensy/teensy-vad-3 |
| Benchmark protocol + full 21-row table | https://huggingface.co/Teensy/teensy-vad-3/raw/main/BENCHMARKS.md |
| Direct model file | https://huggingface.co/Teensy/teensy-vad-v4/resolve/main/teensy-v4.npz |
| ONNX float32 | https://huggingface.co/Teensy/teensy-vad-v4/resolve/main/teensy-v4.onnx |
| ONNX int8 | https://huggingface.co/Teensy/teensy-vad-v4/resolve/main/teensy-v4-int8.onnx |

Other sizes (each repo also ships `-40k`, `-80k`, `-100k` `.npz` +
ONNX float/int8 exports): swap the filename in the URLs above.
There is **no public GitHub yet** — the source (`teensyvad/` package:
audio.py, features.py, model.py, streaming.py, offline.py, quant.py,
~2,500 commented lines) ships with this project locally; copy the
package directory into your app.

## 3. Install

```bash
pip install numpy                        # the only hard dependency
pip install huggingface_hub              # optional: auto-download from HF
# then copy the teensyvad/ package directory from this project
```

## 4. Offline file mode — speech segments (fsmn-vad convention)

```python
from teensyvad import OfflineVAD

vad = OfflineVAD("Teensy/teensy-vad-v4")            # auto-downloads from HF
segments = vad.segments("long_audio.wav")           # [[start_ms, end_ms], ...]
```

- Accepts a local `.npz` path instead of an HF repo id (then
  `huggingface_hub` is not needed).
- Any input sample rate is resampled to 8 kHz internally (whole-file
  FFT resampling).
- Variant selection: `OfflineVAD("Teensy/teensy-vad-v4",
  model_file="teensy-v4-80k.npz")`.

## 5. ASR pipeline (VAD front-end)

```python
from teensyvad import OfflineVAD
from teensyvad.audio import write_wav

vad = OfflineVAD("Teensy/teensy-vad-v4")
segments = vad.segments("meeting_2hours.wav")        # [[start_ms, end_ms], ...]

for i, (start_ms, end_ms) in enumerate(segments):
    seg = vad.slice("meeting_2hours.wav", start_ms, end_ms)   # float32 @ 8 kHz
    write_wav(f"speech_{i}.wav", seg, 8000)
    # text = my_asr.transcribe(f"speech_{i}.wav")    # your ASR here
```

Note: output is 8 kHz (telephony band). For a 16 kHz ASR, load the
original file at its own rate and slice by ms offsets instead.

## 6. Streaming / real-time (telephony)

```python
from teensyvad import StreamingVAD

vad = StreamingVAD("teensy-v4.npz")                  # or any downloaded .npz
for frame in phone_frames:                           # 20 ms PCM16LE @ 8 kHz
    for ev in vad.feed(frame):                       # any chunk size
        print(ev.t, ev.type)                         # speech_start / speech_end
events = vad.flush()                                 # call at hangup/EOF
print("talk time:", vad.speech_seconds)
```

- `feed()` accepts bytes (PCM16LE) or float arrays of any size.
- Offline and streaming decisions are identical by construction
  (`OfflineVAD` is the streaming engine fed whole files).

## 7. Other languages: ONNX

The `.onnx` graph consumes the standardised 400-dim feature vector
(not raw audio) and outputs one logit. Feature frontend spec (all
reimplementable in any language):

1. Mono float32 @ 8 kHz; frames of 200 samples (25 ms), hop 80 (10 ms)
2. Hann window → zero-padded FFT to 256 → power spectrum
3. 20 mel triangles over 80–3800 Hz (rows normalised to sum 1)
4. `log(mel + 1e-10)`, subtract per-frame mean over bands
5. Delta = current − previous frame; concat → 40 dims/frame
6. Stack 10 frames (oldest→newest), standardise with `in_mean`/`in_std`
   stored in the `.npz` (values are tiled per frame — 40-dim stats
   repeated 10×)

Python with ONNX Runtime:

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("teensy-v4.onnx", providers=["CPUExecutionProvider"])
logit = sess.run(["logit"], {"x": features_400dim})[0]
p_speech = 1 / (1 + np.exp(-logit.ravel()[0]))
```

## 8. Choosing a model

| use case | pick | why |
|---|---|---|
| Smallest accurate (embed) | `teensy-v4.npz` (20k, 87 KB) | default; int8 ONNX 22 KB |
| Best real-world AUC | `teensy-v4-80k.npz` (80k, 321 KB) | TEN AUC 0.880 — matches FlashVAD v0.1 (0.882) |
| Known background talkers/rooms | v3/v4 family | trained with babble + real room ambience |
| Legacy baselines | v1 (labels), v2 (Silero-distilled) | v2 has the tightest boundaries |

## 9. Operating thresholds (critical)

AUC (ranking) transfers across domains; **thresholds do not**. Metadata
in v3/v4 `.npz` files ships profiles:

```json
{"profiles": {"close_mic": {"thr_hi": 0.45, "thr_lo": 0.27},
              "distant_room": {"thr_hi": 0.10, "thr_lo": 0.06}}}
```

- Close-mic / PSTN (default): thr_hi 0.45. StreamingVAD reads this.
- Distant rooms (speakerphone): thr_hi 0.10 — re-calibrate on ~1 h of
  your own audio if the domain is neither (see
  `scripts/calibrate_realworld.py`).

## 10. Metrics summary (human-labelled real audio, our harness)

| system | TEN F1* | TEN AUC | AMI F1 | AMI AUC | µs/20 ms |
|---|---|---|---|---|---|
| teensy-v4-80k | 0.896 | 0.880 | 0.880 | 0.862 | 66 |
| Silero VAD (1.77M params) | 0.938 | 0.952 | 0.714 | 0.894 | 89 |
| WebRTC VAD | n/a | n/a | 0.842 | 0.760 | 2 |
| Energy VAD | — | 0.670 | 0.592 | 0.658 | 7 |

\* best-F1 threshold (upper bound). Full table + protocol:
[BENCHMARKS.md](https://huggingface.co/Teensy/teensy-vad-3/raw/main/BENCHMARKS.md).
Silero ranks best by AUC (87× larger); its stock threshold misses 44 %
of room speech, while calibrated teensy-v3/v4 hit F1 ~0.89 in rooms.

## 11. Limitations

- English speech; no music in training (music reads as "activity")
- 100 ms context: fricatives in noise are the hardest frames
- Far-end echo/crosstalk not modelled (no AEC) — use Asterisk's
  echo canceller upstream
- Weights are CC BY-NC-SA 4.0 (ESC-50 NC noise in training) — replace
  training noise for commercial use

## 12. Reproduce / inspect

Every number above is reproducible from the project scripts
(`prepare_data_v3.py --npy`, `distill_label.py`, `train_v3.py`,
`compare_all.py`, `make_charts.py`). Model files are plain `.npz`
(weights + JSON metadata) — open with `numpy.load` and inspect.
