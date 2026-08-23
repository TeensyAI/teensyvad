"""Generate golden fixtures for the Go port parity test.

Writes go/testdata/{input.raw,probs.npy,events.json} using the reference
Python stack on a deterministic synthetic clip (speech-shaped bursts over
noise). Regenerate with:
    .venv/bin/python scripts/gen_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "go" / "testdata"
MODEL = ROOT / "models" / "teensy-v4-80k.npz"

SR = 8000
DUR_S = 12.0


def synth() -> np.ndarray:
    rng = np.random.default_rng(1234)
    t = np.arange(int(SR * DUR_S)) / SR
    x = 0.01 * rng.standard_normal(len(t)).astype(np.float32)
    # five speech-like bursts: formant-ish tones with syllabic AM
    for start, dur in [(0.8, 1.6), (3.0, 2.4), (5.9, 1.1), (7.6, 2.8), (11.0, 0.8)]:
        i0, i1 = int(start * SR), min(int((start + dur) * SR), len(t))
        seg_t = t[i0:i1] - start
        f0 = 120 + 40 * np.sin(2 * np.pi * 1.3 * seg_t)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        seg = 0.4 * np.sin(phase)
        for fm, amp in [(700, 0.25), (1220, 0.15), (2600, 0.08)]:
            seg += amp * np.sin(phase * fm / 150.0)
        am = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * seg_t)  # syllables
        x[i0:i1] += (seg * am).astype(np.float32)
    peak = np.abs(x).max()
    return (0.7 * x / peak).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT))
    from teensyvad.streaming import StreamingVAD

    x = synth()
    pcm = (x * 32767).astype("<i2").tobytes()
    (OUT / "input.raw").write_bytes(pcm)

    vad = StreamingVAD(MODEL)
    probs = []
    events = []
    CH = 160  # 20 ms chunks, like Asterisk
    for i in range(0, len(pcm), CH * 2):
        for ev in vad.feed(pcm[i:i + CH * 2]):
            events.append({"type": ev.type, "t": round(ev.t, 6)})
    probs = [float(p) for p in vad.prob_history]
    for ev in vad.flush():
        events.append({"type": ev.type, "t": round(ev.t, 6)})

    np.save(OUT / "probs.npy", np.asarray(probs, dtype=np.float32))
    (OUT / "events.json").write_text(json.dumps(events))
    print(f"frames={len(probs)} events={events}")


if __name__ == "__main__":
    main()
