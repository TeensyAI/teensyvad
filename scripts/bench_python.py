"""Python-side counterpart of go/cmd/bench — same machine, same pattern:
stream synthetic 20 ms chunks through StreamingVAD, report per-hop cost.

    .venv/bin/python scripts/bench_python.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from teensyvad.streaming import StreamingVAD  # noqa: E402


def synth(seconds: int, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    sr = 8000
    t = np.arange(sr * seconds) / sr
    x = (0.01 * rng.standard_normal(len(t))).astype(np.float32)
    burst = (t % 1.6) < 1.0
    f0 = np.where(burst, 130 + 30 * np.sin(2 * np.pi * 1.2 * t), 130.0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = (np.sin(phase) + 0.34 * np.sin(3 * phase) + 0.17 * np.sin(5 * phase)) / 1.51
    x += np.where(burst, 0.35 * tone, 0.0).astype(np.float32)
    return (np.clip(x, -1, 1) * 8000).astype("<i2").tobytes()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "teensy-v4-80k.npz"))
    ap.add_argument("--secs", type=int, default=60)
    args = ap.parse_args()

    pcm = synth(args.secs)
    vad = StreamingVAD(args.model)
    ch = 320  # 20 ms
    events = 0
    t0 = time.perf_counter()
    for i in range(0, len(pcm) - ch + 1, ch):
        events += len(vad.feed(pcm[i:i + ch]))
    events += len(vad.flush())
    dt = time.perf_counter() - t0

    hops = len(pcm) / 2 / 80
    print(f"model        : {args.model}")
    print(f"audio        : {args.secs} s streamed in 20 ms chunks")
    print(f"total time   : {dt:.3f} s")
    print(f"per hop      : {dt * 1e6 / hops:.3f} µs / 10 ms frame")
    print(f"per 20 ms op : {dt * 1e6 / (hops / 2):.3f} µs")
    print(f"RTF          : {dt / args.secs:.6f} ({args.secs / dt:.0f}x real-time on one core)")
    print(f"events       : {events}, speech {vad.speech_seconds:.2f} s")


if __name__ == "__main__":
    main()
