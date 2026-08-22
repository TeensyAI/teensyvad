"""Speed benchmark: how "ultra-fast" is teensyvad, really?

    .venv/bin/python scripts/benchmark.py

Measures, on this machine:
* per-frame model inference (batch and one-at-a-time);
* full StreamingVAD.feed() on 20 ms telephony chunks (the Asterisk path);
* the energy baseline for scale;
* resulting real-time factors (RTF < 1 → faster than real time; we're
  aiming for RTF ≈ 0.001, i.e. ~1000× real time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import float_to_pcm16  # noqa: E402
from teensyvad.energy_vad import EnergyVAD  # noqa: E402
from teensyvad.model import MLP  # noqa: E402
from teensyvad.streaming import StreamingVAD, load_model  # noqa: E402


def bench(fn, n_iter: int, warmup: int = 5) -> float:
    """Median seconds per call over n_iter runs."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v1.npz"))
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    model = load_model(args.model)
    K = int(model.meta["context"])
    dim = int(model.meta["n_mels"]) * (2 if model.meta.get("deltas", True) else 1)

    print(f"model: {model.sizes}  {model.n_params():,} params  "
          f"({args.model.stat().st_size/1024:.0f} KB on disk)")

    # 1) raw model inference ------------------------------------------------
    X = rng.normal(size=(1, K * dim)).astype(np.float32)
    t1 = bench(lambda: model.probs(X), 2000)
    Xb = rng.normal(size=(1000, K * dim)).astype(np.float32)
    tb = bench(lambda: model.probs(Xb), 200) / 1000
    print(f"inference        : {t1*1e6:8.1f} µs/frame (single)   "
          f"{tb*1e6:8.1f} µs/frame (batched ×1000)")

    # 2) full streaming path on 20 ms telephony chunks ----------------------
    sr = int(model.meta.get("sr", 8000))
    x = (0.1 * rng.normal(size=int(args.seconds * sr))).astype(np.float32)
    pcm = float_to_pcm16(x)
    chunk = 320  # 20 ms @ 8 kHz, 16-bit — exactly one AudioSocket payload
    vad = StreamingVAD(args.model)
    t_chunk = bench(lambda: vad.feed(pcm[:chunk]), n_iter=int(args.seconds * 50), warmup=50)
    frames_per_chunk = 2        # 10 ms feature hop, 20 ms chunks
    per_frame = t_chunk / frames_per_chunk
    rtf = t_chunk / 0.02
    print(f"StreamingVAD     : {t_chunk*1e6:8.1f} µs / 20 ms chunk   "
          f"≈{per_frame*1e6:.1f} µs/frame   RTF={rtf:.5f} "
          f"({1/rtf:,.0f}× real time)")

    # 3) energy baseline ----------------------------------------------------
    ev = EnergyVAD(sr=sr)
    t_ev = bench(lambda: ev.feed(pcm[:chunk]), n_iter=int(args.seconds * 50), warmup=50)
    print(f"EnergyVAD        : {t_ev*1e6:8.1f} µs / 20 ms chunk   RTF={t_ev/0.02:.6f}")

    print(json.dumps({
        "model_params": model.n_params(),
        "model_kb": round(args.model.stat().st_size / 1024, 1),
        "us_per_frame_single": round(t1 * 1e6, 2),
        "us_per_frame_batched": round(tb * 1e6, 2),
        "us_per_20ms_chunk_streaming": round(t_chunk * 1e6, 1),
        "rtf_streaming": round(rtf, 6),
        "times_realtime": round(1 / rtf),
    }, indent=2))


if __name__ == "__main__":
    main()
