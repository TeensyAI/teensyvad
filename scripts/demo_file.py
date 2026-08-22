"""Try teensyvad on a wav file — no Asterisk required.

    .venv/bin/python scripts/demo_file.py path/to/audio.wav

Prints a speech/nonspeech timeline and (with --plot) draws
probability + decisions against the spectrogram.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import float_to_pcm16, load_wav  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import load_model  # noqa: E402
from teensyvad.streaming import StreamingVAD  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav")
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v1.npz"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    args = ap.parse_args()

    x = load_wav(args.wav, sr=8000)
    pcm = float_to_pcm16(x)
    vad = StreamingVAD(args.model)
    chunk = int(len(pcm) * 0 + 8000 * 2 * args.chunk_ms / 1000)  # bytes per chunk

    events = []
    for i in range(0, len(pcm), chunk):
        events += vad.feed(pcm[i:i + chunk])

    # timeline bar
    p = np.array(vad.prob_history)
    n = len(p)
    width = min(100, n)
    col_of = lambda i: int(i * width / max(n, 1))
    cols = [" "] * width
    marks = np.zeros(width, dtype=bool)
    for ev in events:
        marks[col_of(int(ev.t / (float(vad.hop_s))))] = True
    speech = p >= (vad.thr_hi + vad.thr_lo) / 2
    for i in range(width):
        lo, hi = int(i * n / width), int((i + 1) * n / width)
        cols[i] = "█" if speech[lo:hi].mean() > 0.5 else "·"
    print(f"{'0s':>6} {'timeline':^{width}}  {len(x)/8000:.0f}s")
    print(f"{'':>6} {''.join(cols)}")
    print(f"{'':>6} speech: {vad.speech_seconds:.1f}s / {len(x)/8000:.1f}s "
          f"({vad.speech_seconds/ (len(x)/8000) * 100:.0f}%)")
    for ev in events:
        print(f"  {ev.t:8.2f}s  {ev.type}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model = load_model(args.model)
        lm = LogMel(sr=8000)
        F = lm(x)
        t = (np.arange(len(F)) * lm.hop_len + lm.frame_len / 2) / lm.sr
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].imshow(F[:, :20].T, origin="lower", aspect="auto",
                     extent=[0, len(x) / 8000, 0, 20], cmap="magma")
        ax[0].set_ylabel("mel band")
        ax[0].set_title(f"{args.wav}  —  log-mel + teensyvad")
        ax[1].plot(t, p, lw=0.8)
        ax[1].axhline(vad.thr_hi, color="r", ls="--", lw=0.8)
        ax[1].axhline(vad.thr_lo, color="orange", ls=":", lw=0.8)
        ax[1].fill_between(t, 0, 1, where=speech, alpha=0.15, color="g")
        for ev in events:
            ax[1].axvline(ev.t, color="g" if ev.type == "speech_start" else "r",
                          ls="--", lw=0.8)
        ax[1].set_xlabel("seconds"); ax[1].set_ylabel("P(speech)")
        ax[1].set_ylim(-0.02, 1.02)
        out = Path(args.wav).with_suffix(".vad.png")
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"plot → {out}")


if __name__ == "__main__":
    main()
