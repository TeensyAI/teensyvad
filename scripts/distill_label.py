"""Distillation step 1: label every mixture frame with Silero VAD.

    .venv/bin/python scripts/distill_label.py            # → data/prepared/*.distill.npz

Silero v5 runs natively at 8 kHz (256-sample chunks = one decision per
32 ms).  Our student looks at 10 ms frames, so teacher probabilities are
interpolated at each frame's centre time — soft targets, not thresholds.
The student's BCE loss accepts fractional labels directly, which is
textbook knowledge distillation with zero extra machinery.

Also prints where the teacher DISAGREES with the construction labels —
that disagreement is the lesson list: intra-utterance pauses, breathy
tails, and any noise clips the teacher mistakes for speech.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import read_wav  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402

SR = 8000
CHUNK = 256                     # silero @ 8 kHz
TEACHER_THR = 0.5


def load_teacher():
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()   # TorchScript module (8 kHz mode supported)
    model.reset_states()
    model.eval()
    return model, torch


def teacher_probs(model, torch, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run Silero over a clip → (probabilities, chunk-centre times)."""
    model.reset_states()
    probs, times = [], []
    with torch.no_grad():
        for k in range(len(x) // CHUNK):
            chunk = torch.from_numpy(np.ascontiguousarray(x[k * CHUNK:(k + 1) * CHUNK]))
            out = model(chunk, SR)
            if isinstance(out, dict):             # future-proof wrapper styles
                out = out.get("pred", out.get("prob", out.get("probs")))
            probs.append(float(torch.as_tensor(out).ravel()[0]))
            times.append((k * CHUNK + CHUNK / 2) / SR)
    return np.array(probs, dtype=np.float64), np.array(times)


def label_split(model, torch, data_dir: Path, split: str, feat: LogMel) -> Path:
    # v4-style memmap dataset? (clip_len.npy present, audio/ wavs, F.npy)
    npy_mode = (data_dir / "clip_len.npy").exists() and \
               (data_dir / "F.npy").exists()
    if npy_mode:
        clip_len = np.load(data_dir / "clip_len.npy")
        assert split == "train", "npy mode currently handles the train split only"
        wavs = sorted((data_dir / "audio").glob(f"{split}_*.wav"))
        assert len(wavs) == len(clip_len), \
            f"{split}: {len(wavs)} wavs vs {len(clip_len)} clips"
        total = int(clip_len.sum())
        # teacher outputs → SEPARATE files (construction labels stay in y.npy)
        ysoft_m = np.lib.format.open_memmap(data_dir / "ysoft.npy", mode="w+",
                                            dtype=np.float16, shape=(total,))
        yteach_m = np.lib.format.open_memmap(data_dir / "yteach.npy", mode="w+",
                                             dtype=np.float16, shape=(total,))
        pos = 0
        t0 = time.time()
        for j, w in enumerate(wavs):
            x, _ = read_wav(w)
            n = int(clip_len[j])
            centers = (feat.frame_len / 2 + np.arange(n) * feat.hop_len) / SR
            if len(x) >= CHUNK:
                tp, tt = teacher_probs(model, torch, x)
                soft = np.interp(centers, tt, tp)
            else:
                soft = np.zeros(n, dtype=np.float64)
            ysoft_m[pos:pos + n] = soft.astype(np.float16)
            yteach_m[pos:pos + n] = (soft >= TEACHER_THR)
            pos += n
            if (j + 1) % 1000 == 0:
                print(f"  [{split}] {j + 1}/{len(wavs)}  ({time.time()-t0:.0f}s)",
                      flush=True)
        ysoft_m.flush(); yteach_m.flush()
        del ysoft_m, yteach_m

        hard = np.asarray(np.load(data_dir / "yteach.npy") > 0.5)
        print(f"[{split}] {total:,} frames → ysoft.npy + yteach.npy (memmap)")
        print(f"  teacher speech: {hard.mean()*100:5.1f}%")
        y_con = np.load(data_dir / "y.npy", mmap_mode="r")
        dis = hard != (np.asarray(y_con) > 0.5)
        print(f"  disagreement vs construction: {dis.mean()*100:5.1f}%")
        return data_dir / "yteach.npy"

    z = np.load(data_dir / f"{split}.npz")
    F, y_con, snr = z["F"], z["y"], z["snr"]
    clip_len = z["clip_len"]
    audio_dir = data_dir / "audio"
    wavs = sorted(audio_dir.glob(f"{split}_*.wav"))
    assert len(wavs) == len(clip_len), f"{split}: {len(wavs)} wavs vs {len(clip_len)} clips"

    soft_parts, t0 = [], time.time()
    pos = 0
    for j, w in enumerate(wavs):
        x, _ = read_wav(w)
        n = int(clip_len[j])
        # clip-LOCAL frame-centre times (must match teacher's local clock!)
        centers = (feat.frame_len / 2 + np.arange(n) * feat.hop_len) / SR
        pos += n
        if len(x) >= CHUNK:
            tp, tt = teacher_probs(model, torch, x)
            soft = np.interp(centers, tt, tp)     # edge-held at both ends
        else:
            soft = np.zeros(n, dtype=np.float64)
        soft_parts.append(soft)
        if (j + 1) % 200 == 0:
            print(f"  [{split}] {j + 1}/{len(wavs)}  ({time.time()-t0:.0f}s)", flush=True)

    soft = np.concatenate(soft_parts).astype(np.float32)
    hard = (soft >= TEACHER_THR).astype(np.float32)
    out = data_dir / f"{split}.distill.npz"
    np.savez(out, F=F, y=hard, ysoft=soft, y_construct=y_con, snr=snr,
             clip_len=clip_len)

    # ---- disagreement report: teacher vs construction --------------------
    dis = hard != y_con
    print(f"[{split}] {len(F):,} frames → {out.name}")
    print(f"  teacher speech: {hard.mean()*100:5.1f}%   construction: {y_con.mean()*100:5.1f}%")
    print(f"  disagreement  : {dis.mean()*100:5.1f}%")
    both = (y_con > .5) & (hard < .5)
    onlyt = (y_con < .5) & (hard > .5)
    print(f"    construction says speech, teacher silent: {both.sum()/len(F)*100:4.1f}%  "
          f"(intra-utterance pauses, breathy tails)")
    print(f"    teacher says speech, construction silent: {onlyt.sum()/len(F)*100:4.1f}%  "
          f"(teacher hears voice in 'noise', or placement edges)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    feat = LogMel(sr=SR)
    model, torch = load_teacher()
    print("teacher loaded (silero-vad, TorchScript, 8 kHz mode)")
    for s in args.splits:
        label_split(model, torch, args.data, s, feat)
    print("done.")


if __name__ == "__main__":
    main()
