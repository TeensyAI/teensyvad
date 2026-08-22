"""Noise pools for mixture construction — the v3 data engine.

v1/v2 trained on ESC-50 environmental noise only.  Real calls have two
more realities:

* **babble** — other people talking in the background (offices, cafés,
  call centers).  Synthesised by summing N disjoint LibriSpeech speakers
  (the classic recipe: ≥6 talkers sounds like noise, not speech).
* **real room ambience** — AMI distant-mic non-speech stretches: chairs,
  keyboards, laptops, HVAC, paper — actual room tone from actual rooms.

This module builds and caches pools as flat float32 memmap-able arrays
plus per-clip boundaries, with a shared sampling API used by
prepare_data.py.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import read_wav  # noqa: E402

SR = 8000


def convert_to_wav8k(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@8000",
                        str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"afconvert failed {src}: {r.stderr.strip()}")


def mass_convert(files: list[Path], cache: Path, workers: int = 8) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    out = [cache / (f.stem + ".wav") for f in files]
    todo = [(s, d) for s, d in zip(files, out) if not d.exists()]
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda p: convert_to_wav8k(*p), todo))
    return out


def load_pool(wavs: list[Path], min_sec: float = 1.0) -> list[np.ndarray]:
    pool = []
    for w in wavs:
        x, _ = read_wav(w)
        if len(x) >= int(min_sec * SR):
            pool.append(x)
    return pool


def sample_crop(rng: np.random.Generator, pool: list[np.ndarray], n: int) -> np.ndarray:
    """Random contiguous crop of n samples from a random pool clip (tiles if short)."""
    if not pool:
        raise ValueError("empty noise pool")
    src = pool[rng.integers(len(pool))]
    if len(src) >= n:
        s = rng.integers(0, len(src) - n + 1)
        return src[s:s + n].copy()
    reps = int(np.ceil(n / len(src)))
    tiled = np.concatenate([src] * reps)
    s = rng.integers(0, len(tiled) - n + 1)
    return tiled[s:s + n].copy()


# --------------------------------------------------------------------------
# Pool builders (each cached under data/noise_pools/)
# --------------------------------------------------------------------------

def build_esc50_pool(esc50_dir: Path, cache_root: Path, split_folds) -> list[np.ndarray]:
    """ESC-50 clips by fold, minus human-vocal categories (as in v1/v2)."""
    import csv
    meta = esc50_dir / "meta" / "esc50.csv"
    audio = esc50_dir / "audio"
    HUMAN = {"breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
             "snoring", "drinking_sipping", "sneezing"}
    files = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["fold"]) in split_folds and row["category"] not in HUMAN:
                files.append(audio / row["filename"])
    wavs = mass_convert(files, cache_root / "esc50")
    return load_pool(wavs)


def build_babble_pool(libri_root: Path, cache_root: Path,
                      n_talkers: int = 7, n_mixes: int = 400,
                      min_sec: float = 4.0, max_sec: float = 12.0,
                      seed: int = 42, exclude_speakers: set[str] | None = None,
                      cache_name: str = "babble") -> list[np.ndarray]:
    """Sum n_talkers disjoint speakers → babble noise clips.

    `exclude_speakers` keeps babble talkers disjoint from any speech
    sources we might evaluate against (hygiene, not strictly required
    for noise).
    """
    cache_dir = cache_root / cache_name
    done_marker = cache_dir / "DONE"
    if done_marker.exists():
        return load_pool(sorted(cache_dir.glob("*.wav")))
    utts = sorted(libri_root.rglob("*.flac"))
    # group utterances by speaker
    by_spk: dict[str, list[Path]] = {}
    for u in utts:
        spk = u.parent.parent.name
        if exclude_speakers and spk in exclude_speakers:
            continue
        by_spk.setdefault(spk, []).append(u)
    speakers = sorted(by_spk)
    rng = np.random.default_rng(seed)
    files = []
    for i in range(n_mixes):
        chosen = rng.choice(speakers, size=n_talkers, replace=False)
        picks = [by_spk[s][rng.integers(len(by_spk[s]))] for s in chosen]
        files.append(picks)
    # convert once, flat
    flat = sorted({p for grp in files for p in grp})
    converted = {p.stem: q for p, q in zip(
        flat, mass_convert(flat, cache_root / "babble_src"))}
    from teensyvad.audio import write_wav
    out_dir = cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, grp in enumerate(files):
        dur = float(rng.uniform(min_sec, max_sec))
        n = int(dur * SR)
        mix = np.zeros(n, dtype=np.float32)
        for p in grp:
            x, _ = read_wav(converted[p.stem])
            if len(x) < n:
                reps = int(np.ceil(n / len(x)))
                x = np.concatenate([x] * reps)
            s = rng.integers(0, len(x) - n + 1)
            mix += x[s:s + n] / n_talkers
        write_wav(out_dir / f"babble_{i:04d}.wav", mix, SR)
    done_marker.write_text("ok")
    return load_pool(sorted(out_dir.glob("*.wav")))


def build_ambience_pool(ami_wav_dir: Path, ami_manual_dir: Path, cache_root: Path,
                        meetings: list[str] | None = None,
                        min_chunk: float = 2.0, max_chunks_per_meeting: int = 8,
                        seed: int = 7) -> list[np.ndarray]:
    """Non-speech stretches of AMI distant-mic audio → real room ambience.

    Parses the manual segment XMLs (same parser as eval_realworld),
    finds gaps ≥ min_chunk*2 seconds, cuts random chunks from them.
    Only used for TRAINING noise — never for evaluation.
    """
    import re
    cache_dir = cache_root / "ambience"
    done_marker = cache_dir / "DONE"
    if done_marker.exists():
        return load_pool(sorted(cache_dir.glob("*.wav")))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_realworld import parse_ami_segments
    rng = np.random.default_rng(seed)
    out_dir = cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from teensyvad.audio import write_wav, resample_fft
    wavs = sorted(ami_wav_dir.glob("*.Array1-01.wav"))
    if meetings:
        wavs = [w for w in wavs if w.name.split(".")[0] in meetings]
    n_saved = 0
    for w in wavs:
        meeting = w.name.split(".")[0]
        spans = parse_ami_segments(meeting, ami_manual_dir)
        if len(spans) == 0:
            continue
        x, sr_in = read_wav(w)
        if sr_in != SR:
            x = resample_fft(x, sr_in, SR)
        dur = len(x) / SR
        # gaps between speech spans (and before/after)
        gaps = []
        prev_end = 0.0
        for s, e in spans:
            if s - prev_end >= min_chunk * 2:
                gaps.append((prev_end, s))
            prev_end = max(prev_end, e)
        if dur - prev_end >= min_chunk * 2:
            gaps.append((prev_end, dur))
        rng.shuffle(gaps)
        for gi, (gs, ge) in enumerate(gaps[:max_chunks_per_meeting]):
            length = float(rng.uniform(min_chunk, min(ge - gs, 10.0)))
            start = rng.uniform(gs, ge - length)
            i0, i1 = int(start * SR), int((start + length) * SR)
            chunk = x[i0:i1].astype(np.float32)
            write_wav(out_dir / f"amb_{meeting}_{gi:02d}.wav", chunk, SR)
            n_saved += 1
    done_marker.write_text(f"ok {n_saved}")
    return load_pool(sorted(out_dir.glob("*.wav")))
