"""Build the teensyvad training set: speech + noise → labelled frames.

Pipeline
--------
1. Pick LibriSpeech utterances (dev-clean → train/val split **by speaker**,
   test-clean → test; nobody in val/test has ever been heard in train).
2. Convert FLAC → 8 kHz mono WAV with macOS `afconvert` (no deps), cached.
3. ESC-50 supplies noise, split **by fold** (train 1–3, val 4, test 5) so
   evaluation faces noise *types* the model never heard.  Human-vocal
   ESC-50 categories (coughing, laughing, …) are excluded — they'd poison
   the "not speech" labels.
4. Each example: an utterance padded with 0.3–2 s of noise on both sides,
   mixed at a random SNR, 40 % of the time pushed through a µ-law
   round-trip (simulating G.711 telephony), then a random gain.
   Labels: frame centre inside the utterance → speech.
5. Log-mel(+Δ) features extracted once and stored as flat (T, 40) arrays;
   context windows are stacked on the fly at train time (memory-light).

Output: data/prepared/{train,val,test}.npz  +  demo wavs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import read_wav, telephony_roundtrip, write_wav  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402

# ESC-50 categories that contain human vocal tract activity → excluded.
HUMAN_CATEGORIES = {
    "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
    "snoring", "drinking_sipping", "sneezing",
}
FOLD_OF_SPLIT = {"train": {1, 2, 3}, "val": {4}, "test": {5}}
SNRS_DB = [None, 20, 15, 10, 5, 0]      # None = clean (noise still mixed in tiny amount? no: clean)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------

def collect_libri(root: Path, subset: str) -> list[Path]:
    """All .flac utterances under a LibriSpeech subset, sorted."""
    base = root / subset
    utts = sorted(base.rglob("*.flac"))
    if not utts:
        raise FileNotFoundError(f"no .flac under {base}")
    return utts


def split_by_speaker(utts: list[Path], n_val_speakers: int):
    """Deterministic speaker-disjoint split (speakers, not files)."""
    speakers = sorted({u.parent.parent.name for u in utts})
    val_spk = set(speakers[-n_val_speakers:]) if n_val_speakers else set()
    tr = [u for u in utts if u.parent.parent.name not in val_spk]
    va = [u for u in utts if u.parent.parent.name in val_spk]
    return tr, va


def load_esc50(meta_csv: Path, audio_dir: Path, split: str) -> list[Path]:
    """ESC-50 clips for a split (by fold), minus human-vocal categories."""
    folds = FOLD_OF_SPLIT[split]
    out = []
    with open(meta_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["fold"]) in folds and row["category"] not in HUMAN_CATEGORIES:
                out.append(audio_dir / row["filename"])
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ESC-50 files missing, e.g. {missing[0]}")
    return out


def convert_to_wav8k(src: Path, dst: Path) -> Path:
    """FLAC/44.1k wav → 8 kHz 16-bit mono wav via macOS afconvert (cached)."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@8000", str(src), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"afconvert failed on {src}: {r.stderr.strip()}")
    return dst


def mass_convert(files: list[Path], cache: Path, workers: int = 8) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    jobs, out = [], []
    for src in files:
        dst = cache / (src.stem + ".wav")
        jobs.append((src, dst))
        out.append(dst)
    todo = [(s, d) for s, d in jobs if not d.exists()]
    if todo:
        log(f"  afconvert: {len(todo)} files …")
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(convert_to_wav8k, s, d) for s, d in todo]
            for i, fu in enumerate(cf.as_completed(futs)):
                fu.result()
                if (i + 1) % 200 == 0:
                    log(f"    {i + 1}/{len(todo)}")
    return out


# --------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def trim_silence(x: np.ndarray, sr: int, win_ms: float = 20.0,
                 guard_ms: float = 60.0, thr_db: float = 25.0) -> np.ndarray:
    """Trim the studio silence LibriSpeech pads around each utterance.

    Those lead/tail frames contain no speech — labelling them "speech"
    would teach the model that plain noise is voice (and cap every metric).
    We cut everything quieter than (utterance peak − 25 dB), keeping a
    60 ms guard so onsets stay natural.
    """
    win = int(win_ms / 1000 * sr)
    if len(x) < 3 * win:
        return x
    n = len(x) // win
    fr = x[: n * win].reshape(n, win)
    db = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1)) + 1e-10)
    speechy = db > db.max() - thr_db
    if not speechy.any():
        return x
    first = int(np.argmax(speechy))
    last = n - 1 - int(np.argmax(speechy[::-1]))
    guard = int(round(guard_ms / win_ms))
    first, last = max(0, first - guard), min(n - 1, last + guard)
    return x[first * win: (last + 1) * win].copy()


def noise_crop(rng: np.random.Generator, pool: list[np.ndarray], seconds: int, sr: int) -> np.ndarray:
    """Random noise segment of `seconds`, tiled/wrapped from a random clip."""
    src = pool[rng.integers(len(pool))]
    n = seconds * sr
    if len(src) >= n:
        start = rng.integers(0, len(src) - n + 1)
        return src[start:start + n].copy()
    reps = int(np.ceil(n / len(src)))
    tiled = np.concatenate([src] * reps)
    start = rng.integers(0, len(tiled) - n + 1)
    return tiled[start:start + n].copy()


def build_example(rng, speech: np.ndarray, noise_pool, sr: int, feat: LogMel):
    """One training clip: [noise lead | utterance | noise tail] mixed at SNR."""
    lead = float(rng.uniform(0.15, 1.0))
    tail = float(rng.uniform(0.15, 1.0))
    pad = np.concatenate([
        noise_crop(rng, noise_pool, int(np.ceil(lead)) + 1, sr)[: int(lead * sr)],
        speech,
        noise_crop(rng, noise_pool, int(np.ceil(tail)) + 1, sr)[: int(tail * sr)],
    ])

    snr = SNRS_DB[rng.integers(len(SNRS_DB))]
    if snr is None:
        mixed = speech_padded = pad.copy()          # clean path
        snr_val = np.inf
    else:
        noise = noise_crop(rng, noise_pool, int(np.ceil(len(pad) / sr)) + 1, sr)[: len(pad)]
        s_r, n_r = rms(speech), rms(noise)
        if n_r < 1e-6:
            n_r = 1e-6
        # want 20·log10(s_r/(g·n_r)) = snr  →  g = s_r / (n_r · 10^(snr/20))
        g = s_r / (n_r * 10 ** (snr / 20.0))
        mixed = pad + g * noise
        snr_val = float(snr)

    if rng.random() < 0.4:                          # G.711 telephony simulation
        mixed = telephony_roundtrip(mixed.astype(np.float32))
    mixed = (mixed * float(rng.uniform(0.4, 1.0))).astype(np.float32)

    n_speech = len(speech)
    n_lead = int(lead * sr)
    F = feat(mixed)                                 # (T, 40)
    T = len(F)
    # label by frame CENTRE; centres are at win/2 + t·hop samples
    centers = feat.frame_len / 2 + np.arange(T) * feat.hop_len
    y = ((centers >= n_lead) & (centers < n_lead + n_speech)).astype(np.float32)
    snr_col = np.full(T, snr_val if np.isfinite(snr_val) else 99.0, dtype=np.float32)
    return mixed, F, y, snr_col


def build_split(name: str, utts: list[Path], noise_pool_paths: list[Path],
                out_dir: Path, cache: Path, feat: LogMel, rng,
                max_utts: int, save_demo: int = 0):
    log(f"[{name}] {len(utts)} utterances available, using {min(len(utts), max_utts)}")
    utts = utts[:max_utts]
    wavs = mass_convert(utts, cache / "speech")
    noise_wavs = mass_convert(noise_pool_paths, cache / "noise")
    noise_pool = [read_wav(p)[0] for p in noise_wavs]
    noise_pool = [n for n in noise_pool if len(n) >= feat.sr]      # drop degenerate
    log(f"[{name}] noise pool: {len(noise_pool)} clips "
        f"({sum(len(n) for n in noise_pool)/feat.sr/60:.1f} min)")

    Fs, ys, snrs = [], [], []
    demo_dir = out_dir / "demo"
    for i, w in enumerate(wavs):
        speech, _ = read_wav(w)
        if len(speech) < feat.sr:                   # <1 s: skip
            continue
        speech = trim_silence(speech, feat.sr)
        if len(speech) < feat.sr // 2:              # nothing left after trim
            continue
        mixed, F, y, s = build_example(rng, speech, noise_pool, feat.sr, feat)
        Fs.append(F.astype(np.float32))
        ys.append(y)
        snrs.append(s)
        if i < save_demo:
            demo_dir.mkdir(parents=True, exist_ok=True)
            write_wav(demo_dir / f"{name}_{i:02d}.wav", mixed, feat.sr)
            np.savez(demo_dir / f"{name}_{i:02d}.npz", y=y, snr=s)
        if (i + 1) % 200 == 0:
            log(f"  [{name}] {i + 1}/{len(wavs)} clips")

    # 15 % extra pure-noise examples (hard negatives: no speech at all)
    for _ in range(int(0.15 * len(wavs))):
        secs = int(rng.integers(2, 6))
        x = noise_crop(rng, noise_pool, secs, feat.sr)
        F = feat(x)
        Fs.append(F.astype(np.float32))
        ys.append(np.zeros(len(F), dtype=np.float32))
        snrs.append(np.full(len(F), -1.0, dtype=np.float32))  # marker: noise-only

    F = np.concatenate(Fs); y = np.concatenate(ys); s = np.concatenate(snrs)
    p = out_dir / f"{name}.npz"
    np.savez(p, F=F, y=y, snr=s)
    pos = y.mean() * 100
    log(f"[{name}] saved {p.name}: {len(y)} frames, {pos:.1f} % speech, "
        f"{len(F)/1e6:.2f}M×{F.shape[1]} feats")
    return p


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libri", type=Path, default=Path("data/raw/LibriSpeech"))
    ap.add_argument("--esc50", type=Path, default=Path("data/raw/ESC-50-master"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared"))
    ap.add_argument("--cache", type=Path, default=Path("data/prepared/wav_cache"))
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    feat = LogMel(sr=8000)
    args.out.mkdir(parents=True, exist_ok=True)

    dev = collect_libri(args.libri, "dev-clean")
    tst = collect_libri(args.libri, "test-clean")
    tr_utts, va_utts = split_by_speaker(dev, n_val_speakers=8)
    log(f"dev-clean: {len(dev)} utts → {len(tr_utts)} train / {len(va_utts)} val (speaker-disjoint)")

    meta = args.esc50 / "meta" / "esc50.csv"
    audio = args.esc50 / "audio"
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tr_utts)   # deterministic shuffles keep splits reproducible
    rng2 = np.random.default_rng(args.seed + 1)
    rng2.shuffle(va_utts)
    rng3 = np.random.default_rng(args.seed + 2)
    rng3.shuffle(tst)

    build_split("train", tr_utts, load_esc50(meta, audio, "train"),
                args.out, args.cache, feat, np.random.default_rng(100), args.train)
    build_split("val", va_utts, load_esc50(meta, audio, "val"),
                args.out, args.cache, feat, np.random.default_rng(200), args.val, save_demo=2)
    build_split("test", tst, load_esc50(meta, audio, "test"),
                args.out, args.cache, feat, np.random.default_rng(300), args.test, save_demo=3)
    log("done.")


if __name__ == "__main__":
    main()
