"""Prepare the v3 scaled training set — bigger, noisier, more real.

    .venv/bin/python scripts/prepare_data_v3.py --utts 8000

vs v1/v2 (1,200 utterances):
* speech  ~8,000 utterances from train-clean-100 (~25× more frames)
* noise   ESC-50 + **synthetic 7-talker babble** (disjoint LibriSpeech
  speakers) + **real AMI room ambience** (non-speech distant-mic
  stretches — rooms, chairs, keyboards, HVAC)
* SNR     −5 … 20 dB + clean (harder low end)
* labels  teacher (Silero) soft/hard via distill_label.py as before

Speaker hygiene: val/test keep using dev-clean/test-clean speakers —
train-clean-100 speakers are disjoint from both by LibriSpeech design.
Outputs to data/prepared_v3/ with audio saved for distillation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_pools import (build_ambience_pool, build_babble_pool,  # noqa: E402
                         build_esc50_pool, build_musan_pool, sample_crop)
from teensyvad.audio import read_wav, telephony_roundtrip, write_wav  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from prepare_data import rms, trim_silence  # noqa: E402

SR = 8000
SNRS_DB = [None, 20, 15, 10, 5, 0, -5]
# noise source mix per example (weights sum 1)
NOISE_MIX = {"esc50": 0.40, "babble": 0.40, "ambience": 0.20}
# with --musan, the musan pool replaces esc50 1:1 (same 0.40 weight);
# babble+ambience are already commercial-safe (LibriSpeech / AMI, CC BY 4.0)


def mass_convert(files, cache, workers=8):
    from noise_pools import mass_convert as mc
    return mc(files, cache, workers)


def collect_utt_speakers(root: Path, limit: int, rng):
    utts = sorted(root.rglob("*.flac"))
    # exclude dev/test splits when a corpus-wide root is passed
    utts = [u for u in utts if "/dev" not in str(u) and "/test" not in str(u)]
    rng.shuffle(utts)
    return utts[:limit]


def build_example(rng, speech, pools: dict, sr=SR):
    """→ (mixed, snr_value, speech_start_samples, speech_len_samples)."""
    # prior-balanced padding: fixed 0.15-1.0 s padding made long utterances
    # push the speech-frame prior to 91 % (the v7-360h failure). Scale the
    # noise padding so the speech fraction lands in 0.72-0.88 regardless of
    # utterance length.
    f_target = float(rng.uniform(0.72, 0.88))
    sp_sec = len(speech) / sr
    pad_total = min(sp_sec * (1.0 - f_target) / f_target, 6.0)
    lead = float(rng.uniform(0.25, 0.75)) * pad_total
    tail = pad_total - lead

    def noise(n):
        kinds = [k for k in NOISE_MIX if NOISE_MIX[k] > 0]
        w = np.array([NOISE_MIX[k] for k in kinds])
        k = kinds[rng.choice(len(kinds), p=w / w.sum())]
        return sample_crop(rng, pools[k], n)

    n_lead = int(lead * sr)
    n_tail = int(tail * sr)
    pad = np.concatenate([noise(n_lead), speech, noise(n_tail)])
    snr = SNRS_DB[rng.integers(len(SNRS_DB))]
    if snr is None:
        mixed = pad.copy()
        snr_val = np.inf
    else:
        nz = noise(len(pad))
        s_r, n_r = rms(speech), rms(nz)
        g = s_r / (max(n_r, 1e-6) * 10 ** (snr / 20.0))
        mixed = pad + g * nz
        snr_val = float(snr)
    if rng.random() < 0.4:
        mixed = telephony_roundtrip(mixed.astype(np.float32))
    mixed = (mixed * float(rng.uniform(0.4, 1.0))).astype(np.float32)
    return mixed, snr_val, n_lead, len(speech)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--utts", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=Path("data/prepared_v3"))
    ap.add_argument("--libri100", type=Path, default=Path("data/raw/LibriSpeech/train-clean-100"))
    ap.add_argument("--esc50", type=Path, default=Path("data/raw/ESC-50-master"))
    ap.add_argument("--musan", type=Path, default=None,
                    help="MUSAN root (OpenSLR 17). When set, MUSAN noise "
                         "replaces ESC-50 as the environmental noise pool — "
                         "commercial-safe training data (CC BY 4.0).")
    ap.add_argument("--ami-wav", type=Path, default=Path("data/raw/ami/wav"))
    ap.add_argument("--ami-manual", type=Path, default=Path("data/raw/ami/manual"))
    ap.add_argument("--cache", type=Path, default=Path("data/prepared_v3/wav_cache"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--npy", action="store_true",
                    help="memmap-friendly output: F.npy (float16) + y.npy + "
                         "snr.npy + clip_len.npy — required for ~100h sets")
    args = ap.parse_args()

    t0 = time.time()
    feat = LogMel(sr=SR)
    rng = np.random.default_rng(args.seed)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)

    print("building noise pools …")
    env_name = "musan" if args.musan else "esc50"
    pools = {}
    if args.musan:
        pools["musan"] = build_musan_pool(args.musan, Path("data/noise_pools"))
    else:
        pools["esc50"] = build_esc50_pool(args.esc50, Path("data/noise_pools"), {1, 2, 3})
    # AMI ambience ONLY from the 3 calibration meetings — the other 8
    # meetings are the eval set (eval_realworld.py) and must stay unseen.
    pools["ambience"] = build_ambience_pool(
        args.ami_wav, args.ami_manual, Path("data/noise_pools"),
        meetings=["ES2002a", "IS1000a", "TS3003a"],
        max_chunks_per_meeting=12)
    print(f"  {env_name}: {len(pools[env_name])} clips | ambience: {len(pools['ambience'])} clips")
    NOISE_MIX.clear()   # mutate module global — build_example's noise() reads it
    NOISE_MIX.update({env_name: 0.40, "babble": 0.40, "ambience": 0.20})
    try:
        pools["babble"] = build_babble_pool(args.libri100, Path("data/noise_pools"),
                                            n_talkers=7, n_mixes=600)
        print(f"  babble: {len(pools['babble'])} clips")
    except Exception as e:
        print(f"  !! babble unavailable ({e}); proceeding with {env_name}+ambience")
        NOISE_MIX.update({env_name: 0.6, "babble": 0.0, "ambience": 0.4})

    utts = collect_utt_speakers(args.libri100, args.utts, rng)
    print(f"speech: {len(utts)} utterances from train-clean-100")
    wavs = mass_convert(utts, args.cache / "speech")
    print(f"converted {len(wavs)} in {time.time()-t0:.0f}s")

    Fs, ys, snrs, clip_len = [], [], [], []
    audio_dir = args.out / "audio"
    for i, w in enumerate(wavs):
        speech, _ = read_wav(w)
        if len(speech) < SR:
            continue
        speech = trim_silence(speech, SR)
        if len(speech) < SR // 2:
            continue
        mixed, snr_val, n_lead, n_sp = build_example(rng, speech, pools)
        F = feat(mixed)
        centers = (feat.frame_len / 2 + np.arange(len(F)) * feat.hop_len)
        y = ((centers >= n_lead) & (centers < n_lead + n_sp)).astype(np.float32)
        # float16 halves the memory footprint; log-mel precision loss ~1e-3
        Fs.append(F.astype(np.float16 if args.npy else np.float32))
        ys.append(y)
        clip_len.append(len(F))
        snrs.append(np.full(len(F), snr_val if np.isfinite(snr_val) else 99.0,
                            dtype=np.float32))
        write_wav(audio_dir / f"train_{i:05d}.wav", mixed, SR)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}  ({time.time()-t0:.0f}s)")
    print(f"mixtures built: {len(Fs)}")

    if args.npy:
        # chunked flush into a pre-sized float16 memmap keeps peak RAM low
        total = int(sum(clip_len))
        Fm = np.lib.format.open_memmap(args.out / "F.npy", mode="w+",
                                       dtype=np.float16, shape=(total, 40))
        ym = np.lib.format.open_memmap(args.out / "y.npy", mode="w+",
                                       dtype=np.float16, shape=(total,))
        sm = np.lib.format.open_memmap(args.out / "snr.npy", mode="w+",
                                       dtype=np.float32, shape=(total,))
        pos = 0
        for Fc, yc, sc in zip(Fs, ys, snrs):
            n = len(Fc)
            Fm[pos:pos + n] = Fc
            ym[pos:pos + n] = yc
            sm[pos:pos + n] = sc
            pos += n
        Fm.flush(); ym.flush(); sm.flush()
        del Fs, ys, snrs
        np.save(args.out / "clip_len.npy", np.array(clip_len, dtype=np.int64))
        print(f"saved → {args.out}/*.npy  ({total:,} frames, "
              f"{time.time()-t0:.0f}s total)")
        return

    F = np.concatenate(Fs)
    np.savez(args.out / "train.npz", F=F, y=np.concatenate(ys),
             snr=np.concatenate(snrs), clip_len=np.array(clip_len, dtype=np.int64))
    print(f"saved → {args.out/'train.npz'}  ({len(F):,} frames, "
          f"{time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
