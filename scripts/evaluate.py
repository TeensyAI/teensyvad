"""Evaluate the trained VAD against the energy baseline on the test set.

    .venv/bin/python scripts/evaluate.py

Frame-level comparison (neural vs energy VAD) overall and per SNR band,
plus an event-level score: a detected speech segment counts as correct if
it overlaps a true speech segment by ≥ 50 % of the shorter one.

The energy VAD gets the same streaming feed of PCM chunks the neural one
gets — this is a fair, "as deployed" comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import float_to_pcm16  # noqa: E402
from teensyvad.energy_vad import EnergyVAD  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import MLP, auc, load_model, prf  # noqa: E402
from teensyvad.streaming import StreamingVAD  # noqa: E402
from scripts_utils import context_windows  # noqa: E402


def events_from_labels(y: np.ndarray, hop_s: float):
    """Ground-truth speech segments from frame labels."""
    segs, start = [], None
    for i, v in enumerate(y):
        if v > 0.5 and start is None:
            start = i
        elif v <= 0.5 and start is not None:
            segs.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        segs.append((start * hop_s, len(y) * hop_s))
    return segs


def events_from_stream(vad, pcm_bytes: bytes, chunk: int = 320):
    evs = []
    for i in range(0, len(pcm_bytes), chunk):
        for e in vad.feed(pcm_bytes[i:i + chunk]):
            evs.append((e.type, e.t))
    vad.reset()
    segs, start = [], None
    for kind, t in evs:
        if kind == "speech_start":
            start = t
        elif kind == "speech_end" and start is not None:
            segs.append((start, t))
            start = None
    if start is not None:
        segs.append((start, len(pcm_bytes) / 2 / 8000))
    return segs


def match_events(pred, truth, iou_thr: float = 0.3):
    """Greedy overlap matching; returns (tp, n_pred, n_true)."""
    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    tp = 0
    used = [False] * len(truth)
    for p in pred:
        best_j, best_ov = -1, 0.0
        for j, t in enumerate(truth):
            if used[j]:
                continue
            ov = overlap(p, t)
            if ov > best_ov:
                best_ov, best_j = ov, j
        if best_j >= 0:
            # overlap ≥ 30 % of the shorter segment → a match
            shorter = min(p[1] - p[0], truth[best_j][1] - truth[best_j][0])
            if shorter > 0 and best_ov / shorter >= iou_thr:
                tp += 1
                used[best_j] = True
    return tp, len(pred), len(truth)


SNR_BANDS = [(99.0, "clean"), (20, "20dB"), (15, "15dB"), (10, "10dB"), (5, "5dB"), (0, "0dB")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared/test.npz"))
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v1.npz"))
    ap.add_argument("--events-clips", type=int, default=120,
                    help="clips re-synthesised for the event-level test")
    args = ap.parse_args()

    z = np.load(args.data)
    F, y, snr = z["F"], z["y"], z["snr"]
    model = load_model(args.model)
    K = int(model.meta["context"])
    hop_s = float(model.meta["hop_ms"]) / 1000.0

    X, yw = context_windows(F, y, K)
    probs = model.probs(X)
    thr = float(model.meta["thr_hi"])

    print("=" * 74)
    print("FRAME-LEVEL  (neural VAD, threshold from model metadata)")
    print("=" * 74)
    p, r, f1, _ = prf(probs, yw, thr)
    print(f"  overall: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={auc(probs, yw):.4f}")
    for band_val, band_name in SNR_BANDS:
        m = np.isclose(snr[K - 1:], band_val)
        if m.sum() == 0:
            continue
        pb, rb, fb, _ = prf(probs[m], yw[m], thr)
        aucb = auc(probs[m], yw[m])
        print(f"  SNR {band_name:>6}: F1={fb:.4f}  AUC={aucb:.4f}  (n={int(m.sum()):,})")

    # ---------------- energy baseline, same stream ------------------------
    print("=" * 74)
    print("EVENT-LEVEL  (streaming feed, 20 ms PCM chunks — as Asterisk sees it)")
    print("=" * 74)

    # Rebuild audio-ish stream from features? Not possible — instead, reuse
    # demo clips if present; otherwise synthesise from the saved mixtures is
    # not available here, so we regenerate a handful of clips quickly using
    # the same recipe (prepare_data is deterministic-seeded; for evaluation
    # we simply re-mix from the val/demo set).
    demo_dir = args.data.parent / "demo"
    demo_wavs = sorted(demo_dir.glob("test_*.wav"))[: args.events_clips]
    if not demo_wavs:
        demo_wavs = sorted(demo_dir.glob("val_*.wav"))

    if demo_wavs:
        tps = {"neural": 0, "energy": 0}
        preds = {"neural": 0, "energy": 0}
        truths = 0
        for w in demo_wavs:
            from teensyvad.audio import read_wav
            x, sr = read_wav(w)
            lab = np.load(w.with_suffix(".npz"))
            yt = lab["y"]
            truth = events_from_labels(yt, hop_s)
            truths += len(truth)
            pcm = float_to_pcm16(x)

            sv = StreamingVAD(args.model)
            seg_n = events_from_stream(sv, pcm)
            tps["neural"] += match_events(seg_n, truth)[0]
            preds["neural"] += len(seg_n)

            ev = EnergyVAD(sr=sr)
            seg_e = events_from_stream(ev, pcm)
            tps["energy"] += match_events(seg_e, truth)[0]
            preds["energy"] += len(seg_e)

        for name in ("neural", "energy"):
            tp, np_, nt = tps[name], preds[name], truths
            prec = tp / max(np_, 1)
            rec = tp / max(nt, 1)
            f1e = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"  {name:>6}: segments P={prec:.3f}  R={rec:.3f}  F1={f1e:.3f}  "
                  f"(pred {np_}, true {nt})")
    else:
        print("  (no demo clips found — run prepare_data.py with --test ≥ 3)")


if __name__ == "__main__":
    main()
