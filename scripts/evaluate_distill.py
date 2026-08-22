"""Evaluate a distilled teensyvad model against its Silero teacher.

    .venv/bin/python scripts/evaluate_distill.py --model models/teensy-v2.npz

Measures what distillation promises:
1. frame-level agreement with the teacher (corr, F1, disagreement rate);
2. onset/offset timing vs teacher events (median |Δ| in ms);
3. event F1 vs construction labels (did we keep real-world utility?);
4. speed: student vs teacher on the same 60 s of audio — the payoff.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.audio import read_wav  # noqa: E402
from teensyvad.model import load_model, prf  # noqa: E402
from teensyvad.streaming import StreamingVAD  # noqa: E402
from scripts.evaluate import events_from_labels, events_from_stream, match_events  # noqa: E402
from scripts.distill_label import load_teacher, teacher_probs, TEACHER_THR  # noqa: E402

SR = 8000


def boundaries_from_events(evs):
    """[(start,end)] from StreamingVAD events list."""
    segs, start = [], None
    for e in evs:
        if e.type == "speech_start":
            start = e.t
        elif e.type == "speech_end" and start is not None:
            segs.append((start, e.t))
            start = None
    return segs


def onset_offset_deltas(pred_segs, teacher_segs):
    """Match segments greedily by centre distance; return (onset_ds, offset_ds) seconds."""
    on, off = [], []
    used = set()
    for p in pred_segs:
        best, bd = None, 1e9
        for j, t in enumerate(teacher_segs):
            if j in used:
                continue
            d = abs((p[0] + p[1]) / 2 - (t[0] + t[1]) / 2)
            if d < bd:
                bd, best = d, j
        if best is not None and bd < 0.5:
            used.add(best)
            on.append(p[0] - teacher_segs[best][0])
            off.append(p[1] - teacher_segs[best][1])
    return np.array(on), np.array(off)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v2.npz"))
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--clips", type=int, default=100)
    args = ap.parse_args()

    # ---- 1) frame agreement on labelled test features ---------------------
    z = np.load(args.data / "test.distill.npz")
    from scripts_utils import context_windows
    model = load_model(args.model)
    K = int(model.meta["context"])
    X, _ = context_windows(z["F"], z["F"][:, 0], K)     # dummy y
    ysoft_t = z["ysoft"][K - 1:]
    hard_t = z["y"][K - 1:]
    p_s = model.probs(X)

    corr = float(np.corrcoef(p_s, ysoft_t)[0, 1])
    dis = float(np.mean((p_s >= 0.5) != (hard_t >= 0.5)))
    f1_t = prf(p_s, hard_t, 0.5)[2]
    print("=" * 70)
    print("FRAME AGREEMENT WITH TEACHER (test set)")
    print("=" * 70)
    print(f"  Pearson r (soft): {corr:.4f}")
    print(f"  F1 vs teacher decisions: {f1_t:.4f}   frame disagreement: {dis*100:.1f}%")

    # ---- 2) streaming: boundaries + events on real wavs --------------------
    print("=" * 70)
    print(f"STREAMING ON {args.clips} TEST CLIPS (vs teacher run live)")
    print("=" * 70)
    teacher, torch = load_teacher()
    hop_s = float(model.meta["hop_ms"]) / 1000.0

    on_ds, off_ds = [], []
    ev_p, ev_t, ev_true = [], [], []
    wavs = sorted((args.data / "audio").glob("test_*.wav"))[: args.clips]
    for w in wavs:
        x, _ = read_wav(w)
        y = None
        npz = w.with_suffix(".npz")
        lab = np.load(args.data / "test.npz")
        # construction labels come from the distill file, aligned by clip order
        # (audio/test_XXXXX.wav ↔ clip_len order is identical)
        idx = int(w.stem.split("_")[1])
        cl = lab["clip_len"]
        pos = int(np.sum(cl[:idx]))
        y = lab["y"][pos: pos + int(cl[idx])]
        truth = events_from_labels(y, hop_s)

        sv = StreamingVAD(args.model)
        events = []
        import numpy as _np
        pcm_bytes = (x * 32768).astype("<i2").tobytes()
        for i in range(0, len(pcm_bytes), 320):
            events += sv.feed(pcm_bytes[i:i + 320])
        events += sv.flush()                 # close segment open at EOF
        segs_s = boundaries_from_events(events)

        tp, tt = teacher_probs(teacher, torch, x)
        centers = model.meta and (_np.arange(len(x) // 256) * 256 + 128) / SR
        tseg_mask = _np.interp(centers, tt, tp) >= TEACHER_THR
        segs_t = events_from_labels(tseg_mask.astype(float), 256 / SR)

        o, f = onset_offset_deltas(segs_s, segs_t)
        on_ds += list(o); off_ds += list(f)
        ev_t.append(match_events(segs_s, truth))

    tp_all = sum(t for t, _, _ in ev_t); np_all = sum(n for _, n, _ in ev_t)
    nt_all = sum(n for _, _, n in ev_t)
    prec = tp_all / max(np_all, 1); rec = tp_all / max(nt_all, 1)
    f1e = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  events vs construction truth: P={prec:.3f} R={rec:.3f} F1={f1e:.3f} "
          f"({np_all} pred / {nt_all} true)")
    if on_ds:
        on_ds, off_ds = np.array(on_ds), np.array(off_ds)
        print(f"  onset  Δ vs teacher: median {np.median(on_ds)*1000:+7.1f} ms  "
              f"(|Δ| median {np.median(np.abs(on_ds))*1000:5.1f} ms)")
        print(f"  offset Δ vs teacher: median {np.median(off_ds)*1000:+7.1f} ms  "
              f"(|Δ| median {np.median(np.abs(off_ds))*1000:5.1f} ms)")

    # ---- 3) speed ----------------------------------------------------------
    print("=" * 70)
    print("SPEED (60 s of 8 kHz audio, same machine)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.3, 0.3, SR * 60).astype(np.float32)
    pcm = (x * 32768).astype("<i2").tobytes()

    sv = StreamingVAD(args.model)
    sv.feed(pcm[:320])                     # warm
    t0 = time.perf_counter()
    for i in range(0, len(pcm), 320):
        sv.feed(pcm[i:i + 320])
    t_student = time.perf_counter() - t0

    teacher.reset_states()
    with torch.no_grad():
        teacher(torch.from_numpy(x[:256]), SR)   # warm
        t0 = time.perf_counter()
        for k in range(len(x) // 256):
            teacher(torch.from_numpy(np.ascontiguousarray(x[k*256:(k+1)*256])), SR)
    t_teacher = time.perf_counter() - t0

    print(f"  student (teensyvad) : {t_student*1000:7.1f} ms")
    print(f"  teacher (silero)    : {t_teacher*1000:7.1f} ms")
    print(f"  → student is {t_teacher/t_student:.1f}× faster than its teacher")


if __name__ == "__main__":
    main()
