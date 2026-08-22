"""Grand comparison: every teensy model vs Energy / WebRTC / Silero VAD.

    .venv/bin/python scripts/compare_all.py            # → comparison.json + markdown

One harness, one protocol, three real-world scorings:
  * TEN VAD public set (30 human-labelled recordings) — AUC + best-thr F1
  * AMI SDM meetings (8 eval meetings, manual labels)  — F1/AUC at
    AMI-dev-calibrated thresholds (same protocol for every model)
  * speed on 20 ms telephony chunks (RTF) + params/size

Baselines: adaptive energy VAD, Google WebRTC VAD (aggressiveness chosen
on AMI dev meetings, like every other operating point here), and Silero
VAD (the distillation teacher).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensyvad.audio import float_to_pcm16  # noqa: E402
from teensyvad.energy_vad import EnergyVAD  # noqa: E402
from teensyvad.features import LogMel  # noqa: E402
from teensyvad.model import auc, prf  # noqa: E402
from teensyvad.quant import load_any  # noqa: E402
from teensyvad.streaming import StreamingVAD  # noqa: E402
from scripts_utils import context_windows  # noqa: E402
from scripts.eval_realworld import (SR, centers_for, eval_ten,  # noqa: E402
                                    frame_truth, load_any_rate, parse_ami_segments)
from scripts.distill_label import load_teacher, teacher_probs  # noqa: E402

HOP_S = 0.01
EVAL_MEETINGS_EXCL = ("ES2002a", "IS1000a", "TS3003a")   # = calibration meetings


# ---------------------------------------------------------------- webrtc

def webrtc_decisions(x8k: np.ndarray, aggressiveness: int = 2) -> np.ndarray:
    """WebRTC VAD decisions per 10 ms frame @ 8 kHz.

    NOTE: pcm is BYTES — 80 samples × 2 bytes = 160 bytes per 10 ms frame.
    """
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    pcm = float_to_pcm16(x8k)
    n = 160                                    # bytes per 10 ms @ 8 kHz int16
    out = []
    for i in range(0, len(pcm) - n + 1, n):
        try:
            out.append(1.0 if vad.is_speech(pcm[i:i + n], SR) else 0.0)
        except Exception:
            out.append(0.0)
    return np.array(out, dtype=float)


def pick_webrtc_aggressiveness(lm: LogMel) -> int:
    """Choose aggressiveness on AMI dev meetings (frame F1) — same protocol
    used for every model's operating point."""
    wav_dir = Path("data/raw/ami/wav")
    manual = Path("data/raw/ami/manual")
    best = (None, -1.0)
    for agg in range(4):
        P, Y = [], []
        for meeting in EVAL_MEETINGS_EXCL:
            w = wav_dir / f"{meeting}.Array1-01.wav"
            spans = parse_ami_segments(meeting, manual)
            x = load_any_rate(w)
            F = lm(x)
            centers = centers_for(len(F), lm)
            truth = frame_truth(centers, spans)
            d = webrtc_decisions(x, agg)
            n = min(len(d), len(truth))
            P.append(d[:n]); Y.append(truth[:n])
        P = np.concatenate(P); Y = np.concatenate(Y).astype(float)
        f1 = prf(P, Y, 0.5)[2]
        print(f"  webrtc agg={agg}: AMI-dev frame F1 {f1:.4f}")
        if f1 > best[1]:
            best = (agg, f1)
    print(f"  → webrtc aggressiveness {best[0]} chosen on AMI dev")
    return best[0]


# ---------------------------------------------------------------- eval

def eval_ami_models(models: dict, lm: LogMel, verbose=False):
    """AMI SDM eval meetings (calibration meetings excluded), all models."""
    wav_dir = Path("data/raw/ami/wav")
    manual = Path("data/raw/ami/manual")
    out = {name: dict(P=[], y=[]) for name in models}
    out["_silero"] = dict(P=[], y=[])
    out["_energy"] = dict(P=[], y=[])
    out["_webrtc"] = dict(P=[], y=[])
    teacher, torch = load_teacher()
    for w in sorted(wav_dir.glob("*.Array1-01.wav")):
        meeting = w.name.split(".")[0]
        if meeting in EVAL_MEETINGS_EXCL:
            continue
        spans = parse_ami_segments(meeting, manual)
        if len(spans) == 0:
            continue
        x = load_any_rate(w)
        F = lm(x)
        centers = centers_for(len(F), lm)
        truth = frame_truth(centers, spans)
        K = int(list(models.values())[0].meta["context"])
        X, _ = context_windows(F, F[:, 0], K)
        cc = centers[K - 1:]
        yy = truth[K - 1:]
        step = 100_000
        for s in range(0, len(X), step):
            for name, m in models.items():
                out[name]["P"].append(m.probs(X[s:s + step]))
        tp, tt = teacher_probs(teacher, torch, x)
        out["_silero"]["P"].append(np.interp(cc, tt, tp))
        ev = EnergyVAD(sr=SR)
        pcm = float_to_pcm16(x)
        st = []
        for i in range(0, len(pcm), 160):
            ev.feed(pcm[i:i + 160])
            st.append(ev.in_speech)
        out["_energy"]["P"].append(np.array(st, dtype=float)[K - 1:][:len(cc)])
        out["_webrtc"]["P"].append(webrtc_decisions(x, eval_ami_models.agg)[K - 1:][:len(cc)])
        for name in out:
            out[name]["y"].append(yy)
        if verbose:
            print(f"  {meeting}: {len(cc)/6000:.1f} min scored")
    scores = {}
    for name, r in out.items():
        P = np.concatenate(r["P"]); Y = np.concatenate(r["y"]).astype(bool)
        if name.startswith("_"):
            pred = P > 0.5
            tp = int(np.sum(pred & Y)); fp = int(np.sum(pred & ~Y)); fn = int(np.sum(~pred & Y)); tn = int(np.sum(~pred & ~Y))
            scores[name] = dict(f1=prf(P, Y, 0.5)[2], auc=auc(P, Y),
                                far=fp / max(fp + tn, 1), mr=fn / max(fn + tp, 1))
        else:
            meta = models[name].meta
            # models with domain profiles: AMI wants the distant_room point
            if "profiles" in meta and "distant_room" in meta.get("profiles", {}):
                thr = float(meta["profiles"]["distant_room"]["thr_hi"])
            else:
                thr = float(meta.get("thr_hi", 0.5))
            pred = P >= thr
            tp = int(np.sum(pred & Y)); fp = int(np.sum(pred & ~Y)); fn = int(np.sum(~pred & Y)); tn = int(np.sum(~pred & ~Y))
            scores[name] = dict(f1=prf(P, Y, thr)[2], auc=auc(P, Y),
                                far=fp / max(fp + tn, 1), mr=fn / max(fn + tp, 1))
    return scores


def bench_streaming(model_path: Path, seconds: float = 30.0) -> float:
    """µs per 20 ms chunk for the full StreamingVAD path."""
    rng = np.random.default_rng(0)
    x = 0.1 * rng.normal(size=int(SR * seconds)).astype(np.float32)
    pcm = float_to_pcm16(x)
    vad = StreamingVAD(model_path)
    vad.feed(pcm[:320])
    ts = []
    for i in range(0, len(pcm), 320):
        t0 = time.perf_counter()
        vad.feed(pcm[i:i + 320])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("comparison.json"))
    args = ap.parse_args()

    lm = LogMel(sr=SR)

    # discover models: float variants of every family (skip int8/ptq dupes)
    if args.models is None:
        paths = []
        for p in sorted(Path("models").glob("teensy-v*.npz")):
            n = p.stem
            if any(k in n for k in ("-int8", "-wide", "-soft")):
                continue
            paths.append(p)
        args.models = [str(p) for p in paths]
    models = {}
    for p in args.models:
        m = load_any(p)
        models[Path(p).stem.replace("teensy-", "")] = m
    print(f"models ({len(models)}): {', '.join(models)}")

    print("\npicking WebRTC aggressiveness on AMI dev …")
    eval_ami_models.agg = pick_webrtc_aggressiveness(lm)

    print("\n=== TEN VAD public set ===")
    ten, ten_swept = eval_ten(models, lm, ulaw=False)

    print("\n=== AMI SDM meetings (8 eval meetings) ===")
    ami = eval_ami_models(models, lm, verbose=True)

    print("\n=== speed (full streaming path, 20 ms chunks) ===")
    speed = {}
    for name, m in models.items():
        p = Path("models") / f"teensy-{name}.npz"
        speed[name] = dict(us_per_chunk=bench_streaming(p), params=m.n_params(),
                           kb=p.stat().st_size / 1024)
    # baselines speed
    rng = np.random.default_rng(0)
    x = 0.1 * rng.normal(size=int(SR * 30)).astype(np.float32)
    pcm = float_to_pcm16(x)
    ev = EnergyVAD(sr=SR); ev.feed(pcm[:160])
    ts = []
    for i in range(0, len(pcm), 160):
        t0 = time.perf_counter(); ev.feed(pcm[i:i + 160]); ts.append(time.perf_counter() - t0)
    speed["_energy"] = dict(us_per_chunk=float(np.median(ts)) * 1e6, params=0, kb=0)
    import webrtcvad as _w
    vad = _w.Vad(eval_ami_models.agg)
    ts = []
    for i in range(0, len(pcm) - 159, 160):   # 160 bytes = 10 ms @ 8 kHz int16
        t0 = time.perf_counter(); vad.is_speech(pcm[i:i + 160], SR); ts.append(time.perf_counter() - t0)
    speed["_webrtc"] = dict(us_per_chunk=float(np.median(ts)) * 2 * 1e6,  # → per 20 ms
                            params=0, kb=0)
    teacher, torch = load_teacher()
    with torch.no_grad():
        teacher(torch.from_numpy(x[:256]), SR)
        ts = []
        for k in range(len(x) // 256):
            t0 = time.perf_counter()
            teacher(torch.from_numpy(np.ascontiguousarray(x[k * 256:(k + 1) * 256])), SR)
            ts.append(time.perf_counter() - t0)
    speed["_silero"] = dict(us_per_chunk=float(np.median(ts)) * 1e6,   # per 32 ms hop
                            params=1_774_000, kb=2200)

    # ---------------- report ----------------
    disp = {"_silero": "Silero VAD", "_energy": "Energy VAD", "_webrtc": "WebRTC VAD"}
    rows = []
    # deterministic, readable ordering: families by version, sizes ascending
    def order_key(name):
        fam = name.split("-")[0]
        ver = int(fam[1:]) if fam[1:2].isdigit() else 9   # v1..v4 first, baselines last
        size_part = name.split("-")[1] if "-" in name else ""
        size = int(size_part[:-1]) if size_part.endswith("k") and size_part[:-1].isdigit() \
            else (0 if size_part == "" else 55)           # base=0, -qat=55
        return (ver, size, name)
    for name in sorted(list(models) + ["_silero", "_webrtc", "_energy"], key=order_key):
        t = ten.get(name, {}); a = ami.get(name, {}); s = speed.get(name, {})
        sw = ten_swept.get(name, {})
        rows.append(dict(
            model=disp.get(name, name),
            params=s.get("params"), kb=s.get("kb"),
            ten_f1=sw.get("f1"), ten_auc=t.get("auc"),
            ten_far=sw.get("far"), ten_mr=sw.get("mr"),
            ami_f1=a.get("f1"), ami_auc=a.get("auc"),
            ami_far=a.get("far"), ami_mr=a.get("mr"),
            us_per_20ms=s.get("us_per_chunk"),
        ))

    hdr = (f"{'model':<16}{'params':>9}{'KB':>7} | {'TEN F1*':>8}{'TEN AUC':>8} "
           f"{'FAR*':>6}{'MR*':>6} | {'AMI F1':>7}{'AMI AUC':>8}{'FAR':>6}{'MR':>6} | {'µs/20ms':>8}")
    print("\n" + "=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in rows:
        p = f"{r['params']:,}" if r["params"] else "-"
        kb = f"{r['kb']:.0f}" if r["kb"] else "-"
        tf1 = f"{r['ten_f1']:.4f}" if r["ten_f1"] else "-"
        tauc = f"{r['ten_auc']:.4f}" if r["ten_auc"] else "-"
        far = f"{r['ten_far']*100:.0f}%" if r["ten_far"] is not None else "-"
        mr = f"{r['ten_mr']*100:.0f}%" if r["ten_mr"] is not None else "-"
        af1 = f"{r['ami_f1']:.4f}" if r["ami_f1"] else "-"
        aauc = f"{r['ami_auc']:.4f}" if r["ami_auc"] else "-"
        afar = f"{r['ami_far']*100:.0f}%" if r["ami_far"] is not None else "-"
        amr = f"{r['ami_mr']*100:.0f}%" if r["ami_mr"] is not None else "-"
        us = f"{r['us_per_20ms']:.0f}" if r["us_per_20ms"] else "-"
        print(f"{r['model']:<16}{p:>9}{kb:>7} | {tf1:>8}{tauc:>8} {far:>6}{mr:>6} | "
              f"{af1:>7}{aauc:>8}{afar:>6}{amr:>6} | {us:>8}")
    print("*TEN at best-F1 threshold (upper bound); AMI at AMI-dev-calibrated threshold")

    args.out.write_text(json.dumps(
        dict(rows=rows, webrtc_agg=eval_ami_models.agg), indent=2, default=str))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
