"""Calibrate + benchmark a teensy-v7 tiny transformer on the real-world protocol.

    .venv/bin/python scripts/eval_transformer.py --model models/teensy-v7-tt.npz         --calibrate     # sweep thr on AMI dev → save into meta
    .venv/bin/python scripts/eval_transformer.py --model models/teensy-v7-tt.npz         --eval --out models/comparison_v7_tt.json

Streaming model: causal attention is evaluated over fixed 750-frame segments
(7.5 s) with the first 250 frames discarded as warmup per segment — exact
for a causal transformer (no future leakage), O(T·W) instead of O(T²).
Reloads the torch state dict from the npz export written by
scripts/train_transformer.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teensyvad.features import LogMel  # noqa: E402
from scripts.eval_realworld import (SR, centers_for, frame_truth,  # noqa: E402
                                    load_any_rate, parse_ami_segments,
                                    parse_ten_labels)
from teensyvad.model import auc  # noqa: E402

CALIB_MEETINGS = ["ES2002a", "IS1000a", "TS3003a"]
SEG, WARM = 750, 250


class TinyCausalVAD(nn.Module):
    def __init__(self, in_dim, d_model, layers, heads, window, head):
        super().__init__()
        self.window = window
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(window, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                           dim_feedforward=d_model * 2,
                                           batch_first=True, norm_first=True,
                                           dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.h1 = nn.Linear(d_model, head)
        self.h2 = nn.Linear(head, 1)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.in_proj(x)
        pos = torch.arange(T, device=x.device) % self.window
        h = h + self.pos(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        z = self.norm(self.enc(h, mask=mask, is_causal=True))
        return self.h2(torch.relu(self.h1(z))).squeeze(-1)


def load_model(path: Path):
    z = dict(np.load(path, allow_pickle=True))
    meta = json.loads(str(z["meta"]))
    m = TinyCausalVAD(40, meta["d_model"], meta["layers"], meta["heads"],
                      meta["window"], meta["head"])
    sd = {}
    sd["in_proj.weight"] = torch.tensor(z["in_proj/W"].T)
    sd["in_proj.bias"] = torch.tensor(z["in_proj/b"])
    sd["pos.weight"] = torch.tensor(z["pos/table"])
    sd["norm.weight"] = torch.tensor(z["norm/w"]); sd["norm.bias"] = torch.tensor(z["norm/b"])
    sd["h1.weight"] = torch.tensor(z["h1/W"].T); sd["h1.bias"] = torch.tensor(z["h1/b"])
    sd["h2.weight"] = torch.tensor(z["h2/W"].T); sd["h2.bias"] = torch.tensor(z["h2/b"])
    L = meta["layers"]
    for li in range(L):
        p = f"enc.layers.{li}."
        e = f"enc/{li}/"
        sd[f"{p}self_attn.in_proj_weight"] = torch.tensor(z[f"{e}self_attn.in_proj_weight"])
        sd[f"{p}self_attn.in_proj_bias"] = torch.tensor(z[f"{e}self_attn.in_proj_bias"])
        sd[f"{p}self_attn.out_proj.weight"] = torch.tensor(z[f"{e}self_attn.out_proj.weight"])
        sd[f"{p}self_attn.out_proj.bias"] = torch.tensor(z[f"{e}self_attn.out_proj.bias"])
        sd[f"{p}linear1.weight"] = torch.tensor(z[f"{e}linear1.weight"])
        sd[f"{p}linear1.bias"] = torch.tensor(z[f"{e}linear1.bias"])
        sd[f"{p}linear2.weight"] = torch.tensor(z[f"{e}linear2.weight"])
        sd[f"{p}linear2.bias"] = torch.tensor(z[f"{e}linear2.bias"])
        sd[f"{p}norm1.weight"] = torch.tensor(z[f"{e}norm1.weight"])
        sd[f"{p}norm1.bias"] = torch.tensor(z[f"{e}norm1.bias"])
        sd[f"{p}norm2.weight"] = torch.tensor(z[f"{e}norm2.weight"])
        sd[f"{p}norm2.bias"] = torch.tensor(z[f"{e}norm2.bias"])
    m.load_state_dict(sd)
    m.meta = meta
    m.in_mean = z["in_mean"]; m.in_std = z["in_std"]
    m.eval()
    return m


def probs_on(m, lm, x8k):
    """Causal segment streaming: 750-frame segments, drop 250 warmup frames."""
    F = ((lm(x8k) - m.in_mean) / m.in_std).astype(np.float32)
    probs = []
    with torch.no_grad():
        s = 0
        first = True
        while s < len(F):
            seg = F[s:s + SEG]
            lo = m(torch.tensor(seg).unsqueeze(0)).reshape(-1).numpy()
            drop = 0 if first else WARM
            probs.append(lo[drop:])
            s += SEG - WARM
            first = False
    return np.concatenate(probs)


def sweep_best(P, Y):
    best = (0.5, -1.0)
    for thr in np.arange(0.05, 0.96, 0.01):
        pred = P >= thr
        tp = float((pred & Y).sum()); fp = float((pred & ~Y).sum()); fn = float((~pred & Y).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        if f1 > best[1]: best = (float(thr), f1)
    return best


def score(P, Y, thr):
    pred = P >= thr
    tp = float((pred & Y).sum()); fp = float((pred & ~Y).sum()); fn = float((~pred & Y).sum())
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1.0),
                far=fp / max((~Y).sum(), 1.0), mr=fn / max(Y.sum(), 1.0),
                auc=auc(P, Y.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, default=Path("data/raw/ami/wav"))
    ap.add_argument("--manual", type=Path, default=Path("data/raw/ami/manual"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="v7-tt")
    args = ap.parse_args()

    m = load_model(args.model)
    lm = LogMel(sr=SR)

    if args.calibrate:
        P, Y = [], []
        for meeting in CALIB_MEETINGS:
            spans = parse_ami_segments(meeting, args.manual)
            x = load_any_rate(args.wav_dir / f"{meeting}.Array1-01.wav")
            centers = centers_for(*(2 * (None,) )) if False else centers_for(len(lm(x)), lm)
            F = lm(x); centers = centers_for(len(F), lm)
            truth = frame_truth(centers, spans)
            probs = probs_on(m, lm, x)
            n = min(len(probs), len(truth))
            P.append(probs[:n]); Y.append(truth[:n])
            print(f"  {meeting}: {len(F)/6000:.1f} min, {truth.mean()*100:.0f}% speech", flush=True)
        Pc = np.concatenate(P); Yc = np.concatenate(Y)
        thr, f1 = sweep_best(Pc, Yc)
        print(f"best frame F1 on AMI dev: {f1:.4f} @ thr_hi {thr:.2f}")
        m.meta["thr_hi"] = round(thr, 3); m.meta["thr_lo"] = round(0.6 * thr, 3)
        m.meta["calibrated_on"] = "ami-dev:" + ",".join(CALIB_MEETINGS)
        # re-save meta only
        z = dict(np.load(args.model, allow_pickle=True))
        z["meta"] = np.array(json.dumps(m.meta))
        np.savez(str(args.model), **z)
        print(f"saved thresholds → {args.model}")
        return

    if args.eval:
        tdir = Path("data/raw/ten-vad/testset")
        P, Y = [], []
        for w in sorted(tdir.glob("testset-audio-*.wav")):
            lab = parse_ten_labels(w.with_suffix(".scv"))
            x = load_any_rate(w)
            F = lm(x); centers = centers_for(len(F), lm)
            truth = frame_truth(centers, lab)
            probs = probs_on(m, lm, x)
            n = min(len(probs), len(truth))
            P.append(probs[:n]); Y.append(truth[:n])
        Pc = np.concatenate(P); Yc = np.concatenate(Y).astype(bool)
        thr = float(m.meta.get("thr_hi", 0.5))
        at = score(Pc, Yc, thr); best = sweep_best(Pc, Yc)
        rows = dict(model=args.tag, params=m.meta.get("params"),
                    ten_f1=round(best[1], 4), ten_auc=round(at["auc"], 4))
        print(f"TEN: F1* {rows['ten_f1']}, AUC {rows['ten_auc']}")

        P, Y = [], []
        speeds = []
        for w in sorted(args.wav_dir.glob("*.Array1-01.wav")):
            meeting = w.name.split(".")[0]
            if meeting in CALIB_MEETINGS: continue
            spans = parse_ami_segments(meeting, args.manual)
            try:
                x = load_any_rate(w)
            except Exception as e:
                print(f"  !! {meeting}: {e} — skipping"); continue
            F = lm(x); centers = centers_for(len(F), lm)
            truth = frame_truth(centers, spans)
            t0 = time.perf_counter(); probs = probs_on(m, lm, x)
            speeds.append((time.perf_counter() - t0) / max(len(probs), 1))
            n = min(len(probs), len(truth))
            P.append(probs[:n]); Y.append(truth[:n])
        Pc = np.concatenate(P); Yc = np.concatenate(Y).astype(bool)
        thr = float(m.meta.get("thr_hi", 0.5))
        at = score(Pc, Yc, thr)
        rows["ami_f1"] = round(at["f1"], 4); rows["ami_auc"] = round(at["auc"], 4)
        # speed: per-frame × 2 → per 20 ms
        rows["us_per_20ms"] = round(float(np.median(speeds)) * 2 * 1e6, 1)
        print(f"AMI: F1 {rows['ami_f1']} AUC {rows['ami_auc']} | {rows['us_per_20ms']} µs/20ms")

        if args.out:
            args.out.write_text(json.dumps({"rows": [rows]}, indent=2))
            print(f"→ {args.out}")


if __name__ == "__main__":
    main()
