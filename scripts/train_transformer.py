"""Train a tiny causal transformer VAD (Kyutai-style streaming, no recurrence).

    .venv/bin/python scripts/train_transformer.py --data data/prepared_v5         --d-model 64 --layers 3 --out models/teensy-v7-tt.npz

Same data, same Silero soft labels, same early stopping as the GRU. Causal
attention over a --window frame window (default 250 = 2.5 s); ~100k params
at d_model 64 (1/18th of Silero).
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

SR = 8000


class TinyCausalVAD(nn.Module):
    """Causal transformer encoder over mel frames, per-frame logit head."""

    def __init__(self, in_dim: int = 40, d_model: int = 64, layers: int = 3,
                 heads: int = 4, window: int = 250, head: int = 24):
        super().__init__()
        self.window = window
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(window, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 2,
            batch_first=True, norm_first=True, dropout=0.0)
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
        return self.h2(torch.relu(self.h1(z))).squeeze(-1)   # (B, T)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared_v5"))
    ap.add_argument("--val", type=Path, default=Path("data/prepared/val.distill.npz"))
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--head", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batches-per-epoch", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qat-epochs", type=int, default=3,
                    help="QAT fine-tune epochs after float training (0 = off)")
    ap.add_argument("--out", type=Path, default=Path("models/teensy-v7-tt.npz"))
    args = ap.parse_args()

    t0 = time.time()
    F = np.load(args.data / "F.npy", mmap_mode="r")
    yteach = np.load(args.data / "yteach.npy", mmap_mode="r")
    clip_len = np.load(args.data / "clip_len.npy")
    bounds = np.concatenate([[0], np.cumsum(clip_len)])
    zv = np.load(args.val)
    Fv = np.asarray(zv["F"], dtype=np.float32)
    yv = np.asarray(zv["y"], dtype=np.float32)
    rng = np.random.default_rng(23)

    model = TinyCausalVAD(in_dim=F.shape[1], d_model=args.d_model,
                          layers=args.layers, heads=args.heads,
                          window=args.window, head=args.head)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"transformer params: {n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    chunk = np.asarray(F[:1_000_000], dtype=np.float64)
    mean40 = chunk.mean(0); std40 = np.maximum(chunk.std(0), 1e-3); del chunk
    def norm(X):
        return ((X - mean40) / std40).astype(np.float32)

    def eval_val() -> float:
        """Windowed streaming approximation: 750-frame segments, drop the
        first 250 warmup frames per segment (exact for causal attention)."""
        model.eval()
        preds, seg, warm = [], 750, 250
        with torch.no_grad():
            for s in range(0, len(Fv), seg - warm):
                lo = model(torch.tensor(norm(Fv[s:s + seg])).unsqueeze(0)).reshape(-1)
                preds.append(lo.numpy()[(warm if s > 0 else 0):])
        model.train()
        pred = (np.concatenate(preds) > 0).astype(np.float32)
        yy = yv[-len(pred):]
        tp = float(((pred == 1) & (yy == 1)).sum())
        fp = float(((pred == 1) & (yy == 0)).sum())
        fn = float(((pred == 0) & (yy == 1)).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1.0)

    def _export(val_f1: float) -> None:
        sd = model.state_dict()
        npz = {
            "in_proj/W": sd["in_proj.weight"].numpy().T,
            "in_proj/b": sd["in_proj.bias"].numpy(),
            "pos/table": sd["pos.weight"].numpy(),
            "norm/w": sd["norm.weight"].numpy(), "norm/b": sd["norm.bias"].numpy(),
            "h1/W": sd["h1.weight"].numpy().T, "h1/b": sd["h1.bias"].numpy(),
            "h2/W": sd["h2.weight"].numpy().T, "h2/b": sd["h2.bias"].numpy(),
            "in_mean": mean40.astype(np.float32), "in_std": std40.astype(np.float32),
            "meta": np.array(json.dumps(dict(
                arch="transformer", d_model=args.d_model, layers=args.layers,
                heads=args.heads, window=args.window, head=args.head, sr=SR,
                win_ms=25.0, hop_ms=10.0, n_fft=256, n_mels=20,
                fmin=80.0, fmax=3800.0, deltas=False, context=args.window,
                thr_hi=0.5, thr_lo=0.3, hangover_ms=250.0, on_frames=3,
                trained_with="scripts/train_transformer.py", data=str(args.data),
                val_f1=round(val_f1, 4), params=n_params))),
        }
        for li in range(args.layers):
            p = f"enc.layers.{li}."
            for wname in ["self_attn.in_proj_bias", "self_attn.out_proj.weight",
                          "self_attn.out_proj.bias",
                          "linear1.weight", "linear1.bias",
                          "linear2.weight", "linear2.bias",
                          "norm1.weight", "norm1.bias",
                          "norm2.weight", "norm2.bias"]:
                npz[f"enc/{li}/{wname}"] = sd[f"{p}{wname}"].numpy()
            npz[f"enc/{li}/self_attn.in_proj_weight"] = sd[f"{p}self_attn.in_proj_weight"].numpy()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(args.out), **npz)

    best = {"f1": -1.0}
    total_target = args.epochs * args.batches_per_epoch
    next_val = args.batches_per_epoch
    print(f"training tt d={args.d_model} layers={args.layers} window={args.window} … "
          f"{total_target:,} batches", flush=True)
    tot, since = 0.0, 0
    for done in range(1, total_target + 1):
        utt = rng.integers(0, len(clip_len), size=args.batch)
        starts = bounds[utt]; lens = clip_len[utt]
        offs = starts + (rng.random(args.batch) * np.maximum(lens - args.chunk, 1)).astype(np.int64)
        Xb = np.stack([np.asarray(F[o:o + args.chunk], dtype=np.float32) for o in offs])
        yb = np.stack([np.asarray(yteach[o:o + args.chunk], dtype=np.float32) for o in offs])
        opt.zero_grad()
        lo = model(torch.tensor(norm(Xb)))
        loss = bce(lo, torch.tensor(yb))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += float(loss)
        if done % 500 == 0:
            # crash-resilient: full state every 500 batches
            torch.save({"model": model.state_dict(), "done": done},
                       str(args.out) + ".ckpt")
        if done == next_val or done == total_target:
            f1 = eval_val()
            ep = done / args.batches_per_epoch
            print(f"[tt] it {done:5d} ({ep:4.1f}ep)  loss {tot/max(since,1):.4f}  "
                  f"val_f1 {f1:.4f}", flush=True)
            tot, since = 0.0, 0
            if f1 > best["f1"]:
                best = {"f1": f1}
                _export(f1)                    # save-best immediately
            next_val += args.batches_per_epoch
    print("TRAINING_COMPLETE", flush=True)

    # ---- QAT fine-tune: fake-quant all Linear weights each step ----
    if args.qat_epochs > 0:
        def fake_q(w):
            s_ = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
            rq = (w / s_).round().clamp(-127, 127)
            return w + (rq * s_ - w).detach()
        qparams = [mod.weight for mod in model.modules()
                   if isinstance(mod, nn.Linear)]
        qopt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
        model.train()
        print(f"QAT fine-tune {args.qat_epochs} epochs …", flush=True)
        for ep in range(1, args.qat_epochs + 1):
            tot = 0.0
            for it in range(args.batches_per_epoch // 5):
                masters = [pp.data.clone() for pp in qparams]
                for pp in qparams:
                    pp.data = fake_q(pp.data)
                utt = rng.integers(0, len(clip_len), size=args.batch)
                starts = bounds[utt]; lens = clip_len[utt]
                offs = starts + (rng.random(args.batch) * np.maximum(
                    lens - args.chunk, 1)).astype(np.int64)
                Xb = np.stack([np.asarray(F[o:o + args.chunk],
                                          dtype=np.float32) for o in offs])
                yb = np.stack([np.asarray(yteach[o:o + args.chunk],
                                          dtype=np.float32) for o in offs])
                qopt.zero_grad(); opt.zero_grad()
                lo = model(torch.tensor(norm(Xb)))
                loss = bce(lo, torch.tensor(yb))
                loss.backward()
                qopt.step()
                for pp, mast in zip(qparams, masters):
                    pp.data.copy_(mast)
                tot += float(loss)
            f1 = eval_val()
            print(f"[qat] epoch {ep:3d}  loss {tot/(args.batches_per_epoch//5):.4f}  "
                  f"val_f1 {f1:.4f}", flush=True)
            if f1 >= best["f1"]:
                best = {"f1": f1}
                _export(f1)

    print(f"best val_f1 {best['f1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
