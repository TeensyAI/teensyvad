"""Train the tiny GRU VAD (v7) in torch — same data, labels, early stopping.

    .venv/bin/python scripts/train_rnn.py --data data/prepared_v5 \
        --hidden 96 --chunk 250 --out models/teensy-v7-gru96.npz

Training runs in torch (SIMD + autograd; the numpy runtime in
teensyvad/rnn.py is the deployed inference path and is verified to match).
Chunks of `--chunk` frames (default 2.5 s) are drawn from random positions
in random utterances; the GRU state starts at zero per chunk — at inference
the state persists across the whole call, which only helps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.features import LogMel  # noqa: E402

SR = 8000


class TinyGRUTorch(nn.Module):
    def __init__(self, in_dim: int = 40, hidden: int = 96, head: int = 24):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.h1 = nn.Linear(hidden, head)
        self.h2 = nn.Linear(head, 1)

    def forward(self, x, h0=None):
        lt, hn = self.gru(x, h0)
        return self.h2(torch.relu(self.h1(lt))).squeeze(-1), hn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prepared_v5"))
    ap.add_argument("--val", type=Path, default=Path("data/prepared/val.distill.npz"))
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--head", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--batch", type=int, default=256, help="sequences per batch")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batches-per-epoch", type=int, default=1500,
                    help="chunks sampled per epoch (data is huge; sampling "
                         "is with replacement)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=Path, default=Path("models/teensy-v7-gru.npz"))
    args = ap.parse_args()

    t0 = time.time()
    dev = "cpu"
    F = np.load(args.data / "F.npy", mmap_mode="r")
    yteach = np.load(args.data / "yteach.npy", mmap_mode="r")
    clip_len = np.load(args.data / "clip_len.npy")
    bounds = np.concatenate([[0], np.cumsum(clip_len)])
    print(f"train frames {len(yteach):,}  utterances {len(clip_len):,}")

    zv = np.load(args.val)
    Fv = np.asarray(zv["F"], dtype=np.float32)
    yv = np.asarray(zv["y"], dtype=np.float32)

    in_dim = F.shape[1]
    rng = np.random.default_rng(23)
    model = TinyGRUTorch(in_dim, args.hidden, args.head).to(dev)
    with torch.no_grad():                                # input normalisation
        chunk = np.asarray(F[:1_000_000], dtype=np.float64)
        model.gru.flatten_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    # normalisation stats folded into a tensor the numpy runtime can load
    mean40 = chunk.mean(0)
    std40 = np.maximum(chunk.std(0), 1e-3)
    del chunk

    def norm(X):
        return ((X - mean40) / std40).astype(np.float32)

    def eval_val() -> float:
        """Frame F1 @ 0.5 over the val set with stateful streaming."""
        model.eval()
        h = torch.zeros(1, 1, args.hidden)
        preds = []
        with torch.no_grad():
            for s in range(0, len(Fv), 20_000):
                xb = torch.tensor(norm(Fv[s:s + 20_000])).unsqueeze(0)
                lo, h = model(xb, h.contiguous())        # (1, T, H) state carried
                preds.append((lo.reshape(-1) > 0).float().numpy())
        model.train()
        pred = np.concatenate(preds)
        tp = float(((pred == 1) & (yv == 1)).sum())
        fp = float(((pred == 1) & (yv == 0)).sum())
        fn = float(((pred == 0) & (yv == 1)).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1.0)

    best = {"f1": -1.0, "state": None}
    print(f"training gru{args.hidden} chunk={args.chunk} …")
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for it in range(args.batches_per_epoch):
            utt = rng.integers(0, len(clip_len), size=args.batch)
            starts = bounds[utt]
            lens = clip_len[utt]
            offs = starts + (rng.random(args.batch) * np.maximum(lens - args.chunk, 1)).astype(np.int64)
            Xb = np.stack([np.asarray(F[o:o + args.chunk], dtype=np.float32)
                           for o in offs])
            yb = np.stack([np.asarray(yteach[o:o + args.chunk], dtype=np.float32)
                           for o in offs])
            Xb = norm(Xb)
            opt.zero_grad()
            lo, _ = model(torch.tensor(Xb))
            loss = bce(lo, torch.tensor(yb))
            loss.backward()
            opt.step()
            tot += float(loss)
        f1 = eval_val()
        print(f"[gru] epoch {ep:3d}  loss {tot/args.batches_per_epoch:.4f}  "
              f"val_f1 {f1:.4f}", flush=True)
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.detach().clone() for k, v in
                                        model.state_dict().items()}}
    model.load_state_dict(best["state"])

    # ---- export to the numpy runtime format (teensyvad/rnn.py) ----
    sd = model.state_dict()
    H = args.hidden
    # torch gate row order is [r, z, n]; our runtime keys are [z, r, n]
    wiz, wir, win = (sd["gru.weight_ih_l0"].numpy()[H:2*H].T,
                     sd["gru.weight_ih_l0"].numpy()[0:H].T,
                     sd["gru.weight_ih_l0"].numpy()[2*H:3*H].T)
    whz, whr, whn = (sd["gru.weight_hh_l0"].numpy()[H:2*H].T,
                     sd["gru.weight_hh_l0"].numpy()[0:H].T,
                     sd["gru.weight_hh_l0"].numpy()[2*H:3*H].T)
    npz = {
        "W/Wiz": wiz, "W/Wir": wir, "W/Win": win,
        "W/Whz": whz, "W/Whr": whr, "W/Whn": whn,
        "b/z": (sd["gru.bias_ih_l0"].numpy()[H:2*H] + sd["gru.bias_hh_l0"].numpy()[H:2*H]),
        "b/r": (sd["gru.bias_ih_l0"].numpy()[0:H] + sd["gru.bias_hh_l0"].numpy()[0:H]),
        "b/n": (sd["gru.bias_ih_l0"].numpy()[2*H:3*H] + sd["gru.bias_hh_l0"].numpy()[2*H:3*H]),
        "Wh/h1": sd["h1.weight"].numpy().T, "bh/h1": sd["h1.bias"].numpy(),
        "Wh/h2": sd["h2.weight"].numpy().T, "bh/h2": sd["h2.bias"].numpy(),
        "in_mean": mean40.astype(np.float32), "in_std": std40.astype(np.float32),
        "meta": np.array(json_meta := __import__("json").dumps(dict(
            arch="gru", hidden=args.hidden, head=args.head, sr=SR,
            win_ms=25.0, hop_ms=10.0, n_fft=256, n_mels=20,
            fmin=80.0, fmax=3800.0, deltas=False, context=0,
            thr_hi=0.5, thr_lo=0.3, hangover_ms=250.0, on_frames=3,
            trained_with="scripts/train_rnn.py", data=str(args.data),
            val_f1=round(best["f1"], 4)))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), **npz)
    print(f"saved → {args.out}  best val_f1 {best['f1']:.4f}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
