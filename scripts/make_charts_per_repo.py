"""Per-repo comparison charts: accuracy, size and speed for every named
variant of one family vs the public baselines.

    .venv/bin/python scripts/make_charts_per_repo.py     # needs comparison.json

Writes hf_assets/chart_v1.png, chart_v2.png, chart_v3.png — one per HF
repo, each showing that family's named variants plus Silero / WebRTC /
Energy for scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.loads(Path("comparison.json").read_text())["rows"]
BY = {r["model"]: r for r in rows}
OUT = Path("hf_assets")
OUT.mkdir(exist_ok=True)

BASE_COLOR = {"Silero VAD": "#e67e22", "WebRTC VAD": "#8e44ad", "Energy VAD": "#c0392b"}
FAM_COLOR = {"v1": "#7f8c8d", "v2": "#2980b9", "v3": "#27ae60", "v4": "#d35400"}

# named variants per family (display name → comparison.json key)
VARIANTS = {
    "v1": [("teensy-v1 (20k)", "v1"), ("teensy-v1-40k", "v1-40k"),
           ("teensy-v1-80k", "v1-80k"), ("teensy-v1-100k", "v1-100k")],
    "v2": [("teensy-v2 (20k)", "v2"), ("teensy-v2-40k", "v2-40k"),
           ("teensy-v2-80k", "v2-80k"), ("teensy-v2-100k", "v2-100k"),
           ("teensy-v2-qat (int8)", "v2-qat")],
    "v3": [("teensy-v3 (20k)", "v3"), ("teensy-v3-40k", "v3-40k"),
           ("teensy-v3-80k", "v3-80k"), ("teensy-v3-100k", "v3-100k"),
           ("teensy-v3-qat (int8)", "v3-qat")],
    "v4": [("teensy-v4 (20k)", "v4"), ("teensy-v4-40k", "v4-40k"),
           ("teensy-v4-80k", "v4-80k"), ("teensy-v4-100k", "v4-100k"),
           ("teensy-v4-qat (int8)", "v4-qat"), ("teensy-v4-40k-qat", "v4-40k-qat"),
           ("teensy-v4-80k-qat", "v4-80k-qat"), ("teensy-v4-100k-qat", "v4-100k-qat")],
}
BASELINES = ["Silero VAD", "WebRTC VAD", "Energy VAD"]

PANELS = [("TEN VAD set\nAUC", "ten_auc", False),
          ("AMI SDM\nF1 (calibrated)", "ami_f1", False),
          ("AMI SDM\nAUC", "ami_auc", False),
          ("speed\nµs / 20 ms (log)", "us_per_20ms", True),
          ("size\nKB (log)", "kb", True)]

for fam in ("v1", "v2", "v3", "v4"):
    entries = [(name, FAM_COLOR[fam], BY[key]) for name, key in VARIANTS[fam]
               if key in BY]          # skip variants not yet in comparison.json
    entries += [(b.replace(" VAD", ""), BASE_COLOR[b], BY[b]) for b in BASELINES]
    fig, ax = plt.subplots(1, 5, figsize=(15.5, 4.0))
    for a, (title, field, logy) in zip(ax, PANELS):
        vis = [(n, c, r[field]) for n, c, r in entries if r.get(field) is not None]
        xs = np.arange(len(vis))
        vals = [v for _, _, v in vis]
        bars = a.bar(xs, vals, color=[c for _, c, _ in vis])
        a.set_xticks(xs, [n for n, _, _ in vis], fontsize=6.2, rotation=38,
                     ha="right")
        a.set_title(title, fontsize=9.5)
        if logy:
            a.set_yscale("log")
        a.grid(alpha=0.3, axis="y", which="both")
        vmax = max(vals)
        for b, v in zip(bars, vals):
            fmt = f"{v:.3f}" if v < 10 else f"{v:.0f}"
            a.text(b.get_x() + b.get_width() / 2, v * (1.18 if logy else 1.01),
                   fmt, ha="center", fontsize=6)
        a.set_ylim(top=vmax * (2.2 if logy else 1.12))
    fig.suptitle(f"teensy-vad-{fam[-1]} family — named variants vs public baselines "
                 f"(same protocol, human-labelled real audio)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / f"chart_{fam}.png"
    fig.savefig(out, dpi=130)
    print("→", out)
