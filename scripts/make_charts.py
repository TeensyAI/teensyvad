"""Generate comparison charts (PNG) for the HF model cards.

    .venv/bin/python scripts/make_charts.py            # uses comparison.json

Produces in hf_assets/:
  capacity.png      — F1/AUC vs parameter count, v1/v2/v3 families
  realworld.png     — TEN + AMI bars: teensy families vs Silero/WebRTC/Energy
  speed_accuracy.png— µs/20ms vs AMI AUC scatter
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("hf_assets")
OUT.mkdir(exist_ok=True)

rows = json.loads(Path("comparison.json").read_text())["rows"]

FAMILY_OF = {}
for r in rows:
    m = r["model"]
    if m.startswith(("v1", "v2", "v3")):
        fam = m.split("-")[0]
        FAMILY_OF[m] = fam

COLOR = {"v1": "#7f8c8d", "v2": "#2980b9", "v3": "#27ae60", "v4": "#d35400",
         "Silero VAD": "#e67e22", "WebRTC VAD": "#8e44ad", "Energy VAD": "#c0392b"}

# ---------------------------------------------------------------- capacity
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fam in ("v1", "v2", "v3", "v4"):
    pts = ([(r["params"], r) for r in rows if r["model"].startswith(fam + "-")]
           + [(20449, [r for r in rows if r["model"] == fam][0])])
    pts.sort(key=lambda t: t[0])
    ps = [p for p, _ in pts]
    ax[0].plot(ps, [r["ami_f1"] for _, r in pts], "o-", color=COLOR[fam],
               label=f"{fam} family", lw=1.8, ms=5)
    ax[1].plot(ps, [r["ami_auc"] for _, r in pts], "o-", color=COLOR[fam],
               label=f"{fam} family", lw=1.8, ms=5)
for a, t in zip(ax, ["AMI SDM frame F1", "AMI SDM AUC"]):
    a.set_xlabel("parameters"); a.set_title(t)
    a.grid(alpha=0.3); a.legend(fontsize=8)
    a.ticklabel_format(style="plain", axis="x")
ax[0].set_ylabel("F1 @ AMI-calibrated thr")
fig.suptitle("Capacity scaling: more parameters only help with more data "
             "(v1/v2: 1M frames; v3: 10.7M)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "capacity.png", dpi=130)
print("→", OUT / "capacity.png")

# ---------------------------------------------------------------- realworld
base = ["v1", "v2", "v3", "v4"]
entries = []      # (label, color, ten_auc or None, ami_f1, ami_auc)
for b in base:
    fam_rows = [r for r in rows if r["model"] == b or r["model"].startswith(b + "-")]
    best_auc = max(fam_rows, key=lambda r: (r["ami_auc"] or 0))
    best_f1 = max(fam_rows, key=lambda r: (r["ami_f1"] or 0))
    label = f"teensy-{b}\n(best size)"
    entries.append((label, COLOR[b], best_auc["ten_auc"], best_f1["ami_f1"], best_auc["ami_auc"]))
for b in ("Silero VAD", "WebRTC VAD", "Energy VAD"):
    r = [x for x in rows if x["model"] == b][0]
    entries.append((b.replace(" VAD", ""), COLOR[b], r["ten_auc"], r["ami_f1"], r["ami_auc"]))

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
panels = [("TEN VAD public set — AUC", 2), ("AMI SDM — F1 (calibrated thr)", 3),
          ("AMI SDM — AUC", 4)]
for a, (t, col) in zip(ax, panels):
    vis = [(e[0], e[1], e[col]) for e in entries if e[col] is not None]
    xs = np.arange(len(vis))
    bars = a.bar(xs, [v for _, _, v in vis], color=[c for _, c, _ in vis])
    a.set_xticks(xs, [n for n, _, _ in vis], fontsize=8)
    a.set_title(t, fontsize=10)
    a.set_ylim(0.3, 1.0)
    a.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, [v for _, _, v in vis]):
        a.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
               ha="center", fontsize=7)
fig.suptitle("Real-world benchmarks (human-labelled audio) — best size per teensy family "
             "vs public baselines", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "realworld.png", dpi=130)
print("→", OUT / "realworld.png")

# ---------------------------------------------------------------- speed vs accuracy
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in rows:
    if not r["us_per_20ms"]:
        continue
    m = r["model"]
    c = COLOR.get(m.split("-")[0] if m.startswith("v") else m, "#555")
    fam = m.split("-")[0] if m.startswith("v") else m
    size = 30 if not r["params"] else 24 + r["params"] / 4000
    ax.scatter(r["us_per_20ms"], r["ami_auc"], s=size, color=COLOR.get(fam, "#555"),
               alpha=0.85, edgecolor="k", lw=0.4)
    if m in ("v3", "v3-80k", "Silero VAD", "WebRTC VAD", "Energy VAD", "v1", "v2"):
        ax.annotate(m, (r["us_per_20ms"], r["ami_auc"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
ax.set_xlabel("µs per 20 ms chunk (streaming, this machine)")
ax.set_ylabel("AMI SDM AUC")
ax.set_title("Speed vs real-world accuracy")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "speed_accuracy.png", dpi=130)
print("→", OUT / "speed_accuracy.png")
