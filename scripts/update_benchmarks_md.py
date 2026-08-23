"""Regenerate the full-results table in hf_upload/teensy-vad-3/BENCHMARKS.md
from comparison.json — single source of truth, no manual transcription.

    .venv/bin/python scripts/update_benchmarks_md.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROWS = json.loads(Path("comparison.json").read_text())["rows"]
BY = {r["model"]: r for r in ROWS}

def order_key(name):
    fam = name.split("-")[0]
    ver = int(fam[1:]) if fam[1:2].isdigit() else 9
    size_part = name.split("-")[1] if "-" in name else ""
    size = int(size_part[:-1]) if size_part.endswith("k") and size_part[:-1].isdigit() \
        else (0 if size_part == "" else 55)
    return (ver, size, name)

names = sorted([n for n in BY if n.startswith("v")], key=order_key)
baselines = ["Silero VAD", "WebRTC VAD", "Energy VAD"]
# bold the family champion on TEN AUC and AMI F1.
# AMI-F1 champion must have a sane operating point (FAR < 90%) — a model
# that predicts "always speech" can top F1 on speech-heavy meetings
# without detecting anything (the v1 family does exactly that on AMI).
ten_best = max(names, key=lambda n: BY[n]["ten_auc"] or 0)
sane = [n for n in names if (BY[n]["ami_far"] or 1.0) < 0.9]
ami_f1_best = max(sane, key=lambda n: BY[n]["ami_f1"] or 0)

def fmt(name, field, nd=3):
    v = BY[name][field]
    if v is None:
        return "n/a" if field == "ten_auc" else "—"
    s = f"{v:.{nd}f}" if v < 10 else f"{v:.0f}"
    return f"**{s}**" if (name == ten_best and field == "ten_auc") or \
                    (name == ami_f1_best and field == "ami_f1") else s

lines = ["| model | params | KB | TEN F1* | TEN AUC | AMI F1 | AMI AUC | µs/20ms |",
         "|---|---|---|---|---|---|---|---|"]
for n in names + baselines:
    r = BY[n]
    if n in baselines:
        p = f"{r['params']:,}" if r["params"] else ("~6k (C)" if n == "WebRTC VAD" else "—")
        kb = f"{r['kb']:.0f}" if r["kb"] else ("~50" if n == "WebRTC VAD" else "—")
        def cell(v, bold=False, nd=3):
            if v is None:
                return "n/a"
            s = f"{v:.{nd}f}"
            return f"**{s}**" if bold else s
        ten = (cell(r["ten_f1"], bold=(n == "Silero VAD")) + " | "
               + cell(r["ten_auc"], bold=(n == "Silero VAD")) + " | ")
        ami_f1 = f"{r['ami_f1']:.3f}" if r["ami_f1"] is not None else "n/a"
        ami_auc = f"**{r['ami_auc']:.3f}**" if n == "Silero VAD" and r["ami_auc"] is not None \
            else (f"{r['ami_auc']:.3f}" if r["ami_auc"] is not None else "n/a")
        us = f"{r['us_per_20ms']:.0f}" if r["us_per_20ms"] else "—"
        lines.append(f"| {n} | {p} | {kb} | {ten}{ami_f1} | {ami_auc} | {us} |")
    else:
        lines.append(f"| {n} | {r['params']:,} | {r['kb']:.0f} | "
                     f"{fmt(n,'ten_f1')} | {fmt(n,'ten_auc')} | {fmt(n,'ami_f1')} | "
                     f"{fmt(n,'ami_auc')} | {fmt(n,'us_per_20ms')} |")
table = "\n".join(lines)

md = Path("hf_upload/teensy-vad-3/BENCHMARKS.md")
src = md.read_text()
s = src.index("| model | params | KB |")
e = src.index("\n\n", src.index("Energy VAD |"))
md.write_text(src[:s] + table + src[e:])
print(f"table rewritten: {len(names)} teensy models + 3 baselines "
      f"(TEN AUC champ: {ten_best}, AMI F1 champ: {ami_f1_best})")
