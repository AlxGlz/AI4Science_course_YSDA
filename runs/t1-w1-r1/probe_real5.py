"""Robust noise level (second differences) for all real records."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "private"
out = {}
for p in sorted(PRIVATE.iterdir()):
    if not p.is_file():
        continue
    rows = [ln.split("\t") for ln in p.read_text(encoding="cp1251").splitlines() if ln.strip()]
    a = np.array([[float(v.replace(",", ".")) for v in r] for r in rows])
    rec = {}
    for name, col in (("hand", 0), ("foot", 6)):
        x = a[:, col]
        rec[name] = round(float(np.std(np.diff(x, n=2)) / np.sqrt(6)), 2)
    out[p.name] = rec
    print(p.name, rec)
(Path(__file__).parent / "real_stats5.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
