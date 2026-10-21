"""Housekeeping: move debug scaffolding to scratch/, normalise line endings to CRLF."""

from pathlib import Path
import shutil

RUN = Path("runs/t1-w1-r1")
SCRATCH = RUN / "scratch"
SCRATCH.mkdir(exist_ok=True)

for pattern in ("diag_*.py", "patch_*.py", "table.py"):
    for f in RUN.glob(pattern):
        shutil.move(str(f), str(SCRATCH / f.name))
        print("moved", f.name)

targets = [Path("checks/checks.py")]
targets += sorted(RUN.glob("*.py")) + sorted(RUN.glob("*.md")) + sorted(RUN.glob("*.txt"))
targets += sorted(SCRATCH.glob("*.py"))
for f in targets:
    raw = f.read_bytes()
    norm = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if norm != raw:
        f.write_bytes(norm)
        print("crlf", f)
