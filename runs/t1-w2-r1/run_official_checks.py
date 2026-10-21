"""Запуск штатной проверки checks/checks.py против этого прогона.

Файл проверки не меняется: у него жёстко задан каталог прогона
(runs/t1-w1-r1). Здесь модуль загружается как есть, после импорта
подменяются RUN_DIR / SETS / MODULATOR, и вызываются те же самые
функции проверок — то есть проверяется ровно тот код, что лежит
в checks/checks.py, но на данных runs/t1-w2-r1.

Запуск:  .venv\\Scripts\\python runs\\t1-w2-r1\\run_official_checks.py
Код возврата 0 — все проверки прошли.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

spec = importlib.util.spec_from_file_location("official_checks", PROJECT / "checks" / "checks.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.RUN_DIR = HERE
mod.SETS = HERE / "sets"
mod.MODULATOR = HERE / "modulator.py"
mod.RESULTS.clear()

for set_dir in sorted(d for d in mod.SETS.iterdir() if d.is_dir()):
    mod.check_set(set_dir)
mod.check_reproducibility()

failed = [r for r in mod.RESULTS if not r[0]]
print("-" * 72)
print(f"{len(mod.RESULTS) - len(failed)}/{len(mod.RESULTS)} checks passed")
for _, nm, dt in failed:
    print(f"  FAILED: {nm} — {dt}")
sys.exit(1 if failed else 0)
