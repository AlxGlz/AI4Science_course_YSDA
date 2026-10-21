"""Разовое обновление INDEX.md под новый прогон t1-w2-r1 (правится вручную, не агентом)."""

from __future__ import annotations

from pathlib import Path

p = Path("INDEX.md")
text = p.read_bytes().decode("utf-8").replace("\r\n", "\n")

tree_anchor = "    scratch/          черновики и диагностика\n"
tree_add = """  t1-w2-r1/           прогон 1 задачи T1, неделя 2 (независимое решение)
    modulator.py      генератор: Config, build, пресеты, CLI
    run_official_checks.py  запуск checks/checks.py против этого каталога
    sets/             данные прогона: set_clean, set_baseline, set_hard
    figures.py        построение рисунков из сохранённых CSV
    figures/          рис. 1-3 и facts.json (что изображено)
    check_figures.py  численная сверка того, что видно на рисунках
    probe_real.py     замеры по data/private/, обоснование параметров
    real_stats.json   результаты замеров
    report.md         отчёт прогона (9 разделов)
    checks_*.txt      логи проверок: штатная, по каталогу прогона, рисунки
    scratch/          черновики: diag.py (подбор формы), summary.py (сводка)
"""
task_old = "| T1 | `tasks/task_1.md` | в работе | `t1-w1-r1` |"
task_new = "| T1 | `tasks/task_1.md` | на приёмке | `t1-w1-r1`, `t1-w2-r1` |"
run_anchor = "| `t1-w1-r1` | T1 | 1 |"
run_line = ("| `t1-w2-r1` | T1 | 2 | 3 набора синтетических ФПГ (300/420/360 с, "
            "359/560/540 ударов) с разметкой, 3 рисунка, отчёт | "
            "`checks/checks.py` · 83/83 · `run_official_checks.py` · "
            "`check_figures.py` · 40/40 | `проверен` |\n")

assert text.count(tree_anchor) == 1, "tree anchor"
assert text.count(task_old) == 1, "task row"
lines = text.split("\n")
idx = next(i for i, ln in enumerate(lines) if ln.startswith(run_anchor))
lines.insert(idx + 1, run_line.rstrip("\n"))
text = "\n".join(lines)
text = text.replace(tree_anchor, tree_anchor + tree_add)
text = text.replace(task_old, task_new)

p.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
print("INDEX.md обновлён")
