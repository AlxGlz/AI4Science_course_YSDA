from pathlib import Path

p = Path("runs/t1-w1-r1/report.md")
with open(p, encoding="utf-8", newline="") as fh:
    s = fh.read()
old = "  checks_output.txt      полный вывод checks/checks.py\r\n"
new = old + "  scratch/               отладочные скрипты прогона (в результат не входят)\r\n"
assert s.count(old) == 1, s.count(old)
with open(p, "w", encoding="utf-8", newline="") as fh:
    fh.write(s.replace(old, new))
print("ok")
