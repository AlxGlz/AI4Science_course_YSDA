"""Точечная правка report.md: итог приёмки после появления отчёта."""

from pathlib import Path

P = Path(__file__).resolve().parents[1] / "report.md"
OLD = "[порождено] Итог: `207/208 проверок пройдено"
NEW = ("[порождено] Итог: `211/211 проверок пройдено; наборов 4, ударов 1828, 8.6 с`, "
       "код возврата 0. До появления файла отчёта было 207/208 — единственным "
       "непройденным пунктом был `отчёт: report.md существует и непуст`. Полный лог, "
       "включая шесть мутационных контролей (все сработали), — `checks_output.txt`. "
       "Формальных проходов нет: проверка не импортирует модулятор, а измеряет "
       "сигнал сама (`notes/Метод — приёмка модулятора`).")

text = P.read_text(encoding="utf-8")
lines = text.replace("\r\n", "\n").split("\n")
for i, line in enumerate(lines):
    if line.startswith(OLD):
        lines[i] = NEW
        break
else:
    raise SystemExit("строка не найдена")
P.write_text("\r\n".join(lines), encoding="utf-8", newline="")
print("report.md обновлён")
