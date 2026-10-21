"""Генерация заметок классов онтологии проекта из ontology.yaml.

Единственный источник истины — `ontology.yaml`. Скрипт пишет по заметке
`ontology/<Класс>.md` на каждый класс: определение, связь `подклассОт`,
подклассы, свойства (исходящие/входящие), атрибуты и правила.

Заметки классов руками не правятся: правится схема, скрипт запускается заново.

Запуск:
    .venv\\Scripts\\python tools\\build_class_notes.py
    .venv\\Scripts\\python tools\\build_class_notes.py --check   # только сверка
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "ontology.yaml"
OUT_DIR = ROOT / "ontology"


def as_list(value) -> list[str]:
    """domain/range в схеме бывают строкой или списком — приводим к списку."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def load_schema() -> dict:
    with SCHEMA.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def class_note(name: str, spec: dict, schema: dict) -> str:
    classes: dict[str, dict] = schema.get("classes") or {}
    props: dict[str, dict] = schema.get("properties") or {}
    attrs: dict[str, dict] = schema.get("attributes") or {}
    rules: list[dict] = schema.get("rules") or []

    parent = spec.get("parent")
    definition = " ".join(str(spec.get("definition", "")).split())

    children = [child for child, cspec in classes.items() if cspec.get("parent") == name]

    outgoing = [
        (prop, as_list(pspec.get("range")), as_list(pspec.get("characteristics")))
        for prop, pspec in props.items()
        if name in as_list(pspec.get("domain"))
    ]
    incoming = [
        (prop, as_list(pspec.get("domain")))
        for prop, pspec in props.items()
        if name in as_list(pspec.get("range"))
    ]

    my_attrs = [(attr, as_list(aspec.get("values"))) for attr, aspec in attrs.items()
                if aspec.get("domain") == name]

    my_rules = [r for r in rules if (r.get("check") or {}).get("class") == name]

    lines = [
        f"# {name}",
        "",
        f"<!-- [порождено] `tools/build_class_notes.py` из `ontology.yaml`"
        f" {dt.date.today().isoformat()}. Руками не править: правится схема"
        f" и скрипт запускается заново. -->",
        "",
        "## Определение",
        "",
        definition or "—",
        "",
        "## Связи",
        "",
        f"- подклассОт: {f'[[{parent}]]' if parent else '—'}",
        "",
        "## Подклассы",
        "",
    ]
    lines += [f"- [[{child}]]" for child in children] or ["—"]
    lines += ["", "## Свойства", "", "Исходящие (класс в domain):", ""]
    if outgoing:
        for prop, rng, chars in outgoing:
            tail = f" — {', '.join(chars)}" if chars else ""
            lines.append(f"- `{prop}` → {' | '.join(f'[[{r}]]' for r in rng)}{tail}")
    else:
        lines.append("—")
    lines += ["", "Входящие (класс в range):", ""]
    if incoming:
        for prop, dom in incoming:
            lines.append(f"- `{prop}` ← {', '.join(f'[[{d}]]' for d in dom)}")
    else:
        lines.append("—")
    lines += ["", "## Атрибуты", ""]
    if my_attrs:
        for attr, values in my_attrs:
            lines.append(f"- `{attr}`: {', '.join(values) if values else '—'}")
    else:
        lines.append("—")
    lines += ["", "## Правила", ""]
    if my_rules:
        for rule in my_rules:
            lines.append(f"- **{rule.get('id')}** {rule.get('text')}")
            if rule.get("dl"):
                lines.append(f"  `{rule.get('dl')}`")
    else:
        lines.append("—")
    lines.append("")
    return "\n".join(lines)


def verify(name: str, text: str, spec: dict) -> list[str]:
    """Сверка заметки со схемой: определение и подклассОт на месте."""
    problems = []
    definition = " ".join(str(spec.get("definition", "")).split())
    if definition and definition not in " ".join(text.split()):
        problems.append(f"{name}: нет дословного определения")
    parent = spec.get("parent")
    want = f"- подклассОт: {f'[[{parent}]]' if parent else '—'}"
    if want not in text:
        problems.append(f"{name}: ожидалось `{want}`")
    return problems


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    schema = load_schema()
    classes: dict[str, dict] = schema.get("classes") or {}
    if not classes:
        print("ERROR: в ontology.yaml нет classes")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    notes = {name: class_note(name, spec, schema) for name, spec in classes.items()}

    if not check_only:
        for name, text in notes.items():
            (OUT_DIR / f"{name}.md").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    problems: list[str] = []
    for name, text in notes.items():
        path = OUT_DIR / f"{name}.md"
        if not path.exists():
            problems.append(f"{name}: файла нет")
            continue
        disk = path.read_bytes().decode("utf-8")
        if disk != text.replace("\n", "\r\n"):
            problems.append(f"{name}: файл расходится со схемой (перегенерировать)")
        problems += verify(name, disk, classes[name])

    roots = [n for n, s in classes.items() if not s.get("parent")]
    print(f"classes: {len(classes)}; roots: {', '.join(roots)}")
    print(f"notes: {len(notes)} в {OUT_DIR.relative_to(ROOT)}/")
    if problems:
        print(f"FAIL {len(problems)}:")
        for p in problems:
            print(" -", p)
        return 1
    print("OK: заметки классов соответствуют ontology.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
