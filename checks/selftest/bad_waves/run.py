"""Сломанный прогон selftest: стирает 3 волны из signal.csv (число волн != строк truth)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modulator import generate  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    params = json.loads((here / "params.json").read_text())
    generate(out, params, drop_waves=3)


if __name__ == "__main__":
    main()