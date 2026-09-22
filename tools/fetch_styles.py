"""Скачивает список стилей Fooocus в resources/styles.

Стили — это данные, а не код: копировать их в репозиторий руками значит
потерять связь с источником. Скрипт запускается один раз при развёртывании.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

# Корень проекта в sys.path: инструменты запускают по пути, и тогда туда
# попадает каталог скрипта, а не корень.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()  # эти скрипты печатают по-русски; cp1252 их бы уронил

BASE = "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles"
FILES = (
    "sdxl_styles_fooocus.json",
    "sdxl_styles_sai.json",
    "sdxl_styles_mre.json",
    "sdxl_styles_twri.json",
    "sdxl_styles_diva.json",
    "sdxl_styles_marc_k3nt3l.json",
)

TARGET = Path(__file__).resolve().parent.parent / "resources" / "styles"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in FILES:
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            raw = response.read().decode("utf-8")
        entries = json.loads(raw)
        (TARGET / name).write_text(raw, encoding="utf-8")
        total += len(entries)
        print(f"{name}: {len(entries)}")
    print(f"всего стилей: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
