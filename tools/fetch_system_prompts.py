"""Скачивает официальные промты переписывания из репозитория QwenLM.

Эти файлы — часть поставки модели, а не наш текст. Пользователь волен их
править, поэтому уже существующие файлы не перезаписываются без --force.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/prompt_rewrite/prompts"
FILES = ("system_prompt_t2i.txt", "system_prompt_edit.txt")
TARGET = Path(__file__).resolve().parent.parent / "resources" / "prompts"


def main(argv: list[str]) -> int:
    force = "--force" in argv
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = TARGET / name
        if destination.exists() and not force:
            print(f"{name}: уже есть, пропускаю (--force чтобы перезаписать)")
            continue
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            text = response.read().decode("utf-8")
        destination.write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} символов")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
