"""Скачивает каталог поз openposes.com (модель — Эмма Уотсон) в ``poses/``.

Интерфейс делает то же сам при первом открытии окна поз; скрипт нужен, чтобы
подготовить каталог заранее или из уже скачанного архива:

    .venv\\Scripts\\python tools\\fetch_poses.py [--archive путь\\к\\poses.zip]

Качается только недостающее — повторный запуск дозагружает прерванное.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Корень проекта в sys.path: инструменты запускают по пути, и тогда туда
# попадает каталог скрипта, а не корень.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen import config
from fooocus_qwen.logging_setup import use_utf8_console
from fooocus_qwen.poses import library

use_utf8_console()  # скрипт печатает по-русски; cp1252 его бы уронил


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=None, help="уже скачанный poses.zip")
    args = parser.parse_args(argv)

    def report(done: int, total: int) -> None:
        print(f"\rплитки: {done}/{total}", end="", flush=True)

    count = library.fetch_catalog(config.POSE_LIBRARY_DIR, archive=args.archive, progress=report)
    print(f"\nкаталог: {count} поз в {config.POSE_LIBRARY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
