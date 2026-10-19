"""Раскладка результатов по каталогам дат.

Каталог на день — то же решение, что в Fooocus: за месяц работы в одной папке
накапливаются тысячи файлов, и любой файловый менеджер на них спотыкается.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Каталог дня: ГГГГ-ММ-ДД (см. next_path). Рядом бывают и другие подкаталоги —
# свои позы пользователя (``config.USER_POSE_SUBDIR``), — это не результаты.
_DAY_DIR = re.compile(r"\d{4}-\d{2}-\d{2}")


# Что галерея показывает: картинки и ролики вкладки Video. Параметры ролика
# лежат соседним JSON — он сам результатом не считается.
RESULT_SUFFIXES = (".png", ".mp4")
# Контактный лист раскадровки лежит рядом со своим роликом, но результатом не
# считается: в галерее он был бы картинкой без параметров.
SHEET_SUFFIX = ".sheet.png"


def next_path(directory: Path, when: datetime | None = None, suffix: str = ".png") -> Path:
    """Свободное имя файла внутри каталога сегодняшней даты.

    Имя основано на времени с точностью до секунды; если за секунду сохраняется
    несколько результатов, добавляется порядковый номер. Ролик и картинка с
    одним временем не делят имя: занятость проверяется по всем видам
    результатов, иначе у ``12-00-00.png`` и ``12-00-00.mp4`` совпали бы
    соседние JSON-файлы параметров.
    """
    moment = when or datetime.now()
    day_dir = directory / f"{moment:%Y-%m-%d}"
    stem = f"{moment:%H-%M-%S}"

    def taken(name: str) -> bool:
        return any((day_dir / f"{name}{kind}").exists() for kind in {*RESULT_SUFFIXES, suffix})

    name = stem
    index = 1
    while taken(name):
        name = f"{stem}_{index}"
        index += 1
    return day_dir / f"{name}{suffix}"


def recent(directory: Path, limit: int = 60) -> list[Path]:
    """Последние результаты — картинки и ролики, новые первыми."""
    if not directory.is_dir():
        return []

    files: list[Path] = []
    for day_dir in sorted(directory.iterdir(), reverse=True):
        if not day_dir.is_dir() or not _DAY_DIR.fullmatch(day_dir.name):
            continue
        found = [path for path in day_dir.iterdir()
                 if path.suffix.lower() in RESULT_SUFFIXES and not path.name.lower().endswith(SHEET_SUFFIX)]
        files.extend(sorted(found, key=lambda path: path.name, reverse=True))
        if len(files) >= limit:
            break
    return files[:limit]
