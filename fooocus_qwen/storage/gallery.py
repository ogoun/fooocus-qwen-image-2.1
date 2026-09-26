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


def next_path(directory: Path, when: datetime | None = None) -> Path:
    """Свободное имя файла внутри каталога сегодняшней даты.

    Имя основано на времени с точностью до секунды; если за секунду сохраняется
    несколько изображений, добавляется порядковый номер.
    """
    moment = when or datetime.now()
    day_dir = directory / f"{moment:%Y-%m-%d}"
    stem = f"{moment:%H-%M-%S}"

    candidate = day_dir / f"{stem}.png"
    index = 1
    while candidate.exists():
        candidate = day_dir / f"{stem}_{index}.png"
        index += 1
    return candidate


def recent(directory: Path, limit: int = 60) -> list[Path]:
    """Последние изображения, новые первыми."""
    if not directory.is_dir():
        return []

    files: list[Path] = []
    for day_dir in sorted(directory.iterdir(), reverse=True):
        if not day_dir.is_dir() or not _DAY_DIR.fullmatch(day_dir.name):
            continue
        files.extend(sorted(day_dir.glob("*.png"), reverse=True))
        if len(files) >= limit:
            break
    return files[:limit]
