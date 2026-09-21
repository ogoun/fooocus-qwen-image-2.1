"""Именованные пресеты промтов.

Один пресет — один JSON-файл. Имя, которое видит пользователь, хранится внутри
файла, а имя файла получается из него обеззараживанием: пользователь вправе
назвать пресет как угодно, включая символы, недопустимые в путях.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_NAME_KEY = "__name__"


def safe_filename(name: str) -> str:
    cleaned = _UNSAFE.sub("_", name.strip()).strip(". ")
    # Точки в начале и конце и путевые сегменты убираются полностью: иначе
    # имя вида «../../x» увело бы файл за пределы каталога пресетов.
    cleaned = cleaned.replace("..", "_")
    return cleaned or "preset"


def save_prompt(name: str, payload: dict[str, Any], directory: Path) -> Path:
    if not name.strip():
        raise ValueError("Имя пресета не может быть пустым")

    directory.mkdir(parents=True, exist_ok=True)
    clean_name = safe_filename(name)
    name_clean = name.strip()

    # Проверяем коллизии имён: если файл уже существует, но содержит другой дисплей-имя,
    # ищем свободный номер
    path = directory / f"{clean_name}.json"
    attempt = 1
    max_attempts = 100

    while path.is_file() and attempt <= max_attempts:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get(_NAME_KEY) == name_clean:
                # Совпадает дисплей-имя, перезаписываем
                break
        except (OSError, json.JSONDecodeError):
            # Файл повреждён или нечитаем, перезаписываем
            break

        # Имена различаются, ищем свободный номер
        path = directory / f"{clean_name}_{attempt + 1}.json"
        attempt += 1

    if attempt > max_attempts:
        raise RuntimeError(f"Не удалось найти свободное место для пресета {name_clean}: столкновение имён исчерпано")

    stored = dict(payload)
    stored[_NAME_KEY] = name_clean
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_prompt(name: str, directory: Path) -> dict[str, Any]:
    """Загружает пресет по дисплей-имени, ища среди файлов с коллизиями имён."""
    clean_name = safe_filename(name)
    name_clean = name.strip()

    # Ищем файл с соответствующим дисплей-именем среди возможных коллизий
    candidates = [directory / f"{clean_name}.json"]
    for attempt in range(1, 100):
        candidates.append(directory / f"{clean_name}_{attempt + 1}.json")

    for path in candidates:
        if not path.is_file():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            raise ValueError(f"Пресет {path} повреждён: ожидалась сохранённая словарь, получена {type(payload).__name__}")

        if payload.get(_NAME_KEY) == name_clean:
            # Найден нужный файл
            payload.pop(_NAME_KEY, None)
            return payload

    raise FileNotFoundError(f"Пресет промта не найден: {name_clean}")


def list_prompts(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []

    names: list[str] = []
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Пресет %s пропущен: %s", path.name, error)
            continue
        if not isinstance(payload, dict):
            LOGGER.warning("Пресет %s пропущен: не словарь, а %s", path.name, type(payload).__name__)
            continue
        names.append(payload.get(_NAME_KEY, path.stem))
    return sorted(names)


def delete_prompt(name: str, directory: Path) -> bool:
    path = directory / f"{safe_filename(name)}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True
