"""Именованные пресеты промтов.

Один пресет — один JSON-файл. Имя, которое видит пользователь, хранится внутри
файла, а имя файла получается из него обеззараживанием: пользователь вправе
назвать пресет как угодно, включая символы, недопустимые в путях.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_NAME_KEY = "__name__"
_MAX_COLLISIONS = 100


def safe_filename(name: str) -> str:
    cleaned = _UNSAFE.sub("_", name.strip()).strip(". ")
    # Точки в начале и конце и путевые сегменты убираются полностью: иначе
    # имя вида «../../x» увело бы файл за пределы каталога пресетов.
    cleaned = cleaned.replace("..", "_")
    return cleaned or "preset"


@dataclass(frozen=True)
class _CandidateResolution:
    """Результат поиска кандидатов для пресета по дисплей-имени."""
    # (path, payload) если найден пресет с правильным display-имнем
    match: tuple[Path, dict[str, Any]] | None
    # path первого свободного слота (не существующего на диске)
    free: Path | None
    # (path, error_msg) первого повреждённого слота
    damaged: tuple[Path, str] | None


def _resolve_candidates(name: str, directory: Path) -> _CandidateResolution:
    """Ищет пресет по дисплей-имени среди возможных коллизий.

    Возвращает:
    - match: (path, payload) если display-имя совпадает
    - free: первый свободный слот на диске
    - damaged: первый повреждённый слот (путь и сообщение об ошибке)

    Все три функции (save_prompt, load_prompt, delete_prompt) используют
    этот результат согласно своей логике.
    """
    clean_name = safe_filename(name)
    name_clean = name.strip()

    # Строим список кандидатов
    candidates = [directory / f"{clean_name}.json"]
    for attempt in range(1, _MAX_COLLISIONS):
        candidates.append(directory / f"{clean_name}_{attempt + 1}.json")

    match = None
    free = None
    damaged = None

    for path in candidates:
        if not path.is_file():
            # Первый свободный слот
            if free is None:
                free = path
            continue

        # Файл существует, пробуем его прочитать
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            # Файл существует но JSON повреждён
            if damaged is None:
                damaged = (path, f"Пресет {path.name} повреждён: {error}")
            continue

        if not isinstance(payload, dict):
            # Файл существует, JSON валиден, но не словарь
            if damaged is None:
                damaged = (path, f"Пресет {path.name} повреждён: ожидалась словарь, получена {type(payload).__name__}")
            continue

        if payload.get(_NAME_KEY) == name_clean:
            # Найден правильный пресет
            match = (path, payload)
            # Продолжаем поиск, чтобы найти free и damaged если нужны

    return _CandidateResolution(match=match, free=free, damaged=damaged)


def save_prompt(name: str, payload: dict[str, Any], directory: Path) -> Path:
    if not name.strip():
        raise ValueError("Имя пресета не может быть пустым")

    directory.mkdir(parents=True, exist_ok=True)
    name_clean = name.strip()

    resolution = _resolve_candidates(name, directory)

    if resolution.match:
        # Пресет с таким display-имнем уже существует, перезаписываем его
        path, _ = resolution.match
    elif resolution.free:
        # Нет matching display-имени, используем первый свободный слот
        path = resolution.free
    else:
        # Все слоты заняты
        raise RuntimeError(f"Не удалось найти свободное место для пресета {name_clean}: столкновение имён исчерпано")

    stored = dict(payload)
    stored[_NAME_KEY] = name_clean
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_prompt(name: str, directory: Path) -> dict[str, Any]:
    """Загружает пресет по дисплей-имени, ища среди файлов с коллизиями имён."""
    resolution = _resolve_candidates(name, directory)

    if resolution.match:
        # Найден пресет с правильным display-имнем
        _, payload = resolution.match
        payload.pop(_NAME_KEY, None)
        return payload

    if resolution.damaged:
        # Файл найден но повреждён
        _, error_msg = resolution.damaged
        raise ValueError(error_msg)

    # Файл не найден
    raise FileNotFoundError(f"Пресет промта не найден: {name.strip()}")


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
    """Удаляет пресет, найдя его по дисплей-имени среди коллизий."""
    resolution = _resolve_candidates(name, directory)

    if resolution.match:
        # Найден пресет с правильным display-имнем, удаляем его
        path, _ = resolution.match
        path.unlink()
        return True

    # Пресет не найден
    return False
