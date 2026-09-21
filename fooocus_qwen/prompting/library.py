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


def _load_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Читает JSON-файл пресета и решает, повреждён ли он.

    Возвращает (payload, None) при успехе или (None, сообщение) иначе. Что
    именно считать «повреждённым файлом» — правило одно на весь модуль:
    им пользуются и _resolve_candidates, и list_prompts, чтобы решение не
    могло разойтись между функцией, ищущей один пресет, и функцией,
    перечисляющей их все (та же болезнь, что и раунды 1-3, только между
    list_prompts и остальными, а не между save/load/delete).
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Пресет {path.name} повреждён: {error}"

    if not isinstance(payload, dict):
        return None, f"Пресет {path.name} повреждён: ожидался словарь, получен {type(payload).__name__}"

    return payload, None


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
        payload, error_msg = _load_payload(path)
        if error_msg is not None:
            # Файл существует, но повреждён либо содержит не словарь
            if damaged is None:
                damaged = (path, error_msg)
            continue

        if payload.get(_NAME_KEY) == name_clean:
            # Первый слот с нужным display-именем побеждает — то же правило
            # "первый в порядке кандидатов", что уже действует для free и
            # damaged. Без guard'а последний найденный слот тихо подменял бы
            # собой первый при коллизии (было исправлено в раунде 4).
            if match is None:
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
        payload, error_msg = _load_payload(path)
        if error_msg is not None:
            LOGGER.warning("%s — пропущен в списке пресетов", error_msg)
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
