"""Параметры генерации внутри PNG.

Пишутся два текстовых блока. Первый — JSON под своим ключом, он и есть источник
истины при восстановлении параметров. Второй — человекочитаемая строка под
ключом ``parameters``, который умеют показывать сторонние просмотрщики.

Чтение никогда не бросает исключение: пользователь перетащит в окно случайный
PNG, и это нормальная ситуация, а не ошибка.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

LOGGER = logging.getLogger(__name__)

CHUNK_KEY = "fooocus_qwen"
LEGACY_KEY = "parameters"


def to_readable(parameters: dict[str, Any]) -> str:
    """Строка в привычном для просмотрщиков формате."""
    prompt = parameters.get("prompt") or parameters.get("prompt_boosted", "")
    lines = [str(prompt)]

    negative = parameters.get("negative_prompt", "")
    if negative:
        lines.append(f"Negative prompt: {negative}")

    width = parameters.get("width")
    height = parameters.get("height")
    tail = [
        f"Steps: {parameters.get('steps')}",
        f"Seed: {parameters.get('seed')}",
        f"Size: {width}x{height}",
        f"CFG scale: {parameters.get('true_cfg_scale')}",
        f"Model: Qwen-Image-2.1",
    ]
    styles = parameters.get("styles") or []
    if styles:
        tail.append(f"Styles: {', '.join(styles)}")
    lines.append(", ".join(tail))

    return "\n".join(lines)


def save_png(image: Image.Image, path: Path, parameters: dict[str, Any]) -> Path:
    """Сохраняет изображение с параметрами. Режим изображения не меняется."""
    path.parent.mkdir(parents=True, exist_ok=True)

    info = PngImagePlugin.PngInfo()
    info.add_text(CHUNK_KEY, json.dumps(parameters, ensure_ascii=False))
    info.add_text(LEGACY_KEY, to_readable(parameters))

    image.save(path, format="PNG", pnginfo=info)
    return path


def read_png(path: Path) -> dict[str, Any] | None:
    """Возвращает параметры или ``None``, если их нет или они испорчены."""
    try:
        with Image.open(path) as image:
            raw = image.text.get(CHUNK_KEY)
    except (OSError, AttributeError) as error:
        LOGGER.debug("Не удалось прочитать %s: %s", path, error)
        return None

    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        LOGGER.debug("Испорченные метаданные в %s: %s", path, error)
        return None

    return payload if isinstance(payload, dict) else None
