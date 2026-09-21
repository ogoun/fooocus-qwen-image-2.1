"""Каталог стилей Fooocus и их применение к промту.

Стиль — это шаблон с местом для пользовательского промта плюс негативная часть.
Три стиля Fooocus имеют пустой шаблон и существуют только ради негатива; их
нельзя применять подстановкой, иначе пользовательский промт потеряется.

Важно: Qwen-Image-2.1 рассчитана на сэмплирование без guidance, и при
``true_cfg_scale = 1.0`` негативная часть в модель не попадает вовсе. Интерфейс
обязан сказать об этом пользователю — иначе молчание выглядит неисправностью.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PLACEHOLDER = "{prompt}"


@dataclass(frozen=True)
class Style:
    """Один стиль: шаблон положительной части и негативная часть."""

    name: str
    prompt: str
    negative_prompt: str


def load_styles(directory: Path) -> dict[str, Style]:
    """Читает все JSON-файлы каталога. Порядок файлов задаёт порядок в списке."""
    catalogue: dict[str, Style] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Файл стилей %s пропущен: %s", path.name, error)
            continue
        for entry in entries:
            name = entry.get("name", "").strip()
            if not name:
                continue
            catalogue[name] = Style(
                name=name,
                prompt=entry.get("prompt", ""),
                negative_prompt=entry.get("negative_prompt", ""),
            )
    return catalogue


def _join(parts: Sequence[str]) -> str:
    return ", ".join(part for part in parts if part.strip())


def apply_styles(
    prompt: str,
    negative: str,
    names: Sequence[str],
    catalogue: dict[str, Style],
) -> tuple[str, str]:
    """Возвращает пару «положительный промт, негативный промт».

    Несколько стилей дают несколько вариантов описания одного и того же предмета,
    склеенных запятой, — так это устроено в Fooocus.
    """
    positives: list[str] = []
    negatives: list[str] = [negative] if negative.strip() else []

    for name in names:
        style = catalogue.get(name)
        if style is None:
            LOGGER.warning("Стиль %r не найден в каталоге и пропущен", name)
            continue
        if PLACEHOLDER in style.prompt:
            positives.append(style.prompt.replace(PLACEHOLDER, prompt))
        elif style.prompt.strip():
            positives.append(_join([prompt, style.prompt]))
        if style.negative_prompt.strip():
            negatives.append(style.negative_prompt)

    return (_join(positives) if positives else prompt), _join(negatives)
