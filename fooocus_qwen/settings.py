"""Настройки производительности: точность весов трансформера и механизм внимания.

Выбор делается при установке и меняется во вкладке «Настройки», поэтому
живёт в файле, а не в ключах командной строки: ключ пришлось бы помнить при
каждом запуске, а файл запоминает сам. Лежит в ``user/`` — это выбор
человека за машиной, а не проекта, и в репозиторий он не попадает.

Испорченный или устаревший файл — не повод ронять запуск: непонятное поле
заменяется значением по умолчанию, а причина уходит в журнал.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import config

LOGGER = logging.getLogger(__name__)

PRECISION_BF16 = "bf16"
PRECISION_INT8 = "int8"
PRECISIONS: tuple[str, ...] = (PRECISION_BF16, PRECISION_INT8)


@dataclass(frozen=True)
class Settings:
    """Выбор пользователя.

    ``precision`` — в чём хранится трансформер: ``bf16`` (исходные веса,
    13.3 ГиБ видеопамяти) или ``int8`` (7.3 ГиБ, веса Unsloth; качество —
    LPIPS 0.064 к bf16 по их замеру). ``sage_attention`` — считать внимание
    SageAttention, если пакет установлен: −15…25 % времени шага
    (``docs/research/2026-09-24-uskorenie-turbo-sage-int8.md``).
    """

    precision: str = PRECISION_BF16
    sage_attention: bool = False


def load(path: Path | None = None) -> Settings:
    """Читает настройки; отсутствующий файл — значения по умолчанию."""
    path = path or config.SETTINGS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError) as error:
        LOGGER.warning("Файл настроек %s не читается (%s), беру значения по умолчанию", path, error)
        return Settings()
    if not isinstance(raw, dict):
        LOGGER.warning("Файл настроек %s — не объект, беру значения по умолчанию", path)
        return Settings()

    settings = Settings()
    precision = raw.get("precision", settings.precision)
    if precision in PRECISIONS:
        settings = replace(settings, precision=precision)
    else:
        LOGGER.warning("Неизвестная точность %r в %s, беру %s", precision, path, settings.precision)
    sage = raw.get("sage_attention", settings.sage_attention)
    if isinstance(sage, bool):
        settings = replace(settings, sage_attention=sage)
    return settings


def save(settings: Settings, path: Path | None = None) -> None:
    """Записывает настройки целиком, через временный файл.

    Запись через замену: оборванная на середине запись оставила бы
    полуфайл, и следующий запуск молча вернулся бы к значениям по умолчанию.
    """
    if settings.precision not in PRECISIONS:
        raise ValueError(f"неизвестная точность: {settings.precision!r}")
    path = path or config.SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update(path: Path | None = None, **changes) -> Settings:
    """Меняет часть настроек и сохраняет."""
    settings = replace(load(path), **changes)
    save(settings, path)
    return settings
