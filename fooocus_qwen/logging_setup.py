"""Журналирование: одинаковый формат в консоли и в файле.

Файл журнала один на запуск и назван временем старта — так разбор инцидента
не требует гадать, какие строки к какому запуску относятся.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from . import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _configure_console_stream(stream):
    """Настраивает поток консоли на UTF-8, чтобы кириллица не роняла лог.

    Весь проект логирует по-русски, а консоль Windows по умолчанию открыта в
    однобайтовой кодовой странице вроде cp1252, которая кириллицу не
    представляет: без перенастройки первое же сообщение уровня INFO валит
    StreamHandler.emit() с UnicodeEncodeError. ``errors="replace"`` — не
    основной способ починки (UTF-8 покрывает кириллицу целиком), а страховка
    на случай символа, который не представим вообще нигде.

    ``reconfigure`` есть у обычного текстового потока, но не у всего, что
    бывает на месте ``sys.stdout`` (под pytest в некоторых режимах перехвата
    вывода метода может не быть) — тогда просто оставляем поток как есть,
    вместо того чтобы падать на попытке его настроить.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return stream


def setup_logging(verbose: bool = False) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(_configure_console_stream(sys.stdout))
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Пайплайн diffusers печатает прогресс-бар и предупреждения на каждом шаге,
    # что в файле журнала превращается в шум.
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
