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


def setup_logging(verbose: bool = False) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
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
