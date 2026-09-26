"""Пути приложения, значения по умолчанию и разбор аргументов командной строки.

Корень проекта вычисляется от расположения файла, а не от текущего каталога:
оболочку запускают и из проводника, и из планировщика, где рабочий каталог
произвольный.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "Qwen-Image-2.1"
# Дополнительные веса — каждое в своём каталоге рядом с основной моделью:
# INT8-трансформер Unsloth (замена bf16-трансформера, 7.3 ГБ) и адаптер
# дистиллята turbo (6 шагов вместо 16–40, 1.3 ГБ). Качаются по требованию.
INT8_DIR = PROJECT_ROOT / "Qwen-Image-2.1-INT8"
TURBO_DIR = PROJECT_ROOT / "Qwen-Image-2.1-turbo"
# Распознавание позы на фотографии (DWPose, ONNX, 350 МБ) — для плитки
# «Добавить позу»; качается при первом распознавании.
DWPOSE_DIR = PROJECT_ROOT / "DWPose"
RESOURCES_DIR = PROJECT_ROOT / "resources"
STYLES_DIR = RESOURCES_DIR / "styles"
SYSTEM_PROMPT_DIR = RESOURCES_DIR / "prompts"

USER_DIR = PROJECT_ROOT / "user"
OUTPUT_DIR = USER_DIR / "outputs"
PROMPT_DIR = USER_DIR / "prompts"
# Позы: каталог openposes.com (качается при первом открытии окна поз, в
# репозиторий не входит) и позы, распознанные на фотографиях пользователя.
POSE_LIBRARY_DIR = PROJECT_ROOT / "poses"
USER_POSE_DIR = USER_DIR / "poses"

LOG_DIR = PROJECT_ROOT / "logs"
ENDPOINT_FILE = PROJECT_ROOT / "llm_endpoint.txt"
SETTINGS_FILE = USER_DIR / "settings.json"

PRESET_NAMES = ("LowQuality", "MiddleQuality", "MaxQuality", "Turbo")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7865


@dataclass(frozen=True)
class AppConfig:
    """Параметры запуска оболочки."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    lang: str = "ru"
    pin_memory: bool = True
    preload: bool = True
    preset: str = "MiddleQuality"
    verbose: bool = False
    open_browser: bool = False
    model_dir: Path = MODEL_DIR


def ensure_directories() -> None:
    """Создаёт каталоги пользовательских данных. Вызывается при старте."""
    for path in (OUTPUT_DIR, PROMPT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fooocus_qwen", description="Оболочка Qwen-Image-2.1")
    parser.add_argument("--host", default=DEFAULT_HOST, help="адрес прослушивания")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="порт прослушивания")
    parser.add_argument("--lang", choices=("ru", "en"), default="ru", help="язык интерфейса")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="MiddleQuality", help="пресет качества")
    parser.add_argument("--verbose", action="store_true", help="подробный журнал")
    # Закрепление памяти ускоряет переброску весов, но занимает десятки гигабайт
    # неперемещаемой оперативной памяти — на чужой машине это может не подойти.
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="не закреплять копии весов в оперативной памяти",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="не загружать модель в фоне при старте (первая генерация будет дольше)",
    )
    parser.set_defaults(preload=True)
    # Браузер открывают скрипты запуска (run.ps1, run.sh), а не сам модуль:
    # при отладке и в дымовом прогоне лишнее окно только мешает.
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="открыть интерфейс в браузере, когда сервер будет готов",
    )
    # Пара к предыдущему: скрипты запуска ставят --open-browser первым, и
    # этот ключ, пришедший от пользователя следом, его перебивает —
    # argparse берёт последнее значение.
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="не открывать браузер при запуске",
    )
    parser.set_defaults(open_browser=False)
    parser.add_argument("--prompt", help="сгенерировать одно изображение без интерфейса и выйти")
    parser.add_argument("--out", help="куда сохранить результат режима --prompt")
    parser.add_argument("--selftest", action="store_true", help="проверить готовность окружения и выйти")
    # Два режима установки. Логика у них общая для Windows и Linux, поэтому
    # живёт здесь, а install.ps1 и install.sh только зовут её флагом: то же
    # самое, написанное дважды на двух языках оболочки, разъезжается, и
    # первым это замечает пользователь той системы, что реже под рукой.
    parser.add_argument(
        "--fetch-model",
        dest="fetch_model",
        action="store_true",
        help="скачать недостающие веса модели и выйти",
    )
    parser.add_argument(
        "--setup-performance",
        dest="setup_performance",
        action="store_true",
        help="спросить точность весов (bf16/INT8) и SageAttention, записать выбор и выйти",
    )
    parser.add_argument(
        "--setup-llm",
        dest="setup_llm",
        action="store_true",
        help="спросить адрес и токен языковой модели, записать их и выйти",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(
        host=args.host,
        port=args.port,
        lang=args.lang,
        pin_memory=args.pin_memory,
        preload=args.preload,
        preset=args.preset,
        verbose=args.verbose,
        open_browser=args.open_browser,
    )
