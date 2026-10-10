"""Пути приложения, значения по умолчанию и разбор аргументов командной строки.

Корень проекта вычисляется от расположения файла, а не от текущего каталога:
оболочку запускают и из проводника, и из планировщика, где рабочий каталог
произвольный.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "Qwen-Image-2.1"
# Дополнительные веса — каждое в своём каталоге рядом с основной моделью:
# INT8-трансформер Unsloth (замена bf16-трансформера, 7.3 ГБ) и адаптер
# дистиллята turbo (6 шагов вместо 16–40, 1.3 ГБ). Качаются по требованию.
INT8_DIR = PROJECT_ROOT / "Qwen-Image-2.1-INT8"
TURBO_DIR = PROJECT_ROOT / "Qwen-Image-2.1-turbo"
# Для карт на 6–12 ГБ: трансформер в GGUF (Unsloth, 3–7 ГБ по варианту) и
# текстовый энкодер, сжатый в int8 из bf16-весов основной модели (8.4 ГБ;
# собирается установкой, не качается).
GGUF_DIR = PROJECT_ROOT / "Qwen-Image-2.1-GGUF"
TE_INT8_DIR = PROJECT_ROOT / "Qwen-Image-2.1-TE-INT8"
# Распознавание позы на фотографии (DWPose, ONNX, 350 МБ) — для плитки
# «Добавить позу»; качает установка (--fetch-model).
DWPOSE_DIR = PROJECT_ROOT / "DWPose"
# Официальный Qwen-Image-2.1-Turbo: только трансформер (bf16 или GGUF в
# подкаталоге ``gguf``), планировщик и узлы расписания — энкодер и VAE
# берутся у основной модели. Качается при первом выборе пресета.
# Имя — не «…-Turbo»: Windows не различает регистр, и это был бы каталог
# адаптера Viggle «…-turbo» (так и случилось при первой загрузке).
TURBO_OFFICIAL_DIR = PROJECT_ROOT / "Qwen-Image-2.1-TurboOfficial"
# Лора расширения кадра (outpaint) — качается при первом расширении.
OUTPAINT_DIR = PROJECT_ROOT / "Qwen-Image-2.1-outpaint"
# Пользовательские LoRA (``*.safetensors``) — как ``models/loras`` у Fooocus.
# Ключ ``--lora-dir`` указывает другой каталог, например каталог ComfyUI.
LORA_DIR = PROJECT_ROOT / "loras"
RESOURCES_DIR = PROJECT_ROOT / "resources"
STYLES_DIR = RESOURCES_DIR / "styles"
SYSTEM_PROMPT_DIR = RESOURCES_DIR / "prompts"

USER_DIR = PROJECT_ROOT / "user"
OUTPUT_DIR = USER_DIR / "outputs"
PROMPT_DIR = USER_DIR / "prompts"
# Каталог поз openposes.com (модель — Эмма Уотсон) — часть поставки, лежит в
# репозитории. Позы, распознанные на фотографиях пользователя, — его данные и
# живут рядом с его генерациями (см. user_pose_dir).
POSE_LIBRARY_DIR = RESOURCES_DIR / "poses" / "catalog"
USER_POSE_SUBDIR = "poses"

LOG_DIR = PROJECT_ROOT / "logs"
ENDPOINT_FILE = PROJECT_ROOT / "llm_endpoint.txt"
SETTINGS_FILE = USER_DIR / "settings.json"

PRESET_NAMES = (
    "LowQuality", "MiddleQuality", "MaxQuality", "Turbo", "TurboDraft", "Turbo4",
    "QwenTurbo", "QwenTurboHD", "QwenTurbo2K",
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7865

# Секреты можно передать и окружением: аргументы командной строки видны в
# списке процессов всем пользователям машины, переменные окружения — нет.
ENV_PASSWORD = "FOOOCUS_QWEN_PASSWORD"
ENV_API_TOKEN = "FOOOCUS_QWEN_API_TOKEN"


@dataclass(frozen=True)
class AppConfig:
    """Параметры запуска оболочки."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    lang: str = "en"
    pin_memory: bool = True
    preload: bool = True
    preset: str = "MiddleQuality"
    verbose: bool = False
    open_browser: bool = False
    model_dir: Path = MODEL_DIR
    lora_dir: Path = LORA_DIR
    # Только API, без интерфейса (``--api``, ``fooocus_qwen/api``).
    api: bool = False
    # Вход в интерфейс (``--username``/``--password``) и токен API
    # (``--api-token``); ``None`` — без защиты. Секреты не печатаются в repr:
    # конфигурация попадает в журнал при отладке.
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    api_token: str | None = field(default=None, repr=False)

    @property
    def auth(self) -> tuple[str, str] | None:
        """Пара для ``gr.Blocks.launch(auth=...)`` или ``None``."""
        return (self.username, self.password) if self.username and self.password else None


def user_pose_dir() -> Path:
    """Свои позы — в каталоге генераций пользователя, подкаталогом ``poses``.

    Функцией, а не константой: инструменты и тесты перенаправляют
    ``OUTPUT_DIR`` во временный каталог, и позы обязаны уехать вместе с ним.
    Галерея этот подкаталог не показывает — она берёт только каталоги дат.
    """
    return OUTPUT_DIR / USER_POSE_SUBDIR


def ensure_directories() -> None:
    """Создаёт каталоги пользовательских данных. Вызывается при старте."""
    for path in (OUTPUT_DIR, PROMPT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fooocus_qwen", description="Qwen-Image-2.1 web UI")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    parser.add_argument("--lang", choices=("en", "ru"), default="en", help="interface language")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="MiddleQuality", help="quality preset")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument(
        "--lora-dir", type=Path, default=LORA_DIR,
        help="folder with LoRA files (*.safetensors), for example the ComfyUI loras folder",
    )
    # Закрепление памяти ускоряет переброску весов, но занимает десятки гигабайт
    # неперемещаемой оперативной памяти — на чужой машине это может не подойти.
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="do not pin weight copies in RAM",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="do not load the model in the background at startup (the first generation will take longer)",
    )
    parser.set_defaults(preload=True)
    # Браузер открывают скрипты запуска (run.ps1, run.sh), а не сам модуль:
    # при отладке и в дымовом прогоне лишнее окно только мешает.
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        help="open the interface in a browser once the server is ready",
    )
    # Пара к предыдущему: скрипты запуска ставят --open-browser первым, и
    # этот ключ, пришедший от пользователя следом, его перебивает —
    # argparse берёт последнее значение.
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="do not open a browser at startup",
    )
    parser.set_defaults(open_browser=False)
    # Режим API: только HTTP API (FastAPI), без интерфейса. Одинарный дефис —
    # тоже: так ключ записан в задании и так его наберут по памяти.
    parser.add_argument(
        "--api", "-api",
        dest="api",
        action="store_true",
        help="run only the HTTP API, without the web UI (see docs/API.md)",
    )
    parser.add_argument("--username", help="login for the web UI (needs --password)")
    parser.add_argument(
        "--password",
        help=f"password for the web UI; safer in the {ENV_PASSWORD} environment variable",
    )
    parser.add_argument(
        "--api-token",
        dest="api_token",
        help=f"Bearer token required by the API; safer in the {ENV_API_TOKEN} environment variable",
    )
    parser.add_argument("--prompt", help="generate a single image without the interface and exit")
    parser.add_argument("--out", help="where to save the --prompt result")
    parser.add_argument("--selftest", action="store_true", help="check that the environment is ready and exit")
    # Два режима установки. Логика у них общая для Windows и Linux, поэтому
    # живёт здесь, а install.ps1 и install.sh только зовут её флагом: то же
    # самое, написанное дважды на двух языках оболочки, разъезжается, и
    # первым это замечает пользователь той системы, что реже под рукой.
    parser.add_argument(
        "--fetch-model",
        dest="fetch_model",
        action="store_true",
        help="download missing model weights and exit",
    )
    parser.add_argument(
        "--setup-performance",
        dest="setup_performance",
        action="store_true",
        help="ask for weight precision (bf16/INT8) and SageAttention, save the choice and exit",
    )
    parser.add_argument(
        "--setup-llm",
        dest="setup_llm",
        action="store_true",
        help="ask for the language model address and token, save them and exit",
    )
    return parser


def _secret(given: str | None, variable: str) -> str | None:
    """Ключ командной строки, иначе переменная окружения; пустая строка — нет секрета."""
    value = given if given is not None else os.environ.get(variable)
    return value or None


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    password = _secret(args.password, ENV_PASSWORD)
    # Половина пары — почти наверняка ошибка: молча запустить интерфейс
    # открытым, когда человек думал, что закрыл его, хуже отказа.
    if bool(args.username) != bool(password):
        parser.error(f"--username and --password (or {ENV_PASSWORD}) go together")
    return AppConfig(
        host=args.host,
        port=args.port,
        lang=args.lang,
        pin_memory=args.pin_memory,
        preload=args.preload,
        preset=args.preset,
        verbose=args.verbose,
        open_browser=args.open_browser,
        lora_dir=args.lora_dir,
        api=args.api,
        username=args.username or None,
        password=password,
        api_token=_secret(args.api_token, ENV_API_TOKEN),
    )
