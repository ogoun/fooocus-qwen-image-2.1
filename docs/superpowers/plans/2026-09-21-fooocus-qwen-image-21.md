# Fooocus-Qwen-Image-2.1 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Оболочка в духе Fooocus поверх локальной модели Qwen-Image-2.1: генерация, редактирование промтом, редактирование по маске, библиотека промтов, AI-буст через внешнюю LLM, сохранение в PNG.

**Architecture:** Пакет `fooocus_qwen` из четырёх слоёв. Слои `imaging`, `prompting`, `llm`, `storage` не импортируют torch и покрываются тестами без GPU. Слой `engine` владеет пайплайном diffusers, политикой резидентности моделей и кэшем эмбеддингов. Слой `ui` собирает Gradio и не содержит вычислений.

**Tech Stack:** Python 3.12, PyTorch 2.x + CUDA 12.8, diffusers из git (коммит зафиксирован), transformers 5.17+, Gradio 6.5.1, Pillow, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-fooocus-qwen-image-21-design.md`

## Global Constraints

- Веса модели лежат в `E:\Fooocus-Qwen-Image-2.1\Qwen-Image-2.1`. Их **не трогать и не перемещать**; каталог исключён из git.
- `diffusers` ставится строго из git с коммитом `cc8644b447d8f11074d3df06d0ee0e3e7c91bf75`. Пайплайна `QwenImage21Pipeline` нет ни в одном релизе PyPI.
- `transformers>=5.17`.
- `gradio==6.5.1` — версия зафиксирована, API `ImageEditor` от неё зависит.
- Всё в `torch.bfloat16`. Квантование не применяется нигде.
- Сервер слушает `0.0.0.0` по умолчанию, порт 7865.
- Язык кода: имена — английские, докстринги и комментарии — **русские**, по образцу `I:\ASR_LLM_TTS`.
- Комментарий объясняет **почему**, а не что. Комментарии вида «увеличиваем счётчик» запрещены.
- Каждая задача заканчивается коммитом. Работа идёт в ветке `feature/studio`, не в `master`.
- Размеры кадра всегда кратны 32 (`vae_scale_factor * 2`).
- Референсы адресуются в промте тегами `<image1>`…`<imageN>`; при единственном референсе теги не используются.
- Тесты — pytest, лежат в `tests/`, запускаются `.venv\Scripts\python -m pytest tests -q`.

---

### Task 1: Скелет проекта, зависимости, скрипты установки и запуска

**Files:**
- Create: `requirements.txt`
- Create: `install.ps1`, `install.sh`, `run.ps1`, `run.sh`
- Create: `fooocus_qwen/__init__.py`, `fooocus_qwen/config.py`, `fooocus_qwen/logging_setup.py`, `fooocus_qwen/__main__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: ничего.
- Produces: `config.AppConfig` (поля `host: str`, `port: int`, `lang: str`, `pin_memory: bool`, `preset: str`, `model_dir: Path`, `verbose: bool`); `config.parse_args(argv: list[str] | None) -> AppConfig`; константы `config.PROJECT_ROOT`, `MODEL_DIR`, `RESOURCES_DIR`, `USER_DIR`, `OUTPUT_DIR`, `PROMPT_DIR`, `LOG_DIR`, `STYLES_DIR`, `SYSTEM_PROMPT_DIR`; `logging_setup.setup_logging(verbose: bool) -> None`; `__main__.selftest() -> int`.

- [ ] **Step 1: Создать ветку**

```bash
cd /e/Fooocus-Qwen-Image-2.1
git checkout -b feature/studio
```

- [ ] **Step 2: Написать падающий тест на конфигурацию**

Создать `tests/test_config.py`:

```python
"""Проверки разбора аргументов и путей приложения."""

from pathlib import Path

from fooocus_qwen import config


def test_defaults_listen_on_all_interfaces():
    # Оболочка должна быть доступна из локальной сети без дополнительных ключей.
    cfg = config.parse_args([])
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 7865
    assert cfg.lang == "ru"
    assert cfg.pin_memory is True
    assert cfg.preset == "MiddleQuality"


def test_arguments_override_defaults():
    cfg = config.parse_args(["--host", "127.0.0.1", "--port", "8000", "--lang", "en", "--no-pin-memory"])
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.lang == "en"
    assert cfg.pin_memory is False


def test_model_dir_points_at_downloaded_weights():
    assert config.MODEL_DIR == config.PROJECT_ROOT / "Qwen-Image-2.1"
    assert (config.MODEL_DIR / "model_index.json").is_file()


def test_user_directories_are_created_on_demand():
    config.ensure_directories()
    for path in (config.OUTPUT_DIR, config.PROMPT_DIR, config.LOG_DIR):
        assert path.is_dir()


def test_preset_name_is_validated():
    import pytest

    with pytest.raises(SystemExit):
        config.parse_args(["--preset", "Ultra"])
```

- [ ] **Step 3: Убедиться, что тест падает**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen'`

- [ ] **Step 4: Написать `fooocus_qwen/config.py`**

```python
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
RESOURCES_DIR = PROJECT_ROOT / "resources"
STYLES_DIR = RESOURCES_DIR / "styles"
SYSTEM_PROMPT_DIR = RESOURCES_DIR / "prompts"
PRESET_DIR = RESOURCES_DIR / "presets"

USER_DIR = PROJECT_ROOT / "user"
OUTPUT_DIR = USER_DIR / "outputs"
PROMPT_DIR = USER_DIR / "prompts"

LOG_DIR = PROJECT_ROOT / "logs"
ENDPOINT_FILE = PROJECT_ROOT / "llm_endpoint.txt"

PRESET_NAMES = ("LowQuality", "MiddleQuality", "MaxQuality")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7865


@dataclass(frozen=True)
class AppConfig:
    """Параметры запуска оболочки."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    lang: str = "ru"
    pin_memory: bool = True
    preset: str = "MiddleQuality"
    verbose: bool = False
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
    parser.add_argument("--selftest", action="store_true", help="проверить готовность окружения и выйти")
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(
        host=args.host,
        port=args.port,
        lang=args.lang,
        pin_memory=args.pin_memory,
        preset=args.preset,
        verbose=args.verbose,
    )
```

- [ ] **Step 5: Написать `fooocus_qwen/logging_setup.py`**

```python
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
```

Создать `fooocus_qwen/__init__.py` с одной строкой:

```python
"""Оболочка Fooocus-подобного интерфейса поверх Qwen-Image-2.1."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Написать `fooocus_qwen/__main__.py` с режимом самопроверки**

```python
"""Точка входа. По умолчанию поднимает веб-интерфейс, с --selftest проверяет окружение."""

from __future__ import annotations

import logging
import sys

from . import config, logging_setup

LOGGER = logging.getLogger(__name__)


def selftest() -> int:
    """Проверяет всё, без чего оболочка не запустится, и печатает отчёт.

    Возвращает код возврата процесса: ноль, если готово к работе.
    """
    problems: list[str] = []

    try:
        import torch
    except ImportError as error:
        print(f"[нет] torch не импортируется: {error}")
        return 1

    print(f"[ок ] torch {torch.__version__}")
    if not torch.cuda.is_available():
        problems.append("CUDA недоступна — модель пойдёт на процессоре, это неприемлемо медленно")
        print("[нет] CUDA недоступна")
    else:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[ок ] CUDA: {name}, {total:.1f} ГБ")

    try:
        from diffusers import QwenImage21Pipeline  # noqa: F401
    except ImportError as error:
        problems.append("QwenImage21Pipeline недоступен — нужен diffusers из git")
        print(f"[нет] QwenImage21Pipeline не импортируется: {error}")
    else:
        import diffusers

        print(f"[ок ] diffusers {diffusers.__version__}, QwenImage21Pipeline на месте")

    index = config.MODEL_DIR / "model_index.json"
    if index.is_file():
        print(f"[ок ] веса модели: {config.MODEL_DIR}")
    else:
        problems.append(f"не найден {index}")
        print(f"[нет] веса модели не найдены: {index}")

    if problems:
        print("\nНе готово к работе:")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("\nОкружение готово.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = config.build_parser().parse_args(argv)
    logging_setup.setup_logging(args.verbose)
    config.ensure_directories()

    if args.selftest:
        return selftest()

    # Модуль ui.app появляется в задаче 12; до неё запуск без --selftest и без
    # --prompt упадёт на этом импорте, и это правильное поведение: интерфейса ещё нет.
    from .ui.app import launch

    cfg = config.parse_args(argv)
    launch(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Запустить тесты**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS, 5 тестов

- [ ] **Step 8: Написать `requirements.txt`**

```
# torch и torchvision ставятся отдельно, с индекса CUDA — см. install.ps1 / install.sh.

# Пайплайна QwenImage21Pipeline нет ни в одном релизе PyPI, только в main.
# Коммит зафиксирован: без этого сборка невоспроизводима.
diffusers @ git+https://github.com/huggingface/diffusers@cc8644b447d8f11074d3df06d0ee0e3e7c91bf75

# Qwen3-VL как текстовый энкодер требует пятую ветку transformers.
transformers>=5.17
accelerate>=1.0
safetensors>=0.4
huggingface-hub>=0.30
tokenizers>=0.21
protobuf>=4.25
sentencepiece>=0.2

Pillow>=10.3
numpy>=1.26,<3
opencv-python-headless>=4.10

gradio==6.5.1

einops>=0.8
ftfy>=6.2
tqdm>=4.66
psutil>=6.0
nvidia-ml-py>=12.560

pytest>=8.0
```

- [ ] **Step 9: Написать `install.ps1`**

```powershell
<#
.SYNOPSIS
    Сборка окружения: .venv, torch с CUDA, остальные зависимости, проверка готовности.

.DESCRIPTION
    Порядок шагов важен:

    1. torch ставится ПЕРВЫМ и с индекса pytorch. Поставленный следом за
       остальными, он приедет с PyPI — без CUDA, и всё пойдёт на процессоре.
    2. Остальные зависимости из requirements.txt.
    3. torch закрепляется ещё раз: сторонние пакеты способны подменить
       CUDA-сборку на обычную.
    4. Самопроверка: «установилось» должно означать «запустится».

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -Recreate
#>

[CmdletBinding()]
param([switch] $Recreate)

$ErrorActionPreference = 'Stop'
# Консоль Windows по умолчанию не UTF-8, и русский текст выводится мусором.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$torchIndex = 'https://download.pytorch.org/whl/cu128'

function Step($number, $text) {
    Write-Host ''
    Write-Host "[$number/4] $text" -ForegroundColor Cyan
}

Push-Location $root
try {
    if ($Recreate -and (Test-Path $venv)) {
        Write-Host 'Удаляю прежнее окружение…' -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv
    }

    Step 1 'Создаю окружение и ставлю torch с поддержкой CUDA'
    if (-not (Test-Path $python)) {
        py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host '  Python 3.12 не найден, пробую 3.13' -ForegroundColor Yellow
            py -3.13 -m venv $venv
        }
        if (-not (Test-Path $python)) { throw 'Не удалось создать .venv' }
    }
    & $python -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось обновить pip' }

    & $python -m pip install torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось поставить torch с CUDA' }

    Step 2 'Ставлю остальные зависимости'
    & $python -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось поставить зависимости' }

    Step 3 'Возвращаю CUDA-сборку torch, если её заменили'
    $version = & $python -c "import torch, sys; sys.stdout.write(torch.__version__)"
    if ($version -notlike '*cu*') {
        Write-Host "  сейчас стоит $version — переустанавливаю" -ForegroundColor Yellow
        & $python -m pip install --force-reinstall torch torchvision --index-url $torchIndex
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось вернуть torch с CUDA' }
    } else {
        Write-Host "  всё на месте: $version"
    }

    Step 4 'Проверяю готовность'
    & $python -m fooocus_qwen --selftest
    if ($LASTEXITCODE -ne 0) { throw 'Самопроверка не пройдена' }

    Write-Host ''
    Write-Host 'Готово. Запуск:' -ForegroundColor Green
    Write-Host '    .\run.ps1' -ForegroundColor Cyan
} finally {
    Pop-Location
}
```

- [ ] **Step 10: Написать `install.sh`**

```bash
#!/usr/bin/env bash
# Сборка окружения: .venv, torch с CUDA, остальные зависимости, проверка готовности.
# Порядок шагов важен — см. комментарии в install.ps1.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
python="$venv/bin/python"
torch_index="https://download.pytorch.org/whl/cu128"

step() { printf '\n\033[36m[%s/4] %s\033[0m\n' "$1" "$2"; }

cd "$root"

if [ "${1:-}" = "--recreate" ] && [ -d "$venv" ]; then
    echo "Удаляю прежнее окружение…"
    rm -rf "$venv"
fi

step 1 "Создаю окружение и ставлю torch с поддержкой CUDA"
if [ ! -x "$python" ]; then
    python3.12 -m venv "$venv" 2>/dev/null || python3 -m venv "$venv"
fi
"$python" -m pip install --upgrade pip setuptools wheel
"$python" -m pip install torch torchvision --index-url "$torch_index"

step 2 "Ставлю остальные зависимости"
"$python" -m pip install -r "$root/requirements.txt"

step 3 "Возвращаю CUDA-сборку torch, если её заменили"
version="$("$python" -c 'import torch, sys; sys.stdout.write(torch.__version__)')"
case "$version" in
    *cu*) echo "  всё на месте: $version" ;;
    *)    echo "  сейчас стоит $version — переустанавливаю"
          "$python" -m pip install --force-reinstall torch torchvision --index-url "$torch_index" ;;
esac

step 4 "Проверяю готовность"
"$python" -m fooocus_qwen --selftest

printf '\n\033[32mГотово. Запуск:\033[0m\n    ./run.sh\n'
```

- [ ] **Step 11: Написать `run.ps1` и `run.sh`**

`run.ps1`:

```powershell
<#
.SYNOPSIS
    Запуск оболочки в окружении проекта.

.DESCRIPTION
    Приложение работает только в собственном .venv: в системном Python нет ни
    diffusers из git, ни torch с CUDA. Скрипт проверяет не факт установки, а
    работоспособность — окружение могло остаться от прерванной установки.

    Все аргументы передаются приложению как есть.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 --lang en --port 7870
#>

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host 'Окружение .venv не найдено.' -ForegroundColor Yellow
    Write-Host 'Создайте его один раз:' -ForegroundColor Yellow
    Write-Host '    .\install.ps1' -ForegroundColor Cyan
    exit 1
}

$check = & $python -c "import torch, sys; sys.stdout.write('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Окружение .venv повреждено: не импортируется torch.' -ForegroundColor Red
    Write-Host '    Remove-Item -Recurse -Force .venv; .\install.ps1' -ForegroundColor Cyan
    exit 1
}
if ($check -ne 'cuda') {
    Write-Host 'CUDA недоступна — генерация пойдёт на процессоре и займёт часы.' -ForegroundColor Yellow
    Write-Host 'Переустановите torch:' -ForegroundColor Yellow
    Write-Host '    .venv\Scripts\python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128' -ForegroundColor Cyan
}

Push-Location $root
try {
    & $python -m fooocus_qwen @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
```

`run.sh`:

```bash
#!/usr/bin/env bash
# Запуск оболочки в окружении проекта. Все аргументы уходят приложению как есть.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Окружение .venv не найдено. Создайте его один раз: ./install.sh" >&2
    exit 1
fi

if ! "$python" -c 'import torch' >/dev/null 2>&1; then
    echo "Окружение .venv повреждено: не импортируется torch." >&2
    echo "    rm -rf .venv && ./install.sh" >&2
    exit 1
fi

if [ "$("$python" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')" != "cuda" ]; then
    echo "CUDA недоступна — генерация пойдёт на процессоре и займёт часы." >&2
fi

cd "$root"
exec "$python" -m fooocus_qwen "$@"
```

- [ ] **Step 12: Собрать окружение и проверить самопроверку**

Run: `.\install.ps1`
Expected: четыре шага без ошибок, в конце `Окружение готово.`

Если шаг 4 сообщает, что `QwenImage21Pipeline` не импортируется, значит коммит diffusers не содержит пайплайна — остановиться и сообщить, не подбирая другой коммит наугад.

- [ ] **Step 13: Коммит**

```bash
git add requirements.txt install.ps1 install.sh run.ps1 run.sh fooocus_qwen tests
git commit -m "feat: скелет проекта, зависимости и скрипты установки"
```

---

### Task 2: Клиент внешней LLM

**Files:**
- Create: `fooocus_qwen/llm/__init__.py`, `fooocus_qwen/llm/endpoint.py`, `fooocus_qwen/llm/client.py`
- Create: `llm_endpoint.txt` (образец, в git не попадёт — исключён в `.gitignore`)
- Create: `tests/test_llm_endpoint.py`, `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `config.ENDPOINT_FILE`.
- Produces: `endpoint.LlmEndpoint` (поля `base_url: str`, `token: str | None`, `backend: str`; свойства `chat_url`, `models_url`); `endpoint.parse_endpoint_file(text: str, default_port: int = 8000) -> LlmEndpoint`; `endpoint.load_endpoint(path: Path, default_port: int = 8000) -> LlmEndpoint`; `client.LlmClient(endpoint, model="", timeout=180.0)` с методами `ping() -> list[str]` и `complete(system: str, user: str, images: list[Image.Image] | None = None, temperature: float = 0.3, max_tokens: int = 2048) -> str`; исключение `client.LlmError`.

- [ ] **Step 1: Написать падающий тест разбора адреса**

Создать `tests/test_llm_endpoint.py`:

```python
"""Разбор llm_endpoint.txt во всех вариантах свободного формата."""

import pytest

from fooocus_qwen.llm import endpoint as ep


def test_plain_host_gets_scheme_and_default_port():
    result = ep.parse_endpoint_file("llama.cpp\n192.0.2.10\ntoken=abc\n")
    assert result.backend == "llama.cpp"
    assert result.base_url == "http://192.0.2.10:8000"
    assert result.token == "abc"


def test_explicit_port_is_kept():
    result = ep.parse_endpoint_file("192.0.2.10:1234\n")
    assert result.base_url == "http://192.0.2.10:1234"


def test_explicit_scheme_is_kept_and_https_port_not_forced():
    result = ep.parse_endpoint_file("https://api.example.com\n")
    assert result.base_url == "https://api.example.com"


def test_keyword_form_and_arbitrary_line_order():
    text = "token: SECRET\n# комментарий\nurl = http://10.0.0.5:9000\nvllm\n"
    result = ep.parse_endpoint_file(text)
    assert result.base_url == "http://10.0.0.5:9000"
    assert result.token == "SECRET"
    assert result.backend == "vllm"


def test_api_key_alias_is_accepted():
    assert ep.parse_endpoint_file("host\napi_key=k1\n").token == "k1"


def test_missing_host_is_an_error():
    with pytest.raises(ValueError):
        ep.parse_endpoint_file("token=only\n")


def test_urls_are_built_without_double_slashes():
    result = ep.parse_endpoint_file("http://host:8000/\n")
    assert result.chat_url == "http://host:8000/v1/chat/completions"
    assert result.models_url == "http://host:8000/v1/models"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_llm_endpoint.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.llm'`

- [ ] **Step 3: Написать `fooocus_qwen/llm/endpoint.py`**

```python
"""Разбор файла с адресом внешней языковой модели.

Формат свободный и построчный: имя бэкенда, адрес хоста (возможно со схемой и
портом), строка вида ``token=...``. Порядок строк не важен — файл пишет человек,
а не программа, и требовать от него порядка значит собирать ошибки на ровном месте.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_LINE = re.compile(r"^(token|api_key|key)\s*[=:]\s*(\S+)$", re.IGNORECASE)
_URL_LINE = re.compile(r"^(?:url|endpoint|host)\s*[=:]\s*(\S+)$", re.IGNORECASE)
_HOST_LIKE = re.compile(r"^(https?://)?[\w.\-]+(:\d+)?(/.*)?$")


@dataclass(frozen=True)
class LlmEndpoint:
    """Реквизиты доступа к OpenAI-совместимому серверу."""

    base_url: str
    token: str | None = None
    backend: str = ""

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"


def _normalize(candidate: str, default_port: int) -> str:
    """Доводит строку до полного URL.

    Порт дописывается только к голому http-адресу: у https порт по умолчанию
    свой, и навязывать ему 8000 значит ломать обращение к облачным сервисам.
    """
    url = candidate.strip().rstrip("/")
    has_scheme = url.startswith(("http://", "https://"))
    if not has_scheme:
        url = f"http://{url}"

    without_scheme = url.split("://", 1)[1]
    host_part = without_scheme.split("/", 1)[0]
    if ":" not in host_part and url.startswith("http://"):
        url = url.replace(host_part, f"{host_part}:{default_port}", 1)
    return url


def parse_endpoint_file(text: str, default_port: int = 8000) -> LlmEndpoint:
    backend = ""
    token: str | None = None
    candidates: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        token_match = _TOKEN_LINE.match(line)
        if token_match:
            token = token_match.group(2)
            continue

        url_match = _URL_LINE.match(line)
        if url_match:
            candidates.append(url_match.group(1))
            continue

        if _HOST_LIKE.match(line) and "." in line.split("/")[0] or line.startswith(("http://", "https://")):
            candidates.append(line)
        else:
            # Строка не похожа на адрес — значит это имя бэкенда.
            backend = backend or line

    if not candidates:
        raise ValueError("В файле адреса языковой модели не найдено ни одного хоста")

    return LlmEndpoint(base_url=_normalize(candidates[0], default_port), token=token, backend=backend)


def load_endpoint(path: Path, default_port: int = 8000) -> LlmEndpoint:
    return parse_endpoint_file(path.read_text(encoding="utf-8"), default_port)
```

- [ ] **Step 4: Запустить тесты разбора**

Run: `.venv\Scripts\python -m pytest tests/test_llm_endpoint.py -q`
Expected: PASS, 7 тестов

Замечание для исполнителя: условие выбора между адресом и именем бэкенда в `parse_endpoint_file` написано так, что `llama.cpp` (в имени есть точка!) попадает в ветку адреса. Если тест `test_plain_host_gets_scheme_and_default_port` падает на `backend`, нужно уточнить правило: строка считается адресом, если содержит схему, двоеточие с цифрами, или состоит из четырёх числовых групп. Исправить и повторить шаг 4, не переходя дальше.

- [ ] **Step 5: Написать тест клиента на поддельном сервере**

Создать `tests/test_llm_client.py`:

```python
"""Клиент проверяется на настоящем HTTP-сервере в потоке, без заглушек сети."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from fooocus_qwen.llm.client import LlmClient, LlmError
from fooocus_qwen.llm.endpoint import LlmEndpoint

RECEIVED: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # тишина в выводе тестов
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply({"data": [{"id": "qwen3"}, {"id": "llama"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        RECEIVED.append({"body": body, "auth": self.headers.get("Authorization")})
        self._reply({"choices": [{"message": {"content": "переписанный промт"}}]})

    def _reply(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    RECEIVED.clear()
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_ping_returns_model_names(server):
    client = LlmClient(LlmEndpoint(base_url=server))
    assert client.ping() == ["qwen3", "llama"]


def test_complete_sends_system_and_user_messages(server):
    client = LlmClient(LlmEndpoint(base_url=server, token="SECRET"), model="qwen3")
    answer = client.complete("системный", "пользовательский")

    assert answer == "переписанный промт"
    body = RECEIVED[0]["body"]
    assert body["model"] == "qwen3"
    assert body["messages"][0] == {"role": "system", "content": "системный"}
    assert body["messages"][1]["content"] == "пользовательский"
    assert RECEIVED[0]["auth"] == "Bearer SECRET"


def test_images_are_sent_as_data_urls(server):
    client = LlmClient(LlmEndpoint(base_url=server))
    client.complete("س", "опиши", images=[Image.new("RGB", (8, 8), "red")])

    content = RECEIVED[0]["body"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_thinking_is_disabled():
    # Рассуждения вслух добавляют секунды и не нужны для переписывания промта.
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1"))
    body = client._build_body("s", "u", None, 0.3, 100)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_unreachable_server_raises_llm_error():
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1"), timeout=0.5)
    with pytest.raises(LlmError):
        client.ping()
```

- [ ] **Step 6: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_llm_client.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.llm.client'`

- [ ] **Step 7: Написать `fooocus_qwen/llm/client.py`**

```python
"""Клиент OpenAI-совместимого сервера языковой модели.

Только стандартная библиотека: единственное, что нужно оболочке от LLM, — один
запрос «перепиши промт», и тащить ради него клиентскую библиотеку с её версиями
зависимостей не за что.

Потоковая выдача не используется намеренно: ответ переписывателя — это JSON,
который всё равно разбирается целиком, и показывать его по кусочкам нечего.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from .endpoint import LlmEndpoint

LOGGER = logging.getLogger(__name__)


class LlmError(RuntimeError):
    """Сервер языковой модели недоступен или ответил ошибкой."""


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class LlmClient:
    """Один запрос — один ответ. Историю диалога оболочка не ведёт."""

    def __init__(self, endpoint: LlmEndpoint, model: str = "", timeout: float = 180.0) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self._endpoint.token:
            headers["Authorization"] = f"Bearer {self._endpoint.token}"
        return headers

    def ping(self) -> list[str]:
        """Возвращает список моделей на сервере. Пустой список — сервер жив, но пуст."""
        request = urllib.request.Request(self._endpoint.models_url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=min(self._timeout, 15)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise LlmError(f"Сервер языковой модели недоступен: {error}") from error
        return [item.get("id", "") for item in payload.get("data", [])]

    def _build_body(
        self,
        system: str,
        user: str,
        images: list[Image.Image] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if images:
            content: Any = [{"type": "text", "text": user}]
            content.extend({"type": "image_url", "image_url": {"url": _image_to_data_url(i)}} for i in images)
        else:
            content = user

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Режим размышления съедает секунды, а его текст в ответе не нужен.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def complete(
        self,
        system: str,
        user: str,
        images: list[Image.Image] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        body = self._build_body(system, user, images, temperature, max_tokens)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self._endpoint.chat_url, data=data, headers=self._headers())

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise LlmError(f"Ошибка обращения к языковой модели: {error}") from error

        choices = payload.get("choices") or []
        if not choices:
            raise LlmError("Языковая модель вернула пустой ответ")

        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            # reasoning_content игнорируем намеренно: это внутренние рассуждения.
            raise LlmError("Языковая модель вернула ответ без текста")
        return text
```

Создать `fooocus_qwen/llm/__init__.py`:

```python
"""Подключение к внешней языковой модели по OpenAI-совместимому протоколу."""

from .client import LlmClient, LlmError
from .endpoint import LlmEndpoint, load_endpoint, parse_endpoint_file

__all__ = ["LlmClient", "LlmError", "LlmEndpoint", "load_endpoint", "parse_endpoint_file"]
```

- [ ] **Step 8: Запустить тесты клиента**

Run: `.venv\Scripts\python -m pytest tests/test_llm_client.py tests/test_llm_endpoint.py -q`
Expected: PASS, 12 тестов

- [ ] **Step 9: Создать образец `llm_endpoint.txt`**

```
# Адрес внешней языковой модели для AI-буста промтов.
# Формат свободный: имя бэкенда, адрес, токен. Порядок строк не важен.
llama.cpp
192.0.2.10:8000
token=ЗАМЕНИТЕ_НА_СВОЙ_ТОКЕН
```

- [ ] **Step 10: Коммит**

```bash
git add fooocus_qwen/llm tests/test_llm_endpoint.py tests/test_llm_client.py
git commit -m "feat: клиент внешней языковой модели"
```

---

### Task 3: Стили Fooocus

**Files:**
- Create: `fooocus_qwen/prompting/__init__.py`, `fooocus_qwen/prompting/styles.py`
- Create: `resources/styles/*.json` (шесть файлов, скачиваются)
- Create: `tools/fetch_styles.py`
- Create: `tests/test_styles.py`

**Interfaces:**
- Consumes: `config.STYLES_DIR`.
- Produces: `styles.Style` (поля `name: str`, `prompt: str`, `negative_prompt: str`); `styles.load_styles(directory: Path) -> dict[str, Style]`; `styles.apply_styles(prompt: str, negative: str, names: Sequence[str], catalogue: dict[str, Style]) -> tuple[str, str]`.

- [ ] **Step 1: Написать загрузчик стилей `tools/fetch_styles.py`**

```python
"""Скачивает список стилей Fooocus в resources/styles.

Стили — это данные, а не код: копировать их в репозиторий руками значит
потерять связь с источником. Скрипт запускается один раз при развёртывании.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/sdxl_styles"
FILES = (
    "sdxl_styles_fooocus.json",
    "sdxl_styles_sai.json",
    "sdxl_styles_mre.json",
    "sdxl_styles_twri.json",
    "sdxl_styles_diva.json",
    "sdxl_styles_marc_k3nt3l.json",
)

TARGET = Path(__file__).resolve().parent.parent / "resources" / "styles"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in FILES:
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            raw = response.read().decode("utf-8")
        entries = json.loads(raw)
        (TARGET / name).write_text(raw, encoding="utf-8")
        total += len(entries)
        print(f"{name}: {len(entries)}")
    print(f"всего стилей: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Скачать стили**

Run: `.venv\Scripts\python tools/fetch_styles.py`
Expected: шесть строк и `всего стилей: 277`

- [ ] **Step 3: Написать падающий тест**

Создать `tests/test_styles.py`:

```python
"""Загрузка каталога стилей и подстановка промта."""

from fooocus_qwen import config
from fooocus_qwen.prompting import styles as st


def test_catalogue_contains_all_downloaded_styles():
    catalogue = st.load_styles(config.STYLES_DIR)
    assert len(catalogue) == 277
    assert "sai-anime" in catalogue
    assert "Fooocus Sharp" in catalogue


def test_single_style_substitutes_prompt():
    catalogue = {"s": st.Style(name="s", prompt="anime artwork {prompt} . vibrant", negative_prompt="photo")}
    prompt, negative = st.apply_styles("кот", "", ["s"], catalogue)
    assert prompt == "anime artwork кот . vibrant"
    assert negative == "photo"


def test_two_styles_are_joined_and_negatives_accumulate():
    catalogue = {
        "a": st.Style(name="a", prompt="A {prompt}", negative_prompt="na"),
        "b": st.Style(name="b", prompt="B {prompt}", negative_prompt="nb"),
    }
    prompt, negative = st.apply_styles("кот", "", ["a", "b"], catalogue)
    assert prompt == "A кот, B кот"
    assert negative == "na, nb"


def test_style_without_template_only_adds_negative():
    # Три стиля Fooocus содержат пустой prompt и существуют ради негатива.
    catalogue = {"n": st.Style(name="n", prompt="", negative_prompt="deformed")}
    prompt, negative = st.apply_styles("кот", "мутный", ["n"], catalogue)
    assert prompt == "кот"
    assert negative == "мутный, deformed"


def test_user_negative_comes_first():
    catalogue = {"s": st.Style(name="s", prompt="S {prompt}", negative_prompt="ns")}
    _, negative = st.apply_styles("кот", "свой негатив", ["s"], catalogue)
    assert negative == "свой негатив, ns"


def test_unknown_style_is_ignored_not_fatal():
    # Пресет промта мог быть сохранён на другой сборке с иным набором стилей.
    catalogue = {"s": st.Style(name="s", prompt="S {prompt}", negative_prompt="")}
    prompt, _ = st.apply_styles("кот", "", ["s", "нет такого"], catalogue)
    assert prompt == "S кот"


def test_empty_selection_returns_input_unchanged():
    assert st.apply_styles("кот", "мутный", [], {}) == ("кот", "мутный")
```

- [ ] **Step 4: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_styles.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.prompting'`

- [ ] **Step 5: Написать `fooocus_qwen/prompting/styles.py`**

```python
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
```

Создать `fooocus_qwen/prompting/__init__.py`:

```python
"""Работа с промтами: стили, библиотека пресетов, AI-буст."""
```

- [ ] **Step 6: Запустить тесты**

Run: `.venv\Scripts\python -m pytest tests/test_styles.py -q`
Expected: PASS, 7 тестов

- [ ] **Step 7: Коммит**

```bash
git add fooocus_qwen/prompting tools/fetch_styles.py resources/styles tests/test_styles.py
git commit -m "feat: каталог из 277 стилей Fooocus"
```

---

### Task 4: AI-буст — системные промты и разбор ответа

**Files:**
- Create: `fooocus_qwen/prompting/boost.py`
- Create: `tools/fetch_system_prompts.py`
- Create: `resources/prompts/system_prompt_describe.txt`
- Create: `tests/test_boost.py`

**Interfaces:**
- Consumes: `llm.LlmClient`, `llm.LlmError`, `config.SYSTEM_PROMPT_DIR`.
- Produces: `boost.BoostResult` (поля `prompt: str`, `wh_ratio: str | None`, `ratio_follow: str | None`, `raw: str`); `boost.parse_response(text: str) -> BoostResult`; `boost.build_user_message(prompt: str, reference_count: int) -> str`; `boost.boost(client, prompt, *, mode: str, references: list[Image.Image] | None = None, prompt_dir: Path) -> BoostResult`; `boost.describe(client, image, prompt_dir) -> str`.

- [ ] **Step 1: Написать загрузчик официальных системных промтов**

Создать `tools/fetch_system_prompts.py`:

```python
"""Скачивает официальные промты переписывания из репозитория QwenLM.

Эти файлы — часть поставки модели, а не наш текст. Пользователь волен их
править, поэтому уже существующие файлы не перезаписываются без --force.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/prompt_rewrite/prompts"
FILES = ("system_prompt_t2i.txt", "system_prompt_edit.txt")
TARGET = Path(__file__).resolve().parent.parent / "resources" / "prompts"


def main(argv: list[str]) -> int:
    force = "--force" in argv
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = TARGET / name
        if destination.exists() and not force:
            print(f"{name}: уже есть, пропускаю (--force чтобы перезаписать)")
            continue
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            text = response.read().decode("utf-8")
        destination.write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} символов")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 2: Скачать промты и написать свой промт описания**

Run: `.venv\Scripts\python tools/fetch_system_prompts.py`
Expected: два файла, примерно 10045 и 18344 символа

Создать `resources/prompts/system_prompt_describe.txt`:

```
You are looking at an image and writing the prompt that would produce it.

Describe the finished image as an observer reporting what is in the frame: subject,
pose and expression, clothing and materials, setting, lighting direction and quality,
colour palette, camera framing and lens character, and rendering medium (photograph,
illustration, 3D render, painting).

Copy any text visible in the image character for character, inside double quotes.

Write one paragraph of flowing English prose. No lists, no headings, no preamble, no
mention of the image itself or of the act of describing. Do not invent anything the
image does not show. Return the paragraph and nothing else.
```

- [ ] **Step 3: Написать падающий тест разбора ответа**

Создать `tests/test_boost.py`:

```python
"""Разбор ответа переписывателя промтов и сборка сообщения пользователя."""

from fooocus_qwen.prompting import boost


def test_plain_json_is_parsed():
    result = boost.parse_response('{"rewritten_prompt": "a cat", "wh_ratio": "3:2"}')
    assert result.prompt == "a cat"
    assert result.wh_ratio == "3:2"
    assert result.ratio_follow is None


def test_edit_response_carries_ratio_follow():
    text = '{"rewritten_prompt": "swap", "wh_ratio": "", "ratio_follow": "<image1>"}'
    result = boost.parse_response(text)
    assert result.ratio_follow == "<image1>"
    assert result.wh_ratio is None


def test_json_wrapped_in_markdown_fence_is_parsed():
    # Обычные модели любят оборачивать ответ в блок кода вопреки инструкции.
    text = 'Вот результат:\n```json\n{"rewritten_prompt": "a dog", "wh_ratio": "1:1"}\n```\n'
    result = boost.parse_response(text)
    assert result.prompt == "a dog"
    assert result.wh_ratio == "1:1"


def test_non_json_falls_back_to_raw_text():
    result = boost.parse_response("просто текст без json")
    assert result.prompt == "просто текст без json"
    assert result.wh_ratio is None


def test_empty_answer_yields_empty_prompt():
    result = boost.parse_response("   ")
    assert result.prompt == ""


def test_json_without_rewritten_prompt_falls_back_to_raw():
    result = boost.parse_response('{"wh_ratio": "3:2"}')
    assert result.prompt == '{"wh_ratio": "3:2"}'


def test_user_message_without_references_has_no_tags():
    message = boost.build_user_message("кот в шляпе", reference_count=1)
    assert "<image1>" not in message
    assert "кот в шляпе" in message


def test_user_message_lists_tags_for_several_references():
    message = boost.build_user_message("объедини их", reference_count=3)
    assert "<image1>" in message and "<image2>" in message and "<image3>" in message
    assert "<image4>" not in message
```

- [ ] **Step 4: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_boost.py -q`
Expected: FAIL, `ImportError: cannot import name 'boost'`

- [ ] **Step 5: Написать `fooocus_qwen/prompting/boost.py`**

```python
"""AI-буст: переписывание пользовательского промта внешней языковой моделью.

Системные промты взяты из официального репозитория Qwen и лежат файлами —
пользователь волен их править, поэтому они перечитываются при каждом запросе.

Официальные промты писались под дообученную модель Qwen-Image-2.1-PE-T2I и
требуют строгий JSON в ответе. Обычная модель на llama.cpp нередко оборачивает
его в блок кода или отвечает прозой. Терять из-за этого генерацию нельзя:
при неудаче разбора берём текст как есть.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..llm import LlmClient

LOGGER = logging.getLogger(__name__)

MODE_T2I = "t2i"
MODE_EDIT = "edit"

_PROMPT_FILES = {
    MODE_T2I: "system_prompt_t2i.txt",
    MODE_EDIT: "system_prompt_edit.txt",
}
_DESCRIBE_FILE = "system_prompt_describe.txt"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class BoostResult:
    """Результат переписывания.

    ``wh_ratio`` и ``ratio_follow`` взаимоисключающие: первый задаёт соотношение
    явно, второй велит наследовать его от указанного референса.
    """

    prompt: str
    wh_ratio: str | None = None
    ratio_follow: str | None = None
    raw: str = ""


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_response(text: str) -> BoostResult:
    raw = text.strip()
    if not raw:
        return BoostResult(prompt="", raw=text)

    match = _JSON_OBJECT.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            rewritten = _clean(payload.get("rewritten_prompt"))
            if rewritten:
                return BoostResult(
                    prompt=rewritten,
                    wh_ratio=_clean(payload.get("wh_ratio")),
                    ratio_follow=_clean(payload.get("ratio_follow")),
                    raw=text,
                )

    LOGGER.warning("Ответ переписывателя не содержит поля rewritten_prompt, беру текст как есть")
    return BoostResult(prompt=raw, raw=text)


def build_user_message(prompt: str, reference_count: int) -> str:
    """Собирает сообщение пользователя для переписывателя.

    При двух и более референсах спецификация Qwen требует адресовать их тегами
    ``<imageN>``; при единственном теги запрещены.
    """
    if reference_count < 2:
        return prompt
    tags = " ".join(f"<image{index}>" for index in range(1, reference_count + 1))
    return f"Input images: {tags}\n\nInstruction: {prompt}"


def _read_system_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"Не найден системный промт {path}: {error}") from error


def boost(
    client: LlmClient,
    prompt: str,
    *,
    mode: str,
    prompt_dir: Path,
    references: list[Image.Image] | None = None,
) -> BoostResult:
    system = _read_system_prompt(prompt_dir, _PROMPT_FILES[mode])
    user = build_user_message(prompt, len(references or []))
    # Референсы уходят в модель только в режиме редактирования: переписывателю
    # T2I смотреть не на что, а лишние изображения удлиняют запрос.
    images = references if mode == MODE_EDIT else None
    answer = client.complete(system, user, images=images)
    return parse_response(answer)


def describe(client: LlmClient, image: Image.Image, prompt_dir: Path) -> str:
    system = _read_system_prompt(prompt_dir, _DESCRIBE_FILE)
    return client.complete(system, "Describe this image.", images=[image]).strip()
```

- [ ] **Step 6: Запустить тесты**

Run: `.venv\Scripts\python -m pytest tests/test_boost.py -q`
Expected: PASS, 8 тестов

- [ ] **Step 7: Коммит**

```bash
git add fooocus_qwen/prompting/boost.py tools/fetch_system_prompts.py resources/prompts tests/test_boost.py
git commit -m "feat: AI-буст промта на официальных системных промтах Qwen"
```

---

### Task 5: Геометрия кадра

**Files:**
- Create: `fooocus_qwen/imaging/__init__.py`, `fooocus_qwen/imaging/aspect.py`
- Create: `tests/test_aspect.py`

**Interfaces:**
- Consumes: ничего.
- Produces: `aspect.ASPECT_RATIOS: tuple[str, ...]` (порядок для выпадающего списка); `aspect.CANONICAL: dict[str, tuple[int, int]]` (размеры из карточки модели при 2048); `aspect.dimensions(ratio: str, output_resolution: int) -> tuple[int, int]`; `aspect.snap(value: int) -> int`; `aspect.label(ratio: str, output_resolution: int) -> str`; константа `aspect.FOLLOW_REFERENCE: str`.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_aspect.py`:

```python
"""Соотношения сторон и приведение размеров к кратности 32."""

import pytest

from fooocus_qwen.imaging import aspect


def test_canonical_sizes_match_the_model_card():
    assert aspect.CANONICAL["1:1"] == (2048, 2048)
    assert aspect.CANONICAL["16:9"] == (2752, 1536)
    assert aspect.CANONICAL["3:2"] == (2528, 1696)
    assert aspect.CANONICAL["2:3"] == (1696, 2528)


def test_all_canonical_sizes_are_multiples_of_32():
    for width, height in aspect.CANONICAL.values():
        assert width % 32 == 0 and height % 32 == 0


def test_max_quality_reproduces_the_card_exactly():
    # При 2K берём размеры карточки как есть, без пересчёта.
    assert aspect.dimensions("16:9", 2048) == (2752, 1536)
    assert aspect.dimensions("1:1", 2048) == (2048, 2048)


def test_lower_resolution_scales_and_snaps():
    width, height = aspect.dimensions("1:1", 1024)
    assert (width, height) == (1024, 1024)

    width, height = aspect.dimensions("16:9", 1024)
    assert width % 32 == 0 and height % 32 == 0
    assert width > height
    # Площадь держится около квадрата стороны output_resolution.
    assert 0.85 <= (width * height) / (1024 * 1024) <= 1.15


def test_aspect_order_puts_square_first():
    assert aspect.ASPECT_RATIOS[0] == "1:1"
    assert aspect.FOLLOW_REFERENCE in aspect.ASPECT_RATIOS


def test_follow_reference_has_no_dimensions():
    assert aspect.dimensions(aspect.FOLLOW_REFERENCE, 1024) == (None, None)


def test_snap_rounds_to_nearest_multiple_of_32():
    assert aspect.snap(1000) == 992
    assert aspect.snap(1023) == 1024
    assert aspect.snap(16) == 32  # ноль недопустим


def test_unknown_ratio_is_an_error():
    with pytest.raises(KeyError):
        aspect.dimensions("7:5", 1024)


def test_label_shows_actual_pixels():
    assert aspect.label("1:1", 2048) == "1:1 — 2048×2048"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_aspect.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.imaging'`

- [ ] **Step 3: Написать `fooocus_qwen/imaging/aspect.py`**

```python
"""Соотношения сторон и приведение размеров кадра.

Канон — таблица из карточки модели. Внутренняя функция пайплайна
``calculate_dimensions`` считает стороны из площади и для 16:9 при
``output_resolution = 2048`` даёт 2720×1536, а не 2752×1536 из карточки.
Расхождение небольшое, но оно проявилось бы как необъяснимое несовпадение
размеров, поэтому размеры передаются в пайплайн явно, а не выводятся им.
"""

from __future__ import annotations

MULTIPLE = 32
REFERENCE_RESOLUTION = 2048

FOLLOW_REFERENCE = "от референса"

# Размеры из карточки модели Qwen-Image-2.1 для 2K.
CANONICAL: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

ASPECT_RATIOS: tuple[str, ...] = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", FOLLOW_REFERENCE)


def snap(value: int) -> int:
    """Приводит размер к ближайшей кратности 32, но не к нулю."""
    return max(MULTIPLE, round(value / MULTIPLE) * MULTIPLE)


def dimensions(ratio: str, output_resolution: int) -> tuple[int | None, int | None]:
    """Размеры кадра для соотношения и целевого разрешения.

    Для ``FOLLOW_REFERENCE`` возвращает пару ``None``: тогда пайплайн сам выведет
    размеры из соотношения сторон последнего условного изображения.
    """
    if ratio == FOLLOW_REFERENCE:
        return None, None

    width, height = CANONICAL[ratio]
    if output_resolution == REFERENCE_RESOLUTION:
        return width, height

    scale = output_resolution / REFERENCE_RESOLUTION
    return snap(int(width * scale)), snap(int(height * scale))


def label(ratio: str, output_resolution: int) -> str:
    """Подпись для выпадающего списка: соотношение и настоящие пиксели."""
    if ratio == FOLLOW_REFERENCE:
        return FOLLOW_REFERENCE
    width, height = dimensions(ratio, output_resolution)
    return f"{ratio} — {width}×{height}"
```

Создать `fooocus_qwen/imaging/__init__.py`:

```python
"""Операции над изображениями. Этот слой не знает ни о модели, ни о torch."""
```

- [ ] **Step 4: Запустить тесты**

Run: `.venv\Scripts\python -m pytest tests/test_aspect.py -q`
Expected: PASS, 9 тестов

- [ ] **Step 5: Коммит**

```bash
git add fooocus_qwen/imaging tests/test_aspect.py
git commit -m "feat: соотношения сторон по карточке модели"
```

---

### Task 6: Маски, склейка и outpaint

**Files:**
- Create: `fooocus_qwen/imaging/masking.py`, `fooocus_qwen/imaging/outpaint.py`
- Create: `tests/test_masking.py`, `tests/test_outpaint.py`

**Interfaces:**
- Consumes: `aspect.MULTIPLE`, `aspect.snap`.
- Produces:
  - `masking.mask_from_editor(value: Mapping[str, Any]) -> Image.Image` — режим `"L"`, размер фона.
  - `masking.is_empty(mask: Image.Image) -> bool`
  - `masking.refine(mask: Image.Image, grow: int = 0, feather: int = 0) -> Image.Image`
  - `masking.as_condition(mask: Image.Image) -> Image.Image` — RGB, белое = править.
  - `masking.blend(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image`
  - `masking.region_box(mask: Image.Image, padding: float = 0.25, multiple: int = 32) -> tuple[int, int, int, int] | None`
  - `masking.stitch(original: Image.Image, patch: Image.Image, box: tuple[int, int, int, int], mask: Image.Image) -> Image.Image`
  - `outpaint.OutpaintPlan` (поля `canvas_size: tuple[int, int]`, `paste_box: tuple[int, int, int, int]`)
  - `outpaint.plan(size: tuple[int, int], sides: Sequence[str], amount: float) -> OutpaintPlan`
  - `outpaint.expand(image: Image.Image, plan: OutpaintPlan) -> tuple[Image.Image, Image.Image]`
  - константа `outpaint.SIDES: tuple[str, ...] = ("left", "right", "top", "bottom")`

- [ ] **Step 1: Написать падающий тест масок**

Создать `tests/test_masking.py`:

```python
"""Построение маски, её уточнение и склейка результата с оригиналом.

Главное свойство, ради которого эти тесты существуют: пиксели вне маски после
склейки должны совпадать с исходными побайтово. Пользователь правит участок
портрета и вправе рассчитывать, что остальной кадр не «поплывёт».
"""

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.imaging import masking


def solid(size, colour, mode="RGBA"):
    return Image.new(mode, size, colour)


def editor_value(size, painted_box=None):
    """Повторяет структуру, которую отдаёт gr.ImageEditor."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted_box is not None:
        layer.paste((255, 0, 0, 255), painted_box)
    return {"background": solid(size, (10, 20, 30, 255)), "layers": [layer], "composite": None}


def test_mask_is_built_from_layer_alpha():
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 32, 32)))
    assert mask.mode == "L"
    assert mask.size == (64, 64)
    array = np.asarray(mask)
    assert array[20, 20] == 255
    assert array[5, 5] == 0


def test_several_layers_are_united():
    size = (32, 32)
    first = Image.new("RGBA", size, (0, 0, 0, 0))
    first.paste((255, 0, 0, 255), (0, 0, 8, 8))
    second = Image.new("RGBA", size, (0, 0, 0, 0))
    second.paste((0, 255, 0, 255), (16, 16, 24, 24))
    value = {"background": solid(size, (0, 0, 0, 255)), "layers": [first, second], "composite": None}

    array = np.asarray(masking.mask_from_editor(value))
    assert array[4, 4] == 255
    assert array[20, 20] == 255
    assert array[12, 12] == 0


def test_empty_mask_is_detected():
    assert masking.is_empty(masking.mask_from_editor(editor_value((32, 32))))
    assert not masking.is_empty(masking.mask_from_editor(editor_value((32, 32), (4, 4, 8, 8))))


def test_condition_mask_is_white_where_the_edit_goes():
    mask = masking.mask_from_editor(editor_value((32, 32), (8, 8, 16, 16)))
    condition = masking.as_condition(mask)
    assert condition.mode == "RGB"
    array = np.asarray(condition)
    assert tuple(array[10, 10]) == (255, 255, 255)
    assert tuple(array[2, 2]) == (0, 0, 0)


def test_grow_expands_the_mask():
    mask = masking.mask_from_editor(editor_value((64, 64), (28, 28, 36, 36)))
    grown = np.asarray(masking.refine(mask, grow=6, feather=0))
    assert grown[24, 32] == 255  # шесть пикселей выше исходной границы
    assert np.asarray(mask)[24, 32] == 0


def test_feather_softens_the_edge_without_touching_the_core():
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 48, 48)))
    soft = np.asarray(masking.refine(mask, grow=0, feather=5))
    assert soft[32, 32] == 255
    assert 0 < soft[16, 32] < 255


def test_blend_leaves_pixels_outside_the_mask_byte_identical():
    original = Image.new("RGBA", (64, 64), (17, 89, 200, 255))
    generated = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 32, 32)))

    result = masking.blend(original, generated, mask)

    before = np.asarray(original)
    after = np.asarray(result)
    outside = np.asarray(mask) == 0
    assert np.array_equal(after[outside], before[outside])
    assert tuple(after[20, 20]) == (255, 0, 0, 255)


def test_blend_keeps_far_pixels_exact_even_with_feather():
    original = Image.new("RGBA", (64, 64), (17, 89, 200, 255))
    generated = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = masking.refine(masking.mask_from_editor(editor_value((64, 64), (24, 24, 40, 40))), feather=4)

    after = np.asarray(masking.blend(original, generated, mask))
    assert tuple(after[2, 2]) == (17, 89, 200, 255)


def test_blend_resizes_generated_to_the_original():
    # Пайплайн округляет размеры до кратности 32 и может вернуть не тот размер.
    original = Image.new("RGBA", (100, 60), (1, 2, 3, 255))
    generated = Image.new("RGBA", (96, 32), (250, 250, 250, 255))
    mask = Image.new("L", (100, 60), 255)

    assert masking.blend(original, generated, mask).size == (100, 60)


def test_region_box_covers_the_mask_with_padding_and_snaps_to_32():
    mask = masking.mask_from_editor(editor_value((256, 256), (100, 100, 120, 120)))
    left, top, right, bottom = masking.region_box(mask, padding=0.5)

    assert left <= 100 and top <= 100 and right >= 120 and bottom >= 120
    assert (right - left) % 32 == 0 and (bottom - top) % 32 == 0
    assert 0 <= left and 0 <= top and right <= 256 and bottom <= 256


def test_region_box_of_empty_mask_is_none():
    assert masking.region_box(masking.mask_from_editor(editor_value((64, 64)))) is None


def test_region_box_never_leaves_the_canvas():
    mask = masking.mask_from_editor(editor_value((64, 64), (0, 0, 8, 8)))
    left, top, right, bottom = masking.region_box(mask, padding=2.0)
    assert (left, top) == (0, 0)
    assert right <= 64 and bottom <= 64


def test_stitch_puts_the_patch_back_and_keeps_the_rest():
    original = Image.new("RGBA", (128, 128), (10, 10, 10, 255))
    mask = masking.mask_from_editor(editor_value((128, 128), (32, 32, 64, 64)))
    box = masking.region_box(mask, padding=0.25)
    patch = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (200, 100, 50, 255))

    result = masking.stitch(original, patch, box, mask)

    array = np.asarray(result)
    assert result.size == (128, 128)
    assert tuple(array[40, 40]) == (200, 100, 50, 255)
    assert tuple(array[2, 2]) == (10, 10, 10, 255)


def test_mask_from_editor_accepts_plain_image():
    # Пользователь мог передать готовую маску файлом, а не нарисовать её.
    mask = masking.mask_from_editor(Image.new("L", (16, 16), 255))
    assert mask.size == (16, 16)
    assert not masking.is_empty(mask)


def test_mask_from_editor_rejects_value_without_background():
    with pytest.raises(ValueError):
        masking.mask_from_editor({"background": None, "layers": [], "composite": None})
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_masking.py -q`
Expected: FAIL, `ImportError: cannot import name 'masking'`

- [ ] **Step 3: Написать `fooocus_qwen/imaging/masking.py`**

```python
"""Маски: построение из редактора, уточнение края, склейка с оригиналом.

У Qwen-Image-2.1 нет параметра ``mask_image``: маска подаётся вторым условным
изображением. Модель понимает её семантически, поэтому машинерия inpaint из
Fooocus (заливка области, подмена латентов, патч модели) здесь не нужна — она
существовала ради SDXL, который маску не понимает.

Остаётся одна вещь, которую модель гарантировать не может: неприкосновенность
пикселей вне маски. Её обеспечивает ``blend``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .aspect import MULTIPLE

LOGGER = logging.getLogger(__name__)


def mask_from_editor(value: Mapping[str, Any] | Image.Image) -> Image.Image:
    """Собирает маску из значения ``gr.ImageEditor`` или из готового изображения.

    Редактор отдаёт словарь с фоном, слоями и сведённой картинкой. Маска — это
    объединение альфа-каналов слоёв: кисть рисует по прозрачному слою, и всё,
    что непрозрачно, пользователь пометил.
    """
    if isinstance(value, Image.Image):
        return value.convert("L")

    background = value.get("background")
    if background is None:
        raise ValueError("В значении редактора нет фонового изображения")

    size = background.size
    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in value.get("layers") or []:
        if layer is None:
            continue
        alpha = np.asarray(layer.convert("RGBA").resize(size, Image.NEAREST))[..., 3]
        combined = np.maximum(combined, alpha)

    return Image.fromarray(combined, mode="L")


def is_empty(mask: Image.Image) -> bool:
    return not np.asarray(mask).any()


def refine(mask: Image.Image, grow: int = 0, feather: int = 0) -> Image.Image:
    """Расширяет маску и смягчает её край.

    Запас нужен потому, что правка почти всегда должна захватить немного соседних
    пикселей: контур объекта редко совпадает с движением кисти. Растушёвка
    убирает границу склейки — резкий переход виден даже при идеальном совпадении
    содержимого.
    """
    array = np.asarray(mask.convert("L"))

    if grow > 0:
        kernel_size = 2 * grow + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        array = cv2.dilate(array, kernel)

    if feather > 0:
        # Ядро размытия обязано быть нечётным, иначе OpenCV откажется работать.
        kernel_size = 2 * feather + 1
        array = cv2.GaussianBlur(array, (kernel_size, kernel_size), 0)

    return Image.fromarray(array, mode="L")


def as_condition(mask: Image.Image) -> Image.Image:
    """Превращает маску в условное изображение: белое — править, чёрное — оставить."""
    array = np.asarray(mask.convert("L"))
    return Image.fromarray(np.repeat(array[..., None], 3, axis=2), mode="RGB")


def blend(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Накладывает результат на оригинал по маске.

    Там, где маска строго нулевая, берётся исходный пиксель без каких-либо
    вычислений — это и есть гарантия неприкосновенности кадра вне правки.
    """
    base = original.convert("RGBA")
    patch = generated.convert("RGBA")
    if patch.size != base.size:
        patch = patch.resize(base.size, Image.LANCZOS)

    soft = mask.convert("L")
    if soft.size != base.size:
        soft = soft.resize(base.size, Image.LANCZOS)

    base_array = np.asarray(base).astype(np.float32)
    patch_array = np.asarray(patch).astype(np.float32)
    weight = (np.asarray(soft).astype(np.float32) / 255.0)[..., None]

    mixed = np.rint(base_array * (1.0 - weight) + patch_array * weight)
    untouched = weight == 0.0
    result = np.where(untouched, base_array, mixed).astype(np.uint8)

    return Image.fromarray(result, mode="RGBA")


def region_box(
    mask: Image.Image,
    padding: float = 0.25,
    multiple: int = MULTIPLE,
) -> tuple[int, int, int, int] | None:
    """Прямоугольник вокруг маски с контекстным запасом, кратный ``multiple``.

    Запас даёт модели увидеть окружение: вырезанный впритык фрагмент лишён
    контекста, и модель дорисовывает в нём что угодно. Итог всегда лежит внутри
    холста — выход за границы породил бы кадр не того размера.
    """
    array = np.asarray(mask.convert("L"))
    rows = np.flatnonzero(array.any(axis=1))
    columns = np.flatnonzero(array.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None

    height, width = array.shape
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1

    pad_x = int((right - left) * padding)
    pad_y = int((bottom - top) * padding)
    left, right = max(0, left - pad_x), min(width, right + pad_x)
    top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)

    left, right = _snap_span(left, right, width, multiple)
    top, bottom = _snap_span(top, bottom, height, multiple)
    return left, top, right, bottom


def _snap_span(start: int, end: int, limit: int, multiple: int) -> tuple[int, int]:
    """Растягивает отрезок до кратной длины, не вылезая за ``limit``."""
    span = end - start
    target = min(((span + multiple - 1) // multiple) * multiple, (limit // multiple) * multiple)
    target = max(target, multiple)
    if target >= limit:
        return 0, limit

    start = max(0, start - (target - span) // 2)
    if start + target > limit:
        start = limit - target
    return start, start + target


def stitch(
    original: Image.Image,
    patch: Image.Image,
    box: tuple[int, int, int, int],
    mask: Image.Image,
) -> Image.Image:
    """Вклеивает обработанный фрагмент обратно, соблюдая маску."""
    left, top, right, bottom = box
    canvas = original.convert("RGBA").copy()

    resized = patch.convert("RGBA")
    if resized.size != (right - left, bottom - top):
        resized = resized.resize((right - left, bottom - top), Image.LANCZOS)

    full = canvas.copy()
    full.paste(resized, (left, top))
    return blend(canvas, full, mask)
```

- [ ] **Step 4: Запустить тесты масок**

Run: `.venv\Scripts\python -m pytest tests/test_masking.py -q`
Expected: PASS, 15 тестов

- [ ] **Step 5: Написать падающий тест outpaint**

Создать `tests/test_outpaint.py`:

```python
"""Расширение холста и маска для дорисовки."""

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.imaging import outpaint


def test_plan_grows_only_requested_sides():
    result = outpaint.plan((512, 512), ["right"], 0.5)
    width, height = result.canvas_size
    assert height == 512
    assert width > 512
    assert result.paste_box[0] == 0  # оригинал прижат влево


def test_plan_centres_the_original_when_both_sides_grow():
    result = outpaint.plan((512, 512), ["left", "right"], 0.25)
    left, top, right, bottom = result.paste_box
    width, _ = result.canvas_size
    assert left > 0
    assert abs(left - (width - right)) <= 32


def test_canvas_is_snapped_to_multiples_of_32():
    result = outpaint.plan((500, 300), ["top", "bottom", "left", "right"], 0.3)
    width, height = result.canvas_size
    assert width % 32 == 0 and height % 32 == 0


def test_no_sides_means_no_change():
    result = outpaint.plan((256, 256), [], 0.5)
    assert result.canvas_size == (256, 256)
    assert result.paste_box == (0, 0, 256, 256)


def test_expand_places_the_original_and_masks_only_new_area():
    image = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    result = outpaint.plan((64, 64), ["right"], 0.5)
    canvas, mask = outpaint.expand(image, result)

    assert canvas.size == result.canvas_size
    assert mask.size == result.canvas_size

    canvas_array = np.asarray(canvas)
    mask_array = np.asarray(mask)
    assert tuple(canvas_array[32, 10]) == (200, 30, 30, 255)
    assert mask_array[32, 10] == 0            # исходная область не правится
    assert mask_array[32, result.canvas_size[0] - 4] == 255  # новая область правится


def test_new_area_is_filled_by_edge_replication():
    # Пустой холст сбивает модель: край изображения должен продолжаться,
    # а не обрываться в чёрное.
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    image.paste((0, 200, 0, 255), (60, 0, 64, 64))
    canvas, _ = outpaint.expand(image, outpaint.plan((64, 64), ["right"], 0.5))

    array = np.asarray(canvas)
    assert tuple(array[32, 70]) == (0, 200, 0, 255)


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        outpaint.plan((64, 64), ["diagonal"], 0.5)
```

- [ ] **Step 6: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_outpaint.py -q`
Expected: FAIL, `ImportError: cannot import name 'outpaint'`

- [ ] **Step 7: Написать `fooocus_qwen/imaging/outpaint.py`**

```python
"""Расширение холста: подготовка условного изображения и маски.

Дорисовка за границами кадра — это тот же локальный правочный сценарий: новая
площадь объявляется маской, а исходное изображение остаётся нетронутым.

Новая площадь заполняется продолжением краевых пикселей. Пустой холст модель
трактует как часть композиции и дорисовывает границу изображения внутри кадра;
продолжение края такой подсказки не даёт.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .aspect import snap

SIDES: tuple[str, ...] = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class OutpaintPlan:
    """Куда вырастет холст и где на нём окажется оригинал."""

    canvas_size: tuple[int, int]
    paste_box: tuple[int, int, int, int]


def plan(size: tuple[int, int], sides: Sequence[str], amount: float) -> OutpaintPlan:
    unknown = [side for side in sides if side not in SIDES]
    if unknown:
        raise ValueError(f"Неизвестная сторона расширения: {', '.join(unknown)}")

    width, height = size
    if not sides or amount <= 0:
        return OutpaintPlan(canvas_size=(width, height), paste_box=(0, 0, width, height))

    left = int(width * amount) if "left" in sides else 0
    right = int(width * amount) if "right" in sides else 0
    top = int(height * amount) if "top" in sides else 0
    bottom = int(height * amount) if "bottom" in sides else 0

    canvas_width = snap(width + left + right)
    canvas_height = snap(height + top + bottom)

    # Округление холста до кратности 32 съедает или добавляет пиксели; отдаём
    # разницу тем полям, которые и так растут, чтобы оригинал не деформировался.
    offset_x = min(left, max(0, canvas_width - width))
    offset_y = min(top, max(0, canvas_height - height))

    return OutpaintPlan(
        canvas_size=(canvas_width, canvas_height),
        paste_box=(offset_x, offset_y, offset_x + width, offset_y + height),
    )


def expand(image: Image.Image, outpaint_plan: OutpaintPlan) -> tuple[Image.Image, Image.Image]:
    """Возвращает пару «условное изображение, маска новой площади»."""
    canvas_width, canvas_height = outpaint_plan.canvas_size
    left, top, right, bottom = outpaint_plan.paste_box

    source = np.asarray(image.convert("RGBA"))
    padded = cv2.copyMakeBorder(
        source,
        top=top,
        bottom=canvas_height - bottom,
        left=left,
        right=canvas_width - right,
        borderType=cv2.BORDER_REPLICATE,
    )
    canvas = Image.fromarray(padded, mode="RGBA")

    mask_array = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    mask_array[top:bottom, left:right] = 0
    return canvas, Image.fromarray(mask_array, mode="L")
```

- [ ] **Step 8: Запустить все тесты изображений**

Run: `.venv\Scripts\python -m pytest tests/test_masking.py tests/test_outpaint.py tests/test_aspect.py -q`
Expected: PASS, 31 тест

- [ ] **Step 9: Коммит**

```bash
git add fooocus_qwen/imaging tests/test_masking.py tests/test_outpaint.py
git commit -m "feat: маски, склейка с гарантией сохранности и расширение холста"
```

---

### Task 7: Метаданные PNG, библиотека промтов и галерея

**Files:**
- Create: `fooocus_qwen/imaging/metadata.py`
- Create: `fooocus_qwen/prompting/library.py`
- Create: `fooocus_qwen/storage/__init__.py`, `fooocus_qwen/storage/gallery.py`
- Create: `tests/test_metadata.py`, `tests/test_library.py`, `tests/test_gallery.py`

**Interfaces:**
- Consumes: `config.PROMPT_DIR`, `config.OUTPUT_DIR`.
- Produces:
  - `metadata.CHUNK_KEY: str = "fooocus_qwen"`, `metadata.LEGACY_KEY: str = "parameters"`
  - `metadata.save_png(image: Image.Image, path: Path, parameters: dict[str, Any]) -> Path`
  - `metadata.read_png(path: Path) -> dict[str, Any] | None`
  - `metadata.to_readable(parameters: dict[str, Any]) -> str`
  - `library.save_prompt(name: str, payload: dict[str, Any], directory: Path) -> Path`
  - `library.load_prompt(name: str, directory: Path) -> dict[str, Any]`
  - `library.list_prompts(directory: Path) -> list[str]`
  - `library.delete_prompt(name: str, directory: Path) -> bool`
  - `library.safe_filename(name: str) -> str`
  - `gallery.next_path(directory: Path, when: datetime | None = None) -> Path`
  - `gallery.recent(directory: Path, limit: int = 60) -> list[Path]`

- [ ] **Step 1: Написать падающий тест метаданных**

Создать `tests/test_metadata.py`:

```python
"""Запись параметров генерации в PNG и их чтение обратно."""

from PIL import Image

from fooocus_qwen.imaging import metadata

PARAMS = {
    "app_version": "0.1.0",
    "prompt": "кот в шляпе",
    "prompt_boosted": "a cat wearing a hat, studio light",
    "negative_prompt": "",
    "styles": ["sai-anime"],
    "seed": 12345,
    "steps": 28,
    "width": 1024,
    "height": 1024,
    "output_resolution": 1024,
    "true_cfg_scale": 1.0,
    "use_kv_cache": True,
    "mask_mode": "none",
    "references": 0,
    "seconds": 31.4,
}


def test_parameters_survive_a_round_trip(tmp_path):
    path = metadata.save_png(Image.new("RGB", (16, 16), "red"), tmp_path / "a.png", PARAMS)
    assert metadata.read_png(path) == PARAMS


def test_rgba_transparency_is_preserved(tmp_path):
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    path = metadata.save_png(image, tmp_path / "t.png", PARAMS)
    assert Image.open(path).mode == "RGBA"
    assert Image.open(path).getpixel((4, 4))[3] == 0


def test_human_readable_chunk_is_written_too(tmp_path):
    # Сторонние просмотрщики умеют читать только ключ parameters.
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "b.png", PARAMS)
    text = Image.open(path).text
    assert metadata.CHUNK_KEY in text
    assert metadata.LEGACY_KEY in text
    assert "кот в шляпе" in text[metadata.LEGACY_KEY]


def test_foreign_png_returns_none(tmp_path):
    path = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    assert metadata.read_png(path) is None


def test_broken_json_returns_none(tmp_path):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text(metadata.CHUNK_KEY, "{это не json")
    path = tmp_path / "broken.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    assert metadata.read_png(path) is None


def test_missing_file_returns_none(tmp_path):
    assert metadata.read_png(tmp_path / "нет.png") is None


def test_readable_form_lists_key_parameters():
    text = metadata.to_readable(PARAMS)
    assert "Seed: 12345" in text
    assert "Steps: 28" in text
    assert "Size: 1024x1024" in text


def test_non_ascii_keys_survive(tmp_path):
    params = dict(PARAMS, prompt="Тест «кавычки» и emoji 🐈")
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "u.png", params)
    assert metadata.read_png(path)["prompt"] == "Тест «кавычки» и emoji 🐈"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_metadata.py -q`
Expected: FAIL, `ImportError: cannot import name 'metadata'`

- [ ] **Step 3: Написать `fooocus_qwen/imaging/metadata.py`**

```python
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
    prompt = parameters.get("prompt_boosted") or parameters.get("prompt", "")
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
```

- [ ] **Step 4: Запустить тесты метаданных**

Run: `.venv\Scripts\python -m pytest tests/test_metadata.py -q`
Expected: PASS, 8 тестов

- [ ] **Step 5: Написать падающий тест библиотеки промтов и галереи**

Создать `tests/test_library.py`:

```python
"""Именованные пресеты промтов на диске."""

import pytest

from fooocus_qwen.prompting import library

PAYLOAD = {"prompt": "кот", "negative_prompt": "", "styles": ["sai-anime"], "seed": 7}


def test_saved_prompt_is_read_back(tmp_path):
    library.save_prompt("Мой кот", PAYLOAD, tmp_path)
    assert library.load_prompt("Мой кот", tmp_path) == PAYLOAD


def test_listing_is_sorted_and_shows_display_names(tmp_path):
    library.save_prompt("бета", PAYLOAD, tmp_path)
    library.save_prompt("альфа", PAYLOAD, tmp_path)
    assert library.list_prompts(tmp_path) == ["альфа", "бета"]


def test_saving_twice_overwrites(tmp_path):
    library.save_prompt("имя", PAYLOAD, tmp_path)
    library.save_prompt("имя", dict(PAYLOAD, seed=9), tmp_path)
    assert library.load_prompt("имя", tmp_path)["seed"] == 9
    assert library.list_prompts(tmp_path) == ["имя"]


def test_unsafe_characters_do_not_escape_the_directory(tmp_path):
    library.save_prompt("../../побег", PAYLOAD, tmp_path)
    assert list(tmp_path.glob("*.json"))
    assert not (tmp_path.parent.parent / "побег.json").exists()


def test_delete_removes_the_preset(tmp_path):
    library.save_prompt("имя", PAYLOAD, tmp_path)
    assert library.delete_prompt("имя", tmp_path) is True
    assert library.list_prompts(tmp_path) == []
    assert library.delete_prompt("имя", tmp_path) is False


def test_missing_preset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        library.load_prompt("нет такого", tmp_path)


def test_empty_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        library.save_prompt("   ", PAYLOAD, tmp_path)
```

Создать `tests/test_gallery.py`:

```python
"""Раскладка результатов по каталогам дат."""

from datetime import datetime

from PIL import Image

from fooocus_qwen.storage import gallery


def test_path_is_inside_a_dated_directory(tmp_path):
    path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 15, 4, 5))
    assert path.parent.name == "2026-09-21"
    assert path.suffix == ".png"


def test_names_do_not_collide(tmp_path):
    when = datetime(2026, 9, 21, 15, 4, 5)
    first = gallery.next_path(tmp_path, when=when)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"")
    second = gallery.next_path(tmp_path, when=when)
    assert first != second


def test_recent_returns_newest_first(tmp_path):
    for index, day in enumerate((19, 20, 21)):
        path = gallery.next_path(tmp_path, when=datetime(2026, 9, day, 12, 0, index))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)

    recent = gallery.recent(tmp_path)
    assert len(recent) == 3
    assert recent[0].parent.name == "2026-09-21"
    assert recent[-1].parent.name == "2026-09-19"


def test_recent_respects_the_limit(tmp_path):
    for second in range(5):
        path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 12, 0, second))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)
    assert len(gallery.recent(tmp_path, limit=2)) == 2


def test_recent_on_empty_directory_is_empty(tmp_path):
    assert gallery.recent(tmp_path) == []
```

- [ ] **Step 6: Убедиться, что тесты падают**

Run: `.venv\Scripts\python -m pytest tests/test_library.py tests/test_gallery.py -q`
Expected: FAIL, два `ImportError`

- [ ] **Step 7: Написать `fooocus_qwen/prompting/library.py`**

```python
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
    path = directory / f"{safe_filename(name)}.json"
    stored = dict(payload)
    stored[_NAME_KEY] = name.strip()
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_prompt(name: str, directory: Path) -> dict[str, Any]:
    path = directory / f"{safe_filename(name)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Пресет промта не найден: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(_NAME_KEY, None)
    return payload


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
        names.append(payload.get(_NAME_KEY, path.stem))
    return sorted(names)


def delete_prompt(name: str, directory: Path) -> bool:
    path = directory / f"{safe_filename(name)}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True
```

- [ ] **Step 8: Написать `fooocus_qwen/storage/gallery.py`**

```python
"""Раскладка результатов по каталогам дат.

Каталог на день — то же решение, что в Fooocus: за месяц работы в одной папке
накапливаются тысячи файлов, и любой файловый менеджер на них спотыкается.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def next_path(directory: Path, when: datetime | None = None) -> Path:
    """Свободное имя файла внутри каталога сегодняшней даты.

    Имя основано на времени с точностью до секунды; если за секунду сохраняется
    несколько изображений, добавляется порядковый номер.
    """
    moment = when or datetime.now()
    day_dir = directory / f"{moment:%Y-%m-%d}"
    stem = f"{moment:%H-%M-%S}"

    candidate = day_dir / f"{stem}.png"
    index = 1
    while candidate.exists():
        candidate = day_dir / f"{stem}_{index}.png"
        index += 1
    return candidate


def recent(directory: Path, limit: int = 60) -> list[Path]:
    """Последние изображения, новые первыми."""
    if not directory.is_dir():
        return []

    files: list[Path] = []
    for day_dir in sorted(directory.iterdir(), reverse=True):
        if not day_dir.is_dir():
            continue
        files.extend(sorted(day_dir.glob("*.png"), reverse=True))
        if len(files) >= limit:
            break
    return files[:limit]
```

Создать `fooocus_qwen/storage/__init__.py`:

```python
"""Хранение результатов и истории."""
```

- [ ] **Step 9: Запустить все тесты**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: PASS, 83 теста

- [ ] **Step 10: Коммит**

```bash
git add fooocus_qwen/imaging/metadata.py fooocus_qwen/prompting/library.py fooocus_qwen/storage tests
git commit -m "feat: метаданные PNG, библиотека промтов и раскладка галереи"
```

---

### Task 8: Движок — пресеты качества и политика резидентности

**Files:**
- Create: `fooocus_qwen/engine/__init__.py`, `fooocus_qwen/engine/presets.py`, `fooocus_qwen/engine/residency.py`
- Create: `tests/test_presets.py`, `tests/test_residency.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач.
- Produces:
  - `presets.QualityPreset` (поля `name: str`, `output_resolution: int`, `num_inference_steps: int`)
  - `presets.PRESETS: dict[str, QualityPreset]`, `presets.NAMES: tuple[str, ...]`, `presets.DEFAULT: str`
  - `presets.get(name: str) -> QualityPreset`
  - `residency.StagedModule(module, device, pin_memory=True)` с `to_device()`, `to_host()`, свойствами `resident: bool`, `nbytes: int`
  - `residency.ResidencyManager(pipe, device="cuda", pin_memory=True)` с `start()`, контекстным менеджером `text_encoder_resident()`, `stats() -> dict[str, float]`

- [ ] **Step 1: Написать тест пресетов**

Создать `tests/test_presets.py`:

```python
"""Пресеты качества."""

import pytest

from fooocus_qwen.engine import presets


def test_three_presets_exist_in_ascending_order():
    assert presets.NAMES == ("LowQuality", "MiddleQuality", "MaxQuality")
    resolutions = [presets.get(name).output_resolution for name in presets.NAMES]
    steps = [presets.get(name).num_inference_steps for name in presets.NAMES]
    assert resolutions == sorted(resolutions)
    assert steps == sorted(steps)


def test_max_quality_matches_the_model_card_defaults():
    preset = presets.get("MaxQuality")
    assert preset.output_resolution == 2048
    assert preset.num_inference_steps == 40


def test_resolutions_are_multiples_of_32():
    for name in presets.NAMES:
        assert presets.get(name).output_resolution % 32 == 0


def test_default_is_the_middle_one():
    assert presets.DEFAULT == "MiddleQuality"


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        presets.get("Ultra")
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_presets.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.engine'`

- [ ] **Step 3: Написать `fooocus_qwen/engine/presets.py`**

```python
"""Пресеты качества: разрешение и число шагов.

Значения рассчитаны из пропускной способности 3090 и уточняются
``tools/benchmark.py`` — таблица в спецификации ведётся по измерениям, а не по
оценкам. Число шагов — самый честный рычаг ускорения: оно линейно управляет
временем и не трогает веса.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityPreset:
    """Один набор параметров качества."""

    name: str
    output_resolution: int
    num_inference_steps: int


PRESETS: dict[str, QualityPreset] = {
    "LowQuality": QualityPreset("LowQuality", output_resolution=1024, num_inference_steps=16),
    "MiddleQuality": QualityPreset("MiddleQuality", output_resolution=1536, num_inference_steps=28),
    # Значения из карточки модели: 2K и сорок шагов.
    "MaxQuality": QualityPreset("MaxQuality", output_resolution=2048, num_inference_steps=40),
}

NAMES: tuple[str, ...] = ("LowQuality", "MiddleQuality", "MaxQuality")
DEFAULT = "MiddleQuality"


def get(name: str) -> QualityPreset:
    return PRESETS[name]
```

Создать `fooocus_qwen/engine/__init__.py`:

```python
"""Работа с моделью: загрузка, размещение в памяти, генерация."""
```

- [ ] **Step 4: Запустить тест пресетов**

Run: `.venv\Scripts\python -m pytest tests/test_presets.py -q`
Expected: PASS, 5 тестов

- [ ] **Step 5: Написать тест резидентности**

Создать `tests/test_residency.py`:

```python
"""Перемещение весов между хостом и видеопамятью.

Тесты идут на процессоре, если CUDA недоступна: проверяется логика владения
весами, а она от устройства не зависит.
"""

import pytest
import torch

from fooocus_qwen.engine.residency import StagedModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN = torch.cuda.is_available()


def tiny_module():
    module = torch.nn.Linear(8, 4)
    module.register_buffer("scale", torch.ones(4))
    return module


def test_module_starts_on_the_host():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    assert staged.resident is False
    assert staged.module.weight.device.type == "cpu"


def test_to_device_moves_parameters_and_buffers():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    staged.to_device()

    assert staged.resident is True
    assert staged.module.weight.device.type == DEVICE
    assert staged.module.scale.device.type == DEVICE


def test_values_survive_a_round_trip():
    module = tiny_module()
    expected = module.weight.detach().clone()

    staged = StagedModule(module, DEVICE, pin_memory=PIN)
    staged.to_device()
    staged.to_host()

    assert torch.equal(staged.module.weight.detach(), expected)
    assert staged.module.weight.device.type == "cpu"


def test_host_copy_is_canonical_and_reused():
    # Возврат на хост не копирует веса обратно, а возвращает ссылку на исходную
    # копию: иначе закрепление памяти терялось бы после первой же перестановки.
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    host_tensor = staged.module.weight.data
    staged.to_device()
    staged.to_host()
    assert staged.module.weight.data is host_tensor


def test_repeated_calls_are_idempotent():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    staged.to_device()
    staged.to_device()
    staged.to_host()
    staged.to_host()
    assert staged.resident is False


def test_nbytes_counts_parameters_and_buffers():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    # 8*4 весов + 4 смещения + 4 буфера = 40 значений по 4 байта.
    assert staged.nbytes == 40 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
def test_host_copy_is_pinned_when_asked():
    staged = StagedModule(tiny_module(), "cuda", pin_memory=True)
    assert staged.module.weight.data.is_pinned()
```

- [ ] **Step 6: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_residency.py -q`
Expected: FAIL, `ImportError: cannot import name 'residency'`

- [ ] **Step 7: Написать `fooocus_qwen/engine/residency.py`**

```python
"""Размещение весов между хостом и видеопамятью.

Задача: 33 ГБ весов против 24 ГБ видеопамяти. Штатный
``enable_model_cpu_offload`` гоняет по шине всё и на каждую генерацию; при
быстром пресете это треть времени.

Принятая политика: трансформер и VAE резидентны, текстовый энкодер живёт на
хосте и поднимается только при промахе кэша эмбеддингов. Тогда перебор сида,
шагов и разрешения не создаёт трафика по шине вовсе.

Канонической копией весов считается копия на хосте, а не в модуле. Благодаря
этому закрепление памяти делается один раз: возврат «на хост» — это возврат
ссылки на уже закреплённый тензор, а не новое копирование.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

LOGGER = logging.getLogger(__name__)


def _named_tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Параметры и буферы одним потоком.

    Буферы нельзя пропускать: у трансформера в них лежат таблицы поворотных
    вложений, и модуль без них на видеокарте не считается.
    """
    for name, parameter in module.named_parameters(recurse=True):
        yield f"p:{name}", parameter
    for name, buffer in module.named_buffers(recurse=True):
        yield f"b:{name}", buffer


class StagedModule:
    """Модуль, чьи веса хранятся на хосте и по требованию поднимаются на устройство."""

    def __init__(self, module: torch.nn.Module, device: str | torch.device, pin_memory: bool = True) -> None:
        self.module = module
        self._device = torch.device(device)
        self._resident = False
        self._host: dict[str, torch.Tensor] = {}
        self._nbytes = 0

        pin_failed = False
        for name, tensor in _named_tensors(module):
            host = tensor.detach().to("cpu")
            if pin_memory and not pin_failed:
                try:
                    host = host.pin_memory()
                except RuntimeError as error:
                    # Закрепить десятки гигабайт удаётся не всегда; работать без
                    # закрепления медленнее, но полностью корректно.
                    LOGGER.warning("Не удалось закрепить память, продолжаю без неё: %s", error)
                    pin_failed = True
            self._host[name] = host
            self._nbytes += host.numel() * host.element_size()
            tensor.data = host

    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def to_device(self) -> None:
        if self._resident:
            return
        for name, tensor in _named_tensors(self.module):
            tensor.data = self._host[name].to(self._device, non_blocking=True)
        if self._device.type == "cuda":
            # Копирование из закреплённой памяти асинхронное: без синхронизации
            # первый же вызов модуля прочитал бы наполовину заполненные веса.
            torch.cuda.synchronize(self._device)
        self._resident = True

    def to_host(self) -> None:
        if not self._resident:
            return
        for name, tensor in _named_tensors(self.module):
            tensor.data = self._host[name]
        self._resident = False
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


class ResidencyManager:
    """Владеет размещением трёх моделей пайплайна."""

    def __init__(self, pipe, device: str | torch.device = "cuda", pin_memory: bool = True) -> None:
        self._pipe = pipe
        self._device = torch.device(device)
        self._pin_memory = pin_memory
        self._transformer: StagedModule | None = None
        self._text_encoder: StagedModule | None = None
        self._swaps = 0

    def start(self) -> None:
        """Раскладывает модели по местам. Вызывается один раз после загрузки."""
        LOGGER.info("Готовлю копии весов на хосте (закрепление: %s)", "да" if self._pin_memory else "нет")
        self._transformer = StagedModule(self._pipe.transformer, self._device, self._pin_memory)
        self._text_encoder = StagedModule(self._pipe.text_encoder, self._device, self._pin_memory)

        # VAE не переставляется никогда, поэтому копия на хосте ему не нужна:
        # это сэкономленные 1.35 ГБ закреплённой памяти.
        self._pipe.vae.to(self._device)
        self._transformer.to_device()

        LOGGER.info(
            "Резидентно: трансформер %.1f ГБ, VAE на устройстве; на хосте: энкодер %.1f ГБ",
            self._transformer.nbytes / 2**30,
            self._text_encoder.nbytes / 2**30,
        )

    @contextmanager
    def text_encoder_resident(self) -> Iterator[None]:
        """Поднимает энкодер, вытеснив трансформер, и возвращает всё обратно.

        Вместе они не помещаются: 17.5 плюс 14.2 гигабайта против 24 доступных.
        """
        if self._text_encoder is None or self._transformer is None:
            raise RuntimeError("ResidencyManager.start() не вызывался")

        self._swaps += 1
        self._transformer.to_host()
        self._text_encoder.to_device()
        try:
            yield
        finally:
            self._text_encoder.to_host()
            self._transformer.to_device()

    def stats(self) -> dict[str, float]:
        allocated = torch.cuda.memory_allocated(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        reserved = torch.cuda.memory_reserved(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        return {"allocated_gib": allocated, "reserved_gib": reserved, "swaps": float(self._swaps)}
```

- [ ] **Step 8: Запустить тест резидентности**

Run: `.venv\Scripts\python -m pytest tests/test_residency.py -q`
Expected: PASS, 7 тестов (последний пропускается без CUDA)

- [ ] **Step 9: Коммит**

```bash
git add fooocus_qwen/engine tests/test_presets.py tests/test_residency.py
git commit -m "feat: пресеты качества и политика резидентности весов"
```

---

### Task 9: Кэш эмбеддингов и подкласс пайплайна

**Files:**
- Create: `fooocus_qwen/engine/embeds_cache.py`, `fooocus_qwen/engine/pipeline.py`, `fooocus_qwen/engine/loader.py`
- Create: `tests/test_embeds_cache.py`, `tests/test_pipeline_contract.py`

**Interfaces:**
- Consumes: `residency.ResidencyManager`, `config.MODEL_DIR`.
- Produces:
  - `embeds_cache.fingerprint(image: Image.Image) -> str`
  - `embeds_cache.CacheKey` (поля `prompts: tuple[str, ...]`, `images: tuple[str, ...]`)
  - `embeds_cache.EmbedsCache(capacity: int = 4)` с `key(prompt, image) -> CacheKey`, `get(key) -> tuple | None`, `put(key, value) -> None`, `clear()`, свойствами `hits: int`, `misses: int`, `size: int`
  - `pipeline.QwenImage21StudioPipeline` с методом `attach(residency, cache, device)`
  - `pipeline.assert_contract() -> None`
  - `loader.load(model_dir: Path, device: str = "cuda", pin_memory: bool = True, cache_capacity: int = 4) -> tuple[QwenImage21StudioPipeline, ResidencyManager, EmbedsCache]`

- [ ] **Step 1: Написать тест кэша**

Создать `tests/test_embeds_cache.py`:

```python
"""Кэш эмбеддингов промта."""

import torch
from PIL import Image

from fooocus_qwen.engine.embeds_cache import EmbedsCache, fingerprint


def sample():
    return (torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool))


def test_identical_images_have_identical_fingerprints():
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "red")
    assert fingerprint(first) == fingerprint(second)


def test_different_pixels_change_the_fingerprint():
    assert fingerprint(Image.new("RGB", (8, 8), "red")) != fingerprint(Image.new("RGB", (8, 8), "blue"))


def test_different_size_changes_the_fingerprint():
    assert fingerprint(Image.new("RGB", (8, 8), "red")) != fingerprint(Image.new("RGB", (16, 16), "red"))


def test_miss_then_hit():
    cache = EmbedsCache()
    key = cache.key("кот", None)

    assert cache.get(key) is None
    assert cache.misses == 1

    cache.put(key, sample())
    assert cache.get(key) is not None
    assert cache.hits == 1


def test_key_depends_on_prompt_and_images():
    cache = EmbedsCache()
    image = Image.new("RGB", (8, 8), "red")
    assert cache.key("кот", None) != cache.key("пёс", None)
    assert cache.key("кот", None) != cache.key("кот", [image])


def test_prompt_may_be_a_string_or_a_list():
    cache = EmbedsCache()
    assert cache.key("кот", None) == cache.key(["кот"], None)


def test_capacity_evicts_the_least_recently_used():
    cache = EmbedsCache(capacity=2)
    first, second, third = (cache.key(text, None) for text in ("a", "b", "c"))

    cache.put(first, sample())
    cache.put(second, sample())
    cache.get(first)          # first становится свежим, вытеснить должно second
    cache.put(third, sample())

    assert cache.get(first) is not None
    assert cache.get(second) is None
    assert cache.size == 2


def test_stored_tensors_live_on_the_host():
    # Кэш не должен занимать видеопамять: с десятью референсами запись весит
    # десятки мегабайт, а записей несколько.
    cache = EmbedsCache()
    key = cache.key("кот", None)
    cache.put(key, sample())
    for tensor in cache.get(key):
        assert tensor.device.type == "cpu"


def test_clear_empties_the_cache_and_counters():
    cache = EmbedsCache()
    key = cache.key("кот", None)
    cache.put(key, sample())
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0 and cache.misses == 0
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_embeds_cache.py -q`
Expected: FAIL, `ImportError: cannot import name 'embeds_cache'`

- [ ] **Step 3: Написать `fooocus_qwen/engine/embeds_cache.py`**

```python
"""Кэш эмбеддингов промта.

Кодирование промта — единственная операция, ради которой приходится вытеснять
трансформер из видеопамяти. Пока промт и набор условных изображений не менялись,
результат кодирования не меняется тоже, и перестановку можно не делать: перебор
сида, шагов, разрешения и числа изображений становится бесплатным.

Записи хранятся на хосте: с десятью референсами последовательность разрастается,
и держать несколько таких записей в видеопамяти значит отнимать её у генерации.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

Embeds = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def fingerprint(image: Image.Image) -> str:
    """Отпечаток изображения по его пикселям, размеру и режиму.

    Сравнение по пикселям, а не по идентичности объекта: пользователь может
    подать то же изображение повторно другим объектом, и пересчитывать ради
    этого семнадцать гигабайт весов незачем.
    """
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """Ключ: тексты промтов и отпечатки условных изображений."""

    prompts: tuple[str, ...]
    images: tuple[str, ...]


class EmbedsCache:
    """Кэш с вытеснением давно не использованных записей."""

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[CacheKey, Embeds] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        return len(self._entries)

    def key(self, prompt: str | Sequence[str] | None, image: Sequence[Image.Image] | None) -> CacheKey:
        prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt or ())
        images = tuple(fingerprint(item) for item in (image or []))
        return CacheKey(prompts=tuple(prompts), images=images)

    def get(self, key: CacheKey) -> Embeds | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, key: CacheKey, value: Embeds) -> None:
        self._entries[key] = tuple(tensor.detach().to("cpu") for tensor in value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            evicted, _ = self._entries.popitem(last=False)
            LOGGER.debug("Из кэша эмбеддингов вытеснена запись %s", evicted.prompts[:1])

    def clear(self) -> None:
        self._entries.clear()
        self._hits = 0
        self._misses = 0
```

- [ ] **Step 4: Запустить тест кэша**

Run: `.venv\Scripts\python -m pytest tests/test_embeds_cache.py -q`
Expected: PASS, 9 тестов

- [ ] **Step 5: Написать тест контракта пайплайна**

Создать `tests/test_pipeline_contract.py`:

```python
"""Страж совместимости с diffusers.

Кэш эмбеддингов встроен переопределением защищённого метода. Если апстрим
изменит его сигнатуру, оболочка должна сказать об этом внятно при запуске, а не
падать посреди генерации с непонятной ошибкой.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")


def test_parent_method_signature_is_what_we_expect():
    from diffusers import QwenImage21Pipeline

    parameters = list(inspect.signature(QwenImage21Pipeline._get_qwen_prompt_embeds).parameters)
    assert parameters == ["self", "prompt", "image", "device"]


def test_parent_returns_three_values_per_its_own_annotation():
    from diffusers import QwenImage21Pipeline

    source = inspect.getsource(QwenImage21Pipeline._get_qwen_prompt_embeds)
    assert "return prompt_embeds, encoder_attention_mask, image_pad_mask" in source


def test_call_does_not_forward_image_pad_mask():
    # Ровно это ограничение и вынуждает кэшировать на уровне
    # _get_qwen_prompt_embeds, а не передавать готовые prompt_embeds в __call__.
    from diffusers import QwenImage21Pipeline

    source = inspect.getsource(QwenImage21Pipeline.__call__)
    assert "self.encode_prompt(" in source
    assert "image_pad_mask=image_pad_mask" not in source


def test_contract_guard_passes_on_the_installed_version():
    from fooocus_qwen.engine.pipeline import assert_contract

    assert_contract()
```

- [ ] **Step 6: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_pipeline_contract.py -q`
Expected: FAIL на последнем тесте, `ImportError: cannot import name 'pipeline'`

Первые три теста должны пройти сразу. Если не проходят — значит зафиксированный коммит diffusers не тот, что изучался при проектировании: остановиться и сообщить, а не подгонять тесты под новый код.

- [ ] **Step 7: Написать `fooocus_qwen/engine/pipeline.py`**

```python
"""Подкласс официального пайплайна с кэшем эмбеддингов.

Переопределён ровно один защищённый метод. Переписывать ``__call__`` было бы
заманчиво — там пришлось бы прокинуть ``image_pad_mask``, — но это означало бы
скопировать к себе двести строк цикла денойзинга и получить расхождение с
апстримом при первом же его обновлении.

``_get_qwen_prompt_embeds`` возвращает результат кодирования до размножения на
число изображений и до обнуления маски, то есть ровно то, что стоит кэшировать.
Перестановка моделей происходит вокруг настоящего вызова и при попадании в кэш
не выполняется вовсе.
"""

from __future__ import annotations

import inspect
import logging

import torch
from diffusers import QwenImage21Pipeline

from .embeds_cache import EmbedsCache
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)

_EXPECTED_PARAMETERS = ["self", "prompt", "image", "device"]


def assert_contract() -> None:
    """Проверяет, что апстрим не изменил точку встраивания."""
    actual = list(inspect.signature(QwenImage21Pipeline._get_qwen_prompt_embeds).parameters)
    if actual != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Изменилась сигнатура QwenImage21Pipeline._get_qwen_prompt_embeds: "
            f"ожидалось {_EXPECTED_PARAMETERS}, получено {actual}. "
            "Кэш эмбеддингов встроен в этот метод — обновите fooocus_qwen/engine/pipeline.py "
            "или зафиксируйте прежний коммит diffusers в requirements.txt."
        )


class QwenImage21StudioPipeline(QwenImage21Pipeline):
    """Пайплайн Qwen-Image-2.1 с кэшем эмбеддингов и ручным размещением весов."""

    def attach(self, residency: ResidencyManager, cache: EmbedsCache, device: torch.device) -> None:
        self._studio_residency = residency
        self._studio_cache = cache
        self._studio_device = device

    @property
    def _execution_device(self) -> torch.device:
        """Устройство вычислений задаётся явно.

        Штатная реализация выводит его из размещения модулей, а у нас модули
        сознательно живут на разных устройствах — она вернула бы процессор.
        """
        override = getattr(self, "_studio_device", None)
        if override is not None:
            return override
        return QwenImage21Pipeline._execution_device.fget(self)

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        cache: EmbedsCache | None = getattr(self, "_studio_cache", None)
        residency: ResidencyManager | None = getattr(self, "_studio_residency", None)
        target = device or self._execution_device

        if cache is None or residency is None:
            return super()._get_qwen_prompt_embeds(prompt, image, target)

        key = cache.key(prompt, image)
        cached = cache.get(key)
        if cached is not None:
            LOGGER.debug("Эмбеддинги промта взяты из кэша, энкодер не поднимался")
            return tuple(tensor.to(target) for tensor in cached)

        with residency.text_encoder_resident():
            embeds = super()._get_qwen_prompt_embeds(prompt, image, target)

        cache.put(key, embeds)
        return embeds
```

- [ ] **Step 8: Написать `fooocus_qwen/engine/loader.py`**

```python
"""Сборка пайплайна: загрузка весов, размещение, вспомогательные режимы."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from .embeds_cache import EmbedsCache
from .pipeline import QwenImage21StudioPipeline, assert_contract
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)


def load(
    model_dir: Path,
    device: str = "cuda",
    pin_memory: bool = True,
    cache_capacity: int = 4,
) -> tuple[QwenImage21StudioPipeline, ResidencyManager, EmbedsCache]:
    """Загружает модель и раскладывает её по памяти.

    Веса читаются на хост: размещением дальше управляет ResidencyManager, и
    позволить diffusers самому что-то перенести значило бы получить два хозяина
    у одной видеопамяти.
    """
    assert_contract()

    started = time.perf_counter()
    LOGGER.info("Загружаю модель из %s", model_dir)
    pipe = QwenImage21StudioPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16)

    # На 2K скрытое представление декодируется целиком и занимает заметно больше
    # памяти, чем сами веса VAE; тайлинг снимает этот пик.
    pipe.vae.enable_tiling()

    residency = ResidencyManager(pipe, device=device, pin_memory=pin_memory)
    residency.start()

    cache = EmbedsCache(capacity=cache_capacity)
    pipe.attach(residency, cache, torch.device(device))

    LOGGER.info("Модель готова за %.1f с", time.perf_counter() - started)
    return pipe, residency, cache
```

- [ ] **Step 9: Запустить тесты контракта**

Run: `.venv\Scripts\python -m pytest tests/test_pipeline_contract.py tests/test_embeds_cache.py -q`
Expected: PASS, 13 тестов

- [ ] **Step 10: Проверить загрузку модели вживую**

Run: `.venv\Scripts\python -c "from pathlib import Path; from fooocus_qwen import config, logging_setup; logging_setup.setup_logging(True); from fooocus_qwen.engine import loader; p, r, c = loader.load(config.MODEL_DIR); print(r.stats())"`

Expected: модель грузится, в журнале видны размеры резидентных частей, `allocated_gib` около 15.6.

Если видеопамяти не хватает уже на этом шаге — остановиться и сообщить измеренные числа, не переходя к следующей задаче.

- [ ] **Step 11: Коммит**

```bash
git add fooocus_qwen/engine tests/test_embeds_cache.py tests/test_pipeline_contract.py
git commit -m "feat: кэш эмбеддингов и подкласс пайплайна с ручным размещением весов"
```

---

### Task 10: Генератор и запуск из командной строки

**Files:**
- Create: `fooocus_qwen/engine/generator.py`
- Modify: `fooocus_qwen/__main__.py` (добавить безоконный режим генерации)
- Create: `tests/test_generator.py`

**Interfaces:**
- Consumes: `presets.QualityPreset`, `styles.apply_styles`, `aspect.dimensions`, `aspect.FOLLOW_REFERENCE`, `masking.*`, `metadata.save_png`, `gallery.next_path`, `loader.load`.
- Produces:
  - константы `generator.MASK_NONE = "none"`, `MASK_MASK = "mask"`, `MASK_ANNOTATION = "annotation"`, `MASK_REGION = "region"`, `generator.MASK_MODES: tuple[str, ...]`
  - `generator.ConditionSlot` (поля `tag: str`, `role: str`, `image: Image.Image`)
  - `generator.GenerationRequest` — датакласс, поля перечислены в шаге 3
  - `generator.GeneratedImage` (поля `image: Image.Image`, `seed: int`, `parameters: dict[str, Any]`)
  - `generator.build_conditions(request: GenerationRequest) -> list[ConditionSlot]`
  - `generator.resolve_size(request: GenerationRequest) -> tuple[int | None, int | None]`
  - `generator.Generator(pipe, residency, cache, catalogue)` с `generate(request, progress=None) -> list[GeneratedImage]` и `interrupt() -> None`

- [ ] **Step 1: Написать падающий тест сборки условных изображений**

Создать `tests/test_generator.py`:

```python
"""Сборка запроса на генерацию: условные изображения, теги, размеры.

Тесты не трогают видеокарту: проверяется подготовка запроса, а она к модели
отношения не имеет.
"""

from PIL import Image

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.engine import presets
from fooocus_qwen.imaging import aspect


def request(**overrides):
    base = dict(prompt="кот", preset=presets.get("LowQuality"))
    base.update(overrides)
    return gen.GenerationRequest(**base)


def image(size=(64, 64), colour="red"):
    return Image.new("RGBA", size, colour)


def test_text_to_image_has_no_conditions():
    assert gen.build_conditions(request()) == []


def test_single_reference_gets_no_tag():
    # Спецификация Qwen: при одном изображении теги использовать запрещено.
    slots = gen.build_conditions(request(references=(image(),)))
    assert len(slots) == 1
    assert slots[0].tag == ""
    assert slots[0].role == "reference"


def test_several_references_are_tagged_from_one():
    slots = gen.build_conditions(request(references=(image(), image(), image())))
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>", "<image3>"]


def test_mask_edit_puts_source_first_and_mask_second():
    slots = gen.build_conditions(
        request(source=image(), mask=Image.new("L", (64, 64), 255), mask_mode=gen.MASK_MASK)
    )
    assert [slot.role for slot in slots] == ["source", "mask"]
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>"]
    assert slots[1].image.mode == "RGB"  # маска подаётся как изображение


def test_references_are_numbered_after_source_and_mask():
    slots = gen.build_conditions(
        request(
            source=image(),
            mask=Image.new("L", (64, 64), 255),
            mask_mode=gen.MASK_MASK,
            references=(image(), image()),
        )
    )
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>", "<image3>", "<image4>"]
    assert [slot.role for slot in slots] == ["source", "mask", "reference", "reference"]


def test_annotation_mode_sends_one_image_and_no_mask():
    # Пометки нарисованы прямо на изображении, отдельная маска не нужна.
    slots = gen.build_conditions(
        request(source=image(), mask=Image.new("L", (64, 64), 255), mask_mode=gen.MASK_ANNOTATION)
    )
    assert [slot.role for slot in slots] == ["source"]
    assert slots[0].tag == ""


def test_prompt_edit_without_mask_sends_only_the_source():
    slots = gen.build_conditions(request(source=image(), mask_mode=gen.MASK_NONE))
    assert [slot.role for slot in slots] == ["source"]


def test_size_comes_from_the_aspect_and_preset():
    width, height = gen.resolve_size(request(aspect="16:9", preset=presets.get("MaxQuality")))
    assert (width, height) == (2752, 1536)


def test_follow_reference_leaves_size_to_the_pipeline():
    assert gen.resolve_size(request(aspect=aspect.FOLLOW_REFERENCE)) == (None, None)


def test_editing_follows_the_source_size_by_default():
    # При правке кадр не должен менять пропорции без явной просьбы.
    width, height = gen.resolve_size(request(source=image((100, 50)), mask_mode=gen.MASK_NONE))
    assert width is None and height is None


def test_seed_minus_one_is_replaced_by_a_random_one():
    resolved = gen.resolve_seed(-1)
    assert 0 <= resolved < 2**31


def test_explicit_seed_is_kept():
    assert gen.resolve_seed(12345) == 12345
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_generator.py -q`
Expected: FAIL, `ImportError: cannot import name 'generator'`

- [ ] **Step 3: Написать `fooocus_qwen/engine/generator.py`**

```python
"""Единая точка генерации: текст в изображение, правка промтом, правка по маске.

Порядок условных изображений определяет теги ``<imageN>``, которыми промт на них
ссылается, поэтому он зафиксирован: исходное изображение, затем маска, затем
референсы. Один и тот же порядок нужен интерфейсу для подписей миниатюр, поэтому
он вычисляется здесь, а не дублируется в двух местах.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from .. import __version__
from ..imaging import aspect, masking
from ..prompting.styles import Style, apply_styles
from .embeds_cache import EmbedsCache
from .presets import QualityPreset
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)

MASK_NONE = "none"
MASK_MASK = "mask"
MASK_ANNOTATION = "annotation"
MASK_REGION = "region"
MASK_MODES: tuple[str, ...] = (MASK_NONE, MASK_MASK, MASK_ANNOTATION, MASK_REGION)

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class ConditionSlot:
    """Одно условное изображение: его тег в промте и назначение."""

    tag: str
    role: str
    image: Image.Image


@dataclass
class GenerationRequest:
    """Всё, что нужно для одного запуска."""

    prompt: str
    preset: QualityPreset
    negative_prompt: str = ""
    styles: tuple[str, ...] = ()
    references: tuple[Image.Image, ...] = ()

    aspect: str = "1:1"
    seed: int = -1
    image_number: int = 1
    true_cfg_scale: float = 1.0
    use_kv_cache: bool = True

    source: Image.Image | None = None
    mask: Image.Image | None = None
    mask_mode: str = MASK_NONE
    mask_grow: int = 8
    mask_feather: int = 12
    keep_outside: bool = True

    prompt_original: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedImage:
    """Готовое изображение вместе с параметрами, которые его породили."""

    image: Image.Image
    seed: int
    parameters: dict[str, Any]


def resolve_seed(seed: int) -> int:
    """Отрицательное значение означает «выбери случайный и запомни его»."""
    return random.randrange(2**31) if seed < 0 else seed


def build_conditions(request: GenerationRequest) -> list[ConditionSlot]:
    """Условные изображения в том порядке, в каком их увидит модель."""
    slots: list[tuple[str, Image.Image]] = []

    if request.source is not None:
        slots.append(("source", request.source))
        # В режиме аннотации пометки уже нарисованы на самом изображении,
        # а «точная область» подаёт вырезанную пару так же, как обычная маска.
        if request.mask is not None and request.mask_mode in (MASK_MASK, MASK_REGION):
            slots.append(("mask", masking.as_condition(request.mask)))

    slots.extend(("reference", item) for item in request.references)

    tagged = len(slots) >= 2
    return [
        ConditionSlot(tag=f"<image{index}>" if tagged else "", role=role, image=item)
        for index, (role, item) in enumerate(slots, start=1)
    ]


def resolve_size(request: GenerationRequest) -> tuple[int | None, int | None]:
    """Размеры кадра или пара ``None``, если их выводит пайплайн.

    При правке размеры по умолчанию не задаются: кадр должен сохранить
    пропорции исходного изображения, а их пайплайн возьмёт из него сам.
    """
    if request.source is not None and request.aspect == "1:1":
        return None, None
    return aspect.dimensions(request.aspect, request.preset.output_resolution)


class Generator:
    """Выполняет запросы на генерацию по одному."""

    def __init__(
        self,
        pipe,
        residency: ResidencyManager,
        cache: EmbedsCache,
        catalogue: dict[str, Style],
    ) -> None:
        self._pipe = pipe
        self._residency = residency
        self._cache = cache
        self._catalogue = catalogue
        self._interrupted = False

    def interrupt(self) -> None:
        """Просит прервать текущую генерацию. Читается циклом денойзинга."""
        self._interrupted = True
        self._pipe._interrupt = True

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> list[GeneratedImage]:
        self._interrupted = False
        self._pipe._interrupt = False

        prepared = self._prepare(request)
        positive, negative = apply_styles(
            prepared.prompt, prepared.negative_prompt, prepared.styles, self._catalogue
        )
        slots = build_conditions(prepared)
        condition = [slot.image for slot in slots] or None
        width, height = resolve_size(prepared)
        base_seed = resolve_seed(prepared.seed)
        device = self._pipe._execution_device

        results: list[GeneratedImage] = []
        for index in range(max(1, prepared.image_number)):
            if self._interrupted:
                break

            seed = base_seed + index
            started = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(seed)

            def step_callback(_pipe, step: int, _timestep, kwargs, _index=index):
                if progress is not None:
                    progress(_index, step + 1, prepared.preset.num_inference_steps)
                return kwargs

            output = self._pipe(
                prompt=positive,
                image=condition,
                negative_prompt=negative or None,
                true_cfg_scale=prepared.true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=prepared.preset.num_inference_steps,
                output_resolution=prepared.preset.output_resolution,
                use_kv_cache=prepared.use_kv_cache,
                generator=generator,
                callback_on_step_end=step_callback,
            )

            if self._interrupted:
                # Прерванный цикл всё равно декодирует латенты, но это шум.
                LOGGER.info("Генерация прервана пользователем")
                break

            image = self._finish(request, prepared, output.images[0])
            seconds = time.perf_counter() - started
            results.append(
                GeneratedImage(
                    image=image,
                    seed=seed,
                    parameters=self._parameters(request, prepared, positive, negative, slots, seed, seconds),
                )
            )

        return results

    def _prepare(self, request: GenerationRequest) -> GenerationRequest:
        """Готовит маску и, для режима точной области, вырезает фрагмент."""
        if request.source is None or request.mask is None or request.mask_mode == MASK_NONE:
            return request

        refined = masking.refine(request.mask, grow=request.mask_grow, feather=request.mask_feather)
        if request.mask_mode != MASK_REGION:
            return _replace(request, mask=refined)

        box = masking.region_box(refined, padding=0.25)
        if box is None:
            LOGGER.warning("Маска пуста, режим точной области вырождается в правку целого кадра")
            return _replace(request, mask=refined, mask_mode=MASK_MASK)

        request.extras["region_box"] = box
        return _replace(
            request,
            mask=refined.crop(box),
            source=request.source.crop(box),
        )

    def _finish(
        self,
        original_request: GenerationRequest,
        prepared: GenerationRequest,
        produced: Image.Image,
    ) -> Image.Image:
        """Возвращает результат в систему координат исходного изображения."""
        source = original_request.source
        if source is None or prepared.mask_mode == MASK_NONE:
            return produced

        if prepared.mask_mode == MASK_ANNOTATION:
            # Пометки были частью условного изображения; склеивать не по чему.
            return produced

        if not original_request.keep_outside:
            return produced

        box = original_request.extras.get("region_box")
        if box is not None and prepared.mask is not None:
            full_mask = masking.refine(
                original_request.mask,
                grow=original_request.mask_grow,
                feather=original_request.mask_feather,
            )
            return masking.stitch(source, produced, box, full_mask)

        return masking.blend(source, produced, prepared.mask)

    def _parameters(
        self,
        original_request: GenerationRequest,
        prepared: GenerationRequest,
        positive: str,
        negative: str,
        slots: Sequence[ConditionSlot],
        seed: int,
        seconds: float,
    ) -> dict[str, Any]:
        import diffusers

        width, height = resolve_size(prepared)
        return {
            "app_version": __version__,
            "diffusers_version": diffusers.__version__,
            "prompt": original_request.prompt_original or original_request.prompt,
            "prompt_boosted": positive,
            "negative_prompt": negative,
            "styles": list(original_request.styles),
            "seed": seed,
            "steps": prepared.preset.num_inference_steps,
            "preset": prepared.preset.name,
            "width": width,
            "height": height,
            "output_resolution": prepared.preset.output_resolution,
            "true_cfg_scale": prepared.true_cfg_scale,
            # Переключение этого флага меняет результат при том же сиде,
            # поэтому без него параметры невоспроизводимы.
            "use_kv_cache": prepared.use_kv_cache,
            "mask_mode": prepared.mask_mode,
            "mask_grow": prepared.mask_grow,
            "mask_feather": prepared.mask_feather,
            "references": sum(1 for slot in slots if slot.role == "reference"),
            "seconds": round(seconds, 2),
        }


def _replace(request: GenerationRequest, **changes: Any) -> GenerationRequest:
    """Копия запроса с изменёнными полями; словарь extras остаётся общим."""
    import copy

    clone = copy.copy(request)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone
```

- [ ] **Step 4: Запустить тесты генератора**

Run: `.venv\Scripts\python -m pytest tests/test_generator.py -q`
Expected: PASS, 12 тестов

- [ ] **Step 5: Добавить безоконный режим в `fooocus_qwen/__main__.py`**

В `config.build_parser()` добавить перед `--selftest`:

```python
    parser.add_argument("--prompt", help="сгенерировать одно изображение без интерфейса и выйти")
    parser.add_argument("--out", help="куда сохранить результат режима --prompt")
```

В `__main__.py` добавить функцию и её вызов в `main` сразу после `--selftest`:

```python
def generate_once(args) -> int:
    """Одна генерация без интерфейса: для проверки и для скриптов."""
    from pathlib import Path

    from .engine import loader, presets
    from .engine.generator import GenerationRequest, Generator
    from .imaging import metadata
    from .prompting.styles import load_styles
    from .storage import gallery

    pipe, residency, cache = loader.load(config.MODEL_DIR, pin_memory=args.pin_memory)
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    def show(index: int, step: int, total: int) -> None:
        print(f"\rкартинка {index + 1}: шаг {step}/{total}", end="", flush=True)

    results = engine.generate(
        GenerationRequest(prompt=args.prompt, preset=presets.get(args.preset)), progress=show
    )
    print()

    if not results:
        print("Ничего не сгенерировано")
        return 1

    destination = Path(args.out) if args.out else gallery.next_path(config.OUTPUT_DIR)
    metadata.save_png(results[0].image, destination, results[0].parameters)
    print(f"Сохранено: {destination}")
    print(f"Сид: {results[0].seed}, время: {results[0].parameters['seconds']} с")
    print(f"Память: {residency.stats()}")
    return 0
```

И в `main`, после блока `if args.selftest:`:

```python
    if args.prompt:
        return generate_once(args)
```

- [ ] **Step 6: Проверить генерацию вживую**

Run: `.\run.ps1 --preset LowQuality --prompt "a neon shop sign that reads \"QWEN\", rainy night, reflections on wet pavement" --out test_t2i.png`

Expected: прогресс по шагам, файл `test_t2i.png`, в отчёте время и занятая память. Изображение должно содержать читаемую надпись `QWEN`.

- [ ] **Step 7: Проверить, что повторный запуск того же промта не поднимает энкодер**

Run: `.venv\Scripts\python -c "
from fooocus_qwen import config, logging_setup
logging_setup.setup_logging(False)
from fooocus_qwen.engine import loader, presets
from fooocus_qwen.engine.generator import GenerationRequest, Generator
from fooocus_qwen.prompting.styles import load_styles
pipe, residency, cache = loader.load(config.MODEL_DIR)
engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))
request = GenerationRequest(prompt='a red apple on a table', preset=presets.get('LowQuality'), seed=1)
engine.generate(request)
first = residency.stats()['swaps']
request.seed = 2
engine.generate(request)
print('перестановок после первой генерации:', first)
print('перестановок после второй:', residency.stats()['swaps'])
print('кэш: попаданий', cache.hits, 'промахов', cache.misses)
"`

Expected: число перестановок после второй генерации **равно** числу после первой, попаданий в кэш не меньше одного. Если перестановки выросли — кэш не работает, разбираться прежде чем идти дальше.

- [ ] **Step 8: Коммит**

```bash
git add fooocus_qwen/engine/generator.py fooocus_qwen/__main__.py fooocus_qwen/config.py tests/test_generator.py
git commit -m "feat: генератор изображений и безоконный режим запуска"
```

---

### Task 11: Бенчмарк и уточнение пресетов

**Files:**
- Create: `tools/benchmark.py`
- Modify: `fooocus_qwen/engine/presets.py` (подставить измеренные значения)
- Modify: `docs/superpowers/specs/2026-09-21-fooocus-qwen-image-21-design.md` (таблица 4.3)
- Create: `docs/BENCHMARK.md`

**Interfaces:**
- Consumes: `loader.load`, `presets.PRESETS`, `Generator`.
- Produces: `docs/BENCHMARK.md` с измеренными числами; уточнённые значения в `presets.PRESETS`.

- [ ] **Step 1: Написать `tools/benchmark.py`**

```python
"""Измерение времени и памяти по пресетам и по числу референсов.

Оценки в спецификации получены из пропускной способности карты; здесь они
заменяются измерениями. Первый прогон каждого режима отбрасывается: он несёт
разовые расходы на подъём весов и прогрев ядер.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader, presets
from fooocus_qwen.engine.generator import GenerationRequest, Generator
from fooocus_qwen.prompting.styles import load_styles

PROMPT = "a wooden desk with a brass lamp, an open notebook and a cup of tea, warm afternoon light"


def reference_images(count: int) -> tuple[Image.Image, ...]:
    """Синтетические референсы: важен их объём, а не содержание."""
    return tuple(
        Image.new("RGB", (768, 768), (40 * index % 256, 90, 160)) for index in range(count)
    )


def measure(engine, request, repeats: int) -> dict[str, float]:
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []

    for attempt in range(repeats + 1):
        request.seed = 1000 + attempt
        started = time.perf_counter()
        engine.generate(request)
        elapsed = time.perf_counter() - started
        if attempt > 0:  # первый прогон прогревочный
            durations.append(elapsed)

    return {
        "seconds_median": round(statistics.median(durations), 2),
        "seconds_min": round(min(durations), 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк пресетов и референсов")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--references", type=int, nargs="*", default=[0, 1, 5, 10])
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    pipe, residency, cache = loader.load(config.MODEL_DIR)
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    report: dict[str, dict] = {"presets": {}, "references": {}}

    for name in presets.NAMES:
        request = GenerationRequest(prompt=PROMPT, preset=presets.get(name))
        result = measure(engine, request, args.repeats)
        report["presets"][name] = result
        print(f"{name}: {result}")

    for count in args.references:
        request = GenerationRequest(
            prompt=PROMPT if count < 2 else "combine <image1> and <image2> into one scene",
            preset=presets.get("MiddleQuality"),
            references=reference_images(count),
        )
        try:
            result = measure(engine, request, 1)
        except torch.cuda.OutOfMemoryError:
            result = {"error": "нехватка видеопамяти"}
            torch.cuda.empty_cache()
        report["references"][str(count)] = result
        print(f"референсов {count}: {result}")

    report["cache"] = {"hits": cache.hits, "misses": cache.misses}
    report["residency"] = residency.stats()

    destination = Path(config.PROJECT_ROOT) / "docs" / "BENCHMARK.md"
    destination.write_text(_render(report), encoding="utf-8")
    print(f"\nОтчёт: {destination}")
    return 0


def _render(report: dict) -> str:
    lines = [
        "# Измерения производительности",
        "",
        f"Дата: {date.today().isoformat()}. Карта: {torch.cuda.get_device_name(0)}.",
        "",
        "## Пресеты качества",
        "",
        "| Пресет | Медиана, с | Минимум, с | Пик видеопамяти, ГиБ |",
        "|---|---|---|---|",
    ]
    for name, values in report["presets"].items():
        lines.append(
            f"| {name} | {values['seconds_median']} | {values['seconds_min']} | {values['peak_vram_gib']} |"
        )

    lines += ["", "## Референсные изображения (MiddleQuality)", "", "| Референсов | Результат |", "|---|---|"]
    for count, values in report["references"].items():
        lines.append(f"| {count} | {json.dumps(values, ensure_ascii=False)} |")

    lines += [
        "",
        "## Кэш эмбеддингов и перестановки весов",
        "",
        f"- попаданий: {report['cache']['hits']}, промахов: {report['cache']['misses']}",
        f"- перестановок энкодера: {int(report['residency']['swaps'])}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Прогнать бенчмарк**

Run: `.venv\Scripts\python tools/benchmark.py`
Expected: три строки пресетов, четыре строки референсов, файл `docs/BENCHMARK.md`

- [ ] **Step 3: Подставить измеренные значения в пресеты**

Если измеренное время MiddleQuality превышает 60 с, понизить `output_resolution` до 1280 и повторить шаг 2. Если LowQuality укладывается менее чем в 8 с, поднять `num_inference_steps` до 20 и повторить шаг 2. Цель: примерно 10 / 30 / 75 секунд.

Если на десяти референсах случилась нехватка видеопамяти, добавить в `docs/BENCHMARK.md` строку с найденной границей и записать её в раздел рисков спецификации.

- [ ] **Step 4: Обновить таблицу 4.3 спецификации**

Заменить столбец «Расчётная оценка» на «Измерено» с числами из `docs/BENCHMARK.md` и добавить под таблицей строку со ссылкой на отчёт.

- [ ] **Step 5: Коммит**

```bash
git add tools/benchmark.py docs/BENCHMARK.md docs/superpowers/specs fooocus_qwen/engine/presets.py
git commit -m "feat: бенчмарк пресетов и уточнение их по измерениям"
```

---

### Task 12: Интерфейс — каркас, двуязычность, вкладка «Генерация»

**Files:**
- Create: `fooocus_qwen/ui/__init__.py`, `fooocus_qwen/ui/i18n.py`, `fooocus_qwen/ui/state.py`, `fooocus_qwen/ui/style.css`, `fooocus_qwen/ui/tab_generate.py`, `fooocus_qwen/ui/app.py`
- Create: `tests/test_i18n.py`

**Interfaces:**
- Consumes: `generator.Generator`, `generator.GenerationRequest`, `presets`, `aspect`, `styles`, `boost`, `metadata`, `gallery`, `library`.
- Produces:
  - `i18n.Localizer(default: str = "ru")` с `bind(component, **fields) -> component`, свойством `components: list`, `updates(lang: str) -> list`
  - `i18n.T` — словарь текстов вида `{"ключ": ("русский", "english")}`
  - `i18n.pick(key: str, lang: str) -> str`
  - `state.Studio` — объект приложения: ленивая загрузка модели, доступ к генератору, каталогу стилей и клиенту LLM
  - `tab_generate.build(studio, localizer) -> dict[str, Any]` — возвращает словарь ключевых компонентов для переиспользования другими вкладками
  - `app.launch(cfg: config.AppConfig) -> None`

- [ ] **Step 1: Написать тест локализатора**

Создать `tests/test_i18n.py`:

```python
"""Двуязычные подписи."""

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen.ui.i18n import Localizer, pick


def test_pick_returns_the_requested_language():
    assert pick("generate", "ru") != pick("generate", "en")


def test_unknown_key_returns_itself():
    # Пропущенная подпись не должна ронять интерфейс.
    assert pick("нет такого ключа", "ru") == "нет такого ключа"


def test_bind_registers_the_component():
    with gr.Blocks():
        localizer = Localizer()
        box = localizer.bind(gr.Textbox(label="Промт"), label=("Промт", "Prompt"))

    assert box in localizer.components
    assert len(localizer.components) == 1


def test_updates_match_the_registration_order():
    with gr.Blocks():
        localizer = Localizer()
        localizer.bind(gr.Textbox(), label=("Первый", "First"))
        localizer.bind(gr.Button(), value=("Второй", "Second"))

    updates = localizer.updates("en")
    assert len(updates) == 2
    assert updates[0]["label"] == "First"
    assert updates[1]["value"] == "Second"


def test_several_fields_on_one_component():
    with gr.Blocks():
        localizer = Localizer()
        localizer.bind(gr.Slider(), label=("Шаги", "Steps"), info=("Сколько шагов", "How many steps"))

    update = localizer.updates("ru")[0]
    assert update["label"] == "Шаги"
    assert update["info"] == "Сколько шагов"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_i18n.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fooocus_qwen.ui'`

- [ ] **Step 3: Написать `fooocus_qwen/ui/i18n.py`**

```python
"""Двуязычные подписи интерфейса.

Подписи в Gradio задаются при сборке, поэтому смена языка — это массовое
обновление уже созданных компонентов. Локализатор запоминает, какие поля какого
компонента переводимы, и по запросу отдаёт список обновлений в том же порядке,
в каком компоненты регистрировались.

Промты к модели при этом остаются английскими: переписыватель Qwen отвечает
по-английски независимо от языка запроса, и переводить его ответ обратно было бы
потерей качества.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

LANGUAGES = ("ru", "en")

T: dict[str, tuple[str, str]] = {
    "app_title": ("Qwen-Image-2.1 — студия", "Qwen-Image-2.1 Studio"),
    "language": ("Язык", "Language"),
    "tab_generate": ("Генерация", "Generate"),
    "tab_edit": ("Редактирование", "Edit"),
    "tab_gallery": ("Галерея", "Gallery"),
    "tab_settings": ("Настройки", "Settings"),
    "prompt": ("Промт", "Prompt"),
    "prompt_placeholder": ("Опишите изображение…", "Describe the image…"),
    "generate": ("Сгенерировать", "Generate"),
    "stop": ("Прервать", "Stop"),
    "advanced": ("Продвинутое", "Advanced"),
    "quality": ("Качество", "Quality"),
    "aspect": ("Соотношение сторон", "Aspect ratio"),
    "image_number": ("Количество изображений", "Image number"),
    "seed": ("Сид", "Seed"),
    "seed_info": ("−1 — выбрать случайно", "−1 picks a random one"),
    "styles": ("Стили", "Styles"),
    "negative": ("Негативный промт", "Negative prompt"),
    "cfg": ("Сила guidance (true_cfg_scale)", "Guidance strength (true_cfg_scale)"),
    "cfg_info": (
        "Модель рассчитана на работу без guidance. При значении 1.0 негативный промт "
        "и негативные части стилей в модель не попадают вовсе.",
        "The model is meant to be sampled without guidance. At 1.0 the negative prompt "
        "and the negative half of every style are not sent to the model at all.",
    ),
    "kv_cache": ("Кэш ключей и значений", "KV cache"),
    "kv_cache_info": (
        "Ускоряет генерацию. Переключение меняет результат при том же сиде.",
        "Speeds generation up. Toggling it changes the result for the same seed.",
    ),
    "references": ("Референсы", "References"),
    "references_info": (
        "До 10 изображений. Ссылайтесь на них в промте подписями под миниатюрами.",
        "Up to 10 images. Refer to them in the prompt by the caption under each thumbnail.",
    ),
    "reference_add": ("Добавить референсы", "Add references"),
    "reference_clear": ("Очистить референсы", "Clear references"),
    "boost": ("AI буст", "AI boost"),
    "boost_info": (
        "Переписать промт внешней языковой моделью перед генерацией.",
        "Rewrite the prompt with an external language model before generating.",
    ),
    "boost_result": ("Переписанный промт", "Rewritten prompt"),
    "boost_now": ("Переписать сейчас", "Rewrite now"),
    "describe": ("Описать изображение", "Describe image"),
    "result": ("Результат", "Result"),
    "save_prompt": ("Сохранить промт", "Save prompt"),
    "load_prompt": ("Загрузить промт", "Load prompt"),
    "delete_prompt": ("Удалить промт", "Delete prompt"),
    "preset_name": ("Название пресета", "Preset name"),
    "status": ("Состояние", "Status"),
    "source_image": ("Исходное изображение", "Source image"),
    "mask_mode": ("Режим области", "Region mode"),
    "mask_mode_none": ("Без области — править весь кадр", "No region — edit the whole frame"),
    "mask_mode_mask": ("Маска", "Mask"),
    "mask_mode_annotation": ("Аннотация", "Annotation"),
    "mask_mode_region": ("Точная область", "Exact region"),
    "mask_grow": ("Запас маски, пикселей", "Mask grow, pixels"),
    "mask_feather": ("Растушёвка, пикселей", "Feather, pixels"),
    "keep_outside": ("Сохранять кадр вне маски", "Keep pixels outside the mask"),
    "keep_outside_info": (
        "Склеивает результат с оригиналом: вне маски пиксели остаются исходными.",
        "Blends the result with the original so pixels outside the mask stay untouched.",
    ),
    "outpaint": ("Расширить холст", "Outpaint"),
    "outpaint_sides": ("Стороны", "Sides"),
    "outpaint_amount": ("Насколько расширить", "How far to expand"),
    "apply_edit": ("Применить правку", "Apply edit"),
    "send_to_edit": ("Отправить в редактор", "Send to editor"),
    "llm_endpoint": ("Адрес языковой модели", "Language model endpoint"),
    "llm_check": ("Проверить связь", "Check connection"),
    "llm_model": ("Модель", "Model"),
    "refresh": ("Обновить", "Refresh"),
    "open_folder": ("Открыть папку", "Open folder"),
    "restore_params": ("Восстановить параметры из PNG", "Restore parameters from PNG"),
}


def pick(key: str, lang: str) -> str:
    entry = T.get(key)
    if entry is None:
        return key
    return entry[0] if lang == "ru" else entry[1]


class Localizer:
    """Запоминает переводимые поля компонентов и обновляет их разом."""

    def __init__(self, default: str = "ru") -> None:
        self._default = default
        self._entries: list[tuple[Any, dict[str, tuple[str, str]]]] = []

    def bind(self, component: Any, **fields: tuple[str, str]) -> Any:
        self._entries.append((component, fields))
        return component

    @property
    def components(self) -> list[Any]:
        return [component for component, _ in self._entries]

    def updates(self, lang: str) -> list[Any]:
        index = 0 if lang == "ru" else 1
        return [
            gr.update(**{name: values[index] for name, values in fields.items()})
            for _, fields in self._entries
        ]
```

Создать `fooocus_qwen/ui/__init__.py`:

```python
"""Веб-интерфейс на Gradio. Вычислений здесь нет — только сборка и связывание."""
```

- [ ] **Step 4: Запустить тест локализатора**

Run: `.venv\Scripts\python -m pytest tests/test_i18n.py -q`
Expected: PASS, 5 тестов

- [ ] **Step 5: Написать `fooocus_qwen/ui/state.py`**

```python
"""Состояние приложения: модель, каталог стилей, связь с языковой моделью.

Модель грузится лениво и один раз. Держать тридцать три гигабайта весов ради
того, чтобы пользователь открыл вкладку настроек, незачем, а первый запрос всё
равно подождёт загрузки.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from PIL import Image

from .. import config
from ..llm import LlmClient, LlmError, load_endpoint
from ..prompting import boost
from ..prompting.styles import Style, load_styles

LOGGER = logging.getLogger(__name__)


class Studio:
    """Единственный владелец тяжёлых ресурсов."""

    def __init__(self, cfg: config.AppConfig) -> None:
        self.config = cfg
        self.catalogue: dict[str, Style] = load_styles(config.STYLES_DIR)
        self._lock = threading.Lock()
        self._generator: Any = None
        self._residency: Any = None
        self._cache: Any = None

    @property
    def generator(self):
        """Возвращает генератор, загрузив модель при первом обращении."""
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from ..engine import loader
                    from ..engine.generator import Generator

                    pipe, residency, cache = loader.load(
                        self.config.model_dir, pin_memory=self.config.pin_memory
                    )
                    self._generator = Generator(pipe, residency, cache, self.catalogue)
                    self._residency = residency
                    self._cache = cache
        return self._generator

    @property
    def model_loaded(self) -> bool:
        return self._generator is not None

    def memory_report(self) -> str:
        if self._residency is None:
            return "модель ещё не загружена"
        stats = self._residency.stats()
        return (
            f"видеопамять: {stats['allocated_gib']:.1f} ГиБ занято, "
            f"{stats['reserved_gib']:.1f} ГиБ зарезервировано; "
            f"перестановок энкодера: {int(stats['swaps'])}; "
            f"кэш промтов: {self._cache.hits} попаданий / {self._cache.misses} промахов"
        )

    def llm_client(self) -> LlmClient:
        """Создаёт клиента заново: файл адреса правится без перезапуска."""
        endpoint = load_endpoint(config.ENDPOINT_FILE)
        client = LlmClient(endpoint)
        models = client.ping()
        return LlmClient(endpoint, model=models[0] if models else "")

    def boost_prompt(
        self,
        prompt: str,
        mode: str,
        references: list[Image.Image] | None = None,
    ) -> tuple[str, str | None, str]:
        """Возвращает переписанный промт, соотношение сторон и сообщение о результате."""
        try:
            client = self.llm_client()
            result = boost.boost(
                client, prompt, mode=mode, prompt_dir=config.SYSTEM_PROMPT_DIR, references=references
            )
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("AI-буст не выполнен: %s", error)
            return prompt, None, f"AI буст не выполнен: {error}"
        return result.prompt, result.wh_ratio, "AI буст выполнен"

    def describe_image(self, image: Image.Image) -> tuple[str, str]:
        try:
            return boost.describe(self.llm_client(), image, config.SYSTEM_PROMPT_DIR), "Описание готово"
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Описание не выполнено: %s", error)
            return "", f"Описание не выполнено: {error}"
```

- [ ] **Step 6: Написать `fooocus_qwen/ui/tab_generate.py`**

```python
"""Вкладка генерации: промт, референсы, качество, результат.

Компоновка повторяет Fooocus: наверху результат, под ним строка промта и одна
кнопка, а всё остальное спрятано за переключателем «Продвинутое». Референсы
показаны лентой миниатюр, каждая подписана тегом, которым на неё ссылаются
из промта.
"""

from __future__ import annotations

import logging

import gradio as gr

from .. import config
from ..engine import presets
from ..engine.generator import GenerationRequest
from ..imaging import aspect, metadata
from ..prompting import boost as boost_module
from ..prompting import library
from ..storage import gallery
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)

MAX_REFERENCES = 10


def _captions(count: int) -> list[str]:
    """Подписи миниатюр: ровно те теги, которыми промт адресует изображения.

    При единственном референсе теги запрещены спецификацией Qwen, поэтому
    подпись говорит об этом прямо, а не показывает несуществующий «<image1>».
    """
    if count == 1:
        return ["без тега — одно изображение"]
    return [f"<image{index}>" for index in range(1, count + 1)]


def build(studio, localizer: Localizer) -> dict:
    lang = studio.config.lang
    catalogue_names = sorted(studio.catalogue)

    references = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            result = localizer.bind(
                gr.Gallery(
                    label=pick("result", lang),
                    show_label=True,
                    columns=2,
                    height=620,
                    object_fit="contain",
                    format="png",
                ),
                label=("Результат", "Result"),
            )

            with gr.Row():
                prompt = localizer.bind(
                    gr.Textbox(
                        label=pick("prompt", lang),
                        placeholder=pick("prompt_placeholder", lang),
                        lines=3,
                        scale=8,
                    ),
                    label=("Промт", "Prompt"),
                    placeholder=("Опишите изображение…", "Describe the image…"),
                )
                with gr.Column(scale=1, min_width=140):
                    run_button = localizer.bind(
                        gr.Button(pick("generate", lang), variant="primary"),
                        value=("Сгенерировать", "Generate"),
                    )
                    stop_button = localizer.bind(
                        gr.Button(pick("stop", lang), variant="stop"), value=("Прервать", "Stop")
                    )

            with gr.Row():
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("boost", lang), value=False, info=pick("boost_info", lang)),
                    label=("AI буст", "AI boost"),
                    info=(
                        "Переписать промт внешней языковой моделью перед генерацией.",
                        "Rewrite the prompt with an external language model before generating.",
                    ),
                )
                boost_now = localizer.bind(
                    gr.Button(pick("boost_now", lang)), value=("Переписать сейчас", "Rewrite now")
                )

            boosted = localizer.bind(
                gr.Textbox(label=pick("boost_result", lang), lines=3, interactive=True),
                label=("Переписанный промт", "Rewritten prompt"),
            )

            reference_gallery = localizer.bind(
                gr.Gallery(
                    label=pick("references", lang),
                    columns=5,
                    height=170,
                    object_fit="contain",
                    show_label=True,
                ),
                label=("Референсы", "References"),
            )
            with gr.Row():
                reference_upload = localizer.bind(
                    gr.File(
                        label=pick("reference_add", lang),
                        file_count="multiple",
                        file_types=["image"],
                    ),
                    label=("Добавить референсы", "Add references"),
                )
                reference_clear = localizer.bind(
                    gr.Button(pick("reference_clear", lang)),
                    value=("Очистить референсы", "Clear references"),
                )

        with gr.Column(scale=1):
            quality = localizer.bind(
                gr.Radio(
                    choices=list(presets.NAMES),
                    value=studio.config.preset,
                    label=pick("quality", lang),
                ),
                label=("Качество", "Quality"),
            )
            ratio = localizer.bind(
                gr.Dropdown(
                    choices=list(aspect.ASPECT_RATIOS),
                    value="1:1",
                    label=pick("aspect", lang),
                ),
                label=("Соотношение сторон", "Aspect ratio"),
            )
            image_number = localizer.bind(
                gr.Slider(1, 8, value=1, step=1, label=pick("image_number", lang)),
                label=("Количество изображений", "Image number"),
            )
            status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=3),
                label=("Состояние", "Status"),
            )

            with gr.Accordion(pick("advanced", lang), open=False) as advanced:
                styles = localizer.bind(
                    gr.Dropdown(
                        choices=catalogue_names,
                        value=[],
                        multiselect=True,
                        label=pick("styles", lang),
                    ),
                    label=("Стили", "Styles"),
                )
                negative = localizer.bind(
                    gr.Textbox(label=pick("negative", lang), lines=2),
                    label=("Негативный промт", "Negative prompt"),
                )
                cfg = localizer.bind(
                    gr.Slider(
                        1.0, 8.0, value=1.0, step=0.1,
                        label=pick("cfg", lang), info=pick("cfg_info", lang),
                    ),
                    label=("Сила guidance (true_cfg_scale)", "Guidance strength (true_cfg_scale)"),
                    info=(
                        "Модель рассчитана на работу без guidance. При значении 1.0 негативный "
                        "промт и негативные части стилей в модель не попадают вовсе.",
                        "The model is meant to be sampled without guidance. At 1.0 the negative "
                        "prompt and the negative half of every style are not sent at all.",
                    ),
                )
                seed = localizer.bind(
                    gr.Number(value=-1, precision=0, label=pick("seed", lang), info=pick("seed_info", lang)),
                    label=("Сид", "Seed"),
                    info=("−1 — выбрать случайно", "−1 picks a random one"),
                )
                kv_cache = localizer.bind(
                    gr.Checkbox(
                        value=True, label=pick("kv_cache", lang), info=pick("kv_cache_info", lang)
                    ),
                    label=("Кэш ключей и значений", "KV cache"),
                    info=(
                        "Ускоряет генерацию. Переключение меняет результат при том же сиде.",
                        "Speeds generation up. Toggling it changes the result for the same seed.",
                    ),
                )

                preset_name = localizer.bind(
                    gr.Textbox(label=pick("preset_name", lang)),
                    label=("Название пресета", "Preset name"),
                )
                saved = localizer.bind(
                    gr.Dropdown(
                        choices=library.list_prompts(config.PROMPT_DIR),
                        label=pick("load_prompt", lang),
                    ),
                    label=("Загрузить промт", "Load prompt"),
                )
                with gr.Row():
                    save_button = localizer.bind(
                        gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
                    )
                    delete_button = localizer.bind(
                        gr.Button(pick("delete_prompt", lang)), value=("Удалить промт", "Delete prompt")
                    )

    # --- обработчики ---

    def add_references(files, current):
        from PIL import Image as PILImage

        images = list(current or [])
        for item in files or []:
            if len(images) >= MAX_REFERENCES:
                break
            images.append(PILImage.open(item.name).convert("RGB"))
        captioned = list(zip(images, _captions(len(images))))
        return images, captioned, f"Референсов: {len(images)} из {MAX_REFERENCES}"

    def clear_references():
        return [], [], "Референсы очищены"

    def rewrite(prompt_text, current_references, ratio_value):
        mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
        text, wh_ratio, message = studio.boost_prompt(prompt_text, mode, current_references or None)
        chosen = wh_ratio if wh_ratio in aspect.ASPECT_RATIOS else ratio_value
        return text, chosen, message

    def run(
        prompt_text, boosted_text, use_boost, current_references, quality_name, ratio_value,
        count, style_names, negative_text, cfg_value, seed_value, kv_value,
        progress=gr.Progress(),
    ):
        effective = (boosted_text or "").strip() if use_boost else ""
        message = ""
        if use_boost and not effective:
            mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
            effective, wh_ratio, message = studio.boost_prompt(
                prompt_text, mode, current_references or None
            )
            if wh_ratio in aspect.ASPECT_RATIOS:
                ratio_value = wh_ratio

        request = GenerationRequest(
            prompt=effective or prompt_text,
            prompt_original=prompt_text,
            preset=presets.get(quality_name),
            negative_prompt=negative_text or "",
            styles=tuple(style_names or ()),
            references=tuple(current_references or ()),
            aspect=ratio_value,
            seed=int(seed_value),
            image_number=int(count),
            true_cfg_scale=float(cfg_value),
            use_kv_cache=bool(kv_value),
        )

        def report(index: int, step: int, total: int) -> None:
            progress((step, total), desc=f"изображение {index + 1}/{int(count)}")

        produced = studio.generator.generate(request, progress=report)
        if not produced:
            return [], f"{message} Генерация прервана".strip()

        paths = []
        for item in produced:
            destination = gallery.next_path(config.OUTPUT_DIR)
            metadata.save_png(item.image, destination, item.parameters)
            paths.append(str(destination))

        seeds = ", ".join(str(item.seed) for item in produced)
        report_line = f"Готово. Сиды: {seeds}. {studio.memory_report()}"
        return paths, f"{message} {report_line}".strip()

    def stop():
        if studio.model_loaded:
            studio.generator.interrupt()
        return "Останавливаю…"

    def save(name, prompt_text, negative_text, style_names, quality_name, ratio_value, seed_value, cfg_value):
        if not (name or "").strip():
            return gr.update(), "Укажите название пресета"
        library.save_prompt(
            name,
            {
                "prompt": prompt_text,
                "negative_prompt": negative_text,
                "styles": list(style_names or ()),
                "preset": quality_name,
                "aspect": ratio_value,
                "seed": int(seed_value),
                "true_cfg_scale": float(cfg_value),
            },
            config.PROMPT_DIR,
        )
        return gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=name), f"Пресет «{name}» сохранён"

    def load(name):
        if not name:
            return (gr.update(),) * 7 + ("Пресет не выбран",)
        payload = library.load_prompt(name, config.PROMPT_DIR)
        return (
            payload.get("prompt", ""),
            payload.get("negative_prompt", ""),
            payload.get("styles", []),
            payload.get("preset", presets.DEFAULT),
            payload.get("aspect", "1:1"),
            payload.get("seed", -1),
            payload.get("true_cfg_scale", 1.0),
            f"Пресет «{name}» загружен",
        )

    def delete(name):
        removed = library.delete_prompt(name, config.PROMPT_DIR)
        message = f"Пресет «{name}» удалён" if removed else "Пресет не найден"
        return gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=None), message

    reference_upload.change(
        add_references, [reference_upload, references], [references, reference_gallery, status]
    )
    reference_clear.click(clear_references, None, [references, reference_gallery, status])
    boost_now.click(rewrite, [prompt, references, ratio], [boosted, ratio, status])

    run_event = run_button.click(
        run,
        [prompt, boosted, boost_enabled, references, quality, ratio, image_number,
         styles, negative, cfg, seed, kv_cache],
        [result, status],
    )
    # Кнопка остановки должна срабатывать, пока генерация занимает очередь.
    stop_button.click(stop, None, status, queue=False, cancels=None)

    save_button.click(
        save, [preset_name, prompt, negative, styles, quality, ratio, seed, cfg], [saved, status]
    )
    saved.change(load, saved, [prompt, negative, styles, quality, ratio, seed, cfg, status])
    delete_button.click(delete, saved, [saved, status])

    return {
        "prompt": prompt,
        "boosted": boosted,
        "negative": negative,
        "styles": styles,
        "quality": quality,
        "ratio": ratio,
        "seed": seed,
        "cfg": cfg,
        "status": status,
        "result": result,
        "references": references,
        "reference_gallery": reference_gallery,
        "advanced": advanced,
        "run_event": run_event,
    }
```

- [ ] **Step 7: Написать `fooocus_qwen/ui/style.css` и `fooocus_qwen/ui/app.py`**

`style.css`:

```css
/* Результат — главное на экране, поэтому галерея получает максимум высоты. */
.gradio-container { max-width: 1600px !important; }
footer { display: none !important; }
```

`app.py`:

```python
"""Сборка интерфейса и запуск сервера."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from .. import config
from . import tab_generate
from .i18n import LANGUAGES, Localizer, pick
from .state import Studio

LOGGER = logging.getLogger(__name__)


def build(cfg: config.AppConfig) -> gr.Blocks:
    studio = Studio(cfg)
    localizer = Localizer(cfg.lang)
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

    with gr.Blocks(title=pick("app_title", cfg.lang), css=css, analytics_enabled=False) as demo:
        with gr.Row():
            title = localizer.bind(
                gr.Markdown(f"## {pick('app_title', cfg.lang)}"),
                value=("## Qwen-Image-2.1 — студия", "## Qwen-Image-2.1 Studio"),
            )
            language = gr.Dropdown(
                choices=list(LANGUAGES), value=cfg.lang, label="RU / EN", scale=0, min_width=120
            )

        with gr.Tabs():
            with gr.Tab(pick("tab_generate", cfg.lang)) as generate_tab:
                generate = tab_generate.build(studio, localizer)

        # Вкладки редактирования, галереи и настроек добавляются в задачах 13 и 14.

        language.change(localizer.updates, language, localizer.components, queue=False)

    return demo


def launch(cfg: config.AppConfig) -> None:
    demo = build(cfg)
    demo.queue(default_concurrency_limit=1)
    LOGGER.info("Интерфейс на http://%s:%s", cfg.host, cfg.port)
    demo.launch(server_name=cfg.host, server_port=cfg.port, show_api=False, inbrowser=False)
```

- [ ] **Step 8: Запустить интерфейс и проверить генерацию**

Run: `.\run.ps1 --preset LowQuality`

Проверить в браузере по адресу `http://<адрес машины>:7865`:
1. Страница открывается с другой машины в сети — значит слушается `0.0.0.0`.
2. Промт «a red apple on a wooden table», кнопка «Сгенерировать» — появляется прогресс по шагам и результат.
3. Файл появился в `user/outputs/<дата>/`.
4. Переключатель RU/EN меняет подписи.
5. Загрузка двух референсов — под миниатюрами подписи `<image1>` и `<image2>`; при одном референсе подпись говорит, что тег не нужен.
6. Кнопка «Прервать» останавливает генерацию.

- [ ] **Step 9: Коммит**

```bash
git add fooocus_qwen/ui tests/test_i18n.py
git commit -m "feat: интерфейс генерации с двуязычными подписями и референсами"
```

---

### Task 13: Интерфейс — вкладка «Редактирование»

**Files:**
- Create: `fooocus_qwen/ui/tab_edit.py`
- Modify: `fooocus_qwen/ui/app.py` (подключить вкладку)
- Create: `tests/test_edit_collect.py`

**Interfaces:**
- Consumes: `generator.MASK_*`, `masking.mask_from_editor`, `masking.is_empty`, `outpaint.plan`, `outpaint.expand`.
- Produces:
  - `tab_edit.collect(value: dict, mode: str) -> tuple[Image.Image | None, Image.Image | None]` — исходное изображение и маска для запроса
  - `tab_edit.ANNOTATION_COLOURS: tuple[str, ...]`
  - `tab_edit.build(studio, localizer, generate_components: dict) -> dict`

- [ ] **Step 1: Написать падающий тест сбора данных редактора**

Создать `tests/test_edit_collect.py`:

```python
"""Извлечение исходного изображения и маски из значения редактора."""

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.ui import tab_edit


def editor(size=(64, 64), painted=None, colour=(255, 0, 0, 255)):
    background = Image.new("RGBA", size, (12, 34, 56, 255))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted:
        layer.paste(colour, painted)
    composite = background.copy()
    composite.alpha_composite(layer)
    return {"background": background, "layers": [layer], "composite": composite}


def test_mask_mode_takes_clean_background_and_a_mask():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_MASK)
    assert np.asarray(source.convert("RGBA"))[2, 2].tolist() == [12, 34, 56, 255]
    assert mask is not None
    assert np.asarray(mask)[16, 16] == 255


def test_annotation_mode_takes_the_composite_and_no_mask():
    # Пометки должны попасть в модель как часть изображения.
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_ANNOTATION)
    assert np.asarray(source.convert("RGBA"))[16, 16].tolist() == [255, 0, 0, 255]
    assert mask is None


def test_region_mode_behaves_like_mask_mode():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_REGION)
    assert mask is not None
    assert np.asarray(source.convert("RGBA"))[2, 2].tolist() == [12, 34, 56, 255]


def test_no_mask_mode_ignores_the_layers():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_NONE)
    assert mask is None
    assert np.asarray(source.convert("RGBA"))[16, 16].tolist() == [12, 34, 56, 255]


def test_empty_mask_falls_back_to_no_mask():
    # Пользователь выбрал режим маски, но ничего не нарисовал.
    source, mask = tab_edit.collect(editor(), gen.MASK_MASK)
    assert source is not None
    assert mask is None


def test_missing_value_returns_nothing():
    assert tab_edit.collect(None, gen.MASK_MASK) == (None, None)
    assert tab_edit.collect({"background": None, "layers": []}, gen.MASK_MASK) == (None, None)


def test_annotation_palette_has_the_colours_the_blog_uses():
    # Синий, красный и зелёный — цвета из примера с тремя областями.
    for colour in ("#ff0000", "#0000ff", "#00ff00"):
        assert colour in tab_edit.ANNOTATION_COLOURS
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_edit_collect.py -q`
Expected: FAIL, `ImportError: cannot import name 'tab_edit'`

- [ ] **Step 3: Написать `fooocus_qwen/ui/tab_edit.py`**

```python
"""Вкладка редактирования: правка промтом, по маске, по аннотации и расширение холста.

Все три режима области питаются из одного редактора. Разница в том, что уходит
в модель: чистый фон плюс отдельная чёрно-белая маска — или сведённое
изображение с цветными пометками прямо на нём.

Отдельная маска — основной режим: он не портит оригинал. Аннотация нужна тогда,
когда областей несколько и каждой нужна своя инструкция: цвет пометки становится
адресом области внутри одного промта.
"""

from __future__ import annotations

import logging

import gradio as gr
from PIL import Image

from .. import config
from ..engine import presets
from ..engine.generator import (
    MASK_ANNOTATION,
    MASK_MASK,
    MASK_NONE,
    MASK_REGION,
    GenerationRequest,
)
from ..imaging import masking, metadata, outpaint
from ..prompting import boost as boost_module
from ..storage import gallery
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)

# Цвета из примера в блоге: три области, три инструкции в одном промте.
ANNOTATION_COLOURS: tuple[str, ...] = ("#ff0000", "#0000ff", "#00ff00", "#ffff00", "#ffffff")

_MODE_KEYS = {
    MASK_NONE: "mask_mode_none",
    MASK_MASK: "mask_mode_mask",
    MASK_ANNOTATION: "mask_mode_annotation",
    MASK_REGION: "mask_mode_region",
}


def collect(value, mode: str) -> tuple[Image.Image | None, Image.Image | None]:
    """Разбирает значение редактора на исходное изображение и маску."""
    if not value:
        return None, None

    background = value.get("background")
    if background is None:
        return None, None

    if mode == MASK_ANNOTATION:
        composite = value.get("composite")
        if composite is None:
            composite = background.convert("RGBA").copy()
            for layer in value.get("layers") or []:
                composite.alpha_composite(layer.convert("RGBA"))
        return composite.convert("RGBA"), None

    if mode == MASK_NONE:
        return background.convert("RGBA"), None

    mask = masking.mask_from_editor(value)
    if masking.is_empty(mask):
        LOGGER.info("Режим области выбран, но маска пуста — правлю кадр целиком")
        return background.convert("RGBA"), None

    return background.convert("RGBA"), mask


def build(studio, localizer: Localizer, generate_components: dict) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column(scale=3):
            editor = localizer.bind(
                gr.ImageEditor(
                    label=pick("source_image", lang),
                    type="pil",
                    image_mode="RGBA",
                    layers=True,
                    height=620,
                    brush=gr.Brush(colors=list(ANNOTATION_COLOURS), default_color="#ff0000", color_mode="fixed"),
                    eraser=gr.Eraser(),
                    sources=("upload", "clipboard"),
                ),
                label=("Исходное изображение", "Source image"),
            )

            with gr.Row():
                prompt = localizer.bind(
                    gr.Textbox(
                        label=pick("prompt", lang),
                        placeholder=pick("prompt_placeholder", lang),
                        lines=3,
                        scale=8,
                    ),
                    label=("Промт", "Prompt"),
                )
                with gr.Column(scale=1, min_width=140):
                    run_button = localizer.bind(
                        gr.Button(pick("apply_edit", lang), variant="primary"),
                        value=("Применить правку", "Apply edit"),
                    )
                    stop_button = localizer.bind(
                        gr.Button(pick("stop", lang), variant="stop"), value=("Прервать", "Stop")
                    )

            with gr.Row():
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("boost", lang), value=False),
                    label=("AI буст", "AI boost"),
                )
                describe_button = localizer.bind(
                    gr.Button(pick("describe", lang)), value=("Описать изображение", "Describe image")
                )

            result = localizer.bind(
                gr.Gallery(label=pick("result", lang), columns=2, height=400, object_fit="contain", format="png"),
                label=("Результат", "Result"),
            )
            send_back = localizer.bind(
                gr.Button(pick("send_to_edit", lang)), value=("Отправить в редактор", "Send to editor")
            )

        with gr.Column(scale=1):
            mode = localizer.bind(
                gr.Radio(
                    choices=[(pick(_MODE_KEYS[key], lang), key) for key in _MODE_KEYS],
                    value=MASK_MASK,
                    label=pick("mask_mode", lang),
                ),
                label=("Режим области", "Region mode"),
            )
            quality = localizer.bind(
                gr.Radio(choices=list(presets.NAMES), value=studio.config.preset, label=pick("quality", lang)),
                label=("Качество", "Quality"),
            )
            status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=3),
                label=("Состояние", "Status"),
            )

            with gr.Accordion(pick("advanced", lang), open=False):
                grow = localizer.bind(
                    gr.Slider(0, 64, value=8, step=1, label=pick("mask_grow", lang)),
                    label=("Запас маски, пикселей", "Mask grow, pixels"),
                )
                feather = localizer.bind(
                    gr.Slider(0, 64, value=12, step=1, label=pick("mask_feather", lang)),
                    label=("Растушёвка, пикселей", "Feather, pixels"),
                )
                keep_outside = localizer.bind(
                    gr.Checkbox(
                        value=True, label=pick("keep_outside", lang), info=pick("keep_outside_info", lang)
                    ),
                    label=("Сохранять кадр вне маски", "Keep pixels outside the mask"),
                    info=(
                        "Склеивает результат с оригиналом: вне маски пиксели остаются исходными.",
                        "Blends the result with the original so pixels outside the mask stay untouched.",
                    ),
                )
                seed = localizer.bind(
                    gr.Number(value=-1, precision=0, label=pick("seed", lang)), label=("Сид", "Seed")
                )

            with gr.Accordion(pick("outpaint", lang), open=False):
                sides = localizer.bind(
                    gr.CheckboxGroup(
                        choices=[("←", "left"), ("→", "right"), ("↑", "top"), ("↓", "bottom")],
                        label=pick("outpaint_sides", lang),
                    ),
                    label=("Стороны", "Sides"),
                )
                amount = localizer.bind(
                    gr.Slider(0.1, 1.0, value=0.35, step=0.05, label=pick("outpaint_amount", lang)),
                    label=("Насколько расширить", "How far to expand"),
                )
                expand_button = localizer.bind(
                    gr.Button(pick("outpaint", lang)), value=("Расширить холст", "Outpaint")
                )

    # --- обработчики ---

    def expand_canvas(value, chosen_sides, ratio_amount):
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), MASK_MASK, "Сначала загрузите изображение"
        if not chosen_sides:
            return gr.update(), MASK_MASK, "Выберите хотя бы одну сторону"

        canvas, mask = outpaint.expand(source, outpaint.plan(source.size, chosen_sides, ratio_amount))

        # Новая площадь показывается пользователю как нарисованная область,
        # чтобы её было видно и можно было поправить кистью.
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        painted = Image.new("RGBA", canvas.size, (255, 0, 0, 255))
        layer.paste(painted, (0, 0), mask)

        return (
            {"background": canvas, "layers": [layer], "composite": None},
            MASK_MASK,
            f"Холст расширен до {canvas.size[0]}×{canvas.size[1]}",
        )

    def describe(value):
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), "Сначала загрузите изображение"
        text, message = studio.describe_image(source)
        return (text or gr.update()), message

    def run(
        value, prompt_text, use_boost, mode_value, quality_name,
        grow_value, feather_value, keep_value, seed_value,
        progress=gr.Progress(),
    ):
        source, mask = collect(value, mode_value)
        if source is None:
            return [], "Сначала загрузите изображение"

        effective, message = prompt_text, ""
        if use_boost:
            effective, _, message = studio.boost_prompt(
                prompt_text, boost_module.MODE_EDIT, [source]
            )

        request = GenerationRequest(
            prompt=effective or prompt_text,
            prompt_original=prompt_text,
            preset=presets.get(quality_name),
            aspect="1:1",  # при правке размеры наследуются от исходного изображения
            seed=int(seed_value),
            source=source,
            mask=mask,
            mask_mode=mode_value if mask is not None or mode_value == MASK_ANNOTATION else MASK_NONE,
            mask_grow=int(grow_value),
            mask_feather=int(feather_value),
            keep_outside=bool(keep_value),
        )

        def report(index: int, step: int, total: int) -> None:
            progress((step, total), desc="правка")

        produced = studio.generator.generate(request, progress=report)
        if not produced:
            return [], f"{message} Правка прервана".strip()

        paths = []
        for item in produced:
            destination = gallery.next_path(config.OUTPUT_DIR)
            metadata.save_png(item.image, destination, item.parameters)
            paths.append(str(destination))

        return paths, f"{message} Готово. {studio.memory_report()}".strip()

    def take_back(produced):
        if not produced:
            return gr.update(), "Нечего отправлять"
        first = produced[0]
        path = first[0] if isinstance(first, (list, tuple)) else first
        image = Image.open(path).convert("RGBA")
        return {"background": image, "layers": [], "composite": None}, "Результат перенесён в редактор"

    def stop():
        if studio.model_loaded:
            studio.generator.interrupt()
        return "Останавливаю…"

    expand_button.click(expand_canvas, [editor, sides, amount], [editor, mode, status])
    describe_button.click(describe, editor, [prompt, status])
    run_button.click(
        run,
        [editor, prompt, boost_enabled, mode, quality, grow, feather, keep_outside, seed],
        [result, status],
    )
    stop_button.click(stop, None, status, queue=False)
    send_back.click(take_back, result, [editor, status])

    return {"editor": editor, "result": result, "prompt": prompt, "status": status}
```

- [ ] **Step 4: Подключить вкладку в `app.py`**

В `build()` после вкладки генерации добавить:

```python
            with gr.Tab(pick("tab_edit", cfg.lang)) as edit_tab:
                edit = tab_edit.build(studio, localizer, generate)
```

и импорт `from . import tab_edit`. Также зарегистрировать заголовки вкладок в локализаторе:

```python
        localizer.bind(generate_tab, label=("Генерация", "Generate"))
        localizer.bind(edit_tab, label=("Редактирование", "Edit"))
```

- [ ] **Step 5: Запустить тесты**

Run: `.venv\Scripts\python -m pytest tests/test_edit_collect.py -q`
Expected: PASS, 7 тестов

- [ ] **Step 6: Проверить три режима вживую**

Run: `.\run.ps1 --preset LowQuality`

Проверить на фотографии человека:
1. **Режим «Маска»**: закрасить кистью волосы, промт `change the hair in the marked area to platinum blonde`. Результат: волосы изменились, остальной кадр совпадает с оригиналом.
2. Проверить сохранность численно:

```
.venv\Scripts\python -c "
from PIL import Image
import numpy as np
a = np.asarray(Image.open('оригинал.png').convert('RGBA'))
b = np.asarray(Image.open('user/outputs/<дата>/<файл>.png').convert('RGBA'))
print('доля изменённых пикселей:', float((a != b).any(axis=2).mean()))
"
```

Ожидание: доля заметно меньше единицы и примерно равна площади маски с растушёвкой.

3. **Режим «Аннотация»**: обвести две области разными цветами, промт `remove the object in the blue circle and change the shirt in the red circle to green`. Результат — обе правки применены.
4. **Режим «Точная область»**: закрасить мелкую деталь, проверить, что она проработана лучше, чем в режиме «Маска».
5. **Расширение холста**: выбрать «→», нажать «Расширить холст» — редактор показывает увеличенный холст с помеченной новой площадью; промт `continue the scene to the right`, применить правку.
6. **Описать изображение** — кнопка заполняет промт описанием.

- [ ] **Step 7: Коммит**

```bash
git add fooocus_qwen/ui/tab_edit.py fooocus_qwen/ui/app.py tests/test_edit_collect.py
git commit -m "feat: вкладка редактирования с тремя режимами области и расширением холста"
```

---

### Task 14: Интерфейс — галерея, настройки, восстановление параметров из PNG

**Files:**
- Create: `fooocus_qwen/ui/tab_gallery.py`, `fooocus_qwen/ui/tab_settings.py`
- Modify: `fooocus_qwen/ui/app.py`
- Create: `tests/test_restore.py`

**Interfaces:**
- Consumes: `metadata.read_png`, `gallery.recent`, `library`, `config.ENDPOINT_FILE`, `config.SYSTEM_PROMPT_DIR`.
- Produces:
  - `tab_gallery.restore_fields(parameters: dict | None) -> tuple` — значения для полей вкладки генерации
  - `tab_gallery.build(studio, localizer, generate_components: dict) -> dict`
  - `tab_settings.build(studio, localizer) -> dict`

- [ ] **Step 1: Написать падающий тест восстановления полей**

Создать `tests/test_restore.py`:

```python
"""Восстановление параметров генерации из метаданных PNG."""

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen.imaging import metadata
from fooocus_qwen.ui import tab_gallery

PARAMS = {
    "prompt": "кот в шляпе",
    "prompt_boosted": "a cat wearing a hat",
    "negative_prompt": "blurry",
    "styles": ["sai-anime"],
    "preset": "MaxQuality",
    "seed": 4242,
    "true_cfg_scale": 2.5,
    "width": 2048,
    "height": 2048,
}


def test_fields_come_back_in_the_declared_order():
    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields(PARAMS)
    assert prompt == "кот в шляпе"
    assert boosted == "a cat wearing a hat"
    assert negative == "blurry"
    assert styles == ["sai-anime"]
    assert preset == "MaxQuality"
    assert seed == 4242
    assert cfg == 2.5


def test_missing_parameters_fall_back_to_defaults():
    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields({})
    assert prompt == "" and boosted == "" and negative == ""
    assert styles == []
    assert preset == "MiddleQuality"
    assert seed == -1
    assert cfg == 1.0


def test_none_yields_defaults_too():
    assert tab_gallery.restore_fields(None)[0] == ""


def test_round_trip_through_a_real_png(tmp_path):
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "x.png", PARAMS)
    assert tab_gallery.restore_fields(metadata.read_png(path))[0] == "кот в шляпе"


def test_unknown_preset_falls_back():
    # Пресет мог называться иначе в старой сборке.
    assert tab_gallery.restore_fields({"preset": "Ultra"})[4] == "MiddleQuality"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `.venv\Scripts\python -m pytest tests/test_restore.py -q`
Expected: FAIL, `ImportError: cannot import name 'tab_gallery'`

- [ ] **Step 3: Написать `fooocus_qwen/ui/tab_gallery.py`**

```python
"""Вкладка галереи: история генераций и возврат к их параметрам.

Параметры лежат внутри самих PNG, поэтому история переживает и перезапуск, и
перенос файлов на другую машину — отдельной базы для этого не нужно.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from .. import config
from ..engine import presets
from ..imaging import metadata
from ..storage import gallery
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)


def restore_fields(parameters: dict | None) -> tuple:
    """Значения полей вкладки генерации в фиксированном порядке.

    Порядок: промт, переписанный промт, негатив, стили, пресет, сид, guidance.
    """
    data = parameters or {}
    preset = data.get("preset", presets.DEFAULT)
    if preset not in presets.PRESETS:
        preset = presets.DEFAULT

    return (
        data.get("prompt", ""),
        data.get("prompt_boosted", ""),
        data.get("negative_prompt", ""),
        list(data.get("styles", [])),
        preset,
        int(data.get("seed", -1)),
        float(data.get("true_cfg_scale", 1.0)),
    )


def _open_folder(path: Path) -> None:
    """Открывает каталог в файловом менеджере системы."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def build(studio, localizer: Localizer, generate_components: dict) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column(scale=3):
            history = localizer.bind(
                gr.Gallery(
                    label=pick("tab_gallery", lang),
                    columns=6,
                    height=560,
                    object_fit="contain",
                    value=[str(path) for path in gallery.recent(config.OUTPUT_DIR)],
                ),
                label=("Галерея", "Gallery"),
            )
        with gr.Column(scale=1):
            refresh = localizer.bind(gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh"))
            open_button = localizer.bind(
                gr.Button(pick("open_folder", lang)), value=("Открыть папку", "Open folder")
            )
            dropped = localizer.bind(
                gr.File(label=pick("restore_params", lang), file_types=[".png"]),
                label=("Восстановить параметры из PNG", "Restore parameters from PNG"),
            )
            restore_button = localizer.bind(
                gr.Button(pick("restore_params", lang), variant="primary"),
                value=("Восстановить параметры", "Restore parameters"),
            )
            details = localizer.bind(
                gr.JSON(label=pick("status", lang)), label=("Состояние", "Status")
            )

    selected = gr.State(None)

    def refresh_history():
        return [str(path) for path in gallery.recent(config.OUTPUT_DIR)]

    def on_select(event: gr.SelectData):
        path = Path(event.value["image"]["path"] if isinstance(event.value, dict) else event.value)
        return str(path), metadata.read_png(path) or {"сообщение": "в этом PNG нет наших параметров"}

    def open_outputs():
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _open_folder(config.OUTPUT_DIR)
        return {"открыт каталог": str(config.OUTPUT_DIR)}

    def restore(path, uploaded):
        source = Path(uploaded.name) if uploaded is not None else (Path(path) if path else None)
        if source is None:
            return (gr.update(),) * 7 + ({"сообщение": "выберите изображение или перетащите PNG"},)
        parameters = metadata.read_png(source)
        return restore_fields(parameters) + (parameters or {"сообщение": "параметры не найдены"},)

    refresh.click(refresh_history, None, history)
    open_button.click(open_outputs, None, details)
    history.select(on_select, None, [selected, details])
    restore_button.click(
        restore,
        [selected, dropped],
        [
            generate_components["prompt"],
            generate_components["boosted"],
            generate_components["negative"],
            generate_components["styles"],
            generate_components["quality"],
            generate_components["seed"],
            generate_components["cfg"],
            details,
        ],
    )

    return {"history": history, "details": details}
```

- [ ] **Step 4: Написать `fooocus_qwen/ui/tab_settings.py`**

```python
"""Вкладка настроек: адрес языковой модели, системные промты, состояние памяти.

Системные промты правятся прямо здесь и перечитываются при каждом обращении к
модели — переписыватель настраивается без перезапуска оболочки.
"""

from __future__ import annotations

import logging

import gradio as gr

from .. import config
from ..llm import LlmError
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)

_PROMPT_FILES = ("system_prompt_t2i.txt", "system_prompt_edit.txt", "system_prompt_describe.txt")


def build(studio, localizer: Localizer) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column():
            endpoint_text = localizer.bind(
                gr.Textbox(
                    label=pick("llm_endpoint", lang),
                    lines=5,
                    value=_read_endpoint(),
                ),
                label=("Адрес языковой модели", "Language model endpoint"),
            )
            with gr.Row():
                save_endpoint = localizer.bind(
                    gr.Button(pick("save_prompt", lang)), value=("Сохранить", "Save")
                )
                check = localizer.bind(
                    gr.Button(pick("llm_check", lang), variant="primary"),
                    value=("Проверить связь", "Check connection"),
                )
            endpoint_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

        with gr.Column():
            chosen_file = gr.Dropdown(choices=list(_PROMPT_FILES), value=_PROMPT_FILES[0], label="system prompt")
            prompt_text = gr.Textbox(lines=18, value=_read_prompt(_PROMPT_FILES[0]), show_label=False)
            save_prompt_button = localizer.bind(
                gr.Button(pick("save_prompt", lang)), value=("Сохранить", "Save")
            )
            prompt_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

    memory = localizer.bind(
        gr.Textbox(label=pick("status", lang), interactive=False, lines=2, value="модель ещё не загружена"),
        label=("Состояние", "Status"),
    )
    memory_refresh = localizer.bind(gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh"))

    def store_endpoint(text):
        config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
        return "Адрес сохранён"

    def check_connection():
        try:
            client = studio.llm_client()
        except (LlmError, ValueError, OSError) as error:
            return f"Нет связи: {error}"
        return f"Связь есть. Выбранная модель: {client.model or 'сервер не назвал ни одной'}"

    def load_prompt_file(name):
        return _read_prompt(name)

    def store_prompt_file(name, text):
        (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")
        return f"Файл {name} сохранён"

    save_endpoint.click(store_endpoint, endpoint_text, endpoint_status)
    check.click(check_connection, None, endpoint_status)
    chosen_file.change(load_prompt_file, chosen_file, prompt_text)
    save_prompt_button.click(store_prompt_file, [chosen_file, prompt_text], prompt_status)
    memory_refresh.click(studio.memory_report, None, memory)

    return {"memory": memory}


def _read_endpoint() -> str:
    try:
        return config.ENDPOINT_FILE.read_text(encoding="utf-8")
    except OSError:
        return "# Файл не найден. Укажите бэкенд, адрес и token=…\n"


def _read_prompt(name: str) -> str:
    try:
        return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return f"# Файл {name} не найден. Запустите tools/fetch_system_prompts.py\n"
```

- [ ] **Step 5: Подключить вкладки в `app.py`**

Добавить импорты `from . import tab_gallery, tab_settings` и внутри `gr.Tabs()`:

```python
            with gr.Tab(pick("tab_gallery", cfg.lang)) as gallery_tab:
                tab_gallery.build(studio, localizer, generate)

            with gr.Tab(pick("tab_settings", cfg.lang)) as settings_tab:
                tab_settings.build(studio, localizer)
```

и зарегистрировать их заголовки в локализаторе рядом с остальными.

- [ ] **Step 6: Запустить все тесты**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: PASS, 137 тестов

- [ ] **Step 7: Проверить вживую**

1. Сгенерировать изображение, открыть «Галерею» — оно там есть.
2. Выбрать его: справа показаны параметры; «Восстановить параметры» заполняет поля на вкладке генерации.
3. Перетащить PNG из проводника в поле восстановления — параметры читаются.
4. Перетащить чужой PNG — сообщение «параметры не найдены», без падения.
5. «Настройки»: «Проверить связь» — сообщение об успехе или внятная ошибка.
6. Отредактировать `system_prompt_t2i.txt` прямо в окне, сохранить, включить AI буст — правка подействовала без перезапуска.

- [ ] **Step 8: Коммит**

```bash
git add fooocus_qwen/ui tests/test_restore.py
git commit -m "feat: галерея с восстановлением параметров и вкладка настроек"
```

---

### Task 15: Дымовой прогон, документация и завершение ветки

**Files:**
- Create: `README.md`, `docs/ARCHITECTURE.md`, `docs/USAGE.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-09-21-fooocus-qwen-image-21-design.md` (отметить отступления)

**Interfaces:**
- Consumes: всё предыдущее.
- Produces: документацию и отчёт о дымовом прогоне.

- [ ] **Step 1: Прогнать полный сценарий**

Пройти по каждому пункту и записать результат в `docs/SMOKE.md` (создать файл, таблица «пункт — результат — замечание»):

1. `install.ps1` на чистом окружении (`-Recreate`) проходит до «Окружение готово».
2. `run.ps1` поднимает интерфейс, он открывается с другой машины сети.
3. Генерация из текста, все три пресета качества.
4. Генерация с текстом внутри изображения — надпись читаема.
5. Прозрачный RGBA: промт `This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.` — сохранённый PNG имеет альфа-канал.
6. Все семь соотношений сторон дают заявленные размеры.
7. Правка промтом без маски.
8. Правка по маске; вне маски пиксели не изменились.
9. Правка по аннотации: три области, три инструкции, один промт.
10. Правка точной области на мелкой детали.
11. Расширение холста на каждую из четырёх сторон и на две сразу.
12. Десять референсов: генерация не падает по памяти, подписи `<image1>`…`<image10>` на месте.
13. Один референс: подпись сообщает, что тег не нужен, промт без тега работает.
14. AI буст на русском промте возвращает английский текст и подставляет соотношение сторон.
15. «Описать изображение» заполняет промт.
16. Сохранение и загрузка пресета промта.
17. Восстановление параметров из PNG, включая перетаскивание файла.
18. Прерывание генерации кнопкой.
19. Переключение RU/EN на всех четырёх вкладках.
20. Повторная генерация с тем же промтом не увеличивает счётчик перестановок энкодера.

- [ ] **Step 2: Написать `docs/ARCHITECTURE.md`**

Разделы: слои и их границы; почему кэш встроен в `_get_qwen_prompt_embeds`, а не в `__call__`; политика резидентности и её числа по `docs/BENCHMARK.md`; порядок условных изображений и правило тегов; три режима области и гарантия сохранности; полная схема ключей метаданных PNG (перечислить все ключи из `Generator._parameters` с типами); контракт с diffusers и роль `assert_contract`.

- [ ] **Step 3: Написать `docs/USAGE.md` и `README.md`**

`README.md`: что это, требования (24 ГБ VRAM, 64 ГБ ОЗУ и больше), установка в две команды, запуск, ссылка на `docs/USAGE.md`, лицензионная оговорка — модель под Qwen Research License, использование некоммерческое.

`docs/USAGE.md`: сценарии по шагам с указанием, какой режим области когда выбирать; как ссылаться на референсы тегами; как настроить внешнюю LLM; почему негативный промт не действует при `true_cfg_scale = 1.0`; где лежат результаты.

- [ ] **Step 4: Зафиксировать отступления от спецификации**

В спецификации в разделе 7 заменить пункт «Клик по миниатюре вставляет тег в промт в позицию курсора» на фактическое поведение: Gradio не даёт доступа к позиции курсора в текстовом поле, поэтому миниатюры **подписаны** тегами, а вставка делается пользователем. Требование заказчика — подписи — выполнено.

Добавить в спецификацию раздел «Отступления», перечислив все расхождения, найденные при реализации.

- [ ] **Step 5: Обновить `CHANGELOG.md`**

Добавить запись за день реализации: что сделано, какие числа дал бенчмарк, какие отступления зафиксированы.

- [ ] **Step 6: Прогнать все тесты и линт на чистоту**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: PASS, все тесты

Run: `.venv\Scripts\python -m fooocus_qwen --selftest`
Expected: `Окружение готово.`

- [ ] **Step 7: Коммит и слияние ветки**

```bash
git add README.md docs CHANGELOG.md
git commit -m "docs: архитектура, руководство и отчёт о дымовом прогоне"
git checkout master
git merge --no-ff feature/studio -m "feat: оболочка Fooocus-Qwen-Image-2.1"
```

---

## Self-Review

Проверка плана против спецификации, выполнена после написания.

**Покрытие требований спецификации:**

| Требование | Задача |
|---|---|
| 1. Генерация из текста | 10, 12 |
| 2. Редактирование промтом | 10, 13 |
| 3. Редактирование по маске | 6, 10, 13 |
| 4. Сохранение и загрузка промтов | 7, 12 |
| 5. Внешняя LLM, AI буст, системный промт в файле | 2, 4, 12, 14 |
| 6. Сохранение в PNG | 7, 10 |
| 7. Галерея и метаданные PNG | 7, 14 |
| 8. Outpaint | 6, 13 |
| 9. До 10 референсов с тегами | 10, 12 |
| 10. Три пресета качества | 8, 11 |
| 11. Двуязычный интерфейс | 12 |
| Спец. 4.2 политика резидентности | 8, 9 |
| Спец. 4.4 соотношения сторон | 5 |
| Спец. 11 скрипты установки и запуска | 1 |
| Спец. 12.1 тесты без GPU | 2–10 |
| Спец. 12.2 бенчмарк | 11 |
| Спец. 12.3 дымовой прогон | 15 |

**Согласованность имён между задачами:** `QualityPreset` (8) используется в `GenerationRequest` (10) и в UI (12, 13); `masking.as_condition` (6) вызывается в `build_conditions` (10); `masking.refine`, `blend`, `stitch`, `region_box` (6) — в `Generator._prepare` и `_finish` (10); `aspect.FOLLOW_REFERENCE` (5) — в `resolve_size` (10) и в списке UI (12); `metadata.save_png` (7) — в 10, 12, 13; `gallery.next_path` (7) — там же; `library.list_prompts` (7) — в 12; `restore_fields` (14) отдаёт ровно те семь полей, которые объявлены в словаре компонентов вкладки генерации (12).

**Найдено и исправлено при проверке:**

- Вкладка генерации обязана возвращать компоненты `boosted` и `cfg`, иначе задача 14 не сможет их заполнить — добавлено в возвращаемый словарь задачи 12.
- Порядок полей `restore_fields` зафиксирован в докстринге и в тесте, иначе он разошёлся бы со списком выходов в задаче 14.
- `GenerationRequest.extras` объявлен изменяемым словарём и используется для передачи `region_box` между `_prepare` и `_finish`; `_replace` намеренно делает поверхностную копию, чтобы словарь оставался общим.
