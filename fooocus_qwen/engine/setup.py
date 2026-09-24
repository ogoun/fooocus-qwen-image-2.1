"""Опрос о производительности при установке: точность весов и SageAttention.

Два вопроса, оба с разумным ответом по умолчанию, и оба можно поменять
потом во вкладке «Настройки»:

* **точность трансформера** — bf16 (исходные веса, 13.3 ГиБ видеопамяти)
  или INT8 (веса Unsloth, 6.8 ГиБ, скорость почти та же). От ответа
  зависит, что качает следующий шаг установки: при INT8 bf16-шарды
  трансформера (14 ГБ) не нужны вовсе;
* **SageAttention** — внимание на −15…25 % быстрее. Пакет необязательный и
  ставится отдельно: под Windows — готовая сборка
  (github.com/woct0rdho/SageAttention) плюс ``triton-windows`` той версии,
  что совместима с установленным torch.

Отказ отвечать — полноправный ответ: остаётся текущий выбор, а если его
не было — bf16 без SageAttention, то есть поведение до появления этих
настроек.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .. import settings as settings_module

LOGGER = logging.getLogger(__name__)

SAGE_RELEASE = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/"
# Сборки выпуска v2.2.0-windows.post6: для torch 2.10 и новее, CUDA 12.8 и 13.0.
SAGE_WHEELS = {
    "cu128": "sageattention-2.2.0+cu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
    "cu130": "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl",
}
SAGE_SOURCE = "git+https://github.com/thu-ml/SageAttention.git"


def configure(
    path: Path | None = None,
    ask: Callable[[str], str] = input,
    out: Callable[..., None] = print,
    install_sage: Callable[[Callable[..., None]], bool] | None = None,
    sage_available: Callable[[], bool] | None = None,
) -> settings_module.Settings:
    """Задаёт оба вопроса, сохраняет ответы и возвращает итоговые настройки."""
    ask = _forgiving(ask)
    install_sage = install_sage or install_sage_attention
    sage_available = sage_available or _sage_importable
    current = settings_module.load(path)

    out("  Точность весов трансформера:")
    out("    1 — bf16: исходная точность, 13.3 ГиБ видеопамяти, веса ~33 ГБ")
    out("    2 — INT8: 6.8 ГиБ видеопамяти, скорость почти та же, веса ~26 ГБ")
    default = "2" if current.precision == settings_module.PRECISION_INT8 else "1"
    answer = ask(f"  Выбор (Enter — {default}): ").strip() or default
    precision = settings_module.PRECISION_INT8 if answer == "2" else settings_module.PRECISION_BF16
    if answer not in ("1", "2"):
        out(f"  Не понял «{answer}», оставляю {precision}")

    out("  SageAttention ускоряет генерацию на 15–25 %, пакет ставится отдельно.")
    default_sage = "да" if current.sage_attention else "нет"
    reply = ask(f"  Установить и включить SageAttention? [да/нет] (Enter — {default_sage}): ").strip().lower()
    wants_sage = current.sage_attention if not reply else reply in ("да", "д", "yes", "y")

    if wants_sage and not sage_available():
        wants_sage = install_sage(out)

    chosen = settings_module.Settings(precision=precision, sage_attention=wants_sage)
    settings_module.save(chosen, path)
    out(f"  Записано: точность {precision}, SageAttention {'включён' if wants_sage else 'выключен'}")
    return chosen


def install_sage_attention(out: Callable[..., None] = print, python: str | None = None) -> bool:
    """Ставит SageAttention и Triton в текущее окружение. ``True`` — удалось.

    Неудача не роняет установку: SageAttention — ускорение, а не условие
    работы. Причина печатается, приложение работает штатным вниманием.
    """
    python = python or sys.executable
    plan = sage_install_plan()
    if plan is None:
        return False
    if isinstance(plan, str):
        out(f"  {plan}")
        return False
    for step in plan:
        out(f"  pip install {' '.join(step)}")
        result = subprocess.run([python, "-m", "pip", "install", *step], check=False)
        if result.returncode != 0:
            out("  SageAttention не установился; работаю штатным вниманием, это не мешает остальному.")
            return False
    ok = subprocess.run(
        [python, "-c", "from diffusers.models import attention_dispatch as d; raise SystemExit(0 if d._CAN_USE_SAGE_ATTN else 1)"],
        check=False,
    ).returncode == 0
    out("  SageAttention установлен" if ok else "  SageAttention установлен, но diffusers его не видит — остаюсь на штатном")
    return ok


def sage_install_plan(torch_version: str | None = None, cuda: str | None = None, platform: str | None = None):
    """Что ставить: список аргументов pip по шагам или строка-объяснение, почему нечего.

    Разделено с установкой, чтобы подбор сборки проверялся тестами без сети.
    """
    platform = platform or sys.platform
    if torch_version is None or cuda is None:
        try:
            import torch
        except ImportError:
            return "torch не установлен — SageAttention ставить не к чему."
        torch_version = torch_version or torch.__version__
        cuda = cuda if cuda is not None else (torch.version.cuda or "")

    if not platform.startswith("win"):
        return (
            "Готовых сборок SageAttention 2 под Linux нет; соберите из исходников: "
            f"pip install {SAGE_SOURCE} (нужен CUDA Toolkit). Пока — штатное внимание."
        )
    major_minor = _major_minor(torch_version)
    if major_minor is None or major_minor < (2, 10):
        return f"Сборки SageAttention под torch {torch_version} нет (нужен 2.10 и новее)."
    cuda_tag = "cu" + cuda.replace(".", "")
    wheel = SAGE_WHEELS.get(cuda_tag)
    if wheel is None:
        return f"Сборки SageAttention под CUDA {cuda} нет (есть под 12.8 и 13.0)."
    # triton-windows 3.N работает с torch 2.(N+4): 3.6 — 2.10, 3.7 — 2.11.
    triton_minor = major_minor[1] - 4
    triton = f"triton-windows>=3.{triton_minor},<3.{triton_minor + 1}"
    return [[triton], [SAGE_RELEASE + wheel]]


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        major, minor = version.split("+")[0].split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def _sage_importable() -> bool:
    from . import attention

    return attention.sage_available()


def _forgiving(ask: Callable[[str], str]) -> Callable[[str], str]:
    """Конец ввода и Ctrl+C — «оставить как есть» (см. ``llm/setup.py``)."""

    def guarded(question: str) -> str:
        try:
            return ask(question)
        except (EOFError, KeyboardInterrupt):
            return ""

    return guarded
