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
