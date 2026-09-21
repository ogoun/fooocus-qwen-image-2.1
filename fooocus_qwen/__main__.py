"""Точка входа. По умолчанию поднимает веб-интерфейс, с --selftest проверяет окружение."""

from __future__ import annotations

import sys

from . import config, logging_setup


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


def main(argv: list[str] | None = None) -> int:
    args = config.build_parser().parse_args(argv)
    logging_setup.setup_logging(args.verbose)
    config.ensure_directories()

    if args.selftest:
        return selftest()

    if args.prompt:
        return generate_once(args)

    # Модуль ui.app появляется в задаче 12; до неё запуск без --selftest и без
    # --prompt упадёт на этом импорте, и это правильное поведение: интерфейса ещё нет.
    from .ui.app import launch

    cfg = config.parse_args(argv)
    launch(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
