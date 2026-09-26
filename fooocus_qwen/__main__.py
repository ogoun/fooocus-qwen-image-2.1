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

    # Производительность: выбор из user/settings.json и то, что ему нужно.
    from . import settings
    from .engine import attention, fetch

    chosen = settings.load()
    if chosen.precision == settings.PRECISION_INT8:
        if fetch.missing_extra(config.INT8_DIR, (fetch.INT8_FILE,)):
            problems.append("выбрана точность INT8, но нет её весов — запустите --fetch-model")
            print(f"[нет] точность INT8: нет {config.INT8_DIR / fetch.INT8_FILE}")
        else:
            print("[ок ] точность INT8, веса на месте")
    else:
        print("[ок ] точность bf16")
    if chosen.sage_attention:
        state = (
            "[ок ] SageAttention включён" if attention.sage_available()
            else "[--] SageAttention выбран, но не установлен — работаю штатным вниманием"
        )
        print(state)
    from .poses import detect

    if fetch.missing_extra(config.DWPOSE_DIR, detect.FILES):
        problems.append("нет весов распознавания позы (DWPose) — запустите --fetch-model")
        print(f"[нет] распознавание позы: нет весов DWPose в {config.DWPOSE_DIR}")
    else:
        print("[ок ] распознавание позы: веса DWPose на месте")
    turbo_ready = not fetch.missing_extra(config.TURBO_DIR, fetch.TURBO_FILES)
    print("[ок ] Turbo: веса на месте" if turbo_ready else "[--] Turbo: веса скачаются при первом выборе пресета")

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


def fetch_model() -> int:
    """Доводит веса до полного состава под выбранную точность. Зовётся установкой.

    При INT8 bf16-шарды трансформера (14 ГБ) не качаются: их место занимает
    INT8-трансформер Unsloth (7.3 ГБ). Вместе с моделью — веса распознавания
    позы (DWPose, 350 МБ, «Добавить позу»): без них первое распознавание
    ждало бы загрузки посреди работы.
    """
    from . import settings
    from .engine import fetch
    from .poses import detect

    int8 = settings.load().precision == settings.PRECISION_INT8
    try:
        downloaded = fetch.ensure_model(config.MODEL_DIR, include_transformer=not int8)
        if int8:
            downloaded = fetch.ensure_int8(config.INT8_DIR) or downloaded
        poses = fetch.ensure_files(config.DWPOSE_DIR, detect.REPO, detect.FILES)
    except fetch.ModelDownloadError as error:
        print(f"[нет] {error}")
        return 1
    except OSError as error:
        # Сеть, диск, права: причина человеку важнее типа исключения.
        print(f"[нет] не удалось скачать веса: {error}")
        return 1

    if downloaded:
        print(f"[ок ] веса скачаны: {config.MODEL_DIR}")
    else:
        print(f"[ок ] веса на месте: {config.MODEL_DIR}")
    print(f"[ок ] веса распознавания позы {'скачаны' if poses else 'на месте'}: {config.DWPOSE_DIR}")
    return 0


def setup_performance() -> int:
    """Спрашивает точность весов и SageAttention. Отказ отвечать — не ошибка."""
    from .engine import setup

    setup.configure()
    return 0


def setup_llm() -> int:
    """Спрашивает адрес и токен языковой модели.

    Отказ отвечать и отсутствие консоли — не ошибки: AI-буст промтов
    необязателен, без него работает всё остальное. Уронить установку на
    последнем шаге, когда зависимости уже поставлены, было бы худшим из
    возможных исходов.
    """
    from .llm import setup

    if not sys.stdin.isatty():
        print("  Пропускаю настройку языковой модели: установка идёт без консоли.")
        print(f"  Адрес можно задать позже в {config.ENDPOINT_FILE.name} или во вкладке «Настройки».")
        return 0

    setup.configure(config.ENDPOINT_FILE)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = config.build_parser().parse_args(argv)
    logging_setup.setup_logging(args.verbose)
    config.ensure_directories()

    if args.fetch_model:
        return fetch_model()

    if args.setup_llm:
        return setup_llm()

    if args.setup_performance:
        return setup_performance()

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
