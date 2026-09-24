"""Веса модели: проверка наличия и загрузка с Hugging Face.

Тридцать три гигабайта качаются один раз, и вопрос «а всё ли скачалось»
здесь не праздный: закачку прерывают, диск кончается, сеть рвётся. Поэтому
полнота проверяется не по каталогу и даже не по ``model_index.json``, а по
тем файлам, которые модель перечисляет в собственных индексах шардов
(``*.safetensors.index.json``). Список обязательных файлов не зашит в код —
его сообщает сама модель, и он останется верным, когда её состав изменится.

Загрузку делает ``huggingface_hub.snapshot_download``: он докачивает
недостающее, проверяет контрольные суммы и умеет продолжить прерванное. Тем
же путём (``local_dir``) файлы ложатся прямо в каталог проекта, а не в общий
кэш Hugging Face — иначе те же тридцать три гигабайта легли бы на диск дважды.

Кроме основной модели есть два дополнительных набора весов, каждый в своём
каталоге и каждый качается по требованию:

* **INT8-трансформер** (``unsloth/Qwen-Image-2.1-FP8``, файл
  ``Qwen-Image-2.1-INT8.safetensors``, 7.3 ГБ) — замена bf16-трансформера.
  Кто выбрал INT8, bf16-шарды трансформера (14 ГБ) не качает вовсе:
  ``include_transformer=False``.
* **Адаптер turbo** (``Viggle/Qwen-Image-2.1-viggle-turbo``, 1.3 ГБ) —
  дистиллят на 6 шагов, нужен пресету Turbo.

Прогресс скачивания рисует ``tqdm`` внутри ``huggingface_hub``: в консоли
установки он виден как есть, а интерфейс подхватывает его через
``gr.Progress(track_tqdm=True)``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger(__name__)

REPO_ID = "Qwen/Qwen-Image-2.1"
MARKER = "model_index.json"
APPROXIMATE_SIZE_GB = 33
# Шарды bf16-трансформера: без них модель работает на INT8-трансформере.
# Конфигурация и индекс трансформера качаются всегда — по ним строится модель.
TRANSFORMER_SHARDS = "transformer/*.safetensors"

INT8_REPO = "unsloth/Qwen-Image-2.1-FP8"
INT8_FILE = "Qwen-Image-2.1-INT8.safetensors"

TURBO_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
TURBO_LORA = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
TURBO_SCHEDULER = "scheduler/scheduler_config.json"
TURBO_FILES = (TURBO_LORA, TURBO_SCHEDULER)


class ModelDownloadError(RuntimeError):
    """Загрузка закончилась, а весов на месте всё равно нет."""


def missing_files(model_dir: Path, include_transformer: bool = True) -> list[str]:
    """Возвращает пути недостающих файлов относительно каталога модели.

    Пустой список означает, что модель на месте целиком. Файл нулевой длины
    считается отсутствующим: такой остаётся от оборванной записи и весами не
    является. ``include_transformer=False`` — шарды bf16-трансформера не
    нужны (работа на INT8), их отсутствие не считается.
    """
    model_dir = Path(model_dir)
    if not _present(model_dir / MARKER):
        return [MARKER]

    missing: list[str] = []
    for index_path in sorted(model_dir.glob("*/*.safetensors.index.json")):
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError):
            missing.append(index_path.relative_to(model_dir).as_posix())
            continue
        folder = index_path.parent
        if not include_transformer and folder.name == "transformer":
            continue
        for shard in sorted(set(weight_map.values())):
            if not _present(folder / shard):
                missing.append((folder / shard).relative_to(model_dir).as_posix())

    # VAE хранится одним файлом и потому индекса о себе не оставляет —
    # без этой проверки его пропажа прошла бы незамеченной до первой генерации.
    single = model_dir / "vae" / "diffusion_pytorch_model.safetensors"
    if not _present(single):
        missing.append(single.relative_to(model_dir).as_posix())

    return missing


def is_complete(model_dir: Path, include_transformer: bool = True) -> bool:
    return not missing_files(model_dir, include_transformer)


def ensure_model(
    model_dir: Path,
    repo_id: str = REPO_ID,
    downloader: Callable[..., object] | None = None,
    include_transformer: bool = True,
) -> bool:
    """Доводит каталог весов до полного состава.

    Возвращает ``True``, если что-то качалось, и ``False``, если всё было на
    месте. Неполнота после загрузки — ошибка, а не повод продолжить: молча
    вернуть «готово» с половиной шардов значит перенести отказ на первую
    генерацию, где он обойдётся дороже.
    """
    model_dir = Path(model_dir)
    missing = missing_files(model_dir, include_transformer)
    if not missing:
        LOGGER.info("Веса на месте: %s", model_dir)
        return False

    LOGGER.info(
        "Не хватает файлов весов (%d, первый — %s). Качаю %s в %s, это примерно %d ГБ.",
        len(missing),
        missing[0],
        repo_id,
        model_dir,
        APPROXIMATE_SIZE_GB,
    )
    download = downloader or _snapshot_download
    model_dir.mkdir(parents=True, exist_ok=True)
    ignore = [] if include_transformer else [TRANSFORMER_SHARDS]
    download(repo_id=repo_id, local_dir=model_dir, ignore=ignore)

    still_missing = missing_files(model_dir, include_transformer)
    if still_missing:
        raise ModelDownloadError(
            "Загрузка весов не довела дело до конца, не хватает "
            f"{len(still_missing)} файлов, первый — {still_missing[0]}. "
            f"Повторите установку: докачается только недостающее."
        )
    LOGGER.info("Веса скачаны: %s", model_dir)
    return True


def _present(path: Path) -> bool:
    """Файл есть и он не пуст."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _snapshot_download(*, repo_id: str, local_dir: Path, ignore: list[str] | None = None) -> object:
    """Настоящая загрузка. Вынесена отдельно, чтобы тесты её не звали.

    ``.git`` репозитория модели не нужен: он удваивает объём, храня в LFS
    вторую копию каждого шарда.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=[".git*", *(ignore or [])],
        max_workers=4,
    )


# --- дополнительные веса: INT8-трансформер и адаптер turbo ---------------------


def missing_extra(directory: Path, files: tuple[str, ...]) -> list[str]:
    """Недостающие файлы набора — относительные пути внутри каталога."""
    return [name for name in files if not _present(Path(directory) / name)]


def ensure_files(
    directory: Path,
    repo_id: str,
    files: tuple[str, ...],
    downloader: Callable[..., object] | None = None,
) -> bool:
    """Докачивает недостающие файлы набора. ``True`` — что-то качалось.

    Качаются только недостающие файлы: адаптер turbo, скачанный наполовину,
    не заставит заново тянуть готовую конфигурацию планировщика, и наоборот.
    """
    directory = Path(directory)
    missing = missing_extra(directory, files)
    if not missing:
        return False
    LOGGER.info("Качаю %s из %s в %s", ", ".join(missing), repo_id, directory)
    directory.mkdir(parents=True, exist_ok=True)
    download = downloader or _file_download
    for name in missing:
        download(repo_id=repo_id, filename=name, local_dir=directory)
    still_missing = missing_extra(directory, files)
    if still_missing:
        raise ModelDownloadError(
            f"Загрузка не довела дело до конца, не хватает {', '.join(still_missing)}. "
            "Повторите: докачается только недостающее."
        )
    return True


def ensure_int8(directory: Path, downloader: Callable[..., object] | None = None) -> bool:
    return ensure_files(directory, INT8_REPO, (INT8_FILE,), downloader)


def ensure_turbo(directory: Path, downloader: Callable[..., object] | None = None) -> bool:
    return ensure_files(directory, TURBO_REPO, TURBO_FILES, downloader)


def _file_download(*, repo_id: str, filename: str, local_dir: Path) -> object:
    """Один файл репозитория — прямо в каталог, без общего кэша Hugging Face."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
