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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)

REPO_ID = "Qwen/Qwen-Image-2.1"
MARKER = "model_index.json"
APPROXIMATE_SIZE_GB = 33


class ModelDownloadError(RuntimeError):
    """Загрузка закончилась, а весов на месте всё равно нет."""


def missing_files(model_dir: Path) -> list[str]:
    """Возвращает пути недостающих файлов относительно каталога модели.

    Пустой список означает, что модель на месте целиком. Файл нулевой длины
    считается отсутствующим: такой остаётся от оборванной записи и весами не
    является.
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
        for shard in sorted(set(weight_map.values())):
            if not _present(folder / shard):
                missing.append((folder / shard).relative_to(model_dir).as_posix())

    # VAE хранится одним файлом и потому индекса о себе не оставляет —
    # без этой проверки его пропажа прошла бы незамеченной до первой генерации.
    single = model_dir / "vae" / "diffusion_pytorch_model.safetensors"
    if not _present(single):
        missing.append(single.relative_to(model_dir).as_posix())

    return missing


def is_complete(model_dir: Path) -> bool:
    return not missing_files(model_dir)


def ensure_model(
    model_dir: Path,
    repo_id: str = REPO_ID,
    downloader: Callable[..., object] | None = None,
) -> bool:
    """Доводит каталог весов до полного состава.

    Возвращает ``True``, если что-то качалось, и ``False``, если всё было на
    месте. Неполнота после загрузки — ошибка, а не повод продолжить: молча
    вернуть «готово» с половиной шардов значит перенести отказ на первую
    генерацию, где он обойдётся дороже.
    """
    model_dir = Path(model_dir)
    missing = missing_files(model_dir)
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
    download(repo_id=repo_id, local_dir=model_dir)

    still_missing = missing_files(model_dir)
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


def _snapshot_download(*, repo_id: str, local_dir: Path) -> object:
    """Настоящая загрузка. Вынесена отдельно, чтобы тесты её не звали.

    ``.git`` репозитория модели не нужен: он удваивает объём, храня в LFS
    вторую копию каждого шарда.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=[".git*"],
        max_workers=4,
    )
