"""Библиотека поз: каталог openposes.com и позы, добавленные пользователем.

Поза — четыре файла с общим именем:

* ``<имя>.json`` — точки OpenPose (``skeleton.parse``);
* ``<имя>.png`` — скелет, тот самый, что ложится в ячейку референса;
* ``<имя>.jpg`` — плитка: иллюстрация героини в этой позе (1024×1024);
* ``<имя>.thumb.jpg`` — уменьшенная плитка для окна выбора.

**Каталог** (``config.POSE_LIBRARY_DIR``, ``resources/poses/catalog``) —
позы openposes.com с одной моделью, Эммой Уотсон; лежит в репозитории, как
стили и системные промты. Собран ``tools/fetch_poses.py``: архив скелетов и
точек ``poses.zip`` и плитки ``poses/emma_watson/jpg/<имя>.jpg`` из их
хранилища; тем же инструментом каталог обновляется.

**Свои позы** (``config.user_pose_dir()``, ``user/outputs/poses``) —
распознанные на фотографиях, данные пользователя, рядом с его генерациями.
Скелет сохраняется сразу, плитка — когда её нарисует Qwen-Image: до этого
в окне стоит сам скелет, и позой уже можно пользоваться.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import skeleton

LOGGER = logging.getLogger(__name__)

STORAGE = "https://openposes-storage.s3.ca-central-1.amazonaws.com"
ARCHIVE_URL = f"{STORAGE}/poses.zip"
MODEL = "emma_watson"
TILE_URL = STORAGE + "/poses/" + MODEL + "/jpg/{name}.jpg"

THUMB_SIDE = 320
CUSTOM_PREFIX = "custom_"


@dataclass(frozen=True)
class PoseEntry:
    name: str
    folder: Path
    custom: bool

    @property
    def keypoints(self) -> Path:
        return self.folder / f"{self.name}.json"

    @property
    def skeleton(self) -> Path:
        return self.folder / f"{self.name}.png"

    @property
    def tile(self) -> Path:
        return self.folder / f"{self.name}.jpg"

    @property
    def thumb(self) -> Path:
        return self.folder / f"{self.name}.thumb.jpg"

    def preview(self) -> Path:
        """Что показать в окне выбора: плитку, а пока её нет — скелет."""
        return self.thumb if self.thumb.exists() else self.skeleton


def _entries(folder: Path, custom: bool) -> list[PoseEntry]:
    if not folder.is_dir():
        return []
    names = sorted(path.stem for path in folder.glob("*.json"))
    entries = [PoseEntry(name, folder, custom) for name in names]
    return [entry for entry in entries if entry.skeleton.exists()]


def list_poses(catalog: Path, user: Path) -> list[PoseEntry]:
    """Каталог по имени, затем свои позы в порядке добавления."""
    return _entries(catalog, custom=False) + _entries(user, custom=True)


def save_thumb(tile: Image.Image, destination: Path) -> None:
    thumb = tile.convert("RGB")
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    thumb.save(destination, quality=88)


def _download(url: str, timeout: float = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Fooocus-Qwen-Image"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_catalog(
    catalog: Path,
    archive: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    downloader: Callable[[str], bytes] = _download,
) -> int:
    """Скачивает недостающее в каталог; возвращает число поз в каталоге.

    ``archive`` — уже скачанный ``poses.zip`` (не качать его заново). Плитки
    качаются параллельно и только недостающие: прерванная загрузка
    продолжается, а не начинается сначала.
    """
    catalog.mkdir(parents=True, exist_ok=True)
    if not any(catalog.glob("*.json")):
        data = archive.read_bytes() if archive else downloader(ARCHIVE_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            for member in bundle.namelist():
                path = Path(member)
                if path.suffix in (".json", ".png") and path.parent == Path("."):
                    (catalog / path.name).write_bytes(bundle.read(member))

    entries = _entries(catalog, custom=False)
    missing = [entry for entry in entries if not entry.thumb.exists()]

    def one(entry: PoseEntry) -> None:
        if not entry.tile.exists():
            data = downloader(TILE_URL.format(name=entry.name))
            partial = entry.tile.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(entry.tile)
        save_thumb(Image.open(entry.tile), entry.thumb)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(one, missing):
            done += 1
            if progress:
                progress(done, len(missing))
    return len(entries)


def add_custom(user: Path, pose: skeleton.Pose) -> PoseEntry:
    """Сохраняет распознанную позу; плитки у неё пока нет."""
    user.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name, suffix = f"{CUSTOM_PREFIX}{stamp}", 1
    while (user / f"{name}.json").exists():
        suffix += 1
        name = f"{CUSTOM_PREFIX}{stamp}_{suffix}"
    entry = PoseEntry(name, user, custom=True)
    skeleton.render(pose).save(entry.skeleton)
    entry.keypoints.write_text(pose.to_json(), encoding="utf-8")
    return entry


def set_tile(entry: PoseEntry, tile: Image.Image) -> None:
    tile.convert("RGB").save(entry.tile, quality=92)
    save_thumb(tile, entry.thumb)
