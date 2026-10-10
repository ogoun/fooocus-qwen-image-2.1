"""Картинки API: из base64, поз и галереи — в ``PIL.Image``; результаты — в base64 и ссылки.

Путь галереи приходит от клиента, то есть от кого угодно, кто достучался до
порта. Принимается он только внутри каталога генераций (``config.OUTPUT_DIR``),
иначе ``gallery`` превратилось бы в чтение произвольного файла с диска — та
же граница, что у кисти правки (``ui/painter/payload.py``).
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .. import config
from ..imaging import masking
from ..poses import library
from .schemas import ImageInput, ImageRef

# Потолок одной картинки в base64: 4K PNG без сжатия — около 50 МБ; больше —
# уже не картинка, а попытка забить память сервера.
MAX_BASE64_CHARS = 80 * 1024 * 1024
# Потолок стороны: модель работает до 2K, исходник правки бывает крупнее,
# но 16K — уже не фотография.
MAX_SIDE = 16384


class ImageError(ValueError):
    """Картинку не прочитать или она указывает не туда; ответ — 400 или 404."""

    def __init__(self, message: str, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


def decode_base64(text: str) -> Image.Image:
    """PNG/JPEG/WebP из base64 или data URL; поворот по EXIF применяется."""
    if len(text) > MAX_BASE64_CHARS:
        raise ImageError("the image is too large")
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        raw = base64.b64decode(text, validate=False)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as error:
        raise ImageError(f"not a readable image: {error}") from error
    if max(image.size) > MAX_SIDE:
        raise ImageError(f"the image is larger than {MAX_SIDE} px on a side")
    return ImageOps.exif_transpose(image)


def output_path(relative: str) -> Path:
    """Файл внутри каталога генераций по пути из ответа API; наружу — ``ImageError``."""
    root = config.OUTPUT_DIR.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ImageError("the path is outside the outputs folder")
    if not candidate.is_file():
        raise ImageError(f"no such file: {relative}", not_found=True)
    return candidate


def relative_output(path: Path) -> str:
    """Путь картинки для ответа: от каталога генераций, через прямую косую."""
    return Path(path).resolve().relative_to(config.OUTPUT_DIR.resolve()).as_posix()


def pose_entry(name: str) -> library.PoseEntry:
    for entry in library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir()):
        if entry.name == name:
            return entry
    raise ImageError(f"no such pose: {name}", not_found=True)


def load(source: ImageInput) -> Image.Image:
    """Картинка запроса — строка base64 или ``ImageRef``."""
    ref = ImageRef(base64=source) if isinstance(source, str) else source
    if ref.base64:
        return decode_base64(ref.base64)
    if ref.pose:
        with Image.open(pose_entry(ref.pose).skeleton) as opened:
            return opened.convert("RGB")
    with Image.open(output_path(ref.gallery)) as opened:
        opened.load()
        return opened.copy()


def to_mask(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Маска клиента → чёрно-белая маска размера исходника.

    Принимаются оба привычных вида: чёрно-белая картинка (белое — править) и
    прозрачный слой с мазками (непрозрачное — править, как у кисти
    интерфейса). Слой узнаётся по альфе, в которой есть прозрачность.
    Граница — та же половина, что у кисти (``masking.MASK_THRESHOLD``).
    """
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
        if alpha.min() < 255:
            values = alpha
        else:
            values = np.asarray(image.convert("L"))
    else:
        values = np.asarray(image.convert("L"))
    return Image.fromarray(np.where(values >= masking.MASK_THRESHOLD, 255, 0).astype(np.uint8), mode="L")


def crop(image: Image.Image, box: tuple[int, int, int, int] | None) -> Image.Image:
    """Кадрирование по пикселям исходника, как кнопка «Кадрировать» у кисти."""
    if box is None:
        return image
    left, top, right, bottom = box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ImageError(f"crop {list(box)} does not fit the image {image.width}x{image.height}")
    return image.crop(box)


def encode_png(path: Path) -> str:
    """Готовый PNG с метаданными — как есть, без пересжатия."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
