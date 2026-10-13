"""Форматы картинок на входе: всё, что умеют декодеры, — в 8-битный sRGB.

Браузер показывает и Gradio читает немногое: PNG, JPEG, WebP, GIF, BMP.
Фото с телефона (HEIC), камеры (RAW), из редакторов (TIFF 16 бит, PSD, CMYK
с профилем), рендеров (EXR, HDR) и новые форматы (AVIF, JPEG XL) до модели
не доходили: Chrome не отдаёт такой файл полю Gradio (MIME-тип не
``image/*`` — перетаскивание молча отбрасывалось), а Pillow без модулей не
открывает. Здесь они превращаются в обычную картинку — один раз, на входе
(``ui/image_import.py`` для интерфейса, ``api/images.py`` для API).

Декодеры:

* Pillow — TIFF, PSD (сведённое изображение), TGA, ICO/ICNS, DDS, JPEG 2000,
  QOI, AVIF, PCX, PNM, SGI и др.;
* ``pillow-heif`` — HEIC/HEIF (iPhone, Android): libheif сам применяет
  поворот кадра и отдаёт 8 бит;
* ``pillow-jxl-plugin`` — JPEG XL;
* ``rawpy`` (LibRaw) — RAW камер: баланс белого камеры, sRGB, поворот;
* ``OpenEXR`` — EXR (у колеса OpenCV кодек EXR не собран);
* OpenCV — Radiance HDR и запасной путь для того, что Pillow не узнал
  (PFM, Sun Raster).

EXR и HDR — линейный свет: обрезка по белому, как у просмотрщиков при
экспозиции 0, и кривая sRGB.

Порядок разбора — по содержимому, а не по имени, кроме RAW: большинство RAW
внутри — TIFF, и Pillow открыл бы встроенную миниатюру вместо снимка.

Нормализация одна для всех: поворот по EXIF, профиль ICC → sRGB (фото
iPhone — в Display P3, без перевода цвета тускнеют), 16 бит и плавающая
точка → 8 бит, альфа остаётся, только если в ней есть прозрачность.
Многокадровые (GIF, TIFF, серии HEIC) — первый, основной кадр; у ICO Pillow
сам берёт самый крупный.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)

# Показывает и браузер, и Pillow — такие файлы идут как есть.
NATIVE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif", ".bmp")

RAW_EXTENSIONS: tuple[str, ...] = (
    ".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf",
    ".rw2", ".rwl", ".pef", ".srw", ".x3f", ".3fr", ".fff", ".erf", ".kdc", ".dcr", ".mrw",
    ".mos", ".mef", ".iiq", ".raw",
)
HDR_EXTENSIONS: tuple[str, ...] = (".exr", ".hdr", ".pic")

# Требуют перевода на сервере. Список — для поля выбора файла и для
# перехватчика в браузере; сам разбор от расширения не зависит (кроме RAW).
CONVERTED_EXTENSIONS: tuple[str, ...] = (
    ".heic", ".heif", ".hif", ".avif", ".jxl",
    ".tif", ".tiff", ".psd", ".tga", ".ico", ".cur", ".icns", ".dds",
    ".jp2", ".j2k", ".jpx", ".jpf", ".qoi", ".pcx", ".ppm", ".pgm", ".pbm", ".pnm",
    ".sgi", ".rgb", ".rgba", ".bw", ".xbm", ".xpm", ".dib", ".blp", ".ras", ".sr", ".pfm",
    ".svg",  # SVG растеризует браузер, сервер его не читает (см. image_import.js)
    *HDR_EXTENSIONS, *RAW_EXTENSIONS,
)

# Предел пикселей — защита от «бомб» сжатия; 200 Мп с запасом покрывают
# среднеформатные RAW (100 Мп) и панорамы.
MAX_PIXELS = 200_000_000

_EXR_MAGIC = b"\x76\x2f\x31\x01"
_HDR_MAGIC = (b"#?RADIANCE", b"#?RGBE")
_registered = False
# Расширение → пакет, которого не нашлось: ошибка назовёт, что поставить.
_missing: dict[str, str] = {}


class FormatError(ValueError):
    """Файл не картинка или формат не поддержан; текст — для человека."""


class TooLarge(FormatError):
    """Картинка прочитана (или узнана), но больше предела — пробовать другие декодеры незачем."""


def register() -> None:
    """Подключает к Pillow HEIF и JPEG XL; повторный вызов ничего не делает.

    Нет пакета — формат просто не откроется, с понятной ошибкой: отказ
    одного модуля не должен ронять запуск.
    """
    global _registered
    if _registered:
        return
    _registered = True
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, MAX_PIXELS // 2)
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        _missing.update(dict.fromkeys((".heic", ".heif", ".hif"), "pillow-heif"))
        LOGGER.warning("pillow-heif is not installed: HEIC/HEIF images cannot be opened")
    try:
        import pillow_jxl  # noqa: F401 — регистрирует открыватель при импорте
    except ImportError:
        _missing[".jxl"] = "pillow-jxl-plugin"
        LOGGER.warning("pillow-jxl-plugin is not installed: JPEG XL images cannot be opened")


def needs_conversion(name: str) -> bool:
    """Нужно ли переводить файл с таким именем перед показом в браузере."""
    return Path(name).suffix.lower() not in NATIVE_EXTENSIONS


def decode(data: bytes, name: str = "") -> Image.Image:
    """Картинка любого поддержанного формата → RGB или RGBA, 8 бит, sRGB, повёрнутая."""
    register()
    if not data:
        raise FormatError("the file is empty")
    suffix = Path(name).suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return _finish(_decode_raw(data))
    if data.startswith(_EXR_MAGIC):
        return _finish(_decode_exr(data))
    if suffix in HDR_EXTENSIONS or data.startswith(_HDR_MAGIC):
        return _finish(_decode_opencv(data))
    # Pillow, затем RAW без привычного расширения, затем всё, что знает
    # OpenCV. «Слишком большая» — не повод пробовать следующий: формат узнан.
    for decoder in (_decode_pillow, _decode_raw, _decode_opencv):
        try:
            image = decoder(data)
        except TooLarge:
            raise
        except FormatError:
            continue
        return _finish(image)
    if suffix in _missing:
        raise FormatError(f"{suffix} needs the {_missing[suffix]} package (pip install -r requirements.txt)")
    kind = suffix or "this file"
    raise FormatError(f"cannot read {kind}: not an image or an unsupported format")


def to_png(image: Image.Image) -> bytes:
    """PNG без потерь; сжатие слабое — файл идёт по локальной сети, а не в архив."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _decode_pillow(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.seek(0)
        image.load()
    except Image.DecompressionBombError as error:
        raise TooLarge(f"the image is too large: {error}") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise FormatError(str(error)) from error
    except Exception as error:  # noqa: BLE001 — модули форматов (JPEG XL) на битом файле бросают что угодно
        raise FormatError(f"{type(error).__name__}: {error}") from error
    image = ImageOps.exif_transpose(image)
    return _to_srgb(image)


def _decode_raw(data: bytes) -> Image.Image:
    try:
        import rawpy
    except ImportError as error:
        raise FormatError("camera RAW needs the rawpy package") from error
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            # Баланс белого камеры и sRGB — как снимок выглядел в камере;
            # поворот LibRaw применяет сам (флаг ориентации из файла).
            rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, output_bps=8)
    except (rawpy.LibRawError, OSError, ValueError) as error:
        raise FormatError(f"not a camera RAW: {error}") from error
    return Image.fromarray(rgb)


def _decode_exr(data: bytes) -> Image.Image:
    try:
        import OpenEXR
    except ImportError as error:
        raise FormatError("EXR needs the OpenEXR package") from error
    try:
        with OpenEXR.File(io.BytesIO(data)) as exr:
            channels = exr.channels()
            # Первая часть файла; RGB(A) собран библиотекой, иначе — первый
            # канал как серый (глубина, маски).
            for name in ("RGBA", "RGB"):
                if name in channels:
                    pixels = channels[name].pixels
                    break
            else:
                first = next(iter(channels.values()), None)
                if first is None:
                    raise FormatError("EXR without channels")
                pixels = first.pixels
    except (RuntimeError, OSError, ValueError) as error:
        raise FormatError(f"not a readable EXR: {error}") from error
    return _from_array(np.asarray(pixels))


def _decode_opencv(data: bytes) -> Image.Image:
    import cv2

    try:
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise FormatError(f"OpenCV cannot decode it: {error}") from error
    if array is None:
        raise FormatError("OpenCV cannot decode it")
    if array.ndim == 2:
        array = array[:, :, None]
    channels = array.shape[2]
    if channels >= 3:
        order = [2, 1, 0, 3] if channels == 4 else [2, 1, 0]
        array = array[:, :, order]
    return _from_array(array)


def _from_array(array: np.ndarray) -> Image.Image:
    """Массив H×W×C (RGB порядок) любого типа → картинка 8 бит."""
    if array.ndim == 2:
        array = array[:, :, None]
    if np.issubdtype(array.dtype, np.floating):
        # EXR/HDR — линейный свет: обрезка по белому (как экспозиция 0 у
        # просмотрщиков) и кривая sRGB.
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        color = np.clip(array[:, :, :3] if array.shape[2] >= 3 else array, 0.0, 1.0)
        color = np.where(color <= 0.0031308, color * 12.92, 1.055 * np.power(color, 1 / 2.4) - 0.055)
        if array.shape[2] == 4:
            color = np.concatenate([color, np.clip(array[:, :, 3:], 0.0, 1.0)], axis=2)
        array = np.round(color * 255).astype(np.uint8)
    else:
        array = _to_uint8(array)
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0]).convert("RGB")
    if array.shape[2] == 2:
        array = np.concatenate([array[:, :, :1]] * 3 + [array[:, :, 1:]], axis=2)
    image = Image.fromarray(np.ascontiguousarray(array[:, :, :4]))
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] == 255:
        image = image.convert("RGB")
    return image


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    info = np.iinfo(array.dtype) if np.issubdtype(array.dtype, np.integer) else None
    top = float(array.max()) if array.size else 0.0
    if info is not None and top > 255:
        # 16 бит (и шире): шкала по разрядности, а не по максимуму кадра —
        # иначе тёмный снимок «осветлился» бы.
        scale = 65535.0 if top <= 65535 else float(info.max)
        return np.round(array.astype(np.float64) / scale * 255).clip(0, 255).astype(np.uint8)
    if info is None:
        scale = top if top > 1.0 else 1.0
        return np.round(np.clip(array / scale, 0, 1) * 255).astype(np.uint8)
    return array.clip(0, 255).astype(np.uint8)


def _to_srgb(image: Image.Image) -> Image.Image:
    """Любой режим Pillow → RGB/RGBA 8 бит с переводом цвета по профилю ICC."""
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        array = _to_uint8(np.asarray(image))
        image = Image.fromarray(array)
    alpha = None
    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        base = rgba.convert("RGB") if image.mode in ("P", "PA") else image.convert(
            "RGB" if image.mode in ("RGBA", "RGBa") else "L"
        )
    else:
        base = image
    base = _apply_icc(base, image.info.get("icc_profile"))
    if base.mode != "RGB":
        base = base.convert("RGB")
    if alpha is not None and alpha.getextrema()[0] < 255:
        base.putalpha(alpha)
    return base


def _apply_icc(image: Image.Image, profile: bytes | None) -> Image.Image:
    """Перевод в sRGB по встроенному профилю; без профиля — как есть.

    CMYK без профиля Pillow переводит своей простой формулой — грубо, но
    лучше, чем ничего; с профилем (почти всегда у файлов из типографии) —
    точно.
    """
    if not profile or image.mode not in ("RGB", "L", "CMYK", "LAB"):
        return image
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        return ImageCms.profileToProfile(
            image, source, ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB",
        )
    except (ImageCms.PyCMSError, OSError, ValueError) as error:
        LOGGER.debug("ICC profile ignored: %s", error)
        return image


def _finish(image: Image.Image) -> Image.Image:
    if image.width * image.height > MAX_PIXELS:
        raise TooLarge(f"the image is too large: {image.width}×{image.height}")
    return image
