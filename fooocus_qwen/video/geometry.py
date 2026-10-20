"""Геометрия ролика H3: число кадров, длительность, холст.

Правила — те же, что у движка (``FreeVideo/freevideo_engine/geometry.py``) и
у самой модели: 24 кадра в секунду, число кадров вида ``17·n + 5`` с
округлением вверх, стороны кратны 32 и не меньше 256. Повторены здесь, а не
импортированы: движок живёт в чужом окружении, а интерфейсу длительность и
холст нужны до всякого запуска — подписать ползунок, посчитать тайм-коды
промта.

Холст выбирается **классом площади** и соотношением сторон, а не «короткой
стороной»: так у вертикали и квадрата та же цена по памяти и времени, что у
горизонтали того же класса, — число токенов видео пропорционально площади.
Классы взяты по замерам: 480p — рабочая точка стенда ComfyUI (864×480),
768p — родная геометрия H3 и умолчание FreeVideo (1344×768). Значения
совпадают с таблицей ``ResolutionSelector`` из ``E:\\MINIMAX`` (0.4 и 0.98 МП).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FPS = 24
FRAME_UNIT = 17
FRAME_OFFSET = 5
# Короче движок не берёт: латентных кадров меньше двенадцати (1.625 с).
MIN_SECONDS = 1.625
# Одна генерация FreeVideo — до 15 с (BLOCK_SECONDS); длиннее — частями с
# продолжением (``engine``), до минуты в интерфейсе.
SINGLE_MAX_SECONDS = 15.0
MAX_SECONDS = 60.0
MIN_SIDE = 256
SIDE_STEP = 32

# Классы площади: имя → (ширина, высота) горизонтали 16:9.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (864, 480),
    "576p": (1024, 576),
    "768p": (1344, 768),
}
DEFAULT_RESOLUTION = "480p"

ASPECT_RATIOS: tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")
# Соотношение «как у первого кадра»: холст следует снимку, а не списку.
FOLLOW_IMAGE = "image"
DEFAULT_ASPECT = "16:9"


@dataclass(frozen=True)
class Duration:
    """Фактическая длительность: запрошенные секунды, выровненные по сетке кадров."""

    frames: int
    seconds: float

    @property
    def label(self) -> str:
        return f"{self.seconds:.2f}"


def frames_for(seconds: float) -> int:
    """Кадры для длительности — вверх до ``17·n + 5``, как у движка."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be a positive finite number")
    requested = math.ceil(round(seconds * FPS, 6))
    return requested + (FRAME_OFFSET - requested) % FRAME_UNIT


def duration(seconds: float) -> Duration:
    """Длительность, которую ролик получит на самом деле.

    Секунды зажимаются в поддерживаемый диапазон: ползунок интерфейса их и так
    не выпустит, а API и сохранённые промты могли бы.
    """
    seconds = min(max(float(seconds), MIN_SECONDS), MAX_SECONDS)
    frames = frames_for(seconds)
    return Duration(frames=frames, seconds=frames / FPS)


def _round_side(value: float) -> int:
    return max(MIN_SIDE, int(round(value / SIDE_STEP)) * SIDE_STEP)


def ratio_value(aspect: str) -> float:
    """``"16:9"`` → 1.777…"""
    width, _, height = aspect.partition(":")
    return float(width) / float(height)


def canvas(resolution: str, aspect: str = DEFAULT_ASPECT, image_size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Ширина и высота холста: площадь класса при нужном соотношении сторон.

    ``aspect == FOLLOW_IMAGE`` берёт соотношение у ``image_size`` (первого
    кадра); без снимка — умолчание 16:9. Отношение зажимается в 21:9…9:21:
    дальше модель не обучалась.
    """
    base_width, base_height = RESOLUTIONS.get(resolution, RESOLUTIONS[DEFAULT_RESOLUTION])
    area = base_width * base_height
    if aspect == FOLLOW_IMAGE:
        ratio = image_size[0] / image_size[1] if image_size and image_size[1] else ratio_value(DEFAULT_ASPECT)
    else:
        ratio = ratio_value(aspect)
    ratio = min(max(ratio, 9 / 21), 21 / 9)
    return _round_side(math.sqrt(area * ratio)), _round_side(math.sqrt(area / ratio))


def timecode(seconds: float) -> str:
    """``MM:SS.mmm`` — формат тайм-кодов в промте H3."""
    millis = int(round(max(seconds, 0.0) * 1000))
    minutes, millis = divmod(millis, 60_000)
    return f"{minutes:02d}:{millis / 1000:06.3f}"


# --- финальный апскейл латентов ---------------------------------------------------
#
# После всех шагов сэмплинга латенты холста поднимаются обученной сетью
# LBH-123-AI (та же, что у двухпроходного режима FreeVideo) и декодируются в
# больший кадр. Трансформер больший размер не видит — растут только время
# апскейлера и тайлового декодера. Размер выхода задаётся, как у узла
# AGSoft Minimax H3 Latent Upscaler: множителем, точными сторонами или
# площадью в мегапикселях. Стороны кратны 32 (так их проверяет движок),
# множитель — больше 1 и до 4 (диапазон обучения сети).

UPSCALE_MULTIPLIER = "multiplier"
UPSCALE_DIMENSIONS = "dimensions"
UPSCALE_MEGAPIXELS = "megapixels"
UPSCALE_MODES: tuple[str, ...] = (UPSCALE_MULTIPLIER, UPSCALE_DIMENSIONS, UPSCALE_MEGAPIXELS)
MAX_UPSCALE = 4.0
# Мегапиксель — как у узла AGSoft: 1024 × 1024 пикселей.
MEGAPIXEL = 1024 * 1024


@dataclass(frozen=True)
class Upscale:
    """Настройки финального апскейла; размер выхода считает ``upscaled_size``."""

    mode: str = UPSCALE_MULTIPLIER
    factor: float = 2.0
    width: int = 1920
    height: int = 1088
    megapixels: float = 2.0

    def as_dict(self) -> dict:
        return {"mode": self.mode, "factor": self.factor, "width": self.width, "height": self.height,
                "megapixels": self.megapixels}

    @classmethod
    def from_dict(cls, value: dict | None) -> Upscale | None:
        if not isinstance(value, dict):
            return None
        defaults = cls()
        try:
            return cls(
                mode=value.get("mode") if value.get("mode") in UPSCALE_MODES else defaults.mode,
                factor=float(value.get("factor", defaults.factor)),
                width=int(value.get("width", defaults.width)),
                height=int(value.get("height", defaults.height)),
                megapixels=float(value.get("megapixels", defaults.megapixels)),
            )
        except (TypeError, ValueError):
            return None


def _proportional(target_width: float, target_height: float, ratio: float) -> tuple[int, int]:
    """Пара кратных 32 возле цели с наилучшим сохранением пропорции (перебор соседей, как у AGSoft)."""
    width_units = max(1, round(target_width / SIDE_STEP))
    height_units = max(1, round(target_height / SIDE_STEP))
    best = None
    for w in range(max(1, width_units - 2), width_units + 3):
        for h in range(max(1, height_units - 2), height_units + 3):
            width, height = w * SIDE_STEP, h * SIDE_STEP
            ratio_error = abs(math.log(width / height / ratio))
            size_error = abs(width / target_width - 1) + abs(height / target_height - 1)
            score = (ratio_error * 4 + size_error, ratio_error, size_error)
            if best is None or score < best[0]:
                best = (score, width, height)
    return best[1], best[2]


def upscaled_size(width: int, height: int, upscale: Upscale | None) -> tuple[int, int] | None:
    """Размер выхода после апскейла холста ``width × height``; None — апскейла нет.

    Пропорциональные режимы держат форму холста; точные стороны могут её
    сменить — тогда движок поднимает латенты с запасом и обрезает по центру.
    Меньше холста по любой стороне или больше чем в 4 раза — ``ValueError``.
    """
    if upscale is None:
        return None
    if upscale.mode == UPSCALE_DIMENSIONS:
        size = (_round_side(upscale.width), _round_side(upscale.height))
    elif upscale.mode == UPSCALE_MEGAPIXELS:
        area = float(upscale.megapixels) * MEGAPIXEL
        ratio = width / height
        size = _proportional(math.sqrt(area * ratio), math.sqrt(area / ratio), ratio)
    else:
        factor = float(upscale.factor)
        size = _proportional(width * factor, height * factor, width / height)
    if size == (width, height) or (upscale.mode != UPSCALE_DIMENSIONS and size[0] * size[1] <= width * height):
        return None
    if size[0] < width or size[1] < height:
        raise ValueError(f"The upscaled frame {size[0]}×{size[1]} must not be smaller than the video frame "
                         f"{width}×{height}")
    if max(size[0] / width, size[1] / height) > MAX_UPSCALE + 1e-9:
        raise ValueError(f"The upscaled frame {size[0]}×{size[1]} is more than {MAX_UPSCALE:g}× the video frame "
                         f"{width}×{height}")
    return size
