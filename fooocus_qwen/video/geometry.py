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
