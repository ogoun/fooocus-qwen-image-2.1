"""Плитка для своей позы: героиня каталога в этой позе, нарисованная Qwen-Image.

Подача выбрана опытом ``tools/experiments/pose_tile.py`` (разбор — в
``docs/research/2026-09-26-pozy.md``): единственный референс — скелет, а
образ и стиль каталога описаны словами. Второй референс — плитка каталога
как образец — не работает: модель копирует образец вместе с его позой и
скелет игнорирует. Со скелетом одним поза держится: ошибка 0.7–2.6 %
размера фигуры по DWPose на результате.
"""

from __future__ import annotations

from PIL import Image

from ..engine import presets
from ..engine.generator import GenerationRequest

# Стиль каталога openposes.com словами: живопись крупным мазком, фон —
# разводы краски, одежда повседневная, героиня одна.
PROMPT = (
    "Full-body digital painting of Emma Watson, a young woman with shoulder-length wavy light "
    "brown hair, in a casual outfit, loose expressive painterly brushstrokes, soft cinematic light, "
    "the background is an abstract swirl of paint splashes in teal, blue and violet on a pale grey "
    "canvas. She is posed exactly like the pose skeleton in the reference image: the colored lines "
    "mark her head, shoulders, arms, hips and legs. Do not draw the skeleton lines."
)

# Turbo рисует плитку за ~13 с на RTX 3090 и чист именно на 1024², а это и
# есть размер плитки. Без его адаптера — самый быстрый из обычных пресетов:
# качать ради иконки 1.3 ГБ без спроса незачем.
FAST_PRESET = "Turbo"
FALLBACK_PRESET = "LowQuality"


def request(skeleton_image: Image.Image, turbo_ready: bool, seed: int = -1) -> GenerationRequest:
    preset = presets.get(FAST_PRESET if turbo_ready else FALLBACK_PRESET)
    return GenerationRequest(
        prompt=PROMPT,
        prompt_original=PROMPT,
        preset=preset,
        references=(skeleton_image.convert("RGB"),),
        aspect="1:1",
        seed=seed,
        image_number=1,
    )
