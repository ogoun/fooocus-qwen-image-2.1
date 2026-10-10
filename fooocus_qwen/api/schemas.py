"""Тела запросов и ответов API.

Имена полей — как в метаданных PNG и в сохранённых промтах (``negative_prompt``,
``true_cfg_scale``, ``styles``): что записано в картинке, то и отправляется
обратно повтором. Значения по умолчанию — те же, что у вкладок.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import config
from ..engine import generator as generator_module
from ..imaging import aspect, outpaint
from ..settings import MEMORY_PROFILES, PRECISIONS

PresetName = Literal[config.PRESET_NAMES]  # type: ignore[valid-type]
# «Как у исходника/референса» внутри приложения записано по-русски
# (``aspect.FOLLOW_REFERENCE``); снаружи — ``reference``.
FOLLOW = "reference"
AspectName = Literal[tuple(ratio for ratio in aspect.ASPECT_RATIOS if ratio != aspect.FOLLOW_REFERENCE) + (FOLLOW,)]  # type: ignore[valid-type]
OutpaintRatio = Literal[tuple(ratio for ratio in aspect.ASPECT_RATIOS if ratio != aspect.FOLLOW_REFERENCE)]  # type: ignore[valid-type]
MaskMode = Literal[generator_module.MASK_MODES]  # type: ignore[valid-type]
Side = Literal[outpaint.SIDES]  # type: ignore[valid-type]


class Strict(BaseModel):
    """Неизвестное поле — ошибка: опечатка в имени не должна молча стать значением по умолчанию."""

    model_config = ConfigDict(extra="forbid")


class ImageRef(Strict):
    """Картинка: ровно одно из полей.

    ``base64`` — PNG/JPEG/WebP в base64 или data URL; ``pose`` — имя позы из
    ``GET /poses`` (её скелет); ``gallery`` — путь картинки из ``GET /gallery``.
    Вместо объекта можно прислать строку — это ``base64``.
    """

    base64: str | None = None
    pose: str | None = None
    gallery: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self):
        given = [name for name in ("base64", "pose", "gallery") if getattr(self, name)]
        if len(given) != 1:
            raise ValueError("an image needs exactly one of: base64, pose, gallery")
        return self


ImageInput = str | ImageRef


class LoraInput(Strict):
    name: str = Field(description="file name from GET /loras, without .safetensors")
    weight: float = Field(1.0, ge=-4.0, le=4.0)


class Output(Strict):
    """Что вернуть: картинки в base64 (иначе — только ссылки ``url``)."""

    return_base64: bool = True


class GenerateRequest(Output):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    styles: list[str] = []
    preset: PresetName = "MiddleQuality"  # type: ignore[valid-type]
    aspect: AspectName = "1:1"  # type: ignore[valid-type]
    image_number: int = Field(1, ge=1, le=8)
    seed: int = -1
    true_cfg_scale: float = Field(1.0, ge=1.0, le=10.0)
    use_kv_cache: bool = True
    reference_scale: int = Field(0, ge=0, le=2048, description="0 — chosen by the number of images")
    references: list[ImageInput] = Field([], max_length=10)
    loras: list[LoraInput] = Field([], max_length=8)
    boost: bool = Field(False, description="rewrite the prompt with the language model first")


class EditRequest(Output):
    image: ImageInput
    prompt: str = Field(min_length=1)
    mask: ImageInput | None = Field(None, description="white (or opaque) — the area to change")
    mask_mode: MaskMode | None = Field(  # type: ignore[valid-type]
        None, description="mask (default with a mask), region, annotation (marks drawn on the image), none"
    )
    crop: tuple[int, int, int, int] | None = Field(None, description="left, top, right, bottom in source pixels")
    mask_grow: int = Field(8, ge=0, le=64)
    mask_feather: int = Field(12, ge=0, le=64)
    keep_outside: bool = True
    preset: PresetName = "MiddleQuality"  # type: ignore[valid-type]
    image_number: int = Field(1, ge=1, le=8)
    seed: int = -1
    references: list[ImageInput] = Field([], max_length=10)
    loras: list[LoraInput] = Field([], max_length=8)
    boost: bool = False


class OutpaintRequest(Output):
    image: ImageInput
    ratio: OutpaintRatio | None = Field(None, description="target aspect ratio, the original centred")  # type: ignore[valid-type]
    sides: list[Side] = []
    amount: float = Field(0.25, gt=0.0, le=2.0, description="share of the side added on each chosen side")
    prompt: str = Field("", description="optional: what should appear in the new area")
    describe: bool = Field(False, description="let the language model describe the scene first")
    crop: tuple[int, int, int, int] | None = None
    preset: PresetName = "QwenTurbo"  # type: ignore[valid-type]
    seed: int = -1
    loras: list[LoraInput] = Field([], max_length=8)

    @model_validator(mode="after")
    def _ratio_or_sides(self):
        if not self.ratio and not self.sides:
            raise ValueError("give either ratio or sides")
        return self


class BoostRequest(Strict):
    prompt: str = Field(min_length=1)
    mode: Literal["t2i", "edit"] | None = Field(None, description="default: edit when images are given")
    references: list[ImageInput] = Field([], max_length=10)


class DescribeRequest(Strict):
    image: ImageInput


class PoseRequest(Strict):
    image: ImageInput
    title: str = Field("", max_length=60)


class PromptPayload(BaseModel):
    """Сохранённый промт: поля как у вкладки генерации, лишние сохраняются как есть."""

    model_config = ConfigDict(extra="allow")

    prompt: str = ""
    negative_prompt: str = ""
    styles: list[str] = []
    preset: PresetName = "MiddleQuality"  # type: ignore[valid-type]
    aspect: str = "1:1"
    seed: int = -1
    true_cfg_scale: float = 1.0
    loras: list[dict] = []


class SettingsPatch(Strict):
    precision: Literal[PRECISIONS] | None = None  # type: ignore[valid-type]
    sage_attention: bool | None = None
    memory_profile: Literal[MEMORY_PROFILES] | None = None  # type: ignore[valid-type]

