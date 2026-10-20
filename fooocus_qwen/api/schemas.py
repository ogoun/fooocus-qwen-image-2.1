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
from ..video import geometry as video_geometry
from ..video import prompt_h3 as video_prompt
from ..video import request as video_request
from ..video import storyboard as video_storyboard

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
    ``GET /poses`` (её скелет); ``gallery`` — путь картинки из ``GET /gallery``
    (ролик галереи даёт свой последний кадр — «следующий шот»).
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


# --- видео ---------------------------------------------------------------------------
#
# Имена полей — как в параметрах ролика (соседний JSON в галерее): что
# записано у ролика, то и отправляется обратно повтором.

VideoMode = Literal[video_prompt.MODES]  # type: ignore[valid-type]
VideoResolution = Literal[tuple(video_geometry.RESOLUTIONS)]  # type: ignore[valid-type]
VideoAspect = Literal[video_geometry.ASPECT_RATIOS + (video_geometry.FOLLOW_IMAGE,)]  # type: ignore[valid-type]
VideoQuality = Literal[tuple(video_request.QUALITY)]  # type: ignore[valid-type]
VideoStructure = Literal[tuple(video_storyboard.STRUCTURES)]  # type: ignore[valid-type]


def _seconds():
    return Field(5.0, ge=video_geometry.MIN_SECONDS, le=video_geometry.MAX_SECONDS,
                 description="snapped up to the frame grid (17n+5 frames at 24 fps); over 15 s, made in parts")


class VideoReference(Strict):
    """Референс режима ``ref``: ровно одно из ``base64`` и ``gallery``.

    ``kind`` по имени файла галереи (или ``name``) узнаётся сам; у base64 без
    имени он обязателен. ``name`` — имя исходного файла: из него берётся
    расширение для видео и звука.
    """

    kind: Literal["image", "video", "audio"] | None = None
    base64: str | None = None
    gallery: str | None = Field(None, description="a path from GET /gallery (a picture or a video)")
    name: str | None = Field(None, max_length=200, description="original file name, e.g. voice.mp3")

    @model_validator(mode="after")
    def _exactly_one(self):
        if bool(self.base64) == bool(self.gallery):
            raise ValueError("a reference needs exactly one of: base64, gallery")
        return self


class VideoLora(Strict):
    name: str = Field(description="file name from GET /video/loras (loras/video), .safetensors may be omitted")
    strength: float = Field(1.0, ge=-2.0, le=2.0)


class VideoShot(Strict):
    """Шот раскадровки — как строка таблицы вкладки, без номера."""

    prompt: str = Field(min_length=1)
    seconds: float = Field(video_storyboard.DEFAULT_SHOT_SECONDS, gt=0.0, le=video_geometry.SINGLE_MAX_SECONDS)
    transition: str = Field(video_storyboard.DEFAULT_TRANSITION,
                            description=f"into this shot: {', '.join(video_storyboard.TRANSITIONS)}")
    title: str = ""


class VideoUpscale(Strict):
    """Финальный апскейл латентов — как у галочки Upscale вкладки (``video.geometry.Upscale``)."""

    mode: Literal[video_geometry.UPSCALE_MODES] = video_geometry.UPSCALE_MULTIPLIER  # type: ignore[valid-type]
    factor: float = Field(2.0, ge=1.0, le=video_geometry.MAX_UPSCALE, description="mode multiplier")
    width: int = Field(1920, ge=video_geometry.MIN_SIDE, le=16384,
                       description="mode dimensions; rounded to a multiple of 32")
    height: int = Field(1088, ge=video_geometry.MIN_SIDE, le=16384,
                        description="mode dimensions; rounded to a multiple of 32")
    megapixels: float = Field(2.0, gt=0.0, le=64.0, description="mode megapixels (1 MP = 1024 x 1024)")

    def settings(self) -> video_geometry.Upscale:
        return video_geometry.Upscale(mode=self.mode, factor=self.factor, width=self.width, height=self.height,
                                      megapixels=self.megapixels)


class VideoSettings(Strict):
    """Общее у ролика и раскадровки: холст, качество, сид, LoRA, апскейл, возврат файла."""

    resolution: VideoResolution = video_geometry.DEFAULT_RESOLUTION  # type: ignore[valid-type]
    aspect: VideoAspect = video_geometry.DEFAULT_ASPECT  # type: ignore[valid-type]
    seed: int = -1
    quality: VideoQuality = video_request.DEFAULT_QUALITY  # type: ignore[valid-type]
    loras: list[VideoLora] = Field([], max_length=8)
    upscale: VideoUpscale | None = Field(
        None, description="learned latent upscale after sampling, decoded at the larger size; drafts are not "
                          "upscaled, finishing them is")
    return_base64: bool = Field(False, description="also return the MP4 itself in base64")


class VideoGenerateRequest(VideoSettings):
    prompt: str = Field("", description="required, except with continue_from or refine_from")
    mode: VideoMode = video_prompt.MODE_T2V  # type: ignore[valid-type]
    seconds: float = _seconds()
    first: ImageInput | None = Field(None, description="first frame (i2v, flf2v); a gallery video gives its last frame")
    last: ImageInput | None = Field(None, description="last frame (flf2v)")
    references: list[VideoReference] = Field([], max_length=video_request.MAX_IMAGES + video_request.MAX_VIDEOS
                                             + video_request.MAX_AUDIOS)
    boost: bool = Field(False, description="rewrite the prompt into an H3 prompt with the language model first")
    continue_from: str | None = Field(None, description="a gallery video to extend; the result is the whole video, source and new part joined")
    refine_from: str | None = Field(None, description="a gallery draft to finish without generating anew")

    @model_validator(mode="after")
    def _one_follow_up(self):
        if self.continue_from and self.refine_from:
            raise ValueError("give either continue_from or refine_from, not both")
        return self


class VideoBoostRequest(Strict):
    prompt: str = Field(min_length=1)
    mode: VideoMode = video_prompt.MODE_T2V  # type: ignore[valid-type]
    seconds: float = _seconds()
    aspect: VideoAspect = video_geometry.DEFAULT_ASPECT  # type: ignore[valid-type]
    first: ImageInput | None = None
    last: ImageInput | None = None
    references: list[VideoReference] = Field([], max_length=video_request.MAX_IMAGES + video_request.MAX_VIDEOS
                                             + video_request.MAX_AUDIOS)


class StoryboardRequest(Strict):
    brief: str = Field(min_length=1)
    seconds: float = Field(20.0, ge=5.0, le=120.0, description="total length of the film")
    structure: VideoStructure = video_storyboard.DEFAULT_STRUCTURE  # type: ignore[valid-type]
    aspect: VideoAspect = video_geometry.DEFAULT_ASPECT  # type: ignore[valid-type]


class StoryboardRunRequest(VideoSettings):
    shots: list[VideoShot] = Field(min_length=1, max_length=40)
    chain: bool = Field(True, description="each shot starts from the last frame of the previous one")
    first: ImageInput | None = Field(None, description="first frame of the first shot")

