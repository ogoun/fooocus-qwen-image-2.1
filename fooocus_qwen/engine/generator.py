"""Единая точка генерации: текст в изображение, правка промтом, правка по маске.

Порядок условных изображений определяет теги ``<imageN>``, которыми промт на них
ссылается, поэтому он зафиксирован: исходное изображение, затем маска, затем
референсы. Один и тот же порядок нужен интерфейсу для подписей миниатюр, поэтому
он вычисляется здесь, а не дублируется в двух местах.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from .. import __version__
from ..imaging import aspect, masking
from ..prompting.styles import Style, apply_styles
from .embeds_cache import EmbedsCache
from .presets import QualityPreset
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)

MASK_NONE = "none"
MASK_MASK = "mask"
MASK_ANNOTATION = "annotation"
MASK_REGION = "region"
MASK_MODES: tuple[str, ...] = (MASK_NONE, MASK_MASK, MASK_ANNOTATION, MASK_REGION)

ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class ConditionSlot:
    """Одно условное изображение: его тег в промте и назначение."""

    tag: str
    role: str
    image: Image.Image


@dataclass
class GenerationRequest:
    """Всё, что нужно для одного запуска."""

    prompt: str
    preset: QualityPreset
    negative_prompt: str = ""
    styles: tuple[str, ...] = ()
    references: tuple[Image.Image, ...] = ()

    aspect: str = "1:1"
    seed: int = -1
    image_number: int = 1
    true_cfg_scale: float = 1.0
    use_kv_cache: bool = True

    source: Image.Image | None = None
    mask: Image.Image | None = None
    mask_mode: str = MASK_NONE
    mask_grow: int = 8
    mask_feather: int = 12
    keep_outside: bool = True

    prompt_original: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedImage:
    """Готовое изображение вместе с параметрами, которые его породили."""

    image: Image.Image
    seed: int
    parameters: dict[str, Any]


def resolve_seed(seed: int) -> int:
    """Отрицательное значение означает «выбери случайный и запомни его»."""
    return random.randrange(2**31) if seed < 0 else seed


def build_conditions(request: GenerationRequest) -> list[ConditionSlot]:
    """Условные изображения в том порядке, в каком их увидит модель."""
    slots: list[tuple[str, Image.Image]] = []

    if request.source is not None:
        slots.append(("source", request.source))
        # В режиме аннотации пометки уже нарисованы на самом изображении,
        # а «точная область» подаёт вырезанную пару так же, как обычная маска.
        if request.mask is not None and request.mask_mode in (MASK_MASK, MASK_REGION):
            slots.append(("mask", masking.as_condition(request.mask)))

    slots.extend(("reference", item) for item in request.references)

    tagged = len(slots) >= 2
    return [
        ConditionSlot(tag=f"<image{index}>" if tagged else "", role=role, image=item)
        for index, (role, item) in enumerate(slots, start=1)
    ]


def resolve_size(request: GenerationRequest) -> tuple[int | None, int | None]:
    """Размеры кадра или пара ``None``, если их выводит пайплайн.

    При правке размеры по умолчанию не задаются: кадр должен сохранить
    пропорции исходного изображения, а их пайплайн возьмёт из него сам.
    """
    if request.source is not None and request.aspect == "1:1":
        return None, None
    return aspect.dimensions(request.aspect, request.preset.output_resolution)


class Generator:
    """Выполняет запросы на генерацию по одному."""

    def __init__(
        self,
        pipe,
        residency: ResidencyManager,
        cache: EmbedsCache,
        catalogue: dict[str, Style],
    ) -> None:
        self._pipe = pipe
        self._residency = residency
        self._cache = cache
        self._catalogue = catalogue
        self._interrupted = False

    def interrupt(self) -> None:
        """Просит прервать текущую генерацию. Читается циклом денойзинга."""
        self._interrupted = True
        self._pipe._interrupt = True

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> list[GeneratedImage]:
        self._interrupted = False
        self._pipe._interrupt = False

        prepared = self._prepare(request)
        positive, negative = apply_styles(
            prepared.prompt, prepared.negative_prompt, prepared.styles, self._catalogue
        )
        slots = build_conditions(prepared)
        condition = [slot.image for slot in slots] or None
        width, height = resolve_size(prepared)
        base_seed = resolve_seed(prepared.seed)
        device = self._pipe._execution_device

        results: list[GeneratedImage] = []
        for index in range(max(1, prepared.image_number)):
            if self._interrupted:
                break

            seed = base_seed + index
            started = time.perf_counter()
            generator = torch.Generator(device=device).manual_seed(seed)

            def step_callback(_pipe, step: int, _timestep, kwargs, _index=index):
                if progress is not None:
                    progress(_index, step + 1, prepared.preset.num_inference_steps)
                return kwargs

            output = self._pipe(
                prompt=positive,
                image=condition,
                negative_prompt=negative or None,
                true_cfg_scale=prepared.true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=prepared.preset.num_inference_steps,
                output_resolution=prepared.preset.output_resolution,
                use_kv_cache=prepared.use_kv_cache,
                generator=generator,
                callback_on_step_end=step_callback,
            )

            if self._interrupted:
                # Прерванный цикл всё равно декодирует латенты, но это шум.
                LOGGER.info("Генерация прервана пользователем")
                break

            image = self._finish(request, prepared, output.images[0])
            seconds = time.perf_counter() - started
            results.append(
                GeneratedImage(
                    image=image,
                    seed=seed,
                    parameters=self._parameters(request, prepared, positive, negative, slots, seed, seconds),
                )
            )

        return results

    def _prepare(self, request: GenerationRequest) -> GenerationRequest:
        """Готовит маску и, для режима точной области, вырезает фрагмент."""
        if request.source is None or request.mask is None or request.mask_mode == MASK_NONE:
            return request

        refined = masking.refine(request.mask, grow=request.mask_grow, feather=request.mask_feather)
        if request.mask_mode != MASK_REGION:
            return _replace(request, mask=refined)

        box = masking.region_box(refined, padding=0.25)
        if box is None:
            LOGGER.warning("Маска пуста, режим точной области вырождается в правку целого кадра")
            return _replace(request, mask=refined, mask_mode=MASK_MASK)

        request.extras["region_box"] = box
        return _replace(
            request,
            mask=refined.crop(box),
            source=request.source.crop(box),
        )

    def _finish(
        self,
        original_request: GenerationRequest,
        prepared: GenerationRequest,
        produced: Image.Image,
    ) -> Image.Image:
        """Возвращает результат в систему координат исходного изображения."""
        source = original_request.source
        if source is None or prepared.mask_mode == MASK_NONE:
            return produced

        if prepared.mask_mode == MASK_ANNOTATION:
            # Пометки были частью условного изображения; склеивать не по чему.
            return produced

        if not original_request.keep_outside:
            return produced

        box = original_request.extras.get("region_box")
        if box is not None and prepared.mask is not None:
            full_mask = masking.refine(
                original_request.mask,
                grow=original_request.mask_grow,
                feather=original_request.mask_feather,
            )
            return masking.stitch(source, produced, box, full_mask)

        return masking.blend(source, produced, prepared.mask)

    def _parameters(
        self,
        original_request: GenerationRequest,
        prepared: GenerationRequest,
        positive: str,
        negative: str,
        slots: Sequence[ConditionSlot],
        seed: int,
        seconds: float,
    ) -> dict[str, Any]:
        import diffusers

        width, height = resolve_size(prepared)
        return {
            "app_version": __version__,
            "diffusers_version": diffusers.__version__,
            "prompt": original_request.prompt_original or original_request.prompt,
            "prompt_boosted": positive,
            "negative_prompt": negative,
            "styles": list(original_request.styles),
            "seed": seed,
            "steps": prepared.preset.num_inference_steps,
            "preset": prepared.preset.name,
            "width": width,
            "height": height,
            "output_resolution": prepared.preset.output_resolution,
            "true_cfg_scale": prepared.true_cfg_scale,
            # Переключение этого флага меняет результат при том же сиде,
            # поэтому без него параметры невоспроизводимы.
            "use_kv_cache": prepared.use_kv_cache,
            "mask_mode": prepared.mask_mode,
            "mask_grow": prepared.mask_grow,
            "mask_feather": prepared.mask_feather,
            "references": sum(1 for slot in slots if slot.role == "reference"),
            "seconds": round(seconds, 2),
        }


def _replace(request: GenerationRequest, **changes: Any) -> GenerationRequest:
    """Копия запроса с изменёнными полями; словарь extras остаётся общим."""
    import copy

    clone = copy.copy(request)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone
