"""Единая точка генерации: текст в изображение, правка промтом, правка по маске.

Порядок условных изображений определяет теги ``<imageN>``, которыми промт на них
ссылается, поэтому он зафиксирован: исходное изображение, затем маска, затем
референсы. Тот же порядок нужен интерфейсу для подписей миниатюр, поэтому и
порядок, и теги вычисляет одна функция — ``condition_slots()``, — а вкладка
генерации вызывает её, а не повторяет формулу «i-й референс — <imageI>».
Формула верна ровно до тех пор, пока в списке нет исходного изображения.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GeneratedImage:
    """Готовое изображение вместе с параметрами, которые его породили."""

    image: Image.Image
    seed: int
    parameters: dict[str, Any]


def resolve_seed(seed: int) -> int:
    """Отрицательное значение означает «выбери случайный и запомни его»."""
    return random.randrange(2**31) if seed < 0 else seed


def condition_slots(
    source: Image.Image | None = None,
    mask: Image.Image | None = None,
    mask_mode: str = MASK_NONE,
    references: Sequence[Image.Image] = (),
) -> list[ConditionSlot]:
    """Условные изображения в том порядке, в каком их увидит модель.

    Единственное место, где этот порядок и вытекающие из него теги
    вычисляются. Аргументами, а не готовым запросом, — потому что интерфейсу
    подписи миниатюр нужны раньше, чем существует запрос на генерацию, а
    заводить ради подписи фиктивный ``GenerationRequest`` с фиктивным пресетом
    значило бы соврать о том, что здесь важно. Что важно, видно по сигнатуре:
    ровно четыре поля запроса из пятнадцати.
    """
    slots: list[tuple[str, Image.Image]] = []

    if source is not None:
        slots.append(("source", source))
        # В режиме аннотации пометки уже нарисованы на самом изображении,
        # а «точная область» подаёт вырезанную пару так же, как обычная маска.
        if mask is not None and mask_mode in (MASK_MASK, MASK_REGION):
            slots.append(("mask", masking.as_condition(mask)))

    slots.extend(("reference", item) for item in references)

    tagged = len(slots) >= 2
    return [
        ConditionSlot(tag=f"<image{index}>" if tagged else "", role=role, image=item)
        for index, (role, item) in enumerate(slots, start=1)
    ]


def build_conditions(request: GenerationRequest) -> list[ConditionSlot]:
    """Условные изображения запроса. Тонкая обёртка над ``condition_slots``."""
    return condition_slots(request.source, request.mask, request.mask_mode, request.references)


def resolve_size(request: GenerationRequest) -> tuple[int | None, int | None]:
    """Размеры кадра или пара ``None``, если их выводит пайплайн.

    Наследование размеров источника при правке задаётся явным признаком
    ``aspect.FOLLOW_REFERENCE``, а не значением "1:1": иначе осознанный выбор
    квадрата при правке был бы неотличим от «пользователь ничего не выбирал»
    и молча игнорировался бы.
    """
    if request.source is not None and request.aspect == aspect.FOLLOW_REFERENCE:
        return None, None
    return aspect.dimensions(request.aspect, request.preset.output_resolution)


class Generator:
    """Выполняет запросы на генерацию по одному.

    «По одному» — это не пожелание, а инвариант, и он обеспечивается здесь, а
    не проводкой интерфейса. За генератором стоит ровно один пайплайн, один
    ``ResidencyManager`` и один ``EmbedsCache`` на всё приложение, а видеокарта
    одна на 24 ГБ при 13.3 ГБ резидентного трансформера. Второй запрос,
    начавшийся посреди первого, (1) сбросил бы флаг прерывания первого, и
    кнопка «Прервать» перестала бы работать; (2) при промахе кэша эмбеддингов
    вытеснил бы трансформер на хост прямо посреди чужого цикла денойзинга,
    переприсвоив ``tensor.data`` под работающим forward; (3) просто не
    поместился бы в видеопамять.

    Полагаться на ``concurrency_id`` очереди Gradio для этого нельзя: очередь —
    это слой выше, который легко забыть при добавлении новой вкладки, а
    перечисленные последствия наступают молча и недетерминированно.
    """

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
        self._lock = threading.Lock()

    def interrupt(self) -> None:
        """Просит прервать текущую генерацию. Читается циклом денойзинга.

        Замок ``_lock`` здесь намеренно НЕ берётся: он занят ровно тем, что эта
        кнопка должна остановить, и ожидание на нём сделало бы остановку
        недостижимой. Обе записи — выставление булева флага, а не изменение
        структуры данных, поэтому их безопасно делать из чужого потока: цикл
        денойзинга читает ``pipe._interrupt`` между шагами, а ``generate``
        проверяет ``self._interrupted`` между изображениями.
        """
        self._interrupted = True
        self._pipe._interrupt = True

    def generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> list[GeneratedImage]:
        """Выполняет запрос, дождавшись, пока освободится видеокарта.

        Замок держит весь цикл целиком — по причинам из докстринга класса.
        """
        with self._lock:
            return self._generate(request, progress)

    def recover(self) -> None:
        """Возвращает размещение весов после сбоя, взяв тот же замок, что и ``generate()``.

        Без этого замка вызов мог бы вклиниться прямо в середину чужой
        перестановки моделей внутри ``ResidencyManager.text_encoder_resident()``
        — ровно та гонка между потоками, ради которой замок и существует (см.
        докстринг класса). Раньше корректность держалась на том, что общая
        группа очереди Gradio не отпускает второй обработчик, пока первый не
        вернётся, но полагаться на слой очереди в этом инварианте запрещено
        тем же рассуждением, что и для самой генерации: очередь легко забыть
        при добавлении третьей вкладки или обработчика с ``queue=False``.

        Сбой самого восстановления не пробрасывается дальше: он не должен
        ронять приложение сильнее, чем уже уронил исходный сбой генерации,
        из-за которого восстановление вообще понадобилось. Полный стек уходит
        в журнал. Замок освобождается и на этом пути тоже — это гарантирует
        сам ``with``, а не отдельная забота вызывающей стороны.
        """
        if self._residency is None:
            return
        with self._lock:
            try:
                self._residency.restore()
            except Exception:  # noqa: BLE001 — сбой восстановления не должен ронять приложение
                LOGGER.exception("Не удалось вернуть веса на штатные места")

    def _generate(
        self,
        request: GenerationRequest,
        progress: ProgressCallback | None = None,
    ) -> list[GeneratedImage]:
        self._interrupted = False
        self._pipe._interrupt = False

        prepared, region_box = self._prepare(request)
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

            image = self._finish(request, prepared, output.images[0], region_box)
            seconds = time.perf_counter() - started
            results.append(
                GeneratedImage(
                    image=image,
                    seed=seed,
                    parameters=self._parameters(
                        request, prepared, positive, negative, slots, seed, seconds, width, height
                    ),
                )
            )

        return results

    def _prepare(
        self, request: GenerationRequest
    ) -> tuple[GenerationRequest, tuple[int, int, int, int] | None]:
        """Готовит маску и, для режима точной области, вырезает фрагмент.

        Прямоугольник вырезки возвращается явным вторым значением, а не
        сохраняется в самом запросе. Так уже было сделано однажды — через
        изменяемый словарь на объекте запроса — и это дало утечку между
        вызовами: `_replace` копирует запрос поверхностно, поэтому словарь
        оставался общим, и вырезка от прошлого прогона подставлялась в
        следующий, если вызывающая сторона переиспользовала объект. Урок
        общий: промежуточные результаты одного вызова не живут на входном
        объекте, даже когда это удобно.
        """
        if request.source is None or request.mask is None or request.mask_mode == MASK_NONE:
            return request, None

        refined = masking.refine(request.mask, grow=request.mask_grow, feather=request.mask_feather)
        if request.mask_mode != MASK_REGION:
            return _replace(request, mask=refined), None

        box = masking.region_box(refined, padding=0.25)
        if box is None:
            LOGGER.warning("Маска пуста, режим точной области вырождается в правку целого кадра")
            return _replace(request, mask=refined, mask_mode=MASK_MASK), None

        prepared = _replace(
            request,
            mask=refined.crop(box),
            source=request.source.crop(box),
        )
        return prepared, box

    def _finish(
        self,
        original_request: GenerationRequest,
        prepared: GenerationRequest,
        produced: Image.Image,
        region_box: tuple[int, int, int, int] | None,
    ) -> Image.Image:
        """Возвращает результат в систему координат исходного изображения."""
        source = original_request.source
        if source is None or prepared.mask_mode == MASK_NONE:
            return produced

        if prepared.mask_mode == MASK_ANNOTATION:
            # Пометки были частью условного изображения; склеивать не по чему.
            return produced

        if not original_request.keep_outside:
            if region_box is None:
                return produced
            # Режим «точная область» — единственный, где `produced` не
            # полнокадровый: `_prepare` уже вырезал по нему источник и маску.
            # Отдать его как есть значило бы вернуть пользователю, правившему
            # деталь в 64 пикселя на холсте в 4000, крошечную картинку вместо
            # кадра. Склейки по маске здесь нет — её и просили отключить, — но
            # патч обязан вернуться на своё место.
            return masking.paste_region(source, produced, region_box)

        if region_box is not None and prepared.mask is not None:
            full_mask = masking.refine(
                original_request.mask,
                grow=original_request.mask_grow,
                feather=original_request.mask_feather,
            )
            return masking.stitch(source, produced, region_box, full_mask)

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
        width: int | None,
        height: int | None,
    ) -> dict[str, Any]:
        # Размеры уже посчитаны в generate() из того же prepared — вычислять
        # их здесь ещё раз значило бы выводить один и тот же факт дважды.
        import diffusers

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
            # Соотношение сторон пишется рядом с выведенными из него
            # размерами, а не вместо них: при правке width/height равны None
            # (размер наследуется от источника), и без этого поля метаданные
            # не помнили бы выбор пользователя вовсе — восстанавливать из
            # PNG было бы нечего.
            "aspect": prepared.aspect,
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
    """Копия запроса с изменёнными полями.

    Копия поверхностная: изображения (`source`, `mask`, `references`)
    намеренно разделяются с оригиналом, потому что они неизменяемы для нас и
    весят мегабайты. Отсюда правило: у `GenerationRequest` не должно быть
    изменяемых полей — любое такое поле окажется общим у копии и оригинала,
    и запись в него переживёт вызов. Ровно этим когда-то был словарь
    `extras`, через который `_prepare` передавал вырезку в `_finish`.
    """
    import copy

    clone = copy.copy(request)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone
