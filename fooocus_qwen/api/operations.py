"""Работы API — те же шаги, что у обработчиков вкладок, без Gradio.

Каждая работа делится на две части. ``prepare_*`` выполняется сразу, в
потоке запроса: картинки читаются, LoRA проверяются, и ошибка клиента
становится ответом 400 до постановки в очередь, а не провалом задачи через
минуту ожидания. Возвращает она замыкание — саму работу, которая идёт в
очереди видеокарты (``jobs.JobQueue``) и сообщает прогресс задаче.

Генерация идёт через ``Studio`` — ту же дорогу, что у интерфейса: точность
и профиль из настроек, докачка весов пресета, восстановление размещения
весов после сбоя. Сообщения — английские (``lang="en"``): консоль и
API говорят по-английски, как и весь журнал.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from .. import __version__, config
from .. import settings as settings_module
from ..engine import presets
from ..engine.generator import (
    MASK_ANNOTATION,
    MASK_MASK,
    MASK_NONE,
    MASK_REGION,
    GenerationRequest,
    condition_slots,
    resolve_reference_scale,
)
from ..imaging import aspect, masking, metadata, outpaint
from ..llm import LlmError, load_endpoint, parse_endpoint_file
from ..llm import setup as llm_setup
from ..prompting import boost as boost_module
from ..prompting import library as prompt_library
from ..storage import gallery
from . import images, schemas
from .jobs import Job, JobFailed

LOGGER = logging.getLogger(__name__)

LANG = "en"
API_PREFIX = "/api/v1"
SYSTEM_PROMPTS: tuple[str, ...] = ("system_prompt_t2i.txt", "system_prompt_edit.txt", "system_prompt_describe.txt")

Work = Callable[[Job], dict]


class UpstreamError(RuntimeError):
    """Языковая модель недоступна или ответила ошибкой: ответ 502."""


class EngineUnavailable(RuntimeError):
    """Движок видео не установлен или не готов: ответ 503."""


class RequestError(ValueError):
    """Ошибка в запросе клиента: ответ 400 (или 404 при ``not_found``)."""

    def __init__(self, message: str, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


def _load(source: schemas.ImageInput, what: str) -> Image.Image:
    try:
        return images.load(source)
    except images.ImageError as error:
        raise RequestError(f"{what}: {error}", not_found=error.not_found) from error


def _loras(studio, items: list[schemas.LoraInput]) -> tuple[tuple, list[str]]:
    """Проверенные LoRA и строки о пропущенных — как у ячеек LoRA интерфейса."""
    from ..ui import lora_slots

    state = [{"name": item.name, "weight": item.weight} for item in items]
    resolved, skipped = lora_slots.resolve(studio.config.lora_dir, state, LANG)
    return resolved, [skipped] if skipped else []


def _styles(studio, names: list[str]) -> tuple[str, ...]:
    unknown = [name for name in names if name not in studio.catalogue]
    if unknown:
        raise RequestError(f"unknown styles: {', '.join(unknown)} (see GET {API_PREFIX}/styles)")
    return tuple(names)


def _need_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise RequestError("the prompt is empty")


def _stage(job: Job):
    """Прогресс докачки весов: ``weights_for`` зовёт его как ``gr.Progress``."""

    def progress(_value=None, desc: str | None = None, **_ignored) -> None:
        if desc:
            job.stage(desc)

    return progress


def _run(studio, job: Job, request: GenerationRequest, notes: list[str]) -> list:
    """Докачка весов пресета, генерация; неудача — ``JobFailed`` с причиной."""
    job.progress.images = request.image_number
    missing = studio.weights_for(request.preset, LANG, _stage(job))
    if missing:
        raise JobFailed(missing)
    job.stage("loading" if not studio.model_loaded else "preparing")
    produced, failure = studio.run_generation(request, LANG, progress=job.report)
    if failure is not None:
        raise JobFailed(failure)
    if not produced:
        raise JobFailed("interrupted")
    return produced


def _image_entry(path, seed: int, parameters: dict, return_base64: bool) -> dict[str, Any]:
    relative = images.relative_output(path)
    entry: dict[str, Any] = {
        "seed": seed,
        "path": relative,
        "url": f"{API_PREFIX}/files/{relative}",
        "width": parameters.get("width"),
        "height": parameters.get("height"),
        "parameters": parameters,
    }
    if return_base64:
        entry["base64"] = images.encode_png(path)
    return entry


def _save(image: Image.Image, parameters: dict, seed: int, return_base64: bool) -> dict[str, Any]:
    destination = gallery.next_path(config.OUTPUT_DIR)
    metadata.save_png(image, destination, parameters)
    return _image_entry(destination, seed, parameters, return_base64)


def _scale_note(request: GenerationRequest, explicit: int) -> list[str]:
    """Автоматически урезанный масштаб условных изображений — вслух, как у вкладок."""
    from ..ui.i18n import say

    chosen = resolve_reference_scale(request)
    if not explicit and chosen != request.preset.output_resolution:
        return [say("reference_scale_chosen", LANG, scale=chosen)]
    return []


# --- генерация ---------------------------------------------------------------------


def prepare_generate(studio, body: schemas.GenerateRequest) -> Work:
    _need_prompt(body.prompt)
    references = [_load(item, f"references[{index}]") for index, item in enumerate(body.references)]
    styles = _styles(studio, body.styles)
    loras, notes = _loras(studio, body.loras)
    ratio = aspect.FOLLOW_REFERENCE if body.aspect == schemas.FOLLOW else body.aspect
    if ratio == aspect.FOLLOW_REFERENCE and not references:
        raise RequestError("aspect 'reference' needs at least one reference image")

    def work(job: Job) -> dict:
        messages = list(notes)
        prompt, chosen_ratio = body.prompt, ratio
        if body.boost:
            job.stage("boosting the prompt")
            mode = boost_module.MODE_EDIT if references else boost_module.MODE_T2I
            text, wh_ratio, said = studio.boost_prompt(body.prompt, mode, LANG, references or None)
            messages.append(said)
            prompt = text or body.prompt
            # Соотношение от переписчика — только если клиент его не задал.
            if "aspect" not in body.model_fields_set and wh_ratio in aspect.ASPECT_RATIOS:
                chosen_ratio = wh_ratio
        request = GenerationRequest(
            prompt=prompt,
            prompt_original=body.prompt,
            preset=presets.get(body.preset),
            negative_prompt=body.negative_prompt,
            styles=styles,
            references=tuple(references),
            aspect=chosen_ratio,
            seed=body.seed,
            image_number=body.image_number,
            true_cfg_scale=body.true_cfg_scale,
            use_kv_cache=body.use_kv_cache,
            reference_scale=body.reference_scale,
            loras=loras,
        )
        messages += _scale_note(request, body.reference_scale)
        produced = _run(studio, job, request, messages)
        return {
            "images": [_save(item.image, item.parameters, item.seed, body.return_base64) for item in produced],
            "prompt": prompt,
            "conditions": _conditions(None, None, MASK_NONE, references),
            "messages": [line for line in messages if line],
        }

    return work


def _conditions(source, mask, mode: str, references: list) -> list[dict[str, str]]:
    """Теги ``<imageN>``, под которыми модель увидит картинки, — чтобы ссылаться на них в промте."""
    return [{"tag": slot.tag, "role": slot.role} for slot in condition_slots(source, mask, mode, references)]


# --- правка ------------------------------------------------------------------------


def prepare_edit(studio, body: schemas.EditRequest) -> Work:
    # Модель правки работает по инструкции; без неё результат не определён.
    _need_prompt(body.prompt)
    original = _load(body.image, "image")
    try:
        source = images.crop(original, body.crop).convert("RGBA")
    except images.ImageError as error:
        raise RequestError(str(error)) from error
    mode = body.mask_mode or (MASK_MASK if body.mask is not None else MASK_NONE)
    mask = None
    notes: list[str] = []
    if mode in (MASK_MASK, MASK_REGION):
        if body.mask is None:
            raise RequestError(f"mask_mode '{mode}' needs a mask")
        drawn = images.to_mask(_load(body.mask, "mask"), original.size)
        mask = images.crop(drawn, body.crop)
        if masking.is_empty(mask):
            # Как у кисти: пустая маска — правка всего кадра, и об этом сказано.
            notes.append("The mask is empty; editing the whole image.")
            mask, mode = None, MASK_NONE
    elif body.mask is not None:
        notes.append(f"The mask is ignored in mask_mode '{mode}'.")
    references = [_load(item, f"references[{index}]") for index, item in enumerate(body.references)]
    loras, skipped = _loras(studio, body.loras)
    notes += skipped
    slots = condition_slots(source, mask, mode, references)

    def work(job: Job) -> dict:
        messages = list(notes)
        prompt = body.prompt
        if body.boost:
            job.stage("boosting the prompt")
            text, _, said = studio.boost_edit(body.prompt, slots, mode, LANG)
            messages.append(said)
            prompt = text or body.prompt
        request = GenerationRequest(
            prompt=prompt,
            prompt_original=body.prompt,
            preset=presets.get(body.preset),
            aspect=aspect.FOLLOW_REFERENCE,
            seed=body.seed,
            image_number=body.image_number,
            source=source,
            mask=mask,
            mask_mode=mode,
            references=tuple(references),
            mask_grow=body.mask_grow,
            mask_feather=body.mask_feather,
            keep_outside=body.keep_outside,
            loras=loras,
        )
        messages += _scale_note(request, 0)
        produced = _run(studio, job, request, messages)
        return {
            "images": [_save(item.image, item.parameters, item.seed, body.return_base64) for item in produced],
            "prompt": prompt,
            "mask_mode": mode,
            "conditions": [{"tag": slot.tag, "role": slot.role} for slot in slots],
            "messages": [line for line in messages if line],
        }

    return work


# --- расширение кадра --------------------------------------------------------------


def prepare_outpaint(studio, body: schemas.OutpaintRequest) -> Work:
    try:
        source = images.crop(_load(body.image, "image"), body.crop).convert("RGB")
    except images.ImageError as error:
        raise RequestError(str(error)) from error
    if body.ratio:
        plan = outpaint.plan_for_ratio(source.size, body.ratio)
    else:
        plan = outpaint.plan(source.size, body.sides, body.amount)
    if plan.canvas_size == source.size:
        raise RequestError("the image already has this aspect ratio; nothing to extend")
    notes: list[str] = []
    if plan.canvas_size[0] > 3 * source.size[0] or plan.canvas_size[1] > 3 * source.size[1]:
        notes.append("The frame grows more than three times; the new area may drift from the original.")
    preset = presets.get(body.preset)
    user_loras, skipped = _loras(studio, body.loras)
    notes += skipped

    def work(job: Job) -> dict:
        messages = list(notes)
        job_canvas = outpaint.prepare(source, plan, outpaint.generation_area(preset.output_resolution))
        job.stage("preparing")
        outpaint_lora, failure = studio.outpaint_lora(LANG, _stage(job))
        if failure:
            raise JobFailed(failure)
        description = ""
        if body.describe:
            job.stage("describing the scene")
            description, said = studio.describe_for_outpaint(job_canvas.canvas, LANG)
            messages.append(said)
        request = GenerationRequest(
            prompt=outpaint.prompt(description, body.prompt),
            prompt_original=body.prompt,
            preset=preset,
            aspect=aspect.FOLLOW_REFERENCE,
            seed=body.seed,
            source=job_canvas.canvas,
            # Кадр — ровно размера холста, иначе вклейка оригинала разойдётся
            # с нарисованным (как у кнопки «Расширить»).
            size=job_canvas.size,
            reference_scale=job_canvas.resolution,
            loras=(outpaint_lora, *user_loras),
        )
        produced = _run(studio, job, request, messages)
        saved = []
        for item in produced:
            final = outpaint.stitch(source, item.image, job_canvas)
            parameters = dict(item.parameters, width=final.width, height=final.height, outpaint={
                "ratio": body.ratio or "", "sides": [] if body.ratio else list(body.sides),
                "amount": None if body.ratio else body.amount, "source_size": list(source.size),
            })
            saved.append(_save(final, parameters, item.seed, body.return_base64))
        return {"images": saved, "prompt": request.prompt, "messages": [line for line in messages if line]}

    return work


# --- видео -------------------------------------------------------------------------
#
# Те же шаги, что у вкладки Video: ``VideoRequest`` собирается здесь, ролик
# снимает ``Studio.run_video`` (освобождение карты от Qwen, задание движку,
# сохранение в галерею с соседним JSON). Загруженные клиентом референсы
# лежат в каталоге заданий FreeVideo до конца работы: движок всё равно
# копирует их в свой каталог входов.

VIDEO_INSTALL = "python -m fooocus_qwen --setup-video"
_DEFAULT_SUFFIX = {"video": ".mp4", "audio": ".wav"}


def _video_ready(studio) -> None:
    found = studio.video_status()
    if not found.ready:
        raise EngineUnavailable(f"the video engine is not ready: {found.reason}. Install it with: {VIDEO_INSTALL}")


def _video_loras(items: list[schemas.VideoLora]) -> tuple[tuple[str, float], ...]:
    """LoRA видео — файлы ``loras/video``; имя ложится в путь, поэтому только имя, без каталогов."""
    chosen = []
    for item in items:
        name = item.name if item.name.lower().endswith(".safetensors") else f"{item.name}.safetensors"
        if Path(name).name != name or "/" in name or "\\" in name:
            raise RequestError(f"bad video LoRA name: {item.name}")
        path = config.VIDEO_LORA_DIR / name
        if not path.is_file():
            raise RequestError(f"no video LoRA {item.name!r} (see GET {API_PREFIX}/video/loras)", not_found=True)
        chosen.append((str(path), float(item.strength)))
    return tuple(chosen)


def _gallery_video(relative: str, what: str) -> Path:
    try:
        path = images.output_path(relative)
    except images.ImageError as error:
        raise RequestError(f"{what}: {error}", not_found=error.not_found) from error
    if path.suffix.lower() not in images.VIDEO_SUFFIXES:
        raise RequestError(f"{what}: {relative} is not a video")
    return path


class _Uploads:
    """Каталог файлов-референсов из base64 на время одной работы; ``close`` его удаляет."""

    def __init__(self) -> None:
        self.directory: Path | None = None

    def path(self, name: str) -> Path:
        if self.directory is None:
            day = time.strftime("%Y-%m-%d")
            self.directory = config.VIDEO_RUNS_DIR / "api_uploads" / day / uuid.uuid4().hex[:12]
            self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / name

    def close(self) -> None:
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory = None


def _video_references(items: list[schemas.VideoReference], uploads: _Uploads) -> tuple:
    """Референсы клиента — файлами на диске, с видом; пределы набора — до очереди."""
    from ..video.request import Reference, check_references, kind_of

    found = []
    for index, item in enumerate(items):
        what = f"references[{index}]"
        if item.gallery:
            try:
                path = images.output_path(item.gallery)
            except images.ImageError as error:
                raise RequestError(f"{what}: {error}", not_found=error.not_found) from error
            kind = item.kind or kind_of(path)
            if kind is None:
                raise RequestError(f"{what}: {item.gallery} is neither a picture, a video nor a sound")
        else:
            kind = item.kind or (kind_of(item.name) if item.name else None)
            if kind is None:
                raise RequestError(f"{what}: give kind (image, video or audio) or a file name with an extension")
            try:
                if kind == "image":
                    path = uploads.path(f"ref{index:02d}.png")
                    images.decode_base64(item.base64).save(path)
                else:
                    suffix = Path(item.name).suffix.lower() if item.name and kind_of(item.name) == kind \
                        else _DEFAULT_SUFFIX[kind]
                    path = uploads.path(f"ref{index:02d}{suffix}")
                    path.write_bytes(images.decode_bytes(item.base64, what))
            except images.ImageError as error:
                raise RequestError(f"{what}: {error}") from error
        found.append(Reference(kind, str(path)))
    references = tuple(found)
    try:
        check_references(references)
    except ValueError as error:
        raise RequestError(str(error)) from error
    return references


def _video_progress(job: Job):
    """События движка — в прогресс задачи: доля, подпись, шаги сэмплинга, остаток."""
    from ..video.engine import progress_view

    def on_event(event: dict, prefix: str = "", share: tuple[int, int] = (0, 1)) -> None:
        view = progress_view(event)
        if view is None:
            return
        index, total = share
        progress = job.progress
        progress.stage = f"{prefix}{view.label or 'generating'}"
        progress.fraction = None if view.fraction is None else (index + view.fraction) / total
        progress.remaining_seconds = view.remaining_seconds
        if view.step:
            progress.step, progress.steps = view.step

    return on_event


def _video_entry(path, parameters: dict, return_base64: bool) -> dict[str, Any]:
    # Параметры — как в галерее, из соседнего JSON: там и ``engine_output``,
    # нужный доводке и продлению.
    parameters = _read_video(path) or parameters
    relative = images.relative_output(path)
    entry: dict[str, Any] = {
        "path": relative,
        "url": f"{API_PREFIX}/files/{relative}",
        "width": parameters.get("width"),
        "height": parameters.get("height"),
        "seconds": parameters.get("seconds"),
        "seed": parameters.get("seed"),
        "parameters": parameters,
    }
    if return_base64:
        entry["base64"] = images.encode_png(path)
    return entry


def _check_upscale(request) -> None:
    """Размер апскейла — до постановки в очередь: 400, а не сбой задания через минуту."""
    from ..video import geometry
    from ..video.request import frame_size

    try:
        geometry.upscaled_size(*frame_size(request), request.upscale)
    except ValueError as error:
        raise RequestError(f"upscale: {error}") from error


def _framed(mode: str, first, last) -> tuple:
    """Кадры, которые режим действительно берёт: первый — i2v и flf2v, последний — flf2v."""
    from ..video import prompt_h3

    return (first if mode in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V) else None,
            last if mode == prompt_h3.MODE_FLF2V else None)


def prepare_video(studio, body: schemas.VideoGenerateRequest) -> Work:
    from ..ui.i18n import say
    from ..video import board, geometry, prompt_h3
    from ..video.request import VideoRequest

    _video_ready(studio)
    follow = body.continue_from or body.refine_from
    if not follow:
        _need_prompt(body.prompt)
    notes: list[str] = []
    loras = _video_loras(body.loras)
    uploads = _Uploads()
    try:
        if follow:
            # Доводка и продление — по параметрам самого ролика, как кнопки вкладки.
            kind = board.EXTEND if body.continue_from else board.FINISH
            source = _gallery_video(follow, "continue_from" if body.continue_from else "refine_from")
            changes: dict[str, Any] = {}
            if kind == board.EXTEND:
                if body.prompt.strip():
                    changes.update(prompt=body.prompt, prompt_original=body.prompt)
                for name in ("seconds", "seed"):
                    if name in body.model_fields_set:
                        changes[name] = getattr(body, name)
                if loras:
                    changes["loras"] = loras
            elif body.prompt.strip() or loras:
                notes.append("refine_from finishes the draft as it is; prompt and loras are ignored.")
            # Размер выхода продления и доводки — как у самого ролика: части
            # разного размера не склеить, а черновик обещал свой апскейл.
            ignored = [name for name in ("mode", "first", "last", "references", "upscale")
                       if name in body.model_fields_set]
            if ignored:
                notes.append(f"Ignored with {'continue_from' if body.continue_from else 'refine_from'}: "
                             f"{', '.join(ignored)}.")
            try:
                request = board.follow_up_request(source, kind, body.quality, **changes)
            except board.NoEngineOutput as error:
                raise RequestError(f"{follow}: {error}") from error
        else:
            first = _load(body.first, "first") if body.first is not None else None
            last = _load(body.last, "last") if body.last is not None else None
            used_first, used_last = _framed(body.mode, first, last)
            if first is not None and used_first is None:
                notes.append(f"The first frame is ignored in mode '{body.mode}'.")
            if last is not None and used_last is None:
                notes.append(f"The last frame is ignored in mode '{body.mode}'.")
            if body.mode == prompt_h3.MODE_I2V and used_first is None:
                raise RequestError("mode 'i2v' needs a first frame")
            if body.mode == prompt_h3.MODE_FLF2V and used_first is None and used_last is None:
                raise RequestError("mode 'flf2v' needs a first or a last frame")
            references: tuple = ()
            if body.mode == prompt_h3.MODE_REF:
                references = _video_references(body.references, uploads)
                if not references:
                    raise RequestError("mode 'ref' needs at least one reference")
            elif body.references:
                notes.append(f"References are ignored in mode '{body.mode}'.")
            request = VideoRequest(
                prompt=body.prompt, prompt_original=body.prompt, mode=body.mode, resolution=body.resolution,
                aspect=body.aspect, seconds=body.seconds, seed=body.seed, quality=body.quality,
                first=used_first, last=used_last, references=references, loras=loras,
                upscale=body.upscale.settings() if body.upscale else None,
            )
            _check_upscale(request)
    except BaseException:
        uploads.close()
        raise

    def work(job: Job) -> dict:
        try:
            messages = list(notes)
            chosen = request
            own_prompt = body.prompt.strip() and not body.refine_from
            if body.boost and own_prompt:
                job.stage("boosting the prompt")
                frame_first, frame_last = _framed(chosen.mode, chosen.first, chosen.last)
                text, said = studio.boost_video(
                    body.prompt, chosen.mode, geometry.duration(chosen.seconds), chosen.references,
                    frame_first, frame_last, chosen.aspect, LANG,
                )
                messages.append(said)
                if text:
                    chosen = dataclasses.replace(chosen, prompt=text)
            elif own_prompt and prompt_h3.off_language(body.prompt):
                messages.append(say("video_not_english", LANG))
            job.stage("starting")
            path, parameters, failure = studio.run_video(chosen, LANG, _video_progress(job))
            if failure is not None:
                raise JobFailed(failure)
            return {
                "video": _video_entry(path, parameters, body.return_base64),
                "prompt": parameters.get("prompt_h3", chosen.prompt),
                "messages": [line for line in messages if line],
            }
        finally:
            uploads.close()

    return work


def video_boost(studio, body: schemas.VideoBoostRequest) -> dict[str, Any]:
    """Промт H3 языковой моделью — как кнопка «AI буст» вкладки Video."""
    from ..video import geometry, prompt_h3

    first = _load(body.first, "first") if body.first is not None else None
    last = _load(body.last, "last") if body.last is not None else None
    first, last = _framed(body.mode, first, last)
    uploads = _Uploads()
    try:
        references = _video_references(body.references, uploads) if body.mode == prompt_h3.MODE_REF else ()
        text, message = studio.boost_video(
            body.prompt, body.mode, geometry.duration(body.seconds), references, first, last, body.aspect, LANG,
        )
    finally:
        uploads.close()
    if not text:
        raise UpstreamError(message)
    return {"prompt": text, "message": message}


def storyboard_plan(studio, body: schemas.StoryboardRequest) -> dict[str, Any]:
    """Бриф — в шоты языковой моделью; строки таблицы вкладки — объектами."""
    from ..video import board

    rows, message = studio.plan_storyboard(body.brief, body.seconds, body.structure, body.aspect, LANG)
    if rows is None:
        raise UpstreamError(message)
    shots = board.shot_objects(rows)
    return {"shots": shots, "seconds": round(sum(float(shot["seconds"]) for shot in shots), 3), "message": message}


def prepare_storyboard_run(studio, body: schemas.StoryboardRunRequest) -> Work:
    """Все шоты подряд и склейка — та же ``video.board.run_board``, что у кнопки вкладки."""
    from ..ui.i18n import say
    from ..video import board

    _video_ready(studio)
    first = _load(body.first, "first") if body.first is not None else None
    loras = _video_loras(body.loras)
    planned = board.shots_from([shot.model_dump() for shot in body.shots])
    if not planned:
        raise RequestError("the storyboard has no shots with a prompt")
    if body.upscale:
        from ..video.request import VideoRequest

        _check_upscale(VideoRequest(prompt="", resolution=body.resolution, aspect=body.aspect, first=first,
                                    upscale=body.upscale.settings()))

    def work(job: Job) -> dict:
        report = _video_progress(job)
        total = len(planned)

        def on_event(index: int, event: dict) -> None:
            report(event, prefix=f"shot {index + 1} of {total}: ", share=(index, total))

        result = board.run_board(
            studio, planned, chain=body.chain, resolution=body.resolution, aspect=body.aspect, seed=body.seed,
            quality=body.quality, first=first, loras=loras, lang=LANG,
            title=lambda count: say("video_board_title", LANG, count=count),
            on_event=on_event, stopped=lambda: job.cancel_requested,
            on_joining=lambda: job.stage("joining the shots"),
            upscale=body.upscale.settings() if body.upscale else None,
        )
        if result.failure is not None:
            raise JobFailed(f"shot {result.failed_shot} failed: {result.failure}")
        if result.path is None:
            raise JobFailed("interrupted")
        sheet = images.relative_output(result.sheet) if result.sheet is not None else None
        return {
            "video": _video_entry(result.path, result.parameters, body.return_base64),
            "shots": [_video_entry(path, {}, False) for path in result.produced],
            "contact_sheet": {"path": sheet, "url": f"{API_PREFIX}/files/{sheet}"} if sheet else None,
            "seconds": round(result.seconds, 3),
        }

    return work


def _read_video(path) -> dict:
    from ..video.request import read_parameters

    return read_parameters(path)


def video_status(studio) -> dict[str, Any]:
    found = studio.video_status()
    return {"installed": found.cloned, "ready": found.ready, "reason": found.reason, "install": VIDEO_INSTALL}


def video_lora_list() -> list[str]:
    directory = config.VIDEO_LORA_DIR
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.safetensors"))


def video_catalogue() -> dict[str, Any]:
    """Значения полей видео — для клиентов, которые строят форму сами."""
    from ..video import geometry, prompt_h3, storyboard
    from ..video import request as video_request

    return {
        "modes": list(prompt_h3.MODES),
        "resolutions": list(geometry.RESOLUTIONS),
        "aspect_ratios": [*geometry.ASPECT_RATIOS, geometry.FOLLOW_IMAGE],
        "qualities": list(video_request.QUALITY),
        "default_quality": video_request.DEFAULT_QUALITY,
        "seconds": {"min": geometry.MIN_SECONDS, "max": geometry.MAX_SECONDS,
                    "single_max": geometry.SINGLE_MAX_SECONDS, "fps": geometry.FPS},
        "structures": list(storyboard.STRUCTURES),
        "transitions": list(storyboard.TRANSITIONS),
        "lora_dir": str(config.VIDEO_LORA_DIR),
    }


# --- позы --------------------------------------------------------------------------


def pose_entry_info(entry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "title": entry.title,
        "custom": entry.custom,
        "schematic": entry.schematic,
        "skeleton_url": f"{API_PREFIX}/poses/{entry.name}/skeleton",
        "preview_url": f"{API_PREFIX}/poses/{entry.name}/preview",
    }


def list_poses() -> list[dict[str, Any]]:
    from ..poses import library

    return [pose_entry_info(entry) for entry in library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())]


def add_pose(studio, body: schemas.PoseRequest) -> dict[str, Any]:
    """Фото → скелет DWPose → своя поза. Обложку рисует отдельная работа (``prepare_pose_cover``)."""
    from ..poses import detect, library

    photo = _load(body.image, "image")
    try:
        found = detect.to_pose(studio.pose_detector().detect(photo))
    except detect.NoPersonFound as error:
        raise RequestError("no person found in the photo") from error
    user = config.user_pose_dir()
    own = sum(1 for item in library.list_poses(config.POSE_LIBRARY_DIR, user) if item.custom)
    entry = library.add_custom(user, found, body.title.strip() or f"My pose {own + 1}")
    return pose_entry_info(images.pose_entry(entry.name))


def update_pose(name: str, title: str | None, schematic: bool | None) -> dict[str, Any]:
    from ..poses import library

    entry = pose(name)
    if title is not None:
        library.set_title(config.user_pose_dir(), entry.name, title)
    if schematic is not None:
        library.set_schematic(config.user_pose_dir(), entry.name, schematic)
    return pose_entry_info(pose(name))


def delete_pose(name: str) -> None:
    from ..poses import library

    library.delete(config.user_pose_dir(), pose(name))


def pose(name: str):
    try:
        return images.pose_entry(name)
    except images.ImageError as error:
        raise RequestError(str(error), not_found=True) from error


def prepare_pose_cover(studio, name: str) -> Work:
    """Обложка позы — Qwen-Image по скелету, в стиле каталога (как «Перерисовать обложку»)."""
    from ..poses import library, skeleton, tile

    entry = pose(name)

    def work(job: Job) -> dict:
        with Image.open(entry.skeleton) as opened:
            bones = opened.convert("RGB")
        try:
            keypoints = skeleton.load(entry.keypoints)
        except (OSError, ValueError):
            keypoints = None
        request = tile.request(bones, turbo_ready=studio.turbo_weights_present(), pose=keypoints)
        produced = _run(studio, job, request, [])
        chosen = produced[0].image
        if keypoints is not None and len(produced) > 1:
            try:
                chosen = tile.best([item.image for item in produced], keypoints, studio.pose_detector())
            except Exception:  # noqa: BLE001 — без распознавания остаётся первый кандидат
                LOGGER.exception("Could not score the cover candidates; keeping the first one")
        library.set_tile(entry, chosen)
        library.set_schematic(config.user_pose_dir(), entry.name, False)
        return pose_entry_info(pose(entry.name))

    return work


# --- языковая модель ---------------------------------------------------------------


def boost(studio, body: schemas.BoostRequest) -> dict[str, Any]:
    """Переписанный промт; недоступная языковая модель — ``UpstreamError`` (ответ 502)."""
    references = [_load(item, f"references[{index}]") for index, item in enumerate(body.references)]
    mode = body.mode or (boost_module.MODE_EDIT if references else boost_module.MODE_T2I)
    try:
        result = studio.boost_result(body.prompt, mode, references or None)
    except (LlmError, FileNotFoundError, ValueError, OSError) as error:
        raise UpstreamError(f"AI boost failed: {error}") from error
    return {"prompt": result.prompt, "aspect": result.wh_ratio, "images_skipped": result.images_skipped}


def describe(studio, body: schemas.DescribeRequest) -> dict[str, Any]:
    text, message = studio.describe_image(_load(body.image, "image"), LANG)
    if not text:
        raise UpstreamError(message)
    return {"text": text}


def llm_status(studio, check: bool) -> dict[str, Any]:
    """Адрес языковой модели; токен не отдаётся никогда — только, есть ли он."""
    try:
        endpoint = load_endpoint(config.ENDPOINT_FILE)
    except (OSError, ValueError):
        return {"configured": False}
    status: dict[str, Any] = {"configured": True, "address": endpoint.base_url, "token_set": bool(endpoint.token)}
    if check:
        try:
            client = studio.llm_client()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            status.update(reachable=False, error=str(error))
        else:
            status.update(reachable=True, model=client.model)
    return status


def store_llm(studio, address: str, token: str | None, forget_token: bool) -> dict[str, Any]:
    """Как «Сохранить» на вкладке настроек: пустой токен — оставить прежний."""
    try:
        current = load_endpoint(config.ENDPOINT_FILE)
    except (OSError, ValueError):
        current = None
    kept = None if forget_token else (token or (current.token if current else None))
    text = llm_setup.render(address.strip(), kept, current.backend if current else "")
    try:
        parse_endpoint_file(text)
    except ValueError as error:
        raise RequestError(f"bad address: {error}") from error
    config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
    return llm_status(studio, check=False)


def read_system_prompt(name: str) -> str:
    _system_prompt_name(name)
    return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")


def write_system_prompt(name: str, text: str) -> None:
    _system_prompt_name(name)
    (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")


def _system_prompt_name(name: str) -> None:
    # Имя ложится в путь записи: только из списка.
    if name not in SYSTEM_PROMPTS:
        raise RequestError(f"unknown system prompt: {name}; one of {', '.join(SYSTEM_PROMPTS)}", not_found=True)


# --- каталоги ----------------------------------------------------------------------


def preset_list() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "resolution": preset.output_resolution,
            "steps": preset.num_inference_steps,
            "fast": preset.turbo or bool(preset.transformer),
            "default": name == presets.DEFAULT,
        }
        for name, preset in ((name, presets.get(name)) for name in presets.NAMES)
    ]


def lora_list(studio) -> list[dict[str, Any]]:
    from ..engine import lora

    directory = studio.config.lora_dir
    listed = []
    for name in lora.list_names(directory):
        info = lora.inspect(lora.path_for(directory, name), name)
        listed.append({"name": name, "usable": info.usable, "problem": info.problem, "family": info.family,
                       "rank": info.rank, "layers": info.layers, "triggers": list(info.triggers)})
    return listed


def gallery_list(limit: int) -> list[dict[str, Any]]:
    """Картинки и ролики: параметры картинки — из PNG, ролика — из соседнего JSON."""
    from ..video.request import read_parameters as read_video

    listed = []
    for path in gallery.recent(config.OUTPUT_DIR, limit):
        video = path.suffix.lower() in images.VIDEO_SUFFIXES
        parameters = (read_video(path) if video else metadata.read_png(path)) or {}
        relative = images.relative_output(path)
        listed.append({"path": relative, "url": f"{API_PREFIX}/files/{relative}",
                       "kind": "video" if video else "image", "parameters": parameters})
    return listed


def prompt_get(name: str) -> dict[str, Any]:
    try:
        return prompt_library.load_prompt(name, config.PROMPT_DIR)
    except FileNotFoundError as error:
        raise RequestError(f"no saved prompt {name!r}", not_found=True) from error
    except ValueError as error:
        raise RequestError(f"saved prompt {name!r} is unreadable: {error}") from error


def prompt_save(name: str, payload: schemas.PromptPayload) -> None:
    if not name.strip():
        raise RequestError("the prompt name is empty")
    prompt_library.save_prompt(name, payload.model_dump(), config.PROMPT_DIR)


# --- настройки и состояние ---------------------------------------------------------


def settings_state(studio) -> dict[str, Any]:
    from ..engine import attention, hardware

    chosen = settings_module.load()
    vram = hardware.vram_gib()
    return {
        "precision": chosen.precision,
        "precisions": list(settings_module.PRECISIONS),
        "recommended_precision": settings_module.recommended_precision(vram),
        "sage_attention": chosen.sage_attention,
        "sage_available": attention.sage_available(),
        "memory_profile": chosen.memory_profile,
        "memory_profiles": list(settings_module.MEMORY_PROFILES),
        "resolved_profile": studio.memory_profile(),
        "vram_gib": vram,
    }


def prepare_settings(studio, body: schemas.SettingsPatch) -> Work:
    """Смена настроек — работой в очереди: точность и профиль выгружают модель."""

    def work(job: Job) -> dict:
        chosen = settings_module.load()
        reload = False
        if body.precision and body.precision != chosen.precision:
            job.stage(f"downloading {body.precision} weights")
            try:
                studio.ensure_precision_weights(body.precision)
            except Exception as error:  # noqa: BLE001 — сеть и диск: причина клиенту
                raise JobFailed(f"failed to download {body.precision} weights: {error}") from error
            studio.switch_precision(body.precision)
            reload = True
        if body.memory_profile and body.memory_profile != chosen.memory_profile:
            studio.switch_memory_profile(body.memory_profile)
            reload = True
        if body.sage_attention is not None:
            studio.set_sage_attention(body.sage_attention)
        if reload and studio.config.preload:
            studio.preload_in_background()
        return settings_state(studio)

    return work


def info(studio) -> dict[str, Any]:
    return {
        "version": __version__,
        "model_loaded": studio.model_loaded,
        "memory": studio.memory_report(LANG),
        "presets": list(presets.NAMES),
        "default_preset": presets.DEFAULT,
        "aspect_ratios": [ratio for ratio in aspect.ASPECT_RATIOS if ratio != aspect.FOLLOW_REFERENCE]
        + [schemas.FOLLOW],
        "mask_modes": [MASK_MASK, MASK_REGION, MASK_ANNOTATION, MASK_NONE],
        "outpaint_sides": list(outpaint.SIDES),
        "lora_dir": str(studio.config.lora_dir),
        "settings": settings_state(studio),
        "video": video_catalogue(),
    }
