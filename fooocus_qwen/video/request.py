"""Запрос на видео: от полей вкладки — к заданию драйвера и обратно к файлу.

Здесь три шага, и все — чистая логика без видеокарты:

1. ``build_job`` — холст, длительность по сетке кадров, подготовка входов
   (кадры обрезаются под холст, референсы копируются в каталог задания),
   итоговый промт H3 со строкой выравнивания и метками.
2. Задание исполняет ``engine.VideoEngine``.
3. ``save_result`` — ролик ложится в каталог результатов рядом с картинками,
   параметры — соседним JSON (у MP4 нет места, удобного для них, как у PNG).
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .. import __version__
from ..storage import gallery
from . import geometry, prompt_h3

MODES = (prompt_h3.MODE_T2V, prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V, prompt_h3.MODE_REF)

# Уровни качества — уровни самого FreeVideo (``freevideo_engine/tiers.py``),
# плюс черновик: первый проход в половинном размере, который потом можно
# довести до финала, не генерируя заново (``refine_from``). Порядок —
# от быстрого к медленному; так их и показывает вкладка.
QUALITY: dict[str, dict[str, Any]] = {
    "draft": dict(two_pass=True, base_steps=8, refine_steps=3, step_cache=None, upscale=2, preview=True),
    "light": dict(two_pass=True, base_steps=8, refine_steps=3, step_cache=None, upscale=2),
    "medium": dict(two_pass=True, base_steps=8, refine_steps=3, step_cache=None, upscale=1.5),
    "high": dict(two_pass=False, base_steps=16, refine_steps=3, step_cache="easycache-av", upscale=2),
    "max": dict(two_pass=False, base_steps=20, refine_steps=3, step_cache="easycache-av", upscale=2),
}
DEFAULT_QUALITY = "light"

# Референсы: до 32 у движка, но модель обучена на 9 картинках, 3 видео и
# 3 аудио (до 12 файлов) — дальше не обещаем.
MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
# Продолжение ролика берёт из референсов только картинки, не больше трёх.
CONTINUATION_IMAGES = 3
# Крупные снимки не дают модели ничего, кроме токенов: короткая сторона
# референса до 1536 (``max`` ноды ComfyUI берёт до 2048, но лицу хватает и
# 1536, а бюджет памяти — общий с кадром).
REFERENCE_SHORT_SIDE = 1536

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

SIDECAR_SUFFIX = ".json"


@dataclass(frozen=True)
class Reference:
    """Референс: картинка, видео или звук — файлом на диске."""

    kind: str
    path: str


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    mode: str = prompt_h3.MODE_T2V
    resolution: str = geometry.DEFAULT_RESOLUTION
    aspect: str = geometry.DEFAULT_ASPECT
    seconds: float = 5.0
    seed: int = -1
    quality: str = DEFAULT_QUALITY
    first: Image.Image | None = None
    last: Image.Image | None = None
    references: tuple[Reference, ...] = ()
    loras: tuple[tuple[str, float], ...] = ()
    # Продолжить готовый ролик (его хвост — контекст новой части).
    continue_from: str | None = None
    # Довести черновик до финала: путь черновика внутри каталога заданий.
    refine_from: str | None = None
    # Исходный текст до буста — для метаданных.
    prompt_original: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def kind_of(path: str | Path) -> str | None:
    """Вид референса по расширению: ``image``, ``video``, ``audio`` или None."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return None


def resolve_seed(seed: int) -> int:
    return random.randrange(2**31) if seed is None or int(seed) < 0 else int(seed)


def frame_size(request: VideoRequest) -> tuple[int, int]:
    """Холст запроса; «как у кадра» смотрит на первый, иначе на последний кадр."""
    anchor = request.first or request.last
    return geometry.canvas(request.resolution, request.aspect, anchor.size if anchor else None)


def fit_frame(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Кадр точно в холст: обрезка по центру и масштаб, без искажения пропорций.

    Первый кадр модель берёт как якорь геометрии. Растянутый на холст другой
    пропорции (так делает нода ComfyUI), он искажён, и модель развивает уже
    искажённую сцену. Обрезка теряет края, но не геометрию.
    """
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def _reference_image(source: Path, target: Path) -> None:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    short = min(image.size)
    if short > REFERENCE_SHORT_SIDE:
        scale = REFERENCE_SHORT_SIDE / short
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    image.save(target)


def check_references(references: tuple[Reference, ...]) -> None:
    """Пределы набора референсов — до генерации, а не ошибкой движка через минуту."""
    counts = {kind: sum(1 for item in references if item.kind == kind) for kind in ("image", "video", "audio")}
    if counts["image"] > MAX_IMAGES or counts["video"] > MAX_VIDEOS or counts["audio"] > MAX_AUDIOS:
        raise ValueError(
            f"Too many references: up to {MAX_IMAGES} images, {MAX_VIDEOS} videos and {MAX_AUDIOS} audio files"
        )
    if counts["audio"] and not (counts["image"] or counts["video"]):
        raise ValueError("An audio reference needs at least one image or video reference")


def engine_references(references: tuple[Reference, ...]) -> list[Reference]:
    """Референсы в порядке, в котором их нумерует модель: картинки, видео, звук."""
    order = {"image": 0, "video": 1, "audio": 2}
    return sorted(references, key=lambda item: order.get(item.kind, 3))


def build_job(
    request: VideoRequest,
    run_dir: Path,
    runs_root: Path,
    video_has_audio=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Задание драйвера и параметры для метаданных.

    ``run_dir`` — каталог входов этого задания; ``runs_root`` — каталог
    заданий FreeVideo (там же его кэш результатов и черновики).
    ``video_has_audio(path) -> bool`` — есть ли звук у видеореференса: от
    этого зависит нумерация меток ``<Audio j>``.
    """
    if request.mode not in MODES:
        raise ValueError(f"Unknown video mode: {request.mode}")
    quality = QUALITY.get(request.quality, QUALITY[DEFAULT_QUALITY])
    duration = geometry.duration(request.seconds)
    width, height = frame_size(request)
    seed = resolve_seed(request.seed)
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    media: dict[str, Any] = {"version": 1}
    has_first = has_last = False
    references: list[Reference] = []
    if request.mode in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V) and request.first is not None:
        fit_frame(request.first, (width, height)).save(inputs / "first.png")
        media["first"] = str(inputs / "first.png")
        has_first = True
    if request.mode == prompt_h3.MODE_FLF2V and request.last is not None:
        fit_frame(request.last, (width, height)).save(inputs / "last.png")
        media["last"] = str(inputs / "last.png")
        has_last = True
    if request.mode == prompt_h3.MODE_I2V and not has_first:
        raise ValueError("Image to video needs a first frame")
    if request.mode == prompt_h3.MODE_FLF2V and not (has_first or has_last):
        raise ValueError("First and last frame mode needs at least one frame")
    if request.mode == prompt_h3.MODE_REF:
        check_references(request.references)
        references = engine_references(request.references)
        if not references:
            raise ValueError("Reference mode needs at least one reference")
        rows = []
        for index, item in enumerate(references, start=1):
            source = Path(item.path)
            if item.kind == "image":
                target = inputs / f"ref{index:02d}.png"
                _reference_image(source, target)
            else:
                target = inputs / f"ref{index:02d}{source.suffix.lower()}"
                shutil.copyfile(source, target)
            rows.append({"path": str(target), "kind": item.kind})
        media["references"] = rows
    if request.loras:
        media["loras"] = [{"path": str(path), "strength": float(strength)} for path, strength in request.loras]

    kinds = [item.kind for item in references]
    has_audio = [bool(video_has_audio(item.path)) if video_has_audio else True
                 for item in references if item.kind == "video"]
    labels = prompt_h3.reference_labels(kinds, has_audio) if references else []
    prompt = prompt_h3.compose(
        request.prompt, request.mode, duration, has_first=has_first, has_last=has_last, labels=labels,
    )

    # Продолжению ролика нельзя кадров и видео со звуком — только картинки.
    continuation_media = None
    images = [row for row in media.get("references", []) if row["kind"] == "image"][:CONTINUATION_IMAGES]
    if images or request.loras:
        continuation_media = {"version": 1, "references": images, "loras": media.get("loras", [])}

    has_media = any(key in media for key in ("first", "last", "references", "loras"))
    job = {
        "prompts": [prompt],
        "width": width,
        "height": height,
        "seconds": duration.seconds,
        "seed": seed,
        "quality": {key: value for key, value in quality.items() if key != "preview"},
        "preview": bool(quality.get("preview")),
        "media": media if has_media else None,
        "continuation_media": continuation_media,
        "continue_from": request.continue_from,
        "refine_from": request.refine_from,
        "output_dir": str(runs_root),
    }
    parameters = {
        "app_version": __version__,
        "kind": "video",
        "model": "MiniMax-H3 (FreeVideo VDN-H3)",
        "prompt": request.prompt_original or request.prompt,
        "prompt_h3": prompt,
        "mode": request.mode,
        "resolution": request.resolution,
        "aspect": request.aspect,
        "width": width,
        "height": height,
        "seconds": round(duration.seconds, 3),
        "requested_seconds": float(request.seconds),
        "frames": duration.frames,
        "fps": geometry.FPS,
        "seed": seed,
        "quality": request.quality if request.quality in QUALITY else DEFAULT_QUALITY,
        "references": [{"kind": item.kind, "name": Path(item.path).name} for item in references],
        "labels": labels,
        "loras": [{"name": Path(path).name, "strength": float(strength)} for path, strength in request.loras],
        "continue_from": request.continue_from,
        **request.extra,
    }
    return job, parameters


def next_video_path(directory: Path, when: datetime | None = None) -> Path:
    return gallery.next_path(directory, when, suffix=".mp4")


def save_result(produced: str | Path, directory: Path, parameters: dict[str, Any], runs_root: Path) -> Path:
    """Готовый ролик — в каталог результатов, параметры — соседним JSON.

    Копия, а не перенос: в каталоге заданий FreeVideo держит кэш результатов
    (тот же запрос вернётся мгновенно), а черновик нужен на месте для доводки
    — путь к нему относительно ``runs_root`` и сохраняется в параметрах.
    """
    produced = Path(produced)
    destination = next_video_path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, destination)
    stored = dict(parameters)
    try:
        stored["engine_output"] = produced.resolve().relative_to(Path(runs_root).resolve()).as_posix()
    except ValueError:
        stored["engine_output"] = str(produced)
    sidecar(destination).write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def sidecar(video: Path) -> Path:
    return Path(video).with_suffix(SIDECAR_SUFFIX)


def read_parameters(video: str | Path) -> dict[str, Any]:
    """Параметры ролика из соседнего JSON; пустой словарь, если его нет или он испорчен."""
    try:
        value = json.loads(sidecar(Path(video)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
