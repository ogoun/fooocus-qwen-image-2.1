"""Цепочки роликов без интерфейса: доводка черновика, продление и раскадровка целиком.

Общая логика вкладки Video и HTTP API: обе дороги собирают одинаковые
``VideoRequest`` и идут через ``Studio.run_video``. Gradio здесь нет —
строки состояния и подписи на нужном языке передаёт вызывающий.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .. import config
from ..storage import gallery
from . import geometry, prompt_h3, storyboard
from .request import VideoRequest, next_video_path, read_parameters, resolve_seed, save_result

FINISH = "finish"
EXTEND = "extend"
# Черновик доводится уровнем light: черновик — это его первый проход.
FINISH_QUALITY = "light"


class NoEngineOutput(ValueError):
    """У ролика не осталось рабочих файлов движка: доводить и продлевать нечего."""


def follow_up_request(video: str | Path, kind: str, quality: str, **changes: Any) -> VideoRequest:
    """Запрос «Довести до финала» (``FINISH``) или «Продлить ролик» (``EXTEND``) по готовому ролику.

    Параметры берутся у ролика, а не у текущих полей: доводится и
    продлевается именно он. ``changes`` — явные замены полей запроса
    (API позволяет задать продолжению свой промт, длительность, сид).
    """
    parameters = read_parameters(video)
    engine_output = parameters.get("engine_output")
    if not engine_output:
        raise NoEngineOutput("this video has no engine files left; make it again")
    if kind == FINISH:
        follow = dict(refine_from=engine_output, quality=FINISH_QUALITY)
    elif kind == EXTEND:
        # Продолжается последняя часть: у склейки нет записи латентов.
        last = parameters.get("engine_continue")
        follow = dict(continue_from=str((config.VIDEO_RUNS_DIR / (last or engine_output)).resolve()),
                      continue_whole=str((config.VIDEO_RUNS_DIR / engine_output).resolve()) if last else None,
                      quality=FINISH_QUALITY if quality == "draft" else quality)
    else:
        raise ValueError(f"Unknown follow-up: {kind}")
    values: dict[str, Any] = dict(
        prompt=parameters.get("prompt_h3") or parameters.get("prompt") or "",
        prompt_original=parameters.get("prompt") or "",
        mode=prompt_h3.MODE_T2V,
        resolution=parameters.get("resolution", geometry.DEFAULT_RESOLUTION),
        aspect=parameters.get("aspect", geometry.DEFAULT_ASPECT),
        # Из кадров, а не из округлённых секунд: 5.167 с округлились бы
        # вверх до следующей ступени сетки (141 кадр вместо 124). Продление
        # длится столько, сколько прошлая часть, но не дольше одной части:
        # продлённый или длинный ролик иначе удваивался бы целиком.
        seconds=_follow_seconds(parameters, kind),
        seed=int(parameters.get("seed", -1)),
        # Продление и доводка — в размере ролика: склейка частей разного
        # размера невозможна, а доводка черновика делает то, что он обещал.
        upscale=geometry.Upscale.from_dict(parameters.get("upscale")),
        **follow,
    )
    values.update(changes)
    return VideoRequest(**values)


def _follow_seconds(parameters: dict[str, Any], kind: str) -> float:
    frames = int(parameters.get("frames") or geometry.frames_for(5.0))
    if kind == EXTEND:
        frames = min(int(parameters.get("part_frames") or frames), geometry.frames_for(geometry.SINGLE_MAX_SECONDS))
    return frames / geometry.FPS


def shots_from(items: Sequence[dict]) -> list[storyboard.Shot]:
    """Шоты из объектов ``{seconds, transition, title, prompt}`` — та же нормализация, что у таблицы."""
    rows = [[None, item.get("seconds"), item.get("transition"), item.get("title"), item.get("prompt")]
            for item in items]
    return storyboard.from_rows(rows)


def shot_objects(rows: Sequence[Sequence]) -> list[dict[str, Any]]:
    """Строки таблицы (``storyboard.to_rows``) — в объекты без номера."""
    return [
        {"seconds": row[1], "transition": row[2], "title": row[3], "prompt": row[4]}
        for row in rows
    ]


@dataclass
class BoardResult:
    """Итог раскадровки.

    ``produced`` — снятые шоты; ``path`` — склейка (нет, если шот упал или
    сняли ни одного); ``failed_shot`` — номер упавшего шота с единицы.
    """

    produced: list[Path] = field(default_factory=list)
    path: Path | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    sheet: Path | None = None
    seconds: float = 0.0
    failed_shot: int | None = None
    failure: str | None = None


ShotEvent = Callable[[int, dict], None]


def run_board(
    studio,
    planned: Sequence[storyboard.Shot],
    *,
    chain: bool,
    resolution: str,
    aspect: str,
    seed: int,
    quality: str,
    first: Image.Image | None,
    loras: tuple[tuple[str, float], ...],
    lang: str,
    title: Callable[[int], str],
    on_event: ShotEvent | None = None,
    stopped: Callable[[], bool] = lambda: False,
    on_joining: Callable[[], None] | None = None,
    upscale: geometry.Upscale | None = None,
) -> BoardResult:
    """Все шоты подряд (кадр-цепочка по желанию), затем склейка с переходами и контактный лист.

    ``title(count)`` — подпись склейки в параметрах (на языке вызывающего);
    ``on_event(index, event)`` — события движка текущего шота (с нуля);
    ``stopped()`` проверяется перед каждым шотом: прерывание текущего ролика
    не должно запускать следующий.
    """
    from . import assemble

    result = BoardResult()
    previous_frame = first
    base_seed = resolve_seed(seed)
    for index, shot in enumerate(planned):
        if stopped():
            break
        start_frame = previous_frame if (index == 0 or chain) else None
        request = VideoRequest(
            prompt=shot.prompt,
            prompt_original=shot.prompt,
            mode=prompt_h3.MODE_I2V if start_frame is not None else prompt_h3.MODE_T2V,
            resolution=resolution,
            aspect=aspect,
            seconds=float(shot.seconds),
            seed=base_seed + index,
            quality=quality,
            first=start_frame,
            loras=tuple(loras),
            upscale=upscale,
            extra={"storyboard_shot": index + 1, "storyboard_title": shot.title},
        )
        callback = None if on_event is None else (lambda event, _index=index: on_event(_index, event))
        path, _parameters, failure = studio.run_video(request, lang, callback)
        if failure is not None:
            result.failed_shot, result.failure = index + 1, failure
            return result
        result.produced.append(Path(path))
        if chain:
            previous_frame = assemble.last_frame(path)
    if not result.produced:
        return result
    if on_joining is not None:
        on_joining()
    run_dir = config.VIDEO_RUNS_DIR / "storyboards"
    run_dir.mkdir(parents=True, exist_ok=True)
    joined = next_video_path(run_dir)
    joined.parent.mkdir(parents=True, exist_ok=True)
    count = len(result.produced)
    transitions = [shot.transition for shot in planned[1:count]]
    assemble.concat([str(path) for path in result.produced], joined, transitions=transitions)
    result.parameters = {
        "kind": "video", "storyboard": [vars(shot).copy() for shot in planned[:count]],
        "shots": [path.name for path in result.produced], "quality": quality,
        "resolution": resolution, "aspect": aspect, "seed": base_seed, "prompt": title(count),
        "upscale": upscale.as_dict() if upscale else None,
    }
    result.path = save_result(joined, config.OUTPUT_DIR, result.parameters, config.VIDEO_RUNS_DIR)
    # Контактный лист — рядом с роликом в папке результатов: там его и ищут.
    # Галерея его не показывает (``gallery.SHEET_SUFFIX``).
    result.sheet = assemble.contact_sheet([str(path) for path in result.produced],
                                          result.path.with_suffix(gallery.SHEET_SUFFIX))
    result.seconds = assemble.probe(result.path).seconds
    return result
