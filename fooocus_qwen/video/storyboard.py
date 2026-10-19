"""Раскадровка: бриф ролика → список шотов, каждый — готовый промт H3.

Порядок работы взят из «вайб-монтажа»: сначала структура ролика (типовые
схемы ниже), потом шоты, потом исполнение. Одна генерация H3 — до 15 секунд,
поэтому ролик длиннее собирается из шотов, и каждый шот — отдельный
самодостаточный T2V-промт. Модель между генерациями ничего не помнит, так
что согласованность героя и стиля держится только текстом: одно и то же
описание внешности, одежды, света и палитры повторяется в каждом шоте
дословно — этого требует системный промт.

Ответ языковой модели разбирается терпимо: JSON бывает в блоке кода, с
текстом вокруг, после рассуждений ``<think>`` и с лишними запятыми.
Длительности приводятся к сетке кадров H3 и диапазону одной генерации,
неизвестные переходы становятся склейкой.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..llm import LlmClient
from . import geometry, prompt_h3

LOGGER = logging.getLogger(__name__)

# Переход В НАЧАЛО шота от предыдущего; у первого шота игнорируется. Имена —
# как у фильтра xfade ffmpeg, которым шоты склеиваются.
TRANSITIONS: tuple[str, ...] = ("cut", "dissolve", "fade", "wipeleft", "wiperight")
DEFAULT_TRANSITION = "cut"

# Обычная длина шота: 5–8 секунд. Короче — шот не успевает «постоять», длиннее
# — в одном шоте скапливается несколько событий.
SHOT_MIN_SECONDS = 5.0
SHOT_MAX_SECONDS = 8.0
DEFAULT_SHOT_SECONDS = 6.0

COLUMNS: tuple[str, ...] = ("№", "seconds", "transition", "title", "prompt")

PLAN_TEMPERATURE = 0.5
# Шот — промт H3 на 150–300 слов; минута ролика — до десятка шотов.
PLAN_MAX_TOKENS = 8000


@dataclass
class Shot:
    """Шот раскадровки: промт H3, длительность и переход в его начало."""

    prompt: str
    seconds: float
    transition: str = DEFAULT_TRANSITION
    title: str = ""


@dataclass(frozen=True)
class Structure:
    """Типовая структура ролика: подпись (ru, en), смысловые такты, типичная длина."""

    labels: tuple[str, str]
    beats: tuple[str, ...] = ()
    seconds: float | None = None


STRUCTURES: dict[str, Structure] = {
    "free": Structure(("Свободно — решает модель", "Free — the model decides")),
    "problem_action_result": Structure(
        ("Проблема → действие → результат (3 сцены, ~15 с)", "Problem → action → result (3 scenes, ~15 s)"),
        ("problem", "action", "result"),
        15.0,
    ),
    "promise_action_result_next": Structure(
        (
            "Обещание → действие → результат → что дальше (4 сцены, ~20 с)",
            "Promise → action → result → what's next (4 scenes, ~20 s)",
        ),
        ("promise", "action", "result", "what's next"),
        20.0,
    ),
    "explainer": Structure(
        (
            "Объяснение: вопрос, состав, действие, результат, ограничение (5 сцен, ~40 с)",
            "Explainer: question, ingredients, action, result, limitation (5 scenes, ~40 s)",
        ),
        ("question", "what it is made of", "how it works in action", "result", "limitation"),
        40.0,
    ),
    "story": Structure(
        ("История: завязка, развитие, кульминация, развязка", "Story: setup, development, climax, resolution"),
        ("setup", "development", "climax", "resolution"),
    ),
}
DEFAULT_STRUCTURE = "free"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[\]}])")
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_TIMECODE = re.compile(r"\b(\d{2}):(\d{2}\.\d{3})\b")

_PROMPT_KEYS = ("prompt", "h3_prompt", "description", "text")
_SECONDS_KEYS = ("seconds", "duration", "duration_seconds", "length")
_TITLE_KEYS = ("title", "name", "beat")
_TRANSITION_ALIASES = {
    "cut": "cut",
    "hard cut": "cut",
    "dissolve": "dissolve",
    "cross dissolve": "dissolve",
    "cross-dissolve": "dissolve",
    "crossfade": "dissolve",
    "fade": "fade",
    "fade to black": "fade",
    "fade through black": "fade",
    "wipeleft": "wipeleft",
    "wipe left": "wipeleft",
    "wipe": "wipeleft",
    "wiperight": "wiperight",
    "wipe right": "wiperight",
}


# --- Длительности -----------------------------------------------------------------------------

_STEP = geometry.FRAME_UNIT / geometry.FPS
# Ступени сетки 17·k + 5 в диапазоне одной генерации.
_K_MIN = math.ceil((geometry.MIN_SECONDS * geometry.FPS - geometry.FRAME_OFFSET) / geometry.FRAME_UNIT)
_K_MAX = (geometry.frames_for(geometry.SINGLE_MAX_SECONDS) - geometry.FRAME_OFFSET) // geometry.FRAME_UNIT
_TRAINED_K_MIN = (prompt_h3.TRAINED_MIN_FRAMES - geometry.FRAME_OFFSET) // geometry.FRAME_UNIT


def _step_seconds(k: int) -> float:
    return (geometry.FRAME_UNIT * k + geometry.FRAME_OFFSET) / geometry.FPS


def _nearest_step(seconds: float) -> int:
    k = round((seconds * geometry.FPS - geometry.FRAME_OFFSET) / geometry.FRAME_UNIT)
    return min(max(k, _K_MIN), _K_MAX)


def fit_seconds(seconds: float) -> float:
    """Длительность шота на ближайшей ступени сетки в диапазоне одной генерации.

    Ближайшей, а не следующей вверх, как у движка: длина шота — пожелание, и
    округление вверх систематически раздувало бы ролик на ~0.35 с за шот.
    Значение уже на сетке (5.17) остаётся на месте.
    """
    if not math.isfinite(seconds):
        seconds = DEFAULT_SHOT_SECONDS
    return _step_seconds(_nearest_step(seconds))


def fit_total(shots: Sequence[Shot], total_seconds: float) -> list[Shot]:
    """Длительности под общую длину ролика: масштаб, сетка, добор по ступени.

    Сумма попадает в ``total_seconds`` с точностью до полушага сетки
    (~0.35 с), насколько это позволяет диапазон шота. Добор идёт по одной
    ступени к самому короткому (или от самого длинного) шоту — так длины
    остаются ровными, а не копят разницу в одном шоте.

    Шот не укорачивается ниже обученного диапазона H3 (124 кадра, 5.17 с),
    если модель сама не просила короче: качество там не проверено, и лишние
    полсекунды ролика дешевле. Поэтому три шота на 15 секунд дают 15.5.
    """
    if not shots:
        return []
    raw = [max(shot.seconds, geometry.MIN_SECONDS) for shot in shots]
    floors = [min(_TRAINED_K_MIN, _nearest_step(seconds)) for seconds in raw]
    scale = total_seconds / sum(raw) if total_seconds > 0 else 1.0
    steps = [max(_nearest_step(seconds * scale), floor) for seconds, floor in zip(raw, floors)]
    for _ in range(len(steps) * (_K_MAX - _K_MIN) + 1):
        diff = total_seconds - sum(_step_seconds(k) for k in steps)
        if total_seconds <= 0 or abs(diff) <= _STEP / 2:
            break
        if diff > 0:
            candidates = [i for i, k in enumerate(steps) if k < _K_MAX]
            pick = min(candidates, key=lambda i: steps[i], default=None)
            delta = 1
        else:
            candidates = [i for i, k in enumerate(steps) if k > floors[i]]
            pick = max(candidates, key=lambda i: steps[i], default=None)
            delta = -1
        if pick is None:
            break
        steps[pick] += delta
    return [_retime(shot, _step_seconds(k)) for shot, k in zip(shots, steps)]


def _retime(shot: Shot, seconds: float) -> Shot:
    """Новая длительность шота; тайм-коды промта сжимаются, если вылезли за конец.

    Модель писала склейки под свою длину шота. Если подгонка под общую длину
    шот укоротила и склейка оказалась за концом, тайм-коды масштабируются
    пропорционально — иначе промт стал бы негодным.
    """
    codes = [int(minutes) * 60 + float(rest) for minutes, rest in _TIMECODE.findall(shot.prompt)]
    prompt = shot.prompt
    if codes and max(codes) >= seconds - prompt_h3.DIALOGUE_MARGIN and shot.seconds > 0:
        factor = seconds / max(shot.seconds, max(codes) + prompt_h3.DIALOGUE_MARGIN)
        prompt = _TIMECODE.sub(lambda m: geometry.timecode((int(m.group(1)) * 60 + float(m.group(2))) * factor), prompt)
    return replace(shot, prompt=prompt, seconds=seconds)


# --- Разбор ответа ----------------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    match = _NUMBER.search(_text(value))
    return float(match.group(0).replace(",", ".")) if match else None


def normalize_transition(value: Any) -> str:
    """Переход из списка ``TRANSITIONS``; всё незнакомое — склейка."""
    key = " ".join(_text(value).lower().replace("_", " ").split())
    return _TRANSITION_ALIASES.get(key, DEFAULT_TRANSITION)


def _first(item: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        for name, value in item.items():
            if str(name).lower() == key:
                return value
    return None


def _prompt_text(value: Any) -> str:
    # Модель иногда отдаёт поля H3 объектом, а не строкой.
    if isinstance(value, dict):
        return "\n\n".join(f"{name}: {_text(content)}" for name, content in value.items() if _text(content))
    return _text(value)


def _shot(item: Any) -> Shot | None:
    if isinstance(item, str):
        item = {"prompt": item}
    if not isinstance(item, dict):
        return None
    prompt = _prompt_text(_first(item, _PROMPT_KEYS))
    if not prompt:
        return None
    seconds = _number(_first(item, _SECONDS_KEYS))
    return Shot(
        prompt=prompt,
        seconds=fit_seconds(seconds if seconds is not None else DEFAULT_SHOT_SECONDS),
        transition=normalize_transition(_first(item, ("transition",))),
        title=_text(_first(item, _TITLE_KEYS)),
    )


def _shots_from(value: Any) -> list[Shot]:
    if isinstance(value, dict):
        nested = _first(value, ("shots", "storyboard", "scenes"))
        value = nested if isinstance(nested, list) else [value]
    if not isinstance(value, list):
        return []
    return [shot for shot in map(_shot, value) if shot is not None]


def _candidates(text: str) -> list[str]:
    raw = _THINK.sub("", text or "")
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    return [*_FENCE.findall(raw), raw]


def _decode(candidate: str) -> list[Shot]:
    """Первый JSON-массив (или объект со списком шотов) в тексте, где бы он ни стоял."""
    decoder = json.JSONDecoder()
    sources = (candidate, _TRAILING_COMMA.sub(r"\1", candidate))
    # Сначала массивы, потом объекты: в массиве с лишней запятой первым
    # разбирается его же первый элемент, и раскадровка сократилась бы до шота.
    for opener in "[{":
        for source in sources:
            for start, char in enumerate(source):
                if char != opener:
                    continue
                try:
                    value, _ = decoder.raw_decode(source, start)
                except json.JSONDecodeError:
                    continue
                shots = _shots_from(value)
                if shots:
                    return shots
    return []


def parse(text: str) -> list[Shot]:
    """Шоты из ответа модели; ``ValueError``, если шотов в ответе нет."""
    for candidate in _candidates(text):
        shots = _decode(candidate)
        if shots:
            shots[0] = replace(shots[0], transition=DEFAULT_TRANSITION)
            return shots
    preview = " ".join((text or "").split())[:120]
    raise ValueError(f"Storyboard response contains no JSON array of shots: {preview!r}")


# --- Планирование -----------------------------------------------------------------------------


def shot_count(total_seconds: float, structure: str = DEFAULT_STRUCTURE) -> tuple[int, int]:
    """Сколько шотов просить: по 5–8 с, не меньше числа тактов структуры."""
    beats = len(STRUCTURES.get(structure, STRUCTURES[DEFAULT_STRUCTURE]).beats)
    low = max(1, math.ceil(total_seconds / geometry.SINGLE_MAX_SECONDS), round(total_seconds / SHOT_MAX_SECONDS), beats)
    high = max(low, math.floor(total_seconds / SHOT_MIN_SECONDS))
    return low, high


def build_plan_message(brief: str, total_seconds: float, structure: str, aspect: str) -> str:
    """Сообщение пользователя для раскадровщика: бриф, длина, структура, формат ответа."""
    chosen = STRUCTURES.get(structure, STRUCTURES[DEFAULT_STRUCTURE])
    low, high = shot_count(total_seconds, structure)
    count = f"{low}" if low == high else f"{low}-{high}"
    lines = [
        f"Total duration: about {total_seconds:.0f} seconds; the shot lengths must add up to it.",
        f"Shots: {count}, each {geometry.MIN_SECONDS:g}-{geometry.SINGLE_MAX_SECONDS:g} seconds, normally "
        f"{SHOT_MIN_SECONDS:g}-{SHOT_MAX_SECONDS:g}.",
    ]
    if chosen.beats:
        lines.append(
            "Structure, in this order (one shot per beat, or a short group of shots for a long beat): "
            + " → ".join(chosen.beats)
            + "."
        )
    else:
        lines.append("Structure: choose the clearest order for this brief.")
    lines.append(prompt_h3.aspect_line(aspect))
    lines.append(
        "Return only a JSON array: "
        '[{"title": "...", "seconds": 6.0, "transition": "cut", "prompt": "integrated_multimodal_description: ..."}]'
    )
    lines.append(f"Brief:\n{brief.strip()}")
    return "\n\n".join(lines)


def plan(
    client: LlmClient,
    brief: str,
    total_seconds: float,
    *,
    structure: str = DEFAULT_STRUCTURE,
    aspect: str = geometry.DEFAULT_ASPECT,
    prompt_dir: Path,
) -> list[Shot]:
    """Раскадровка брифа языковой моделью: шоты с готовыми T2V-промтами H3.

    Длины подгоняются под ``total_seconds`` и сетку кадров, промт каждого
    шота проходит ``prompt_h3.compose`` — поля и строка запретов ставятся
    кодом. Нечитаемый ответ — ``ValueError`` с понятной причиной.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"Unknown storyboard structure: {structure!r}")
    total = min(max(float(total_seconds), geometry.MIN_SECONDS), geometry.MAX_SECONDS)
    system = prompt_h3.system_prompt(prompt_h3.STORYBOARD, prompt_dir)
    user = build_plan_message(brief, total, structure, aspect)
    answer = client.complete(system, user, temperature=PLAN_TEMPERATURE, max_tokens=PLAN_MAX_TOKENS)
    shots = fit_total(parse(answer), total)
    LOGGER.info("Storyboard: %d shots, %.2f s in total", len(shots), sum(shot.seconds for shot in shots))
    return [
        replace(shot, prompt=prompt_h3.compose(shot.prompt, prompt_h3.MODE_T2V, shot.seconds)) for shot in shots
    ]


# --- Таблица интерфейса -----------------------------------------------------------------------


def to_rows(shots: Sequence[Shot]) -> list[list]:
    """Строки для ``gr.Dataframe`` в порядке ``COLUMNS``."""
    return [
        [number, round(shot.seconds, 2), shot.transition, shot.title, shot.prompt]
        for number, shot in enumerate(shots, start=1)
    ]


def from_rows(rows: Any) -> list[Shot]:
    """Шоты из таблицы интерфейса.

    Принимает список строк или DataFrame (``gr.Dataframe`` отдаёт pandas).
    Номер в первой колонке не используется — порядок задают строки. Пустые
    строки пропускаются, секунды-строки («6,5», «6 s») читаются, длительность
    ставится на сетку: в таблице видно то, что получится на самом деле.
    """
    if rows is None:
        return []
    if hasattr(rows, "columns") and hasattr(rows, "values"):
        rows = rows.values.tolist()
    shots: list[Shot] = []
    for row in rows:
        cells = [*list(row or ()), *[""] * len(COLUMNS)][: len(COLUMNS)]
        _, seconds, transition, title, prompt = cells
        prompt = _text(prompt)
        if not prompt:
            continue
        value = _number(seconds)
        shots.append(
            Shot(
                prompt=prompt,
                seconds=fit_seconds(value if value is not None else DEFAULT_SHOT_SECONDS),
                transition=normalize_transition(transition),
                title=_text(title),
            )
        )
    return shots
