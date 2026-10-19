"""Промт MiniMax H3: метки референсов, строки выравнивания, сборка, проверка и AI-буст.

В облачном конвейере MiniMax между человеком и моделью стоит модуль
``H3-Context-IR``: он превращает свободный запрос в структурированный
документ, и DiT обучался именно на его выходе. Локально модуля нет, поэтому
промт обязан воспроизводить тот же формат буквально — имена полей, порядок
секций, пустые строки, тайм-коды ``MM:SS.mmm``, теги ``<d>`` и метки
``<Picture N>``. Правила взяты из официальных гайдов MiniMax и замеров стенда
(скилл ``minimax-h3-prompt``); здесь они превращены в код, чтобы промт,
написанный вручную или языковой моделью, уходил в движок проверенным.

Свободный текст ``compose`` не переводит и не дописывает по смыслу — только
оборачивает в поля режима. Перевод и раскадровка — работа LLM-буста
(``boost``), системные промты которого лежат файлами в ``resources/prompts``
и перечитываются при каждом запросе, как у буста изображений.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..llm import LlmClient, LlmImagesRejected
from . import geometry
from .geometry import Duration

LOGGER = logging.getLogger(__name__)

MODE_T2V = "t2v"
MODE_I2V = "i2v"
# Первый и/или последний кадр. Только последний — L2VA: тот же вход ноды,
# другая строка выравнивания.
MODE_FLF2V = "flf2v"
MODE_REF = "ref"
MODES: tuple[str, ...] = (MODE_T2V, MODE_I2V, MODE_FLF2V, MODE_REF)
# Не режим генерации, а системный промт раскадровки (``storyboard``).
STORYBOARD = "storyboard"

KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_AUDIO = "audio"

# Запреты — отдельной строкой в самом конце, после non_diegetic_music.
# Negative prompt у H3 нет, а отрицания внутри описания сцены вызывают то,
# что запрещают (замер на стенде: +55 % раздувания от списка «не должно»).
# В официальном формате строки нет — это наше добавление против водяных знаков.
NEGATIVE_LINE = "No text, subtitles, logos or watermarks."

BASE_FIELDS: tuple[str, ...] = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF_FIELDS: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# Шаблоны строк выравнивания — дословно из официального гайда. У FL2VA метки
# без угловых скобок, у I2VA и L2VA со скобками: это расхождение первоисточника,
# модель училась на нём, поэтому его не унифицируем.
I2VA_LINE = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
FL2VA_LINE = (
    "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
    "0.00-second mark of the target video; Picture 2 (from Shot {shot}) aligns with the {seconds}-second mark "
    "of the target video."
)
L2VA_LINE = (
    "How the reference pictures align with the target video — <Picture 1> (from [Shot {shot}]) aligns with the "
    "{seconds}-second mark of the target video."
)

# Умолчания для свободного текста: поля обязаны быть, а звук пользователь
# описывает редко. N/A в overall_soundscape гайд разрешает только при явной
# тишине, поэтому фон — нейтральной фразой; закадровой музыки по умолчанию нет.
DEFAULT_SOUNDSCAPE = "Natural ambient sound of the scene, with soft physical sounds synchronized to the visible actions."
DEFAULT_MUSIC = "N/A"

# Запас после последней реплики: впритык к концу её срезает.
DIALOGUE_MARGIN = 0.3
# Темп речи для оценки длины реплики: ~2.5 слова в секунду — средний темп
# спокойной речи; русская медленнее, оценка от этого только осторожнее.
WORDS_PER_SECOND = 2.5
# Обученный диапазон H3: 124…362 кадра. Короче движок примет, но качество
# там не проверено.
TRAINED_MIN_FRAMES = 124

BOOST_TEMPERATURE = 0.4
BOOST_MAX_TOKENS = 3000

_PROMPT_FILES = {
    MODE_T2V: "video_system_t2v.md",
    MODE_I2V: "video_system_i2v.md",
    MODE_FLF2V: "video_system_flf2v.md",
    MODE_REF: "video_system_ref.md",
    STORYBOARD: "video_system_storyboard.md",
}

# Плейсхолдеры системных промтов: числа и строка запретов живут в коде, и
# файл, поправленный пользователем, не разойдётся с проверкой.
_PLACEHOLDERS = {
    "{{FPS}}": str(geometry.FPS),
    "{{MIN_SECONDS}}": f"{geometry.MIN_SECONDS:g}",
    "{{MAX_SECONDS}}": f"{geometry.SINGLE_MAX_SECONDS:g}",
    "{{NEGATIVE_LINE}}": NEGATIVE_LINE,
}

_FIELD = re.compile(r"^[ \t]*(" + "|".join(sorted(set(BASE_FIELDS + REF_FIELDS))) + r")[ \t]*:", re.MULTILINE)
_MAIN_FIELD = re.compile(r"^[ \t]*(integrated_multimodal_description|detailed_description)[ \t]*:", re.MULTILINE)
_ALIGNMENT = re.compile(
    r"^[ \t]*(?:For the target video, at |How the reference pictures align with the target video).*$\n?",
    re.MULTILINE,
)
_NEGATIVE = re.compile(r"[ \t]*No text, subtitles, logos,? (?:and|or) watermarks\.?[ \t]*", re.IGNORECASE)
_SHOT = re.compile(r"\[Shot (\d+)\]")
_SHOT_CUT = re.compile(r"\s*At (\S+?),")
_TIMECODE = re.compile(r"^\d{2}:\d{2}\.\d{3}$")
# Всё, похожее на время в тексте: «00:03.500», «0:03.5», «03:05».
_ANY_TIME = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}(?:\.\d+)?(?![\d:])")
_LABEL = re.compile(r"<(Picture|Video|Audio|Subject) (\d+)>")
_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»')
# Облачный синтаксис Hailuo: вход продукта до Context-IR, локальной модели
# ничего не говорит.
_CLOUD = re.compile(
    r"@\s*(?:image|picture|video|audio)\s*\d+|\b(?:input|example|reference)\s+(?:image|video|audio)\s+\d+"
    r"|(?<![<\w])image\s+\d+\b",
    re.IGNORECASE,
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.DOTALL)

# Пометка к запросу, когда модель не приняла картинки: как и в бусте
# изображений, без неё переписыватель выдумывает содержимое кадра.
BLIND_NOTE = (
    "\n\nNote: the reference images are NOT available to you — you cannot see them. Do not guess, describe or "
    "name anything they might contain. Refer to them only by their labels and describe the requested motion, "
    "action and sound."
)


# --- Метки референсов ---------------------------------------------------------------------------


_ENGINE_RANK = {KIND_IMAGE: 0, KIND_VIDEO: 1, KIND_AUDIO: 2}


def _check_kinds(kinds: Sequence[str]) -> None:
    for kind in kinds:
        if kind not in _ENGINE_RANK:
            raise ValueError(f"Unknown reference kind: {kind!r}")


def engine_order(kinds: Sequence[str]) -> list[int]:
    """Перестановка входов в порядок подачи движку: индексы ``kinds``.

    Нумерацию меток задаёт порядок подачи в ноду, а не порядок упоминания в
    промте: сначала все изображения, затем видео, затем самостоятельные аудио.
    Внутри типа порядок пользователя сохраняется.
    """
    _check_kinds(kinds)
    return sorted(range(len(kinds)), key=lambda index: (_ENGINE_RANK[kinds[index]], index))


def labels_by_input(kinds: Sequence[str], video_has_audio: Sequence[bool] | None = None) -> list[tuple[str, ...]]:
    """Метки каждого входа в порядке пользователя — подписать файл в интерфейсе.

    Видео со звуковой дорожкой получает две метки: ``<Audio j>`` дорожки и
    ``<Video k>`` (дорожка подаётся перед видео). ``video_has_audio`` — по
    одному флагу на каждое видео в порядке пользователя; ``None`` — звук
    ни у одного не используется: обычное видео ``<Audio N>`` не порождает
    только оттого, что в файле есть звук.
    """
    _check_kinds(kinds)
    videos = sum(1 for kind in kinds if kind == KIND_VIDEO)
    flags = list(video_has_audio) if video_has_audio is not None else [False] * videos
    if len(flags) != videos:
        raise ValueError(f"video_has_audio has {len(flags)} flags for {videos} videos")
    video_flag = {}
    for index in (i for i, kind in enumerate(kinds) if kind == KIND_VIDEO):
        video_flag[index] = bool(flags[len(video_flag)])

    labels: list[tuple[str, ...]] = [()] * len(kinds)
    counters = {KIND_IMAGE: 0, KIND_VIDEO: 0, KIND_AUDIO: 0}
    # Нумерация по порядку подачи: дорожки видео идут раньше самостоятельных
    # аудио, поэтому и номера <Audio j> у них младше.
    for index in engine_order(kinds):
        kind = kinds[index]
        if kind == KIND_IMAGE:
            counters[KIND_IMAGE] += 1
            labels[index] = (f"<Picture {counters[KIND_IMAGE]}>",)
        elif kind == KIND_VIDEO:
            counters[KIND_VIDEO] += 1
            video = f"<Video {counters[KIND_VIDEO]}>"
            if video_flag[index]:
                counters[KIND_AUDIO] += 1
                labels[index] = (f"<Audio {counters[KIND_AUDIO]}>", video)
            else:
                labels[index] = (video,)
        else:
            counters[KIND_AUDIO] += 1
            labels[index] = (f"<Audio {counters[KIND_AUDIO]}>",)
    return labels


def reference_labels(kinds: Sequence[str], video_has_audio: Sequence[bool] | None = None) -> list[str]:
    """Все метки в порядке подачи движку (``engine_order``), а не в порядке ``kinds``.

    Пример: ``["audio", "video", "image"]`` с звуком у видео →
    ``["<Picture 1>", "<Audio 1>", "<Video 1>", "<Audio 2>"]``.
    """
    per_input = labels_by_input(kinds, video_has_audio)
    return [label for index in engine_order(kinds) for label in per_input[index]]


# --- Строка выравнивания ----------------------------------------------------------------------


def _as_duration(value: Duration | float) -> Duration:
    # Секунды пересчитываются через сетку: «5» становится 5.17. Уже
    # фактическая длительность, округлённая до сотых (5.17), остаётся на своём
    # кадре: сетка ``duration`` округляет вверх, и 5.17·24 = 124.08 ушло бы на
    # следующую ступень, 141 кадр.
    if isinstance(value, Duration):
        return value
    seconds = float(value)
    frames = round(seconds * geometry.FPS)
    if frames % geometry.FRAME_UNIT == geometry.FRAME_OFFSET and abs(frames / geometry.FPS - seconds) < 0.006:
        return Duration(frames=frames, seconds=frames / geometry.FPS)
    return geometry.duration(seconds)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown video mode: {mode!r}")


def alignment_line(
    mode: str,
    duration: Duration | float,
    has_first: bool = False,
    has_last: bool = False,
    *,
    last_shot: int = 1,
) -> str:
    """Первая строка промта для режимов с опорными кадрами; пустая для T2V и Ref.

    ``S.SS`` — фактическая длительность с двумя знаками: «5 секунд» — это
    5.17, и строка с 5.00 разведёт конец описания с концом ролика.
    ``last_shot`` — номер последнего шота: последний кадр принадлежит ему.
    Режим i2v даёт I2VA независимо от флагов; flf2v — по флагам: оба кадра —
    FL2VA, только последний — L2VA, только первый — I2VA.
    """
    _check_mode(mode)
    seconds = _as_duration(duration).label
    if mode == MODE_I2V or (mode == MODE_FLF2V and has_first and not has_last):
        return I2VA_LINE
    if mode == MODE_FLF2V and has_first and has_last:
        return FL2VA_LINE.format(shot=last_shot, seconds=seconds)
    if mode == MODE_FLF2V and has_last:
        return L2VA_LINE.format(shot=last_shot, seconds=seconds)
    return ""


# --- Сборка -----------------------------------------------------------------------------------


def is_structured(text: str) -> bool:
    """Текст уже в формате H3: есть основное поле одного из режимов."""
    return bool(_MAIN_FIELD.search(text or ""))


def _last_shot(text: str) -> int:
    return max((int(number) for number in _SHOT.findall(text)), default=1)


def _shot_list(text: str) -> str:
    count = _last_shot(text)
    return ", ".join(f"[Shot {number}]" for number in range(1, count + 1))


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_service_lines(text: str) -> str:
    """Убирает строки выравнивания и запретов — их ставит ``compose``."""
    return _tidy(_NEGATIVE.sub("", _ALIGNMENT.sub("", text)))


def _as_shot(text: str) -> str:
    return text if text.startswith("[Shot 1]") else f"[Shot 1] {text}".rstrip()


def _wrap_base(text: str, mode: str, has_first: bool, has_last: bool) -> str:
    """Свободный текст → три поля базового режима с якорями опорных кадров.

    Якоря нужны модели, чтобы связать текст с кадрами: в I2VA описание
    начинается от первого кадра, в L2VA сходится к последнему. В FL2VA метки
    без скобок — как в официальном примере режима.
    """
    body = re.sub(r"^\[Shot 1\]\s*", "", text)
    both = mode == MODE_FLF2V and has_first and has_last
    if mode == MODE_I2V or (mode == MODE_FLF2V and has_first and not has_last):
        body = f"The shot begins exactly from <Picture 1>, keeping its subjects, clothing, composition and lighting. {body}"
    elif both:
        body = f"The shot begins in the position and framing established by Picture 1. {body}"
    body = body.strip()
    if both:
        body += " At the end of the shot, everything settles into the pose, spacing and composition established by Picture 2."
    elif mode == MODE_FLF2V and has_last:
        body += (
            " Toward the end, everything converges into the exact composition, poses and lighting established "
            "by <Picture 1>."
        )
    return (
        f"integrated_multimodal_description: {_as_shot(body.strip())}\n\n"
        f"overall_soundscape: {DEFAULT_SOUNDSCAPE}\n\n"
        f"non_diegetic_music: {DEFAULT_MUSIC}"
    )


def _wrap_ref(text: str, labels: Sequence[str]) -> str:
    """Свободный текст → шесть секций Ref2VA с минимально корректными определениями.

    Изображение по умолчанию задаёт персонажа, предмет или стиль, а не кадр:
    по гайду такое изображение не получает отдельной строки, а цитируется в
    определении ``<Subject N>``. Референс без упоминания не работает, поэтому
    каждая метка, которой нет в тексте, получает фразу в описании.
    """
    shots = _shot_list(text)
    definitions: list[str] = []
    retention: list[str] = []
    named: list[str] = []
    mentions: list[str] = []
    subject = 0
    for label in labels:
        match = _LABEL.fullmatch(label)
        kind = match.group(1) if match else ""
        if kind == "Picture":
            subject += 1
            own = f"<Subject {subject}>"
            definitions.append(
                f"{own} is the visible content of {label} (a person, object, environment or style), "
                "reused with its referenced appearance."
            )
            retention.append(f"{own} (appears in {shots}): fully_preserved - its referenced appearance is retained.")
            named.append(own)
            if own not in text and label not in text:
                mentions.append(f"{own} appears with the appearance referenced from {label}.")
        elif kind == "Video":
            definitions.append(f"{label} is a reference video whose motion, camera movement and pacing guide the target video.")
            retention.append(f"{label} (motion and pacing structure): weak_reference - only its motion and pacing are followed.")
            named.append(label)
            if label not in text:
                mentions.append(f"The motion and pacing follow {label}.")
        elif kind == "Audio":
            definitions.append(f"{label} is a reference audio track whose sound guides the target video.")
            retention.append(f"{label}: reference - its sound character is followed without copying the signal.")
            named.append(label)
            if label not in text:
                mentions.append(f"The sound follows {label}.")
    has_audio = any(label.startswith("<Audio") for label in labels)
    task = "[reference generation + audio reference]" if has_audio else "[reference generation]"
    summary = f"{task} The target video follows the detailed description below"
    summary += f", using {', '.join(named)}." if named else "."
    description = _as_shot(" ".join([text.strip(), *mentions]).strip())
    sections = (
        ("subject_definitions", "\n".join(definitions) or "N/A"),
        ("summary", summary),
        ("retention_analysis", "\n".join(retention) or "N/A"),
        ("detailed_description", f"The target video keeps the visual style of the references.\n{description}"),
        ("overall_soundscape", DEFAULT_SOUNDSCAPE),
        ("non_diegetic_music", DEFAULT_MUSIC),
    )
    return "\n\n".join(f"{name}:\n{content}" for name, content in sections)


def compose(
    text: str,
    mode: str,
    duration: Duration | float,
    has_first: bool = False,
    has_last: bool = False,
    labels: Sequence[str] = (),
) -> str:
    """Промт, готовый для движка.

    Структурированный текст сохраняется, а строка выравнивания ставится или
    заменяется на правильную для режима (с номером последнего шота и
    фактической длительностью). Свободный текст оборачивается в поля режима.
    В конце всегда одна строка запретов.
    """
    _check_mode(mode)
    body = _strip_service_lines(text or "")
    if not is_structured(body):
        body = _wrap_ref(body, labels) if mode == MODE_REF else _wrap_base(body, mode, has_first, has_last)
    line = alignment_line(mode, duration, has_first, has_last, last_shot=_last_shot(body))
    return "\n\n".join([*([line] if line else []), body, NEGATIVE_LINE])


# --- Проверка ---------------------------------------------------------------------------------


def _sections(text: str) -> tuple[list[tuple[str, int]], dict[str, str]]:
    """Поля в порядке появления и их содержимое (первое вхождение каждого)."""
    found = [(match.group(1), match.start()) for match in _FIELD.finditer(text)]
    contents: dict[str, str] = {}
    bounds = [match.end() for match in _FIELD.finditer(text)]
    for number, (name, _) in enumerate(found):
        end = found[number + 1][1] if number + 1 < len(found) else len(text)
        contents.setdefault(name, _NEGATIVE.sub("", text[bounds[number] : end]).strip())
    return found, contents


def _seconds(code: str) -> float:
    minutes, _, seconds = code.partition(":")
    return int(minutes) * 60 + float(seconds)


def off_language(prompt: str) -> bool:
    """Основной текст не по-английски.

    Язык оригинала разрешён только внутри ``<d>`` и в кавычках (надписи в
    кадре). Английский узнаётся по письменности: все буквы вне этих мест
    латинские.
    """
    prose = _QUOTED.sub("", _DIALOGUE.sub("", prompt))
    return any(char.isalpha() and not unicodedata.name(char, "").startswith("LATIN") for char in prose)


def _check_fields(text: str, mode: str, found: list[tuple[str, int]], contents: dict[str, str]) -> list[str]:
    problems: list[str] = []
    expected = REF_FIELDS if mode == MODE_REF else BASE_FIELDS
    names = [name for name, _ in found]
    for name in dict.fromkeys(names):
        if name not in expected:
            problems.append(f"Field '{name}' does not belong to {mode} prompts")
        elif names.count(name) > 1:
            problems.append(f"Field '{name}' appears more than once")
    for name in expected:
        if name not in names:
            problems.append(f"Missing field '{name}'")
        elif not contents.get(name):
            problems.append(f"Field '{name}' is empty")
    present = [name for name in dict.fromkeys(names) if name in expected]
    if present != [name for name in expected if name in present]:
        problems.append("Fields are out of order; expected: " + ", ".join(expected))
    if found and _ALIGNMENT.sub("", text[: found[0][1]]).strip():
        problems.append("Unexpected text before the first field")
    return problems


def _check_alignment(
    text: str, mode: str, duration: Duration, has_first: bool | None, has_last: bool | None
) -> list[str]:
    present = [match.group(0).strip() for match in _ALIGNMENT.finditer(text)]
    last_shot = _last_shot(text)
    if mode == MODE_FLF2V and has_first is None and has_last is None:
        # Флаги не переданы — годится любой вариант режима, но с верными
        # длительностью и шотом.
        expected = {alignment_line(mode, duration, *flags, last_shot=last_shot) for flags in ((True, True), (False, True))}
    else:
        expected = {alignment_line(mode, duration, bool(has_first), bool(has_last), last_shot=last_shot)} - {""}
    if not expected:
        return [f"Alignment line does not belong to {mode} prompts"] if present else []
    if not present:
        return ["Missing alignment line; expected: " + " | ".join(sorted(expected))]
    problems: list[str] = []
    if len(present) > 1:
        problems.append("More than one alignment line")
    if " ".join(present[0].split()) not in expected:
        problems.append("Alignment line does not match the mode or duration; expected: " + " | ".join(sorted(expected)))
    elif not text.lstrip().startswith(present[0]):
        problems.append("Alignment line must be the first line")
    return problems


def _check_timeline(body: str, duration: Duration) -> list[str]:
    problems: list[str] = []
    end = duration.seconds
    shots = list(_SHOT.finditer(body))
    numbers = [int(match.group(1)) for match in shots]
    if numbers and numbers != list(range(1, len(numbers) + 1)):
        problems.append("Shots are not numbered sequentially from [Shot 1]")
    previous = 0.0
    for match in shots:
        cut = _SHOT_CUT.match(body, match.end())
        if match.group(1) == "1":
            if cut:
                problems.append("[Shot 1] must not have a cut timecode")
            continue
        if not cut or not _TIMECODE.match(cut.group(1)):
            problems.append(f"{match.group(0)} has no cut timecode in MM:SS.mmm format")
            continue
        moment = _seconds(cut.group(1))
        if moment <= previous:
            problems.append(f"{match.group(0)} cut at {cut.group(1)} is not after the previous cut")
        previous = max(previous, moment)
    for code in _ANY_TIME.findall(body):
        if not _TIMECODE.match(code):
            problems.append(f"Timecode '{code}' is not in MM:SS.mmm format")
        elif _seconds(code) >= end:
            problems.append(f"Timecode {code} is beyond the video duration of {duration.label} s")
    return problems


def _check_dialogue(body: str, duration: Duration) -> list[str]:
    problems: list[str] = []
    if body.count("<d>") != body.count("</d>"):
        problems.append("Unbalanced <d> tags")
    lines = list(_DIALOGUE.finditer(body))
    for line in lines:
        if not re.match(r"\s*\[[^\]]+\]", line.group(1)):
            problems.append(f"Dialogue without a language tag: <d>{line.group(1)[:40]}</d>")
    if lines and "<cutoff>" not in lines[-1].group(1):
        last = lines[-1]
        before = body[: last.start()]
        starts = [_seconds(code) for code in _ANY_TIME.findall(before) if _TIMECODE.match(code)]
        start = max(starts, default=0.0)
        words = len(re.sub(r"^\s*\[[^\]]+\]", "", last.group(1)).split())
        if start + words / WORDS_PER_SECOND > duration.seconds - DIALOGUE_MARGIN:
            problems.append(
                f"Warning: the last spoken line may end less than {DIALOGUE_MARGIN} s before the end of the video"
            )
    return problems


def _check_labels(contents: dict[str, str], labels: Sequence[str]) -> list[str]:
    problems: list[str] = []
    definitions = contents.get("subject_definitions", "")
    description = contents.get("detailed_description", "")
    for label in labels:
        if label not in definitions:
            problems.append(f"Label {label} is not defined in subject_definitions")
        if label in description:
            continue
        # Изображение-источник цитируется в определении <Subject N>, а в
        # описании работает уже сам субъект.
        subjects = [
            match.group(1)
            for line in definitions.splitlines()
            if label in line and (match := re.match(r"\s*(<Subject \d+>)", line))
        ]
        if not any(subject in description for subject in subjects):
            problems.append(f"Label {label} is not used in detailed_description")
    known = set(labels)
    if known:
        used = {match.group(0) for text in contents.values() for match in _LABEL.finditer(text)}
        for label in sorted(used):
            if not label.startswith("<Subject") and label not in known:
                problems.append(f"Label {label} has no matching reference")
    defined = {match.group(0) for match in _LABEL.finditer(definitions)}
    for label in sorted({match.group(0) for match in _LABEL.finditer(contents.get("summary", ""))} - defined):
        problems.append(f"summary introduces label {label} that subject_definitions does not define")
    return problems


def check(
    prompt: str,
    mode: str,
    duration: Duration | float,
    labels: Sequence[str] = (),
    *,
    has_first: bool | None = None,
    has_last: bool | None = None,
) -> list[str]:
    """Проблемы промта по чек-листу H3; пустой список — промт годен.

    Строки с префиксом ``Warning:`` — предупреждения, а не ошибки. Для flf2v
    без флагов годится любой из вариантов FL2VA/L2VA с верной длительностью.
    """
    _check_mode(mode)
    text = (prompt or "").strip()
    if not text:
        return ["Prompt is empty"]
    actual = _as_duration(duration)
    found, contents = _sections(text)
    main = contents.get("detailed_description" if mode == MODE_REF else "integrated_multimodal_description", "")

    problems = _check_fields(text, mode, found, contents)
    problems += _check_alignment(text, mode, actual, has_first, has_last)
    problems += _check_timeline(main, actual)
    problems += _check_dialogue(main, actual)
    if mode == MODE_REF:
        problems += _check_labels(contents, labels)
    for match in _CLOUD.finditer(text):
        problems.append(f"Cloud syntax '{match.group(0)}' is not understood locally; use <Picture N>, <Video N>, <Audio N>")
    if off_language(text):
        problems.append("Text outside <d> and quotes is not in English")
    negatives = len(_NEGATIVE.findall(text))
    if not text.endswith(NEGATIVE_LINE):
        problems.append(f"The prompt must end with the line: {NEGATIVE_LINE}")
    elif negatives > 1:
        problems.append("The prohibition line appears more than once")
    if actual.frames < TRAINED_MIN_FRAMES:
        problems.append(f"Warning: {actual.label} s is shorter than the trained range ({TRAINED_MIN_FRAMES} frames)")
    return problems


# --- AI-буст ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoostedVideoPrompt:
    """Промт после буста и проблемы, найденные в нём проверкой."""

    prompt: str
    problems: list[str] = field(default_factory=list)
    # Картинки были, но модель их не приняла: промт написан по одному тексту.
    images_skipped: bool = False


def system_prompt(mode: str, prompt_dir: Path) -> str:
    """Системный промт буста или раскадровки; файл читается при каждом вызове.

    Пользователь волен править файлы, поэтому их не кешируем. Плейсхолдеры
    ``{{…}}`` подставляются простой заменой: в тексте много фигурных скобок
    примеров JSON, и ``str.format`` на них споткнулся бы.
    """
    if mode not in _PROMPT_FILES:
        raise ValueError(f"Unknown video prompt mode: {mode!r}")
    path = Path(prompt_dir) / _PROMPT_FILES[mode]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"System prompt not found {path}: {error}") from error
    for placeholder, value in _PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return text


def shot_range(seconds: float) -> str:
    """Плотность склеек: 5 с выдерживают 1–2 шота, 8 с — 2–3, 15 с — 3–5."""
    if seconds <= 6.0:
        return "1-2"
    if seconds <= 10.0:
        return "2-3"
    return "3-5"


def _mode_line(mode: str, has_first: bool, has_last: bool) -> str:
    if mode == MODE_T2V:
        return "Mode: T2VA (text only, no reference frames)."
    if mode == MODE_REF:
        return "Mode: Ref2VA (full-reference mode, six sections)."
    if mode == MODE_FLF2V and has_first and has_last:
        return "Mode: FL2VA (Picture 1 is the first frame, Picture 2 is the last frame)."
    if mode == MODE_FLF2V and has_last:
        return "Mode: L2VA (<Picture 1> is the last frame; infer a plausible earlier state and land on it)."
    return "Mode: I2VA (<Picture 1> is the first frame; develop the scene forward from it)."


def aspect_line(aspect: str) -> str:
    """Строка о соотношении сторон для LLM; вертикаль — отдельная композиция.

    Правило «вайб-монтажа»: вертикальный кадр — не обрезанная горизонталь,
    а новая композиция — один главный объект, крупно.
    """
    try:
        ratio = geometry.ratio_value(aspect)
    except (ValueError, ZeroDivisionError):
        return "Aspect ratio: follows the reference frame."
    if ratio < 1:
        return (
            f"Aspect ratio: {aspect} (vertical). A vertical frame is a new composition, not a cropped landscape: "
            "one main subject, large in frame, with room above the head; no wide establishing shots."
        )
    shape = "square" if ratio == 1 else "horizontal"
    return f"Aspect ratio: {aspect} ({shape})."


def build_boost_message(
    text: str,
    mode: str,
    duration: Duration | float,
    labels: Sequence[str] = (),
    has_first: bool = False,
    has_last: bool = False,
    aspect: str = geometry.DEFAULT_ASPECT,
) -> str:
    """Сообщение пользователя для переписывателя: условия ролика и сама идея.

    Длительность — фактическая, с кадрами: модель считает тайм-коды от неё,
    и «5 секунд» без пояснения дали бы строку выравнивания с 5.00.
    """
    _check_mode(mode)
    actual = _as_duration(duration)
    lines = [
        _mode_line(mode, has_first, has_last),
        f"Duration: {actual.label} seconds ({actual.frames} frames at {geometry.FPS} fps). Every timecode must be "
        f"below {geometry.timecode(actual.seconds)}; the last spoken line must end by "
        f"{geometry.timecode(max(actual.seconds - DIALOGUE_MARGIN, 0.0))}.",
    ]
    if mode == MODE_FLF2V and has_first and has_last:
        lines.append("Shots: prefer a single continuous shot unless the idea explicitly asks for cuts.")
    else:
        lines.append(f"Shots: {shot_range(actual.seconds)} shots.")
    line = alignment_line(mode, duration, has_first, has_last)
    if line:
        note = " (the last frame belongs to the final shot: put its number there)" if has_last else ""
        lines.append(f"First line, verbatim{note}:\n{line}")
    lines.append(aspect_line(aspect))
    if labels:
        lines.append(
            f"Reference labels, exactly as numbered: {', '.join(labels)}. Mention every label where it applies; "
            "never invent other labels or use '@image' syntax."
        )
    lines.append(
        "Return only the finished prompt: no explanations, headings or code fences. Write everything in English; "
        "keep spoken lines inside <d> and visible on-screen text in their original language, verbatim."
    )
    lines.append(f"User idea:\n{text.strip()}")
    return "\n\n".join(lines)


def clean_answer(text: str) -> str:
    """Ответ модели без рассуждений ``<think>`` и обёртки блока кода."""
    raw = _THINK.sub("", text or "")
    # Незакрытый блок рассуждений: ответ — то, что после </think>.
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    fence = _FENCE.search(raw)
    if fence:
        raw = fence.group(1)
    return raw.strip()


def _ask(client: LlmClient, system: str, user: str, images: list[Image.Image] | None) -> str:
    """Запрос и один повтор, если ответ пустой или не по-английски.

    Как у буста изображений: сбой редкий, второй ответ почти всегда верный,
    а дольше задерживать генерацию незачем — остальное покажет ``check``.
    """

    def request() -> str:
        return clean_answer(
            client.complete(system, user, images=images, temperature=BOOST_TEMPERATURE, max_tokens=BOOST_MAX_TOKENS)
        )

    answer = request()
    if not answer:
        LOGGER.warning("Video prompt rewriter returned an empty prompt, retrying")
        answer = request()
    elif off_language(answer):
        LOGGER.warning("Video prompt is not in English, retrying")
        answer = request() or answer
    return answer


def boost(
    client: LlmClient,
    text: str,
    mode: str,
    duration: Duration | float,
    *,
    prompt_dir: Path,
    labels: Sequence[str] = (),
    has_first: bool = False,
    has_last: bool = False,
    aspect: str = geometry.DEFAULT_ASPECT,
    images: list[Image.Image] | None = None,
) -> BoostedVideoPrompt:
    """Переписывает идею в промт H3; картинки — подспорье, а не условие.

    Ответ модели проходит через ``compose``: строка выравнивания и строка
    запретов ставятся кодом, а не доверяются модели. Модель без зрения
    отвечает на картинки ошибкой — тогда тот же запрос уходит одним текстом
    с пометкой ``BLIND_NOTE``.
    """
    system = system_prompt(mode, prompt_dir)
    user = build_boost_message(text, mode, duration, labels, has_first, has_last, aspect)
    skipped = False
    if images:
        try:
            answer = _ask(client, system, user, images)
        except LlmImagesRejected as error:
            LOGGER.warning("Language model rejected the images, rewriting the video prompt from text only: %s", error)
            answer = _ask(client, system, user + BLIND_NOTE, None)
            skipped = True
    else:
        answer = _ask(client, system, user, None)
    if not answer:
        LOGGER.warning("Video prompt rewriter returned an empty prompt again, keeping the original text")
        answer = text
    prompt = compose(answer, mode, duration, has_first, has_last, labels)
    problems = check(prompt, mode, duration, labels, has_first=has_first, has_last=has_last)
    return BoostedVideoPrompt(prompt=prompt, problems=problems, images_skipped=skipped)
