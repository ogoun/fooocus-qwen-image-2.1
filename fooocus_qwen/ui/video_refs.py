"""Плитки референсов вкладки Video: картинки, видео и звук — по ячейке на файл.

Раньше референсы шли одним полем «файлы», и метки (``<Picture 1>``,
``<Video 1>``, ``<Audio 1>``) приходилось искать в строке под ним. Плитка
подписана своей меткой — той, под которой файл увидит модель, — как ячейки
референсов на вкладках изображений.

Пустая плитка каждого вида — одна, следующая открывается, когда в эту лёг
файл: пятнадцать пустых ячеек в узкой колонке были бы стеной.

Нумерация — порядок подачи движку (``prompt_h3.labels_by_input``):
картинки, видео (дорожка видео — ``<Audio j>`` перед ним), отдельные звуки.
В режиме «видео → видео» видео одно — исходный ролик, ``<Video 1>``; плитки
видео скрыты, а их значения в запрос не идут (``request.edit_references``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gradio as gr

from ..video import prompt_h3
from ..video.request import MAX_AUDIOS, MAX_IMAGES, MAX_VIDEOS
from . import layout
from .i18n import pick, say

KINDS = (("image", MAX_IMAGES), ("video", MAX_VIDEOS), ("audio", MAX_AUDIOS))
SLOT_COUNT = sum(count for _, count in KINDS)
# Где плитки на виду. «Текст → видео» — тоже: колонка слева от плеера там
# иначе пустовала, и человек не находил, куда класть референсы («область, где
# они на вкладках изображений, пустая»). Первый файл переключает режим на
# референсы (``promote``): текст плюс референсы — это и есть Ref2VA. В
# режимах с кадрами колонку занимают кадры, а чекпойнт FL2VA референсов не
# принимает.
TILE_MODES = (prompt_h3.MODE_T2V, *prompt_h3.REF_MODES)
# Подписи пустых плиток — ключами таблицы: проверка мёртвых переводов ищет
# ключ литералом в коде.
EMPTY_LABELS = {"image": "video_tile_image", "video": "video_tile_video", "audio": "video_tile_audio"}


@dataclass
class Tiles:
    """Плитки референсов и исходный ролик режима «видео → видео»."""

    box: gr.Column
    source: gr.Video
    note: gr.Markdown
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    audios: list = field(default_factory=list)
    # Колонка вокруг каждой плитки — её и прячут: ``visible`` самого
    # ``gr.Video`` в Gradio 6.5 обновлением не прячется (видно на снимке
    # режима «видео → видео»), а контейнер прячется всегда.
    cells: list = field(default_factory=list)
    # Значения всех плиток по порядку (``None`` — пусто): его и читают
    # обработчики, как ``grid.state`` у сетки изображений.
    state: gr.State | None = None

    @property
    def slots(self) -> list:
        return [*self.images, *self.videos, *self.audios]


def _has_audio(path: str) -> bool:
    try:
        from ..video import assemble

        return assemble.probe(path).has_audio
    except Exception:  # noqa: BLE001 — нечитаемый файл: метку дорожки не обещаем
        return False


def _path(value) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "path", None) or getattr(value, "name", None)


def values_by_kind(values) -> dict[str, list[str | None]]:
    """Значения плиток по видам, в порядке плиток."""
    values = [_path(value) for value in (values or [])]
    result, start = {}, 0
    for kind, count in KINDS:
        chunk = values[start:start + count]
        result[kind] = chunk + [None] * (count - len(chunk))
        start += count
    return result


def labels(mode: str, source, values, has_audio=_has_audio) -> tuple[dict[str, list[str]], str]:
    """Метки каждой плитки (пустая строка — плитка пуста) и метка исходника.

    Видео — исходник в «видео → видео», иначе — плитки видео.
    """
    by_kind = values_by_kind(values)
    editing = mode == prompt_h3.MODE_V2V
    source_path = _path(source) if editing else None
    entries: list[tuple[str, int | None, str]] = []
    for index, path in enumerate(by_kind["image"]):
        if path:
            entries.append(("image", index, path))
    if editing:
        if source_path:
            entries.append(("video", None, source_path))
    else:
        for index, path in enumerate(by_kind["video"]):
            if path:
                entries.append(("video", index, path))
    for index, path in enumerate(by_kind["audio"]):
        if path:
            entries.append(("audio", index, path))
    kinds = [kind for kind, _, _ in entries]
    flags = [has_audio(path) for kind, _, path in entries if kind == "video"]
    found = prompt_h3.labels_by_input(kinds, flags) if entries else []
    result = {kind: [""] * count for kind, count in KINDS}
    source_label = ""
    for (kind, index, _path_value), tags in zip(entries, found):
        text = " + ".join(tags)
        if index is None:
            source_label = text
        else:
            result[kind][index] = text
    return result, source_label


def _shown(paths: list[str | None]) -> list[bool]:
    """Заполненные плитки и одна пустая — первая свободная."""
    shown, offered = [], False
    for path in paths:
        if path:
            shown.append(True)
        elif not offered:
            shown.append(True)
            offered = True
        else:
            shown.append(False)
    return shown


def view(mode: str, source, values, lang: str, has_audio=_has_audio) -> list:
    """Обновления: состояние, подписи плиток, видимость их колонок, исходник, подсказка.

    Рамки плиток здесь нет — её показывает ``frame`` отдельным, более ранним
    событием (см. ``wire``).
    """
    by_kind = values_by_kind(values)
    tags, source_label = labels(mode, source, values, has_audio)
    editing = mode == prompt_h3.MODE_V2V
    updates: list = [[_path(value) for value in values]]
    shown_all: list[bool] = []
    for kind, _count in KINDS:
        shown = _shown(by_kind[kind]) if not (editing and kind == "video") else [False] * len(by_kind[kind])
        shown_all += shown
        for path, tag in zip(by_kind[kind], tags[kind]):
            updates.append(gr.update(label=tag if path else pick(EMPTY_LABELS[kind], lang)))
    updates += [gr.update(visible=visible) for visible in shown_all]
    title = pick("video_source", lang)
    # Подпись — метки, как у плиток: «Исходный ролик · <Audio 1> + <Video 1>»
    # не помещалась рядом с кнопками плеера.
    updates.append(gr.update(visible=editing, label=source_label or title))
    hint = "video_refs_t2v" if mode == prompt_h3.MODE_T2V else "video_refs_empty"
    updates.append(say(hint, lang))
    return updates


def build(lang: str) -> Tiles:
    """Плитки — в текущий контейнер (колонку входов вкладки)."""
    source = gr.Video(label=pick("video_source", lang), sources=["upload"], visible=False,
                      elem_classes=[layout.VIDEO_SOURCE])
    # По плитке в ряд: колонка входов узкая (--qs-refs-width), и в двух
    # столбцах подпись <Picture 1> не помещалась рядом с кнопками плитки.
    cells: list = []

    def cell(index: int, make):
        with gr.Column(visible=index == 0, min_width=0, elem_classes=[layout.VIDEO_TILE]) as column:
            component = make()
        cells.append(column)
        return component

    with gr.Column(visible=False, elem_classes=[layout.VIDEO_REFS]) as box:
        images = [
            cell(index, lambda: gr.Image(label=pick("video_tile_image", lang), type="filepath",
                                         sources=["upload"], buttons=["fullscreen"]))
            for index in range(MAX_IMAGES)
        ]
        videos = [
            cell(index, lambda: gr.Video(label=pick("video_tile_video", lang), sources=["upload"]))
            for index in range(MAX_VIDEOS)
        ]
        audios = [
            cell(index, lambda: gr.Audio(label=pick("video_tile_audio", lang), type="filepath", sources=["upload"]))
            for index in range(MAX_AUDIOS)
        ]
        note = gr.Markdown(say("video_refs_empty", lang), elem_classes=[layout.VIDEO_NOTE])
    tiles = Tiles(box=box, source=source, note=note, images=images, videos=videos, audios=audios, cells=cells)
    tiles.state = gr.State([None] * len(tiles.slots))
    return tiles


def frame(mode: str):
    """Рамка плиток видна в «текст → видео» и в режимах с референсами."""
    return gr.update(visible=mode in TILE_MODES)


def promote(mode: str, lang: str, values) -> tuple:
    """Файл в плитке при «текст → видео» — режим «по референсам»: ``(режим, сообщение)``."""
    keep = gr.update()
    if mode == prompt_h3.MODE_T2V and any(_path(value) for value in values):
        return prompt_h3.MODE_REF, say("video_switched_to_ref", lang)
    return keep, keep


def wire(tiles: Tiles, mode, language, status) -> None:
    """Любая перемена плитки, исходника, режима или языка — метки и видимость заново."""

    def refresh(mode_value, source_value, lang, *values):
        return view(mode_value, source_value, list(values), lang)

    def switch(mode_value, lang, *values):
        return promote(mode_value, lang, values)

    inputs = [mode, tiles.source, language, *tiles.slots]
    outputs = [tiles.state, *tiles.slots, *tiles.cells, tiles.source, tiles.note]
    for slot in tiles.slots:
        slot.change(refresh, inputs, outputs, queue=False, show_progress="hidden")
        slot.change(switch, [mode, language, *tiles.slots], [mode, status], queue=False, show_progress="hidden")
    for trigger in (tiles.source.change, language.change):
        trigger(refresh, inputs, outputs, queue=False, show_progress="hidden")
    # Рамка — первой, плитки — после неё. В Gradio 6.5 обновление
    # visible=True у колонки снова показывает дочерние колонки, спрятанные
    # раньше (опыт: колонка-рамка с двумя колонками внутри, 2026-10-25):
    # в одном ответе или параллельным событием плитки видео в режиме
    # «видео → видео» оставались на виду.
    mode.change(frame, mode, tiles.box, queue=False, show_progress="hidden").then(
        refresh, inputs, outputs, queue=False, show_progress="hidden")


def place_image(values, path: str) -> tuple[int | None, list]:
    """Номер первой пустой плитки картинки для «В видео» и новое состояние."""
    values = list(values or [None] * SLOT_COUNT)
    for index in range(MAX_IMAGES):
        if not _path(values[index]):
            values[index] = path
            return index, values
    return None, values
