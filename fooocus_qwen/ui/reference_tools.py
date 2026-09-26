"""Инструменты ячейки референса: окно выбора позы и окно эскиза.

На каждой ячейке сетки — две кнопки-значка. Первая открывает библиотеку поз
(плитки openposes.com и свои позы, в конце — плитка «Добавить позу»): выбор
кладёт в ячейку скелет позы. Вторая открывает холст для наброска: «Принять»
кладёт эскиз в ячейку, «Отмена» закрывает окно.

Окна — колонки поверх страницы (``layout.MODAL``), а не отдельные вкладки:
результат, промт и сетка остаются под ними, и после выбора человек сразу
видит ячейку, куда легла картинка. Своего модального окна в Gradio 6.5.1
нет, поэтому это обычная колонка, которую показывают и прячут, а место на
экране ей даёт CSS.

«Добавить позу» — три шага подряд: окно показывает поле загрузки, по фото
DWPose (``poses.detect``) строит скелет, поза сохраняется в ``user/poses`` и
сразу ложится в ячейку, а затем Qwen-Image рисует для неё плитку в стиле
каталога (``poses.tile``). Плитка — последним шагом и в общей очереди GPU:
скелет нужен человеку сразу, а плитке не к спеху.
"""

from __future__ import annotations

import json
import logging

import gradio as gr
from PIL import Image

from .. import config
from ..engine.generator import MASK_ANNOTATION
from ..poses import detect, library, tile
from . import layout, painter
from .i18n import Localizer, painter_labels, pick, say
from .painter import payload
from .references import MAX_REFERENCES, place
from .state import GPU_CONCURRENCY_ID

LOGGER = logging.getLogger(__name__)

ADD_POSE_TILE = config.RESOURCES_DIR / "poses" / "add_pose.png"

SKETCH_PAINTER_ID = "qs-sketch-painter"
# Холст эскиза: квадрат, как плитки поз, — рисовать на нём можно что угодно,
# а форма результата задаётся не референсом, а соотношением сторон вкладки.
SKETCH_SIDE = 1024
# Палитра наброска: карандаш и основные цвета. Белый — это «замазать»;
# ластик кисти стирает мазки до холста, то есть тоже до белого.
SKETCH_COLOURS: tuple[str, ...] = (
    "#000000", "#6b7280", "#ffffff", "#ef4444", "#f97316", "#facc15",
    "#22c55e", "#3b82f6", "#8b5cf6", "#92400e",
)

# Подсказки к значкам: у ``gr.Button`` нет своего ``title``, и его ставит
# скрипт — при загрузке страницы и при смене языка.
TOOL_TITLES = {
    layout.REF_POSE: ("Поза: выбрать из библиотеки или по фото", "Pose: pick from the library or from a photo"),
    layout.REF_SKETCH: ("Эскиз: нарисовать от руки", "Sketch: draw by hand"),
}
TITLES_JS = """
(lang) => {
    const titles = __TITLES__;
    for (const [cls, pair] of Object.entries(titles)) {
        document.querySelectorAll('.' + cls).forEach(node => {
            node.title = pair[lang === 'en' ? 1 : 0];
            node.setAttribute('aria-label', node.title);
        });
    }
    return [];
}
""".replace("__TITLES__", json.dumps(TOOL_TITLES, ensure_ascii=False))


def pose_tiles(entries: list[library.PoseEntry], lang: str) -> list[tuple[str, str | None]]:
    """Значение галереи поз: плитки и последней — «Добавить позу»."""
    tiles: list[tuple[str, str | None]] = [(str(entry.preview()), None) for entry in entries]
    tiles.append((str(ADD_POSE_TILE), pick("pose_add", lang)))
    return tiles


def selected_index(event: gr.EventData) -> int | None:
    """Номер выбранной плитки. ``gr.SelectData`` падает на событии без
    значения (Gradio 6.5.1), поэтому индекс берётся из сырых данных."""
    data = getattr(event, "_data", None) or {}
    index = data.get("index")
    if isinstance(index, (list, tuple)):
        index = index[0] if index else None
    return index if isinstance(index, int) else None


def build(
    studio,
    localizer: Localizer,
    lang: str,
    language,
    references,
    reference_targets: list,
    pose_buttons: list,
    sketch_buttons: list,
) -> dict:
    """Собирает оба окна и связывает с кнопками ячеек.

    ``reference_targets`` — выходы записи в сетку (состояние, десять ячеек,
    десять тегов, строка состояния вкладки), как у ``references.place``.
    ``lang`` — язык сборки, ``language`` — компонент с текущим языком.
    """
    target = gr.State(0)
    entries_state = gr.State([])

    # --- окно поз ---
    with gr.Column(visible=False, elem_classes=[layout.MODAL]) as pose_window:
        with gr.Column(elem_classes=[layout.MODAL_BOX]):
            pose_title = gr.Markdown(elem_classes=[layout.MODAL_TITLE])
            pose_grid = gr.Gallery(
                columns=8,
                allow_preview=False,
                object_fit="cover",
                show_label=False,
                interactive=False,
                buttons=[],
                elem_classes=[layout.POSE_GRID],
            )
            with gr.Column(visible=False) as add_panel:
                pose_photo = localizer.bind(
                    gr.Image(
                        type="pil",
                        sources=["upload", "clipboard"],
                        label=pick("pose_photo", lang),
                        buttons=[],
                        elem_classes=[layout.POSE_PHOTO],
                    ),
                    label=("Фото с нужной позой", "A photo with the pose"),
                )
            pose_message = gr.Markdown(elem_classes=[layout.MODAL_MESSAGE])
            with gr.Row():
                pose_close = localizer.bind(
                    gr.Button(pick("modal_close", lang), size="sm"), value=("Закрыть", "Close")
                )

    # --- окно эскиза ---
    with gr.Column(visible=False, elem_classes=[layout.MODAL]) as sketch_window:
        with gr.Column(elem_classes=[layout.MODAL_BOX, layout.SKETCH_BOX]):
            sketch_title = gr.Markdown(elem_classes=[layout.MODAL_TITLE])
            sketch = localizer.bind(
                painter.MaskPainter(
                    lang=lang,
                    region=MASK_ANNOTATION,
                    labels=painter_labels(),
                    palette=list(SKETCH_COLOURS),
                    elem_id=SKETCH_PAINTER_ID,
                    elem_classes=[layout.SKETCH],
                ),
                lang=("ru", "en"),
            )
            sketch_message = gr.Markdown(elem_classes=[layout.MODAL_MESSAGE])
            with gr.Row():
                sketch_cancel = localizer.bind(
                    gr.Button(pick("modal_cancel", lang)), value=("Отмена", "Cancel")
                )
                sketch_accept = localizer.bind(
                    gr.Button(pick("modal_accept", lang), variant="primary"), value=("Принять", "Accept")
                )

    # --- обработчики: позы ---

    def pose_opener(index: int):
        def open_pose_window(lang):
            return (
                index,
                gr.Column(visible=True),
                gr.Column(visible=False),
                say("pose_title", lang, cell=index + 1),
                "",
            )

        return open_pose_window

    def load_tiles(lang, progress=gr.Progress()):
        """Плитки в окно; при первом открытии — скачать каталог."""
        catalog = config.POSE_LIBRARY_DIR
        message = ""
        if not library.catalog_ready(catalog):
            progress(0, desc=say("pose_catalog_downloading", lang))
            try:
                library.fetch_catalog(
                    catalog, progress=lambda done, total: progress((done, total), desc=say("pose_catalog_downloading", lang))
                )
            except Exception as error:  # noqa: BLE001 — сеть и диск: строка в окне, не трейсбек
                LOGGER.exception("Каталог поз не скачался")
                message = say("pose_catalog_failed", lang, error=error)
        entries = library.list_poses(catalog, config.USER_POSE_DIR)
        return [entry.name for entry in entries], pose_tiles(entries, lang), message

    def pick_pose(index, names, current, lang, event: gr.EventData):
        """Выбор плитки: поза — в ячейку и окно закрыть; «Добавить позу» — поле загрузки."""
        chosen = selected_index(event)
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        if chosen is None:
            return (*keep, gr.update(), gr.update(), "")
        if chosen >= len(names):
            return (*keep, gr.update(), gr.Column(visible=True), say("pose_add_hint", lang))
        entry = _entry(names[chosen])
        if entry is None:
            return (*keep, gr.update(), gr.update(), say("pose_missing", lang))
        with Image.open(entry.skeleton) as opened:
            image = opened.convert("RGB")
        placed = place(current, index, image, lang, say("pose_placed", lang, cell=index + 1))
        return (*placed, gr.Column(visible=False), gr.Column(visible=False), "")

    def add_pose(photo, index, current, lang):
        """Фото → скелет → своя поза → в ячейку. Плитку рисует следующий шаг."""
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        if photo is None:
            return (*keep, gr.update(), gr.update(), gr.update(), "", None)
        try:
            found = detect.to_pose(studio.pose_detector().detect(photo))
        except detect.NoPersonFound:
            return (*keep, gr.update(), gr.update(), gr.update(), say("pose_not_found", lang), None)
        except Exception as error:  # noqa: BLE001 — веса, сеть, onnxruntime: строка в окне
            LOGGER.exception("Поза не распознана")
            return (*keep, gr.update(), gr.update(), gr.update(), say("pose_failed", lang, error=error), None)

        entry = library.add_custom(config.USER_POSE_DIR, found)
        with Image.open(entry.skeleton) as opened:
            image = opened.convert("RGB")
        placed = place(current, index, image, lang, say("pose_placed", lang, cell=index + 1))
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.USER_POSE_DIR)
        return (
            *placed,
            [item.name for item in entries],
            pose_tiles(entries, lang),
            gr.Column(visible=False),
            say("pose_added", lang),
            entry.name,
        )

    def draw_tile(name, lang, progress=gr.Progress()):
        """Плитка новой позы — Qwen-Image по скелету, в стиле каталога."""
        entry = _entry(name) if name else None
        if entry is None:
            return gr.update(), gr.update(), gr.update()
        progress(0, desc=say("pose_tile_drawing", lang))
        with Image.open(entry.skeleton) as opened:
            bones = opened.convert("RGB")
        request = tile.request(bones, turbo_ready=studio.turbo_weights_present())

        def report(_index: int, step: int, total: int) -> None:
            progress((step, total), desc=say("pose_tile_drawing", lang))

        produced, failure = studio.run_generation(request, lang, progress=report)
        if failure is not None or not produced:
            return gr.update(), gr.update(), say("pose_tile_failed", lang, error=failure or "—")
        library.set_tile(entry, produced[0].image)
        entries = library.list_poses(config.POSE_LIBRARY_DIR, config.USER_POSE_DIR)
        return [item.name for item in entries], pose_tiles(entries, lang), say("pose_tile_done", lang)

    def _entry(name: str) -> library.PoseEntry | None:
        for entry in library.list_poses(config.POSE_LIBRARY_DIR, config.USER_POSE_DIR):
            if entry.name == name:
                return entry
        return None

    # --- обработчики: эскиз ---

    def sketch_opener(index: int):
        def open_sketch_window(lang):
            blank = Image.new("RGB", (SKETCH_SIDE, SKETCH_SIDE), "white")
            return (
                index,
                gr.Column(visible=True),
                say("sketch_title", lang, cell=index + 1),
                payload.encode(blank),
                "",
            )

        return open_sketch_window

    def accept_sketch(value, index, current, lang):
        keep = (gr.update(),) * (1 + 2 * MAX_REFERENCES + 1)
        try:
            canvas = payload.decode(value)
        except payload.PayloadError as error:
            return (*keep, gr.update(), say("sketch_failed", lang, error=error))
        if canvas.background is None:
            return (*keep, gr.update(), say("sketch_empty", lang))
        image = canvas.background.convert("RGBA")
        if canvas.layer is not None:
            image = Image.alpha_composite(image, canvas.layer.convert("RGBA"))
        placed = place(current, index, image.convert("RGB"), lang, say("sketch_placed", lang, cell=index + 1))
        return (*placed, gr.Column(visible=False), "")

    def close():
        return gr.Column(visible=False)

    # --- связи ---
    pose_outputs = [target, pose_window, add_panel, pose_title, pose_message]
    for index, button in enumerate(pose_buttons):
        button.click(
            pose_opener(index), language, pose_outputs, queue=False, show_progress="hidden",
        ).then(
            load_tiles, language, [entries_state, pose_grid, pose_message], show_progress="minimal",
        )
    pose_grid.select(
        pick_pose,
        [target, entries_state, references, language],
        [*reference_targets, pose_window, add_panel, pose_message],
        show_progress="hidden",
    )
    new_pose = gr.State(None)
    pose_photo.upload(
        add_pose,
        [pose_photo, target, references, language],
        [*reference_targets, entries_state, pose_grid, add_panel, pose_message, new_pose],
        show_progress="minimal",
    ).then(
        draw_tile,
        [new_pose, language],
        [entries_state, pose_grid, pose_message],
        concurrency_id=GPU_CONCURRENCY_ID,
    )
    pose_close.click(close, None, pose_window, queue=False)

    sketch_outputs = [target, sketch_window, sketch_title, sketch, sketch_message]
    for index, button in enumerate(sketch_buttons):
        button.click(sketch_opener(index), language, sketch_outputs, show_progress="hidden")
    sketch_accept.click(
        accept_sketch,
        [sketch, target, references, language],
        [*reference_targets, sketch_window, sketch_message],
        js=painter.flush_js(SKETCH_PAINTER_ID),
        show_progress="hidden",
    )
    sketch_cancel.click(close, None, sketch_window, queue=False)

    return {
        "pose_window": pose_window,
        "pose_grid": pose_grid,
        "pose_photo": pose_photo,
        "sketch_window": sketch_window,
        "sketch": sketch,
    }

