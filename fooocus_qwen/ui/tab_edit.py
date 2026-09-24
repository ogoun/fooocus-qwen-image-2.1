"""Вкладка редактирования: правка промтом, по маске, по аннотации и расширение холста.

Все три режима области питаются из одного редактора. Разница в том, что уходит
в модель: чистый фон плюс отдельная чёрно-белая маска — или сведённое
изображение с цветными пометками прямо на нём.

Отдельная маска — основной режим: он не портит оригинал. Аннотация нужна тогда,
когда областей несколько и каждой нужна своя инструкция: цвет пометки становится
адресом области внутри одного промта.
"""

from __future__ import annotations

import logging

import gradio as gr
from PIL import Image

from .. import config
from ..engine import presets
from ..engine.generator import (
    MASK_ANNOTATION,
    MASK_MASK,
    MASK_NONE,
    MASK_REGION,
    GenerationRequest,
    resolve_reference_scale,
)
from ..imaging import aspect as aspect_module
from ..imaging import masking, metadata, outpaint
from ..prompting import boost as boost_module
from ..storage import gallery
from . import layout, painter
from .i18n import Localizer, painter_labels, pick, say, sentences
from .state import GPU_CONCURRENCY_ID, describe_failure

LOGGER = logging.getLogger(__name__)

# Цвета из примера в блоге: три области, три инструкции в одном промте.
ANNOTATION_COLOURS: tuple[str, ...] = ("#ff0000", "#0000ff", "#00ff00", "#ffff00", "#ffffff")

# По этому имени скрипты кнопок находят кисть на странице.
PAINTER_ID = "qs-edit-painter"

# Доля кадра вне маски, начиная с которой шов виден и о нём стоит сказать.
# Ниже — обычная точечная правка, где обрезать почти нечего.
CLIPPED_WARNING_PCT = 25.0

_MODE_KEYS = {
    MASK_NONE: "mask_mode_none",
    MASK_MASK: "mask_mode_mask",
    MASK_ANNOTATION: "mask_mode_annotation",
    MASK_REGION: "mask_mode_region",
}


def collect(value, mode: str) -> tuple[Image.Image | None, Image.Image | None]:
    """Разбирает значение редактора на исходное изображение и маску.

    Три режима области — это три разных ответа на один и тот же разбор:
    «маска» и «точная область» отдают нетронутый фон плюс отдельную
    чёрно-белую маску, «аннотация» — сведённую с пометками картинку и вовсе
    без маски. Спутать их значило бы либо отправить в модель испорченный
    «оригинал» (маска), либо продублировать уже нарисованные пометки вторым
    условным изображением (аннотация).
    """
    if not value:
        return None, None

    background = value.get("background")
    if background is None:
        return None, None

    if mode == MASK_ANNOTATION:
        composite = value.get("composite")
        if composite is None:
            composite = background.convert("RGBA").copy()
            for layer in value.get("layers") or []:
                composite.alpha_composite(layer.convert("RGBA"))
        return composite.convert("RGBA"), None

    if mode == MASK_NONE:
        return background.convert("RGBA"), None

    mask = masking.mask_from_editor(value)
    if masking.is_empty(mask):
        LOGGER.info("Режим области выбран, но маска пуста — правлю кадр целиком")
        return background.convert("RGBA"), None

    return background.convert("RGBA"), mask


def read_painter(raw, lang: str) -> tuple[dict | None, str | None]:
    """Значение кисти → словарь формы ``gr.ImageEditor`` и сообщение об ошибке.

    Ошибка разбора — это строка состояния, а не исключение: значение пришло
    из браузера, и испорченное или подделанное значение не должно ронять
    обработчик.
    """
    try:
        return painter.decode(raw).as_editor_value(), None
    except painter.PayloadError as error:
        LOGGER.warning("Значение кисти отклонено: %s", error)
        return None, say("painter_bad_value", lang, error=error)


def build(studio, localizer: Localizer, language=None) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком; см. докстринг
    ``tab_generate.build``, правило здесь то же.
    """
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.CANVAS_COL]):
            # Правка — это сравнение: было и стало. Пока результат лежал под
            # редактором, одновременно они на экран не помещались, и человек
            # мотал страницу вверх-вниз, держа разницу в голове. Рядом они
            # видны разом, и на широком мониторе для этого есть место.
            with gr.Row(elem_classes=[layout.BOARDS_ROW]):
                with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.SLOT_EDITOR]):
                    # Своя кисть вместо gr.ImageEditor: у того стоимость
                    # движения мыши растёт с длиной мазка, и на крупном кадре
                    # кисть заметно отстаёт от руки (см. ui/painter).
                    editor = localizer.bind(
                        painter.MaskPainter(
                            lang=lang,
                            region=MASK_MASK,
                            labels=painter_labels(),
                            palette=list(ANNOTATION_COLOURS),
                            elem_id=PAINTER_ID,
                            elem_classes=[layout.BOARD, layout.PAINTER],
                        ),
                        lang=("ru", "en"),
                    )

                with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.SLOT_RESULT]):
                    result = localizer.bind(
                        gr.Gallery(
                            label=pick("result", lang),
                            columns=1,
                            object_fit="contain",
                            format="png",
                            preview=True,
                            # Выход, а не вход: без этого Gradio делал поле
                            # интерактивным (оно же вход «Отправить в
                            # редактор») и зазывал загрузить в него файл.
                            interactive=False,
                            elem_classes=[layout.PREVIEW, layout.BOARD],
                        ),
                        label=("Результат", "Result"),
                    )
                    send_back = localizer.bind(
                        gr.Button(pick("send_to_edit", lang)),
                        value=("Отправить в редактор", "Send to editor"),
                    )

            with gr.Row(elem_classes=[layout.PROMPT_BAR, layout.SLOT_PROMPT]):
                prompt = localizer.bind(
                    gr.Textbox(
                        label=pick("prompt", lang),
                        placeholder=pick("prompt_placeholder", lang),
                        lines=3,
                        scale=8,
                    ),
                    label=("Промт", "Prompt"),
                    placeholder=("Опишите изображение…", "Describe the image…"),
                )
                with gr.Column(scale=1, min_width=140):
                    run_button = localizer.bind(
                        gr.Button(pick("apply_edit", lang), variant="primary"),
                        value=("Применить правку", "Apply edit"),
                    )
                    stop_button = localizer.bind(
                        gr.Button(pick("stop", lang), variant="stop"), value=("Прервать", "Stop")
                    )

            with gr.Row(elem_classes=[layout.SLOT_ACTIONS]):
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("boost", lang), value=False),
                    label=("AI буст", "AI boost"),
                )
                describe_button = localizer.bind(
                    gr.Button(pick("describe", lang)), value=("Описать изображение", "Describe image")
                )

        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.SIDE_COL]):
            mode = localizer.bind(
                gr.Radio(
                    choices=[(pick(_MODE_KEYS[key], lang), key) for key in _MODE_KEYS],
                    value=MASK_MASK,
                    label=pick("mask_mode", lang),
                ),
                label=("Режим области", "Region mode"),
                # Подписи вариантов — не служебные идентификаторы вроде имён
                # пресетов, а переводимый текст из T; без этого поля переключатель
                # языка обновил бы всё вокруг радиокнопок, но не их подписи.
                choices=(
                    [(pick(_MODE_KEYS[key], "ru"), key) for key in _MODE_KEYS],
                    [(pick(_MODE_KEYS[key], "en"), key) for key in _MODE_KEYS],
                ),
            )
            quality = localizer.bind(
                gr.Radio(choices=list(presets.NAMES), value=studio.config.preset, label=pick("quality", lang)),
                label=("Качество", "Quality"),
            )
            status = localizer.bind(
                gr.Textbox(
                    label=pick("status", lang), interactive=False, lines=3,
                    elem_classes=[layout.STATUS],
                ),
                label=("Состояние", "Status"),
            )

            advanced = localizer.bind(
                gr.Accordion(pick("advanced", lang), open=False),
                label=("Продвинутое", "Advanced"),
            )
            with advanced:
                grow = localizer.bind(
                    gr.Slider(0, 64, value=8, step=1, label=pick("mask_grow", lang)),
                    label=("Запас маски, пикселей", "Mask grow, pixels"),
                )
                feather = localizer.bind(
                    gr.Slider(0, 64, value=12, step=1, label=pick("mask_feather", lang)),
                    label=("Растушёвка, пикселей", "Feather, pixels"),
                )
                keep_outside = localizer.bind(
                    gr.Checkbox(
                        value=True, label=pick("keep_outside", lang), info=pick("keep_outside_info", lang)
                    ),
                    label=("Сохранять кадр вне маски", "Keep pixels outside the mask"),
                    info=(
                        "Склеивает результат с оригиналом: вне маски пиксели остаются исходными.",
                        "Blends the result with the original so pixels outside the mask stay untouched.",
                    ),
                )
                seed = localizer.bind(
                    gr.Number(value=-1, precision=0, label=pick("seed", lang)), label=("Сид", "Seed")
                )

            outpaint_accordion = localizer.bind(
                gr.Accordion(pick("outpaint", lang), open=False),
                label=("Расширить холст", "Outpaint"),
            )
            with outpaint_accordion:
                sides = localizer.bind(
                    gr.CheckboxGroup(
                        choices=[("←", "left"), ("→", "right"), ("↑", "top"), ("↓", "bottom")],
                        label=pick("outpaint_sides", lang),
                    ),
                    label=("Стороны", "Sides"),
                )
                amount = localizer.bind(
                    gr.Slider(0.1, 1.0, value=0.35, step=0.05, label=pick("outpaint_amount", lang)),
                    label=("Насколько расширить", "How far to expand"),
                )
                expand_button = localizer.bind(
                    gr.Button(pick("outpaint", lang)), value=("Расширить холст", "Outpaint")
                )

    # --- обработчики ---

    def expand_canvas(raw, chosen_sides, ratio_amount, lang):
        value, failure = read_painter(raw, lang)
        if failure:
            return gr.update(), MASK_MASK, failure
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), MASK_MASK, say("upload_first", lang)
        if not chosen_sides:
            return gr.update(), MASK_MASK, say("choose_a_side", lang)

        canvas, mask = outpaint.expand(source, outpaint.plan(source.size, chosen_sides, ratio_amount))

        # Новая площадь показывается пользователю как нарисованная область,
        # чтобы её было видно и можно было поправить кистью.
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        painted = Image.new("RGBA", canvas.size, (255, 0, 0, 255))
        layer.paste(painted, (0, 0), mask)

        return (
            painter.encode(canvas, layer),
            MASK_MASK,
            say("canvas_expanded", lang, width=canvas.size[0], height=canvas.size[1]),
        )

    def describe(raw, lang):
        value, failure = read_painter(raw, lang)
        if failure:
            return gr.update(), failure
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), say("upload_first", lang)
        text, message = studio.describe_image(source, lang)
        return (text or gr.update()), message

    def run(
        raw, prompt_text, use_boost, mode_value, quality_name,
        grow_value, feather_value, keep_value, seed_value, lang,
        progress=gr.Progress(),
    ):
        # Обработчик целиком под try по тем же причинам, что и на вкладке
        # генерации: сбой модели обязан стать строкой состояния.
        try:
            return _apply(
                raw, prompt_text, use_boost, mode_value, quality_name,
                grow_value, feather_value, keep_value, seed_value, lang, progress,
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Обработчик правки не выполнен")
            return [], describe_failure(error, lang)

    def _apply(
        raw, prompt_text, use_boost, mode_value, quality_name,
        grow_value, feather_value, keep_value, seed_value, lang, progress,
    ):
        value, failure = read_painter(raw, lang)
        if failure:
            return [], failure
        source, mask = collect(value, mode_value)
        if source is None:
            return [], say("upload_first", lang)
        # Пустой промт — не «правь на своё усмотрение»: модель правки работает
        # по инструкции, и без неё результат не определён. Хуже всего с
        # расширением холста: новая площадь выходит прозрачной, с пурпуром
        # декодера под альфой (увидено на снимках для README: человек,
        # нажавший «Расширить» и «Применить», получал именно это).
        if not (prompt_text or "").strip():
            return [], say("edit_needs_prompt", lang)

        effective, message = prompt_text, ""
        if use_boost:
            effective, _, message = studio.boost_prompt(
                prompt_text, boost_module.MODE_EDIT, lang, [source]
            )

        request = GenerationRequest(
            prompt=effective or prompt_text,
            prompt_original=prompt_text,
            preset=presets.get(quality_name),
            # Явный признак «наследовать размеры от исходного изображения». Раньше эту
            # роль играло значение "1:1", из-за чего осознанный выбор квадрата при правке
            # был неотличим от отсутствия выбора и молча игнорировался.
            aspect=aspect_module.FOLLOW_REFERENCE,
            seed=int(seed_value),
            source=source,
            mask=mask,
            # Режим переигрывается по факту, а не по выбору пользователя: если
            # mask is None, collect() уже решил, что маскировать нечего (пустая
            # маска в mask/region), и это решение — источник истины, а не
            # напоминание. Аннотация — исключение: там маски нет по определению
            # режима, а не из-за пустого рисунка, поэтому её оставляем как есть.
            mask_mode=mode_value if mask is not None or mode_value == MASK_ANNOTATION else MASK_NONE,
            mask_grow=int(grow_value),
            mask_feather=int(feather_value),
            keep_outside=bool(keep_value),
        )

        # Автоматически урезанный масштаб условных изображений обязан быть
        # виден и здесь: исходник при правке — такое же условное
        # изображение, как референс, и на среднем пресете полный масштаб не
        # помещается в карту вовсе. Молчать о подмене значило бы оставить
        # необъяснимую потерю детальности исходника.
        chosen = resolve_reference_scale(request)
        if chosen != request.preset.output_resolution:
            message = sentences(message, say("reference_scale_chosen", lang, scale=chosen))

        def report(index: int, step: int, total: int) -> None:
            progress((step, total), desc=say("progress_edit", lang))

        # Стадия до первого шага: прогресс из пайплайна приходит только
        # после шага, а загрузка модели и кодирование промта идут раньше
        # и молча. Отличить работу от зависания пользователь не мог.
        progress(0, desc=say(
            "stage_loading" if not studio.model_loaded else "stage_preparing", lang
        ))
        produced, failure = studio.run_generation(request, lang, progress=report)
        if failure is not None:
            return [], sentences(message, failure)
        if not produced:
            return [], sentences(message, say("edit_interrupted", lang))

        paths = []
        for item in produced:
            destination = gallery.next_path(config.OUTPUT_DIR)
            metadata.save_png(item.image, destination, item.parameters)
            paths.append(str(destination))

        edit_done = say("edit_done", lang, memory=studio.memory_report(lang))

        # Обрезанная склейкой работа модели — единственный случай, когда
        # результат формально безупречен (обещание о неприкосновенности
        # выполнено), а глазу виден шов. Молчать о нём значило бы оставить
        # пользователя гадать; штатный выход у случая есть, и он в одном
        # переключателе.
        clipped = max((item.parameters.get("clipped_outside_pct", 0.0) for item in produced),
                      default=0.0)
        if clipped >= CLIPPED_WARNING_PCT:
            edit_done = sentences(edit_done, say("edit_clipped", lang, share=clipped))

        # Результат открывается крупно, а не сеткой. ``preview=True`` у галереи
        # действует только при первой загрузке страницы: при новом значении Gradio
        # возвращается к сетке, и единственная картинка становилась квадратной
        # миниатюрой выше окна — видна была средняя полоса кадра (найдено на снимках
        # для README). ``selected_index=0`` открывает первую картинку в просмотре.
        return gr.Gallery(value=paths, selected_index=0), sentences(message, edit_done)

    def take_back(produced, lang):
        if not produced:
            return gr.update(), say("nothing_to_send", lang)
        first = produced[0]
        path = first[0] if isinstance(first, (list, tuple)) else first
        with Image.open(path) as opened:
            image = opened.convert("RGBA")
        return painter.encode(image), say("sent_to_editor", lang)

    def show_region(mode_value):
        # Режим области меняет вид кисти: в «маске» пометки полупрозрачные и
        # одного цвета, в «аннотации» — палитра и полная непрозрачность.
        return gr.update(region=mode_value)

    def stop(lang):
        if studio.model_loaded:
            studio.generator.interrupt()
        return say("stopping", lang)

    # Каждое событие, которому нужна маска, сначала забирает у кисти свежее
    # значение: синхронизация со страницы отложенная, и без этого последний
    # мазок перед нажатием мог не успеть. Кисть — первый вход у всех трёх.
    flush = painter.flush_js(PAINTER_ID)
    expand_button.click(
        expand_canvas, [editor, sides, amount, language], [editor, mode, status], js=flush
    )
    describe_button.click(describe, [editor, language], [prompt, status], js=flush)
    run_button.click(
        run,
        [editor, prompt, boost_enabled, mode, quality, grow, feather, keep_outside, seed,
         language],
        [result, status],
        # Та же группа очереди, что и у «Сгенерировать»: видеокарта одна.
        concurrency_id=GPU_CONCURRENCY_ID,
        js=flush,
    )
    stop_button.click(stop, language, status, queue=False)
    send_back.click(take_back, [result, language], [editor, status])
    mode.change(show_region, mode, editor, queue=False)

    return {"editor": editor, "result": result, "prompt": prompt, "status": status}
