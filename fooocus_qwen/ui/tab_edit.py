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
)
from ..imaging import aspect as aspect_module
from ..imaging import masking, metadata, outpaint
from ..prompting import boost as boost_module
from ..storage import gallery
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)

# Цвета из примера в блоге: три области, три инструкции в одном промте.
ANNOTATION_COLOURS: tuple[str, ...] = ("#ff0000", "#0000ff", "#00ff00", "#ffff00", "#ffffff")

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


def build(studio, localizer: Localizer) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column(scale=3):
            editor = localizer.bind(
                gr.ImageEditor(
                    label=pick("source_image", lang),
                    type="pil",
                    image_mode="RGBA",
                    layers=True,
                    height=620,
                    brush=gr.Brush(colors=list(ANNOTATION_COLOURS), default_color="#ff0000", color_mode="fixed"),
                    eraser=gr.Eraser(),
                    sources=("upload", "clipboard"),
                ),
                label=("Исходное изображение", "Source image"),
            )

            with gr.Row():
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

            with gr.Row():
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("boost", lang), value=False),
                    label=("AI буст", "AI boost"),
                )
                describe_button = localizer.bind(
                    gr.Button(pick("describe", lang)), value=("Описать изображение", "Describe image")
                )

            result = localizer.bind(
                gr.Gallery(label=pick("result", lang), columns=2, height=400, object_fit="contain", format="png"),
                label=("Результат", "Result"),
            )
            send_back = localizer.bind(
                gr.Button(pick("send_to_edit", lang)), value=("Отправить в редактор", "Send to editor")
            )

        with gr.Column(scale=1):
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
                gr.Textbox(label=pick("status", lang), interactive=False, lines=3),
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

    def expand_canvas(value, chosen_sides, ratio_amount):
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), MASK_MASK, "Сначала загрузите изображение"
        if not chosen_sides:
            return gr.update(), MASK_MASK, "Выберите хотя бы одну сторону"

        canvas, mask = outpaint.expand(source, outpaint.plan(source.size, chosen_sides, ratio_amount))

        # Новая площадь показывается пользователю как нарисованная область,
        # чтобы её было видно и можно было поправить кистью.
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        painted = Image.new("RGBA", canvas.size, (255, 0, 0, 255))
        layer.paste(painted, (0, 0), mask)

        return (
            {"background": canvas, "layers": [layer], "composite": None},
            MASK_MASK,
            f"Холст расширен до {canvas.size[0]}×{canvas.size[1]}",
        )

    def describe(value):
        source, _ = collect(value, MASK_NONE)
        if source is None:
            return gr.update(), "Сначала загрузите изображение"
        text, message = studio.describe_image(source)
        return (text or gr.update()), message

    def run(
        value, prompt_text, use_boost, mode_value, quality_name,
        grow_value, feather_value, keep_value, seed_value,
        progress=gr.Progress(),
    ):
        source, mask = collect(value, mode_value)
        if source is None:
            return [], "Сначала загрузите изображение"

        effective, message = prompt_text, ""
        if use_boost:
            effective, _, message = studio.boost_prompt(
                prompt_text, boost_module.MODE_EDIT, [source]
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
            mask_mode=mode_value if mask is not None or mode_value == MASK_ANNOTATION else MASK_NONE,
            mask_grow=int(grow_value),
            mask_feather=int(feather_value),
            keep_outside=bool(keep_value),
        )

        def report(index: int, step: int, total: int) -> None:
            progress((step, total), desc="правка")

        produced = studio.generator.generate(request, progress=report)
        if not produced:
            return [], f"{message} Правка прервана".strip()

        paths = []
        for item in produced:
            destination = gallery.next_path(config.OUTPUT_DIR)
            metadata.save_png(item.image, destination, item.parameters)
            paths.append(str(destination))

        return paths, f"{message} Готово. {studio.memory_report()}".strip()

    def take_back(produced):
        if not produced:
            return gr.update(), "Нечего отправлять"
        first = produced[0]
        path = first[0] if isinstance(first, (list, tuple)) else first
        image = Image.open(path).convert("RGBA")
        return {"background": image, "layers": [], "composite": None}, "Результат перенесён в редактор"

    def stop():
        if studio.model_loaded:
            studio.generator.interrupt()
        return "Останавливаю…"

    expand_button.click(expand_canvas, [editor, sides, amount], [editor, mode, status])
    describe_button.click(describe, editor, [prompt, status])
    run_button.click(
        run,
        [editor, prompt, boost_enabled, mode, quality, grow, feather, keep_outside, seed],
        [result, status],
    )
    stop_button.click(stop, None, status, queue=False)
    send_back.click(take_back, result, [editor, status])

    return {"editor": editor, "result": result, "prompt": prompt, "status": status}
