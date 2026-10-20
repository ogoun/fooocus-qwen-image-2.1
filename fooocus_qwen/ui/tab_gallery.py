"""Вкладка галереи: история генераций и возврат к их параметрам.

Выбранная картинка описывается карточкой, а не сырым JSON: человеку нужны
промт, размер, сид и качество, а не ключи словаря. Из карточки два
действия — «Повторить параметры» (поля вкладки генерации заполняются, и
интерфейс сам переходит на неё) и «Открыть в редакторе» (картинка уходит в
кисть маски, переход на правку). Раньше до обоих было не дотянуться:
параметры восстанавливались только из перетащенного файла, а в редактор
картинку из галереи было не отправить вовсе.

Параметры генерации живут внутри самих PNG (см. ``imaging.metadata``), поэтому
история переживает и перезапуск оболочки, и перенос файлов результатов на
другую машину — отдельная база для этого не нужна: любой наш PNG сам себе
запись в журнале.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from .. import config
from ..engine import lora, presets
from ..imaging import aspect as aspect_module
from ..imaging import metadata
from ..storage import gallery
from . import layout, lora_slots
from .i18n import Localizer, pick, say
from .tab_edit import editor_value_for, selected_path

# Порядок обязан совпадать с порядком выходов кнопки «Восстановить» в build():
# восстановление читает по этому же порядку значения из словаря параметров, а
# build() раскладывает их по тем же RESTORED_FIELDS полям вкладки генерации. Оба места
# проверяет test_restore.test_restore_output_order_matches_generate_components,
# построенный на реально собранном графе Gradio, а не на переписанном вручную
# списке ожиданий — так что рассинхронизация здесь не пройдёт тесты молча.


def _safe_int(value: object, default: int) -> int:
    """``int(value)``, но не роняет обработчик на нечисловых/чужих метаданных."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float) -> float:
    """``float(value)``, с тем же снисхождением к испорченному значению."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


DEFAULT_ASPECT = "1:1"

# Сколько полей вкладки генерации восстанавливает restore_fields(). Держится
# рядом с самой функцией, чтобы «сколько gr.update() вернуть, когда
# восстанавливать нечего» не приходилось пересчитывать руками в двух местах.
RESTORED_FIELDS = 8 + 3 * lora.SLOTS


def restore_fields(parameters: dict | None, lora_dir: Path | None = None) -> tuple:
    """Значения полей вкладки генерации в фиксированном порядке.

    Порядок: промт, переписанный промт, негатив, стили, пресет, сид, guidance,
    соотношение сторон, затем ячейки LoRA (вкл., файл, вес на каждую;
    ``lora_slots.values_for`` — файл узнаётся и по отпечатку). Кадр без LoRA
    очищает ячейки: параметры повторяются такими, какими были. Отсутствующий или незнакомый пресет (например, из
    старой сборки) тихо заменяется дефолтным — таблица пресетов меняется
    быстрее, чем метаданные в уже сохранённых PNG; с соотношением сторон
    поступаем так же, и по той же причине к нему добавляется ещё одна: PNG,
    сохранённые до появления ключа ``aspect``, его просто не содержат. Сид и
    guidance читаются со снисхождением: PNG с нашим ключом чанка, но
    нечисловым значением (правленный руками файл или чужой инструмент,
    переиспользовавший ключ) не должен ронять обработчик кнопки — он просто
    получит дефолт вместо этого поля.
    """
    data = parameters or {}
    preset = data.get("preset", presets.DEFAULT)
    if preset not in presets.PRESETS:
        preset = presets.DEFAULT

    ratio = data.get("aspect", DEFAULT_ASPECT)
    if ratio not in aspect_module.ASPECT_RATIOS:
        ratio = DEFAULT_ASPECT

    return (
        data.get("prompt", ""),
        data.get("prompt_boosted", ""),
        data.get("negative_prompt", ""),
        list(data.get("styles", [])),
        preset,
        _safe_int(data.get("seed", -1), -1),
        _safe_float(data.get("true_cfg_scale", 1.0), 1.0),
        ratio,
        *lora_slots.values_for(lora_dir or config.LORA_DIR, data.get("loras")),
    )


VIDEO_FIELDS = ("prompt", "mode", "resolution", "aspect", "seconds", "quality", "seed",
                "upscale_enabled", "upscale_mode", "upscale_factor", "upscale_width", "upscale_height",
                "upscale_megapixels")


def video_restore_outputs(video_components: dict | None) -> list:
    """Поля вкладки Video, которые восстанавливает ролик, — в порядке ``VIDEO_FIELDS``."""
    if video_components is None:
        return []
    return [video_components[name] for name in VIDEO_FIELDS]


def restore_video_fields(parameters: dict) -> tuple:
    """Значения полей вкладки Video по параметрам ролика; чужие значения — умолчаниями."""
    from ..video import geometry, prompt_h3
    from ..video.request import DEFAULT_QUALITY, QUALITY

    def text(name: str) -> str | None:
        # Только строки: список или словарь из чужого JSON не хешируется, и
        # проверка «in QUALITY» уронила бы обработчик.
        value = parameters.get(name)
        return value if isinstance(value, str) else None

    mode, resolution, aspect_value, quality = text("mode"), text("resolution"), text("aspect"), text("quality")
    frames = _safe_int(parameters.get("frames"), 0)
    seconds = frames / geometry.FPS if frames > 0 else _safe_float(parameters.get("requested_seconds"), 5.0)
    # Ползунок вкладки — от MIN_SECONDS до MAX_SECONDS; значение вне него Gradio не примет.
    seconds = min(max(seconds, geometry.MIN_SECONDS), geometry.MAX_SECONDS) if math.isfinite(seconds) else 5.0
    # Апскейл: ролик без него снимает галочку, но поля оставляет умолчаниями.
    upscale = geometry.Upscale.from_dict(parameters.get("upscale"))
    shown = upscale or geometry.Upscale()
    return (
        text("prompt") or "",
        mode if mode in prompt_h3.MODES else prompt_h3.MODE_T2V,
        resolution if resolution in geometry.RESOLUTIONS else geometry.DEFAULT_RESOLUTION,
        aspect_value if aspect_value in (*geometry.ASPECT_RATIOS, geometry.FOLLOW_IMAGE) else geometry.DEFAULT_ASPECT,
        seconds,
        quality if quality in QUALITY else DEFAULT_QUALITY,
        _safe_int(parameters.get("seed", -1), -1),
        upscale is not None,
        shown.mode,
        min(max(shown.factor, 1.0), geometry.MAX_UPSCALE) if math.isfinite(shown.factor) else 2.0,
        shown.width,
        shown.height,
        min(max(shown.megapixels, 0.5), 8.0) if math.isfinite(shown.megapixels) else 2.0,
    )


def _open_folder(path: Path) -> None:
    """Открывает каталог в файловом менеджере системы."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _escape(text: str) -> str:
    """Текст пользователя в Markdown карточки — как есть, без разметки.

    Промт пишет человек, и звёздочка или подчёркивание в нём — не курсив.
    """
    return "".join(f"\\{char}" if char in "\\`*_{}[]<>()#+-.!|" else char for char in text)


def is_video(path: Path | str | None) -> bool:
    return path is not None and Path(path).suffix.lower() == ".mp4"


def read_parameters(path: Path) -> dict:
    """Параметры результата: из PNG — чанком, у ролика — из соседнего JSON."""
    if is_video(path):
        from ..video.request import read_parameters as read_video

        return read_video(path)
    return metadata.read_png(path)


def describe_video(parameters: dict, path: Path, lang: str) -> str:
    """Карточка ролика: сценарий, промт H3, режим, кадр, длительность, сид."""
    lines = []
    if parameters.get("prompt"):
        lines += [f"**{say('card_prompt', lang)}**", "", _escape(str(parameters["prompt"])), ""]
    rows = []
    if parameters.get("mode"):
        rows.append((say("card_mode", lang), str(parameters["mode"])))
    if parameters.get("width") and parameters.get("height"):
        rows.append((say("card_size", lang), f"{parameters['width']}×{parameters['height']}"))
        output = (parameters.get("output_width"), parameters.get("output_height"))
        if all(output) and output != (parameters["width"], parameters["height"]):
            rows.append((say("card_output", lang), f"{output[0]}×{output[1]}"))
    if parameters.get("frames"):
        rows.append((say("card_duration", lang), say("card_frames", lang, seconds=parameters.get("seconds"),
                                                      frames=parameters["frames"])))
    if "seed" in parameters:
        rows.append((say("card_seed", lang), str(parameters["seed"])))
    if parameters.get("quality"):
        rows.append((say("card_quality", lang), str(parameters["quality"])))
    if parameters.get("seconds_taken") is not None:
        rows.append((say("card_time", lang), say("card_seconds", lang, seconds=parameters["seconds_taken"])))
    rows.append((say("card_file", lang), path.name))
    lines += ["| | |", "|---|---|"]
    lines += [f"| {_escape(key)} | {_escape(value)} |" for key, value in rows]
    return "\n".join(lines)


def describe(parameters: dict | None, path: Path | None, lang: str) -> str:
    """Карточка выбранной картинки или ролика в Markdown."""
    if path is None:
        return say("gallery_pick", lang)
    if is_video(path) and parameters:
        return describe_video(parameters, path, lang)
    if not parameters:
        return f"**{_escape(path.name)}**\n\n{say('png_without_parameters', lang)}"

    lines = []
    if parameters.get("prompt"):
        lines += [f"**{say('card_prompt', lang)}**", "", _escape(str(parameters["prompt"])), ""]
    boosted = parameters.get("prompt_boosted")
    if boosted and boosted != parameters.get("prompt"):
        lines += [f"**{say('card_boosted', lang)}**", "", _escape(str(boosted)), ""]
    if parameters.get("negative_prompt"):
        lines += [f"**{say('card_negative', lang)}**", "", _escape(str(parameters["negative_prompt"])), ""]

    rows = []
    if parameters.get("width") and parameters.get("height"):
        rows.append((say("card_size", lang), f"{parameters['width']}×{parameters['height']}"))
    if "seed" in parameters:
        rows.append((say("card_seed", lang), str(parameters["seed"])))
    if parameters.get("preset"):
        steps = parameters.get("steps")
        quality = str(parameters["preset"]) + (f" · {say('card_steps', lang, steps=steps)}" if steps else "")
        rows.append((say("card_quality", lang), quality))
    if "true_cfg_scale" in parameters:
        rows.append((say("card_guidance", lang), str(parameters["true_cfg_scale"])))
    if parameters.get("styles"):
        rows.append((say("card_styles", lang), ", ".join(map(str, parameters["styles"]))))
    if parameters.get("seconds") is not None:
        rows.append((say("card_time", lang), say("card_seconds", lang, seconds=parameters["seconds"])))
    rows.append((say("card_file", lang), path.name))

    lines += ["| | |", "|---|---|"]
    lines += [f"| {_escape(key)} | {_escape(value)} |" for key, value in rows]
    return "\n".join(lines)


def build(
    studio,
    localizer: Localizer,
    generate_components: dict,
    language=None,
    edit_components: dict | None = None,
    tabs=None,
    video_components: dict | None = None,
) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком; см. докстринг
    ``tab_generate.build``. ``edit_components`` и ``tabs`` нужны для
    действий, которые уводят с вкладки: «Открыть в редакторе» кладёт картинку
    в кисть правки, и оба действия переключают вкладку сами. Без них вкладка
    собирается и работает, только без этих переходов — так её собирают тесты.
    """
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.CANVAS_COL]):
            history = localizer.bind(
                gr.Gallery(
                    label=pick("tab_gallery", lang),
                    columns=6,
                    elem_classes=[layout.BROWSE],
                    object_fit="contain",
                    interactive=False,
                    buttons=layout.GALLERY_BUTTONS,
                    value=[str(path) for path in gallery.recent(config.OUTPUT_DIR)],
                ),
                label=("Галерея", "Gallery"),
            )
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.SIDE_COL]):
            details = gr.Markdown(say("gallery_pick", lang), elem_classes=[layout.CARD])
            with gr.Row():
                reuse = localizer.bind(
                    gr.Button(pick("gallery_reuse", lang), variant="primary"),
                    value=("Повторить параметры", "Reuse parameters"),
                )
                to_editor = localizer.bind(
                    gr.Button(pick("gallery_to_editor", lang), visible=edit_components is not None),
                    value=("Открыть в редакторе", "Open in editor"),
                )
            # Восстановление из файла — одна кнопка: выбрал PNG, и параметры
            # уже во вкладке генерации. Раньше это были поле загрузки и
            # отдельная кнопка с той же подписью, что сбивало с толку.
            from_file = localizer.bind(
                gr.UploadButton(pick("gallery_from_file", lang), file_types=[".png"], size="sm"),
                label=("Параметры из PNG-файла…", "Parameters from a PNG file…"),
            )
            open_button = localizer.bind(
                gr.Button(pick("open_folder", lang), size="sm"),
                value=("Открыть папку", "Open folder"),
            )
            status = gr.Markdown("", elem_classes=[layout.STATUS])

    # Путь последней выбранной картинки: действия карточки работают с ним.
    selected = gr.State(None)

    def refresh_history():
        return [str(path) for path in gallery.recent(config.OUTPUT_DIR)]

    def on_select(items, lang, event: gr.EventData):
        # EventData, а не SelectData: см. tab_edit.selected_path — у события
        # выбора в Gradio 6.5.1 бывает только индекс, без значения.
        found = selected_path(items, event)
        if found is None:
            return gr.update(), gr.update()
        path = Path(found)
        return str(path), describe(read_parameters(path), path, lang)

    def open_outputs(lang):
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _open_folder(config.OUTPUT_DIR)
        return say("opened_folder", lang, path=config.OUTPUT_DIR)

    def _switch(target: str | None) -> tuple:
        """Переход на вкладку — только если есть куда: число значений обязано
        совпадать с числом выходов, иначе Gradio отвергнет ответ целиком."""
        if tabs is None:
            return ()
        return (gr.Tabs(selected=target) if target else gr.update(),)

    video_fields = video_restore_outputs(video_components)
    keep_video = (gr.update(),) * len(video_fields)

    def _restore_from(source: Path | None, lang):
        if source is None:
            return (gr.update(),) * RESTORED_FIELDS + keep_video + (say("gallery_pick", lang),) + _switch(None)
        parameters = read_parameters(source)
        if not parameters:
            return (gr.update(),) * RESTORED_FIELDS + keep_video + (say("parameters_not_found", lang),) \
                + _switch(None)
        message = say("parameters_restored", lang, name=source.name)
        if is_video(source):
            if not video_fields:
                return (gr.update(),) * RESTORED_FIELDS + (say("parameters_not_found", lang),) + _switch(None)
            return (gr.update(),) * RESTORED_FIELDS + restore_video_fields(parameters) \
                + (say("video_parameters_restored", lang, name=source.name),) + _switch(layout.TAB_VIDEO)
        return restore_fields(parameters, studio.config.lora_dir) + keep_video + (message,) \
            + _switch(layout.TAB_GENERATE)

    def restore(path, lang):
        return _restore_from(Path(path) if path else None, lang)

    def restore_from_file(uploaded, lang):
        return _restore_from(Path(uploaded) if uploaded else None, lang)

    def send_to_editor(path, lang):
        if not path:
            return (gr.update(), say("gallery_pick", lang)) + _switch(None)
        if is_video(path):
            # В кисть уходит картинка: у ролика — его последний кадр (с него
            # обычно и продолжают — поправить и снять следующий шот).
            from ..video import assemble

            frame = assemble.last_frame(path)
            from . import painter

            return (painter.encode(frame.convert("RGBA")), say("sent_to_editor", lang)) + _switch(layout.TAB_EDIT)
        return (editor_value_for(path), say("sent_to_editor", lang)) + _switch(layout.TAB_EDIT)

    restore_outputs = [
        generate_components["prompt"],
        generate_components["boosted"],
        generate_components["negative"],
        generate_components["styles"],
        generate_components["quality"],
        generate_components["seed"],
        generate_components["cfg"],
        generate_components["ratio"],
        *generate_components["loras"].components,
        *video_fields,
        status,
    ]
    switch_output = [tabs] if tabs is not None else []

    open_button.click(open_outputs, language, status)
    history.select(on_select, [history, language], [selected, details])
    reuse.click(restore, [selected, language], restore_outputs + switch_output)
    from_file.upload(restore_from_file, [from_file, language], restore_outputs + switch_output)
    if edit_components is not None:
        to_editor.click(
            send_to_editor, [selected, language],
            [edit_components["editor"], status] + switch_output,
        )

    return {"history": history, "details": details, "selected": selected, "refresh": refresh_history}
