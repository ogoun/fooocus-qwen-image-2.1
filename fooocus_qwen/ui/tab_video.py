"""Вкладка Video: ролики MiniMax H3 через движок FreeVideo.

Компоновка — как у генерации: слева входы и результат, под ними промт и
кнопка, справа режим, размер, длительность и качество. Режимов четыре, как у
самой модели: текст → видео, первый кадр → видео, первый и последний кадр,
референсы (персонажи, стиль, голос; картинки, видео и звук), и правка
готового ролика — «видео → видео» (Ref2VA с исходником ``<Video 1>``).
Видны только входы выбранного режима; референсы — плитками со своими
метками (``ui.video_refs``).

Автоматизация — по «Вайб-монтажу с ИИ» (docs/research/2026-10-19-video-freevideo.md):

* промт H3 собирается из свободного текста сам — поля, строка выравнивания
  с фактической длительностью, метки референсов (``video.prompt_h3``);
  AI-буст переписывает его по официальным правилам H3;
* длительность — секундами, кадры и фактическое время показываются;
  длиннее 15 с — частями с бесшовным продолжением;
* черновик → финал: черновик — первый проход в половинном размере, и
  «Довести до финала» продолжает именно его, без новой генерации;
* «Последний кадр → следующий шот» и «Продлить ролик» — цепочка шотов;
* раскадровка: бриф → шоты (языковая модель, типовые структуры) → все
  шоты подряд с цепочкой кадров → склейка с переходами и выравниванием
  громкости → контактный лист для приёмки.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import gradio as gr

from .. import config
from ..prompting import library
from ..video import board as video_board
from ..video import geometry, prompt_h3, storyboard
from ..video.request import (
    DEFAULT_QUALITY,
    QUALITY,
    Reference,
    VideoRequest,
    edit_seconds,
    kind_of,
)
from . import layout, prompt_library, video_refs
from .i18n import Localizer, T, pick, say, sentences
from .state import GPU_CONCURRENCY_ID, describe_failure
from .tab_edit import chosen_path

LOGGER = logging.getLogger(__name__)

MODES = prompt_h3.MODES


# Подписи режимов и уровней — ключами таблицы, а не собранными строкой:
# проверка мёртвых переводов ищет ключ литералом в коде.
MODE_LABELS = {
    prompt_h3.MODE_T2V: "video_mode_t2v",
    prompt_h3.MODE_I2V: "video_mode_i2v",
    prompt_h3.MODE_FLF2V: "video_mode_flf2v",
    prompt_h3.MODE_REF: "video_mode_ref",
    prompt_h3.MODE_V2V: "video_mode_v2v",
}
UPSCALE_LABELS = {
    geometry.UPSCALE_MULTIPLIER: "video_upscale_multiplier",
    geometry.UPSCALE_DIMENSIONS: "video_upscale_dimensions",
    geometry.UPSCALE_MEGAPIXELS: "video_upscale_megapixels",
}
QUALITY_LABELS = {
    "draft": "video_quality_draft",
    "light": "video_quality_light",
    "medium": "video_quality_medium",
    "high": "video_quality_high",
    "max": "video_quality_max",
}


def mode_choices(lang: str) -> list[tuple[str, str]]:
    return [(pick(MODE_LABELS[mode], lang), mode) for mode in MODES]


def quality_choices(lang: str) -> list[tuple[str, str]]:
    return [(pick(QUALITY_LABELS[name], lang), name) for name in QUALITY]


def upscale_choices(lang: str) -> list[tuple[str, str]]:
    return [(pick(UPSCALE_LABELS[name], lang), name) for name in geometry.UPSCALE_MODES]


def upscale_from(enabled, mode, factor, width, height, megapixels) -> geometry.Upscale | None:
    """Настройки апскейла из полей; None — галочка снята."""
    if not enabled:
        return None
    defaults = geometry.Upscale()

    def number(value, default, kind=float):
        try:
            return kind(value)
        except (TypeError, ValueError):
            return default

    return geometry.Upscale(
        mode=mode if mode in geometry.UPSCALE_MODES else defaults.mode,
        factor=number(factor, defaults.factor),
        width=number(width, defaults.width, int),
        height=number(height, defaults.height, int),
        megapixels=number(megapixels, defaults.megapixels),
    )


def upscale_visibility(enabled, mode) -> tuple:
    """Какие поля апскейла видны: блок, множитель, стороны, мегапиксели."""
    return (
        gr.update(visible=bool(enabled)),
        gr.update(visible=mode == geometry.UPSCALE_MULTIPLIER),
        gr.update(visible=mode == geometry.UPSCALE_DIMENSIONS),
        gr.update(visible=mode == geometry.UPSCALE_MEGAPIXELS),
    )


def aspect_choices(lang: str) -> list[tuple[str, str]]:
    return [(value, value) for value in geometry.ASPECT_RATIOS] + [
        (pick("video_aspect_image", lang), geometry.FOLLOW_IMAGE)
    ]


def structure_choices(lang: str) -> list[tuple[str, str]]:
    index = 0 if lang == "ru" else 1
    return [(item.labels[index], key) for key, item in storyboard.STRUCTURES.items()]


def lora_files(directory: Path | None = None) -> list[str]:
    directory = directory or config.VIDEO_LORA_DIR
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.safetensors"))


def lora_paths(names, strength) -> tuple[tuple[str, float], ...]:
    """Выбранные LoRA видео — пути с общей силой, как их ждёт ``VideoRequest``."""
    return tuple((str(config.VIDEO_LORA_DIR / name), float(strength)) for name in (names or []))


def visibility(mode: str) -> tuple:
    """Какие кадры видны: первый и последний. Плитки референсов — ``video_refs.view``."""
    return (
        gr.update(visible=mode in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V)),
        gr.update(visible=mode == prompt_h3.MODE_FLF2V),
    )


def source_info(path):
    """Размер и длительность исходного ролика; ``None`` — нет файла или он нечитаем."""
    path = video_refs._path(path)
    if not path:
        return None
    try:
        from ..video import assemble

        return assemble.probe(path)
    except Exception:  # noqa: BLE001 — нечитаемый ролик: холст и длительность по полям
        return None


def references_from(files) -> tuple[Reference, ...]:
    """Файлы поля референсов — в референсы с видом по расширению; чужие пропускаются."""
    items = []
    for item in files or []:
        path = item if isinstance(item, str) else getattr(item, "name", None) or getattr(item, "path", None)
        kind = kind_of(path) if path else None
        if kind:
            items.append(Reference(kind, str(path)))
    return tuple(items)


def describe_size(mode, resolution, aspect, seconds, first, last, lang: str,
                  upscale: geometry.Upscale | None = None, quality: str | None = None, source=None) -> str:
    """Строка под ползунком: холст, кадры, фактическая длительность, части и размер после апскейла.

    Правка ролика берёт пропорцию и предел длительности у исходника
    (``source`` — путь к нему), как ``request.build_job``.
    """
    requested = float(seconds or geometry.MIN_SECONDS)
    if mode == prompt_h3.MODE_V2V:
        info = source_info(source)
        size = (info.width, info.height) if info else None
        width, height = geometry.canvas(resolution, geometry.FOLLOW_IMAGE if size else aspect, size)
        duration = geometry.duration(edit_seconds(requested, info.seconds if info else None))
    else:
        anchor = first if mode in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V) else None
        anchor = anchor or (last if mode == prompt_h3.MODE_FLF2V else None)
        width, height = geometry.canvas(resolution, aspect, anchor.size if anchor is not None else None)
        duration = geometry.duration(requested)
    key = "video_size_parts" if duration.seconds > geometry.SINGLE_MAX_SECONDS else "video_size"
    line = say(key, lang, width=width, height=height, frames=duration.frames, seconds=duration.label)
    if upscale is None:
        return line
    try:
        size = geometry.upscaled_size(width, height, upscale)
    except ValueError as error:
        return sentences(line, say("video_size_upscale_invalid", lang, error=error))
    if size is None:
        return sentences(line, say("video_size_upscale_none", lang))
    if quality == "draft":
        return sentences(line, say("video_size_upscale_draft", lang, width=size[0], height=size[1]))
    factor = max(size[0] / width, size[1] / height)
    return sentences(line, say("video_size_upscaled", lang, width=size[0], height=size[1], factor=f"{factor:.2f}"))


def build(studio, localizer: Localizer, language=None) -> dict:
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)
    # Останов раскадровки: кнопка «Прервать» шлёт отмену и в текущий ролик,
    # и в цикл шотов — иначе прервался бы один шот, а следующий начался бы.
    stop_requested = threading.Event()

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.CANVAS_COL]):
            with gr.Row(equal_height=True, elem_classes=[layout.RESULT_ROW]):
                with gr.Column(scale=1, min_width=220, elem_classes=[layout.VIDEO_INPUTS]):
                    first = localizer.bind(
                        gr.Image(label=pick("video_first", lang), type="pil", visible=False,
                                 sources=["upload", "clipboard"], buttons=["fullscreen"]),
                        label=T["video_first"],
                    )
                    last = localizer.bind(
                        gr.Image(label=pick("video_last", lang), type="pil", visible=False,
                                 sources=["upload", "clipboard"], buttons=["fullscreen"]),
                        label=T["video_last"],
                    )
                    tiles = video_refs.build(lang)
                    references = tiles.state
                    source = tiles.source
                result = localizer.bind(
                    gr.Video(label=pick("video_result", lang), interactive=False, loop=True,
                             elem_classes=[layout.VIDEO_BOARD], scale=3),
                    label=T["video_result"],
                )
            with gr.Row():
                to_next = localizer.bind(
                    gr.Button(pick("video_to_next", lang), size="sm"), value=T["video_to_next"]
                )
                extend = localizer.bind(
                    gr.Button(pick("video_extend", lang), size="sm"), value=T["video_extend"]
                )
                finish = localizer.bind(
                    gr.Button(pick("video_finish", lang), size="sm", visible=False), value=T["video_finish"]
                )
            # Параметры последнего ролика: «Довести», «Продлить», «Следующий шот»
            # работают с ним, а не с тем, что сейчас в полях.
            last_result = gr.State(None)

            with gr.Row(elem_classes=[layout.PROMPT_BAR]):
                prompt = localizer.bind(
                    gr.Textbox(label=pick("video_prompt", lang), placeholder=pick("video_prompt_placeholder", lang),
                               lines=3, scale=8),
                    label=T["video_prompt"], placeholder=T["video_prompt_placeholder"],
                )
                with gr.Column(scale=1, min_width=140):
                    run_button = localizer.bind(
                        gr.Button(pick("video_generate", lang), variant="primary"), value=T["video_generate"]
                    )
                    stop_button = localizer.bind(gr.Button(pick("stop", lang), variant="stop"), value=T["stop"])
            prompts = prompt_library.build(localizer, lang)
            with gr.Row():
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("video_boost", lang), value=False, info=pick("video_boost_info", lang)),
                    label=T["video_boost"], info=T["video_boost_info"],
                )
                boost_now = localizer.bind(gr.Button(pick("boost_now", lang)), value=T["boost_now"])
            boosted = localizer.bind(
                gr.Textbox(label=pick("video_boosted", lang), lines=6, interactive=True, visible=False),
                label=T["video_boosted"],
            )

        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.SIDE_COL]):
            engine_note = gr.Markdown("", visible=False, elem_classes=[layout.CARD])
            install_button = localizer.bind(
                gr.Button(pick("video_install", lang), visible=False), value=T["video_install"]
            )
            mode = localizer.bind(
                gr.Radio(choices=mode_choices(lang), value=prompt_h3.MODE_T2V, label=pick("video_mode", lang),
                         elem_classes=[layout.CHOICES]),
                label=T["video_mode"], choices=(mode_choices("ru"), mode_choices("en")),
            )
            with gr.Row():
                resolution = localizer.bind(
                    gr.Dropdown(choices=list(geometry.RESOLUTIONS), value=geometry.DEFAULT_RESOLUTION,
                                label=pick("video_resolution", lang), min_width=120),
                    label=T["video_resolution"],
                )
                aspect = localizer.bind(
                    gr.Dropdown(choices=aspect_choices(lang), value=geometry.DEFAULT_ASPECT,
                                label=pick("aspect", lang), min_width=120),
                    label=T["aspect"], choices=(aspect_choices("ru"), aspect_choices("en")),
                )
            seconds = localizer.bind(
                gr.Slider(geometry.MIN_SECONDS, geometry.MAX_SECONDS, value=5.0, step=0.125,
                          label=pick("video_seconds", lang)),
                label=T["video_seconds"],
            )
            size_note = gr.Markdown(
                describe_size(prompt_h3.MODE_T2V, geometry.DEFAULT_RESOLUTION, geometry.DEFAULT_ASPECT, 5.0,
                              None, None, lang),
                elem_classes=[layout.VIDEO_NOTE],
            )
            quality = localizer.bind(
                gr.Radio(choices=quality_choices(lang), value=DEFAULT_QUALITY, label=pick("video_quality", lang),
                         elem_classes=[layout.CHOICES]),
                label=T["video_quality"], choices=(quality_choices("ru"), quality_choices("en")),
            )
            upscale_enabled = localizer.bind(
                gr.Checkbox(label=pick("video_upscale", lang), value=False, info=pick("video_upscale_info", lang)),
                label=T["video_upscale"], info=T["video_upscale_info"],
            )
            # Параметры как у узла AGSoft Minimax H3 Latent Upscaler: видны,
            # только когда галочка стоит, и только поля выбранного режима.
            with gr.Column(visible=False) as upscale_box:
                upscale_mode = localizer.bind(
                    gr.Radio(choices=upscale_choices(lang), value=geometry.UPSCALE_MULTIPLIER,
                             label=pick("video_upscale_mode", lang), elem_classes=[layout.CHOICES]),
                    label=T["video_upscale_mode"], choices=(upscale_choices("ru"), upscale_choices("en")),
                )
                # Ползунки — в своих контейнерах: в Gradio 6.5 обновление
                # visible у самого Slider прячет его не до конца (остаются
                # поле числа и подпись), а контейнер прячется целиком.
                with gr.Column() as upscale_factor_box:
                    upscale_factor = localizer.bind(
                        gr.Slider(1.0, geometry.MAX_UPSCALE, value=2.0, step=0.05,
                                  label=pick("video_upscale_factor", lang)),
                        label=T["video_upscale_factor"],
                    )
                with gr.Row(visible=False) as upscale_dims:
                    upscale_width = localizer.bind(
                        gr.Number(value=1920, precision=0, step=geometry.SIDE_STEP, minimum=geometry.MIN_SIDE,
                                  label=pick("video_upscale_width", lang), min_width=100),
                        label=T["video_upscale_width"],
                    )
                    upscale_height = localizer.bind(
                        gr.Number(value=1088, precision=0, step=geometry.SIDE_STEP, minimum=geometry.MIN_SIDE,
                                  label=pick("video_upscale_height", lang), min_width=100),
                        label=T["video_upscale_height"],
                    )
                with gr.Column(visible=False) as upscale_mp_box:
                    upscale_megapixels = localizer.bind(
                        gr.Slider(0.5, 8.0, value=2.0, step=0.1, label=pick("video_upscale_mp", lang)),
                        label=T["video_upscale_mp"],
                    )
            status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=3, elem_classes=[layout.STATUS]),
                label=T["status"],
            )
            advanced = localizer.bind(gr.Accordion(pick("advanced", lang), open=False), label=T["advanced"])
            with advanced:
                seed = localizer.bind(
                    gr.Number(value=-1, precision=0, label=pick("seed", lang), info=pick("seed_info", lang)),
                    label=T["seed"], info=T["seed_info"],
                )
                loras = localizer.bind(
                    gr.Dropdown(choices=lora_files(), value=[], multiselect=True,
                                label=pick("video_loras", lang), info=pick("video_loras_info", lang)),
                    label=T["video_loras"], info=T["video_loras_info"],
                )
                lora_strength = localizer.bind(
                    gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label=pick("lora_weight", lang)),
                    label=T["lora_weight"],
                )
            board = localizer.bind(gr.Accordion(pick("video_storyboard", lang), open=False),
                                   label=T["video_storyboard"])
            with board:
                brief = localizer.bind(
                    gr.Textbox(label=pick("video_brief", lang), lines=4,
                               placeholder=pick("video_brief_placeholder", lang)),
                    label=T["video_brief"], placeholder=T["video_brief_placeholder"],
                )
                with gr.Row():
                    structure = localizer.bind(
                        gr.Dropdown(choices=structure_choices(lang), value="free",
                                    label=pick("video_structure", lang), min_width=140),
                        label=T["video_structure"], choices=(structure_choices("ru"), structure_choices("en")),
                    )
                    total = localizer.bind(
                        gr.Slider(5, 120, value=20, step=1, label=pick("video_total", lang), min_width=140),
                        label=T["video_total"],
                    )
                chain = localizer.bind(
                    gr.Checkbox(value=True, label=pick("video_chain", lang), info=pick("video_chain_info", lang)),
                    label=T["video_chain"], info=T["video_chain_info"],
                )
                plan_button = localizer.bind(gr.Button(pick("video_plan", lang)), value=T["video_plan"])
                shots = localizer.bind(
                    gr.Dataframe(
                        headers=list(storyboard.COLUMNS), datatype=["number", "number", "str", "str", "str"],
                        column_count=(len(storyboard.COLUMNS), "fixed"), interactive=True, wrap=True,
                        label=pick("video_shots", lang),
                    ),
                    label=T["video_shots"],
                )
                board_button = localizer.bind(
                    gr.Button(pick("video_board_run", lang), variant="primary"), value=T["video_board_run"]
                )
                contact = localizer.bind(
                    gr.Image(label=pick("video_contact", lang), interactive=False, visible=False,
                             buttons=["fullscreen"]),
                    label=T["video_contact"],
                )

    # --- обработчики ------------------------------------------------------------

    def engine_state(lang):
        """Есть ли движок: без него — карточка и кнопка установки вместо молчаливого сбоя."""
        found = studio.video_status()
        if found.ready:
            return gr.update(visible=False), gr.update(visible=False)
        return gr.update(value=say("video_engine_missing", lang, reason=found.reason), visible=True), \
            gr.update(visible=True)

    def install(lang, progress=gr.Progress()):
        from ..video import engine

        lines: list[str] = []

        def on_line(line: str) -> None:
            lines.append(line)
            LOGGER.info("[FreeVideo setup] %s", line)
            progress(None, desc=line[-120:])

        try:
            engine.install(on_line)
        except Exception as error:  # noqa: BLE001 — сеть, диск, установка: строка состояния
            LOGGER.exception("FreeVideo setup failed")
            return gr.update(), gr.update(), say("video_install_failed", lang, error=str(error)[-300:])
        note, button = engine_state(lang)
        return note, button, say("video_install_done", lang)

    def update_size(mode_value, resolution_value, aspect_value, seconds_value, first_image, last_image,
                    quality_value, use_upscale, upscale_mode_value, factor, width, height, megapixels, lang,
                    source_value=None):
        upscale = upscale_from(use_upscale, upscale_mode_value, factor, width, height, megapixels)
        return describe_size(mode_value, resolution_value, aspect_value, seconds_value, first_image, last_image,
                             lang, upscale=upscale, quality=quality_value, source=source_value)

    def source_loaded(source_value, lang):
        """Исходник правки задаёт длительность: ползунок — на его длину, но не больше 15 с."""
        info = source_info(source_value)
        if info is None:
            return gr.update(), gr.update()
        seconds_value = edit_seconds(geometry.SINGLE_MAX_SECONDS, info.seconds)
        return seconds_value, say("video_source_loaded", lang, seconds=f"{info.seconds:.2f}",
                                  used=f"{geometry.duration(seconds_value).seconds:.2f}")

    def change_mode(mode_value):
        return visibility(mode_value)

    def request_from(prompt_text, mode_value, resolution_value, aspect_value, seconds_value, seed_value,
                     quality_value, first_image, last_image, files, lora_names, strength, source_value=None,
                     **changes):
        loras = lora_paths(lora_names, strength)
        values = dict(
            prompt=prompt_text or "",
            prompt_original=prompt_text or "",
            mode=mode_value,
            resolution=resolution_value,
            aspect=aspect_value,
            seconds=float(seconds_value),
            seed=int(seed_value if seed_value is not None else -1),
            quality=quality_value,
            first=first_image,
            last=last_image,
            references=references_from(files) if mode_value in prompt_h3.REF_MODES else (),
            source_video=video_refs._path(source_value) if mode_value == prompt_h3.MODE_V2V else None,
            loras=loras,
        )
        values.update(changes)
        return VideoRequest(**values)

    def reporter(progress, lang):
        def on_event(event):
            from ..video.engine import progress_view

            view = progress_view(event)
            if view is None:
                return
            label = view.label
            if view.step:
                label = say("video_progress_step", lang, label=label, done=view.step[0], total=view.step[1])
            if view.remaining_seconds:
                minutes, rest = divmod(int(view.remaining_seconds), 60)
                label = say("video_progress_left", lang, label=label, left=f"{minutes}:{rest:02d}")
            progress(view.fraction, desc=label)
        return on_event

    def finish_result(path, parameters, failure, message, lang):
        if failure is not None:
            return gr.update(), sentences(message, failure), gr.update(), gr.update()
        draft = parameters.get("quality") == "draft"
        line = say("video_done", lang, seed=parameters.get("seed"),
                   seconds=f"{float(parameters.get('seconds') or 0):.2f}",
                   # Размер файла: после апскейла он больше холста.
                   width=parameters.get("output_width") or parameters.get("width"),
                   height=parameters.get("output_height") or parameters.get("height"),
                   taken=parameters.get("seconds_taken"))
        if draft:
            line = sentences(line, say("video_draft_hint", lang))
        return str(path), sentences(message, line), str(path), gr.update(visible=draft)

    def boost_text(prompt_text, mode_value, seconds_value, aspect_value, first_image, last_image, files, lang,
                   source_value=None):
        """Промт H3 языковой моделью: ``(текст, сообщение)``."""
        references = references_from(files) if mode_value in prompt_h3.REF_MODES else ()
        duration = geometry.duration(float(seconds_value))
        if mode_value == prompt_h3.MODE_V2V:
            from ..video.request import edit_references

            info = source_info(source_value)
            references = edit_references(VideoRequest(prompt="", source_video=video_refs._path(source_value),
                                                      references=references))
            duration = geometry.duration(edit_seconds(float(seconds_value), info.seconds if info else None))
        return studio.boost_video(
            prompt_text, mode_value, duration, references,
            first_image if mode_value in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V) else None,
            last_image if mode_value == prompt_h3.MODE_FLF2V else None,
            aspect_value, lang,
        )

    def rewrite(prompt_text, mode_value, seconds_value, aspect_value, first_image, last_image, files, lang,
                source_value):
        if mode_value == prompt_h3.MODE_V2V and not video_refs._path(source_value):
            return gr.update(), gr.update(), say("video_source_required", lang)
        text, message = boost_text(prompt_text, mode_value, seconds_value, aspect_value, first_image,
                                   last_image, files, lang, source_value)
        return text or gr.update(), gr.update(value=bool(text)), message

    def run(prompt_text, boosted_text, use_boost, mode_value, resolution_value, aspect_value, seconds_value,
            seed_value, quality_value, first_image, last_image, files, lora_names, strength,
            use_upscale, upscale_mode_value, factor, width, height, megapixels, lang, source_value=None,
            progress=gr.Progress()):
        try:
            stop_requested.clear()
            message = ""
            if mode_value == prompt_h3.MODE_V2V and not video_refs._path(source_value):
                return gr.update(), say("video_source_required", lang), gr.update(), gr.update(), gr.update()
            effective = (boosted_text or "").strip() if use_boost else ""
            if use_boost and not effective:
                progress(0, desc=say("video_boosting", lang))
                effective, message = boost_text(prompt_text, mode_value, seconds_value, aspect_value,
                                                first_image, last_image, files, lang, source_value)
            elif not use_boost and prompt_text and prompt_h3.off_language(prompt_text):
                # H3 учили на английских описаниях (реплики — на любом языке).
                # Ролик всё равно снимется, но человек должен знать, почему
                # сцена может выйти не той.
                message = say("video_not_english", lang)
            if not (effective or (prompt_text or "").strip()):
                return gr.update(), say("video_prompt_required", lang), gr.update(), gr.update(), gr.update()
            request = request_from(effective or prompt_text, mode_value, resolution_value, aspect_value,
                                   seconds_value, seed_value, quality_value, first_image, last_image, files,
                                   lora_names, strength, source_value, prompt_original=prompt_text or "",
                                   upscale=upscale_from(use_upscale, upscale_mode_value, factor, width, height,
                                                        megapixels))
            progress(0, desc=say("video_starting", lang))
            path, parameters, failure = studio.run_video(request, lang, reporter(progress, lang))
            video, line, state, finish_button = finish_result(path, parameters, failure, message, lang)
            return video, line, state, finish_button, effective or gr.update()
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Video handler failed")
            return gr.update(), describe_failure(error, lang), gr.update(), gr.update(), gr.update()

    def follow_up(kind, current, quality_value, lang, progress=gr.Progress()):
        """«Довести до финала» и «Продлить ролик» — по параметрам последнего ролика."""
        if not current:
            return gr.update(), say("video_nothing", lang), gr.update(), gr.update()
        try:
            request = video_board.follow_up_request(current, kind, quality_value)
        except video_board.NoEngineOutput:
            return gr.update(), say("video_no_engine_output", lang), gr.update(), gr.update()
        progress(0, desc=say("video_starting", lang))
        path, produced, failure = studio.run_video(request, lang, reporter(progress, lang))
        return finish_result(path, produced, failure, "", lang)

    def finish_draft(current, quality_value, lang, progress=gr.Progress()):
        return follow_up(video_board.FINISH, current, quality_value, lang, progress)

    def extend_video(current, quality_value, lang, progress=gr.Progress()):
        return follow_up(video_board.EXTEND, current, quality_value, lang, progress)

    def next_shot(current, lang):
        """Последний кадр ролика — первым кадром следующего шота."""
        if not current:
            return gr.update(), gr.update(), say("video_nothing", lang)
        from ..video import assemble

        frame = assemble.last_frame(current)
        return frame, prompt_h3.MODE_I2V, say("video_next_ready", lang)

    def stop(lang):
        stop_requested.set()
        studio.stop_video()
        return say("stopping", lang)

    def plan_shots(brief_text, total_seconds, structure_value, aspect_value, lang):
        rows, message = studio.plan_storyboard(brief_text, float(total_seconds), structure_value,
                                               aspect_value, lang)
        return (rows if rows is not None else gr.update()), message

    def run_board(rows, chain_frames, resolution_value, aspect_value, seed_value, quality_value,
                  first_image, lora_names, strength, use_upscale, upscale_mode_value, factor, width, height,
                  megapixels, lang, progress=gr.Progress()):
        """Все шоты подряд, затем склейка с переходами и контактный лист (``video.board``)."""
        try:
            planned = storyboard.from_rows(rows)
        except ValueError as error:
            return gr.update(), say("video_invalid", lang, error=error), gr.update(), gr.update()
        if not planned:
            return gr.update(), say("video_board_empty", lang), gr.update(), gr.update()
        stop_requested.clear()

        def on_event(index, event):
            def shot_progress(fraction, desc=""):
                overall = None if fraction is None else (index + fraction) / len(planned)
                progress(overall, desc=say("video_board_progress", lang, index=index + 1,
                                           total=len(planned), label=desc))

            reporter(shot_progress, lang)(event)

        result = video_board.run_board(
            studio, planned, chain=bool(chain_frames), resolution=resolution_value, aspect=aspect_value,
            seed=int(seed_value if seed_value is not None else -1), quality=quality_value, first=first_image,
            loras=lora_paths(lora_names, strength), lang=lang,
            title=lambda count: say("video_board_title", lang, count=count),
            on_event=on_event, stopped=stop_requested.is_set,
            on_joining=lambda: progress(0.99, desc=say("video_board_joining", lang)),
            upscale=upscale_from(use_upscale, upscale_mode_value, factor, width, height, megapixels),
        )
        if result.failure is not None:
            return (str(result.produced[-1]) if result.produced else gr.update()), \
                say("video_board_failed", lang, index=result.failed_shot, error=result.failure), \
                gr.update(), gr.update()
        if result.path is None:
            return gr.update(), say("video_cancelled", lang), gr.update(), gr.update()
        message = say("video_board_done", lang, count=len(result.produced), seconds=f"{result.seconds:.2f}")
        return str(result.path), message, gr.update(value=str(result.sheet), visible=True), str(result.path)

    def save(name, prompt_text, mode_value, resolution_value, aspect_value, seconds_value, quality_value,
             seed_value, lang):
        if not (name or "").strip():
            return gr.update(), say("preset_name_required", lang)
        try:
            payload = library.load_prompt(name, config.PROMPT_DIR)
        except (FileNotFoundError, ValueError, OSError):
            payload = {}
        # Слияние, а не перезапись: под тем же именем могли сохранить
        # настройки генерации картинок.
        payload.update(prompt=prompt_text, video={
            "mode": mode_value, "resolution": resolution_value, "aspect": aspect_value,
            "seconds": float(seconds_value), "quality": quality_value, "seed": int(seed_value),
        })
        library.save_prompt(name, payload, config.PROMPT_DIR)
        return prompt_library.after_save(name, lang)

    def load(name, lang):
        keep = gr.update()
        payload, message = prompt_library.load(name, lang)
        if payload is None:
            return (keep,) * 7 + (message,)
        settings = payload.get("video") or {}
        fields = ("mode", "resolution", "aspect", "seconds", "quality", "seed")
        values = [settings[key] if key in settings else keep for key in fields]
        text = payload["prompt"] if "prompt" in payload else keep
        return (text, *values, message)

    # --- связи ------------------------------------------------------------------

    upscale_inputs = [upscale_enabled, upscale_mode, upscale_factor, upscale_width, upscale_height,
                      upscale_megapixels]
    size_inputs = [mode, resolution, aspect, seconds, first, last, quality, *upscale_inputs, language, source]
    # Исходник правки сначала ставит ползунок на свою длину, подпись — после.
    source.change(source_loaded, [source, language], [seconds, status], queue=False,
                  show_progress="hidden").then(update_size, size_inputs, size_note, queue=False,
                                               show_progress="hidden")
    for trigger in (mode.change, resolution.change, aspect.change, seconds.release, first.change, last.change,
                    quality.change, upscale_enabled.change, upscale_mode.change, upscale_factor.release,
                    upscale_width.change, upscale_height.change, upscale_megapixels.release):
        trigger(update_size, size_inputs, size_note, queue=False, show_progress="hidden")
    for trigger in (upscale_enabled.change, upscale_mode.change):
        trigger(upscale_visibility, [upscale_enabled, upscale_mode],
                [upscale_box, upscale_factor_box, upscale_dims, upscale_mp_box], queue=False,
                show_progress="hidden")
    mode.change(change_mode, mode, [first, last], queue=False, show_progress="hidden")
    video_refs.wire(tiles, mode, language, status)

    def show_rewritten(enabled, text):
        return gr.update(visible=bool(enabled) or bool((text or "").strip()))

    for trigger in (boost_enabled.change, boosted.change):
        trigger(show_rewritten, [boost_enabled, boosted], boosted, queue=False, show_progress="hidden")

    boost_inputs = [prompt, mode, seconds, aspect, first, last, references, language, source]
    boost_now.click(rewrite, boost_inputs, [boosted, boost_enabled, status])

    run_inputs = [prompt, boosted, boost_enabled, mode, resolution, aspect, seconds, seed, quality,
                  first, last, references, loras, lora_strength, *upscale_inputs, language, source]
    run_button.click(run, run_inputs, [result, status, last_result, finish, boosted],
                     concurrency_id=GPU_CONCURRENCY_ID)
    stop_button.click(stop, language, status, queue=False)
    finish.click(finish_draft, [last_result, quality, language], [result, status, last_result, finish],
                 concurrency_id=GPU_CONCURRENCY_ID)
    extend.click(extend_video, [last_result, quality, language], [result, status, last_result, finish],
                 concurrency_id=GPU_CONCURRENCY_ID)
    to_next.click(next_shot, [last_result, language], [first, mode, status])
    install_button.click(install, language, [engine_note, install_button, status],
                         concurrency_id=GPU_CONCURRENCY_ID)
    plan_button.click(plan_shots, [brief, total, structure, aspect, language], [shots, status])
    board_button.click(
        run_board,
        [shots, chain, resolution, aspect, seed, quality, first, loras, lora_strength, *upscale_inputs, language],
        [result, status, contact, last_result],
        concurrency_id=GPU_CONCURRENCY_ID,
    )

    prompts.save.click(save, [prompts.saved, prompt, mode, resolution, aspect, seconds, quality, seed, language],
                       [prompts.saved, status])
    prompts.saved.select(load, [prompts.saved, language],
                         [prompt, mode, resolution, aspect, seconds, quality, seed, status])
    prompts.saved.focus(prompt_library.refresh, None, prompts.saved, queue=False)
    prompts.delete.click(prompt_library.delete, [prompts.saved, language], [prompts.saved, status])

    return {
        "prompt": prompt,
        "boosted": boosted,
        "mode": mode,
        "resolution": resolution,
        "aspect": aspect,
        "seconds": seconds,
        "quality": quality,
        "seed": seed,
        "first": first,
        "last": last,
        "references": references,
        "reference_tiles": tiles.images,
        "source": source,
        "result": result,
        "status": status,
        "last_result": last_result,
        "engine_note": engine_note,
        "install_button": install_button,
        "engine_state": engine_state,
        "shots": shots,
        "prompts": prompts,
        "loras": loras,
        "upscale_enabled": upscale_enabled,
        "upscale_mode": upscale_mode,
        "upscale_factor": upscale_factor,
        "upscale_width": upscale_width,
        "upscale_height": upscale_height,
        "upscale_megapixels": upscale_megapixels,
    }


def send_to_video(produced, selected, mode_value, first_image, files, lang: str) -> tuple:
    """«В видео» с вкладок изображений: картинка — в нужный вход по режиму.

    Режимы с референсами — картинка ложится в первую свободную плитку
    картинки. Первый и последний кадр — в последний, если первый уже есть.
    Иначе — первым кадром, и режим «текст → видео» сам переключается на
    «кадр → видео»: человек отправил кадр, значит, хочет ролик от него.
    Выходы: режим, первый кадр, последний кадр, плитки картинок, строка
    состояния.
    """
    from PIL import Image

    keep = gr.update()
    tiles = [keep] * video_refs.MAX_IMAGES
    path = chosen_path(produced, selected)
    if path is None:
        return keep, keep, keep, *tiles, say("nothing_to_send", lang)
    if mode_value in prompt_h3.REF_MODES:
        index, _values = video_refs.place_image(files, path)
        if index is None:
            return keep, keep, keep, *tiles, say("video_refs_full", lang)
        # Плитка уже видна: пустая плитка картинки всегда одна на виду, а
        # занятые видны и так; её колонку откроет ``video_refs.refresh``.
        tiles[index] = gr.update(value=path)
        return keep, keep, keep, *tiles, say("video_sent_reference", lang)
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    # Видимость — в том же ответе: поле, скрытое в момент записи, значение не
    # рисует, а смена режима открыла бы его только следующим событием.
    if mode_value == prompt_h3.MODE_FLF2V and first_image is not None:
        return keep, keep, gr.update(value=image, visible=True), *tiles, say("video_sent_last", lang)
    target_mode = prompt_h3.MODE_I2V if mode_value == prompt_h3.MODE_T2V else mode_value
    return target_mode, gr.update(value=image, visible=True), keep, *tiles, say("video_sent_first", lang)


def open_video_tab():
    """Первый шаг отправки «В видео»: переход на вкладку (см. ``tab_generate.open_generate_tab``)."""
    return gr.Tabs(selected=layout.TAB_VIDEO)
