"""Вкладка генерации: промт, референсы, качество, результат.

Компоновка повторяет Fooocus: наверху результат, под ним строка промта и одна
кнопка, а всё остальное спрятано за переключателем «Продвинутое». Референсы
показаны лентой миниатюр, каждая подписана тегом, которым на неё ссылаются
из промта.
"""

from __future__ import annotations

import logging

import gradio as gr

from .. import config
from ..engine import presets
from ..engine.generator import GenerationRequest
from ..imaging import aspect, metadata
from ..prompting import boost as boost_module
from ..prompting import library
from ..storage import gallery
from .i18n import Localizer, pick
from .state import GPU_CONCURRENCY_ID, describe_failure

LOGGER = logging.getLogger(__name__)

MAX_REFERENCES = 10


def _captions(count: int) -> list[str]:
    """Подписи миниатюр: ровно те теги, которыми промт адресует изображения.

    При единственном референсе теги запрещены спецификацией Qwen, поэтому
    подпись говорит об этом прямо, а не показывает несуществующий «<image1>».
    """
    if count == 1:
        return ["без тега — одно изображение"]
    return [f"<image{index}>" for index in range(1, count + 1)]


def build(studio, localizer: Localizer) -> dict:
    lang = studio.config.lang
    catalogue_names = sorted(studio.catalogue)

    references = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            result = localizer.bind(
                gr.Gallery(
                    label=pick("result", lang),
                    show_label=True,
                    columns=2,
                    height=620,
                    object_fit="contain",
                    format="png",
                ),
                label=("Результат", "Result"),
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
                        gr.Button(pick("generate", lang), variant="primary"),
                        value=("Сгенерировать", "Generate"),
                    )
                    stop_button = localizer.bind(
                        gr.Button(pick("stop", lang), variant="stop"), value=("Прервать", "Stop")
                    )

            with gr.Row():
                boost_enabled = localizer.bind(
                    gr.Checkbox(label=pick("boost", lang), value=False, info=pick("boost_info", lang)),
                    label=("AI буст", "AI boost"),
                    info=(
                        "Переписать промт внешней языковой моделью перед генерацией.",
                        "Rewrite the prompt with an external language model before generating.",
                    ),
                )
                boost_now = localizer.bind(
                    gr.Button(pick("boost_now", lang)), value=("Переписать сейчас", "Rewrite now")
                )

            boosted = localizer.bind(
                gr.Textbox(label=pick("boost_result", lang), lines=3, interactive=True),
                label=("Переписанный промт", "Rewritten prompt"),
            )

            reference_gallery = localizer.bind(
                gr.Gallery(
                    label=pick("references", lang),
                    columns=5,
                    height=170,
                    object_fit="contain",
                    show_label=True,
                ),
                label=("Референсы", "References"),
            )
            with gr.Row():
                reference_upload = localizer.bind(
                    gr.File(
                        label=pick("reference_add", lang),
                        file_count="multiple",
                        file_types=["image"],
                    ),
                    label=("Добавить референсы", "Add references"),
                )
                reference_clear = localizer.bind(
                    gr.Button(pick("reference_clear", lang)),
                    value=("Очистить референсы", "Clear references"),
                )

        with gr.Column(scale=1):
            quality = localizer.bind(
                gr.Radio(
                    choices=list(presets.NAMES),
                    value=studio.config.preset,
                    label=pick("quality", lang),
                ),
                label=("Качество", "Quality"),
            )
            ratio = localizer.bind(
                gr.Dropdown(
                    choices=list(aspect.ASPECT_RATIOS),
                    value="1:1",
                    label=pick("aspect", lang),
                ),
                label=("Соотношение сторон", "Aspect ratio"),
            )
            image_number = localizer.bind(
                gr.Slider(1, 8, value=1, step=1, label=pick("image_number", lang)),
                label=("Количество изображений", "Image number"),
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
                styles = localizer.bind(
                    gr.Dropdown(
                        choices=catalogue_names,
                        value=[],
                        multiselect=True,
                        label=pick("styles", lang),
                    ),
                    label=("Стили", "Styles"),
                )
                negative = localizer.bind(
                    gr.Textbox(label=pick("negative", lang), lines=2),
                    label=("Негативный промт", "Negative prompt"),
                )
                cfg = localizer.bind(
                    gr.Slider(
                        1.0, 8.0, value=1.0, step=0.1,
                        label=pick("cfg", lang), info=pick("cfg_info", lang),
                    ),
                    label=("Сила guidance (true_cfg_scale)", "Guidance strength (true_cfg_scale)"),
                    info=(
                        "Модель рассчитана на работу без guidance. При значении 1.0 негативный "
                        "промт и негативные части стилей в модель не попадают вовсе.",
                        "The model is meant to be sampled without guidance. At 1.0 the negative "
                        "prompt and the negative half of every style are not sent to the model at all.",
                    ),
                )
                seed = localizer.bind(
                    gr.Number(value=-1, precision=0, label=pick("seed", lang), info=pick("seed_info", lang)),
                    label=("Сид", "Seed"),
                    info=("−1 — выбрать случайно", "−1 picks a random one"),
                )
                kv_cache = localizer.bind(
                    gr.Checkbox(
                        value=True, label=pick("kv_cache", lang), info=pick("kv_cache_info", lang)
                    ),
                    label=("Кэш ключей и значений", "KV cache"),
                    info=(
                        "Ускоряет генерацию. Переключение меняет результат при том же сиде.",
                        "Speeds generation up. Toggling it changes the result for the same seed.",
                    ),
                )

                preset_name = localizer.bind(
                    gr.Textbox(label=pick("preset_name", lang)),
                    label=("Название пресета", "Preset name"),
                )
                saved = localizer.bind(
                    gr.Dropdown(
                        choices=library.list_prompts(config.PROMPT_DIR),
                        label=pick("load_prompt", lang),
                    ),
                    label=("Загрузить промт", "Load prompt"),
                )
                with gr.Row():
                    save_button = localizer.bind(
                        gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
                    )
                    delete_button = localizer.bind(
                        gr.Button(pick("delete_prompt", lang)), value=("Удалить промт", "Delete prompt")
                    )

    # --- обработчики ---

    def add_references(files):
        """Пересобирает список референсов из виджета целиком.

        Виджет ``gr.File(file_count="multiple")`` в каждом событии ``change``
        отдаёт не добавленную дельту, а весь текущий набор выбранных файлов —
        так же ведёт себя и удаление файла крестиком внутри виджета. Если бы
        обработчик добавлял ``files`` поверх ранее накопленного состояния, то
        второе добавление референса задваивало бы первый (проверено вручную:
        два референса, добавленные по одному, давали три — сама Gradio уже
        включала первый файл во второй вызов).
        """
        from PIL import Image as PILImage

        images = [PILImage.open(item.name).convert("RGB") for item in (files or [])[:MAX_REFERENCES]]
        captioned = list(zip(images, _captions(len(images))))
        return images, captioned, f"Референсов: {len(images)} из {MAX_REFERENCES}"

    def clear_references():
        # Сбрасываем и сам виджет: иначе следующее добавление файла принесёт
        # с собой прежний набор, который виджет продолжает хранить внутри себя.
        return [], [], None, "Референсы очищены"

    def rewrite(prompt_text, current_references, ratio_value):
        mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
        text, wh_ratio, message = studio.boost_prompt(prompt_text, mode, current_references or None)
        chosen = wh_ratio if wh_ratio in aspect.ASPECT_RATIOS else ratio_value
        return text, chosen, message

    def run(
        prompt_text, boosted_text, use_boost, current_references, quality_name, ratio_value,
        count, style_names, negative_text, cfg_value, seed_value, kv_value,
        progress=gr.Progress(),
    ):
        # Обработчик целиком под try: отсутствующие веса, испорченный
        # model_index.json, нехватка видеопамяти и отказ записи PNG — всё это
        # обязано становиться строкой состояния, а не сырым трейсбеком в тосте.
        try:
            effective = (boosted_text or "").strip() if use_boost else ""
            message = ""
            if use_boost and not effective:
                mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
                effective, wh_ratio, message = studio.boost_prompt(
                    prompt_text, mode, current_references or None
                )
                if wh_ratio in aspect.ASPECT_RATIOS:
                    ratio_value = wh_ratio

            request = GenerationRequest(
                prompt=effective or prompt_text,
                prompt_original=prompt_text,
                preset=presets.get(quality_name),
                negative_prompt=negative_text or "",
                styles=tuple(style_names or ()),
                references=tuple(current_references or ()),
                aspect=ratio_value,
                seed=int(seed_value),
                image_number=int(count),
                true_cfg_scale=float(cfg_value),
                use_kv_cache=bool(kv_value),
            )

            def report(index: int, step: int, total: int) -> None:
                progress((step, total), desc=f"изображение {index + 1}/{int(count)}")

            produced, failure = studio.run_generation(request, progress=report)
            if failure is not None:
                return [], f"{message} {failure}".strip()
            if not produced:
                return [], f"{message} Генерация прервана".strip()

            paths = []
            for item in produced:
                destination = gallery.next_path(config.OUTPUT_DIR)
                metadata.save_png(item.image, destination, item.parameters)
                paths.append(str(destination))

            seeds = ", ".join(str(item.seed) for item in produced)
            report_line = f"Готово. Сиды: {seeds}. {studio.memory_report()}"
            return paths, f"{message} {report_line}".strip()
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Обработчик генерации не выполнен")
            return [], describe_failure(error)

    def stop():
        if studio.model_loaded:
            studio.generator.interrupt()
        return "Останавливаю…"

    def save(name, prompt_text, negative_text, style_names, quality_name, ratio_value, seed_value, cfg_value):
        if not (name or "").strip():
            return gr.update(), "Укажите название пресета"
        library.save_prompt(
            name,
            {
                "prompt": prompt_text,
                "negative_prompt": negative_text,
                "styles": list(style_names or ()),
                "preset": quality_name,
                "aspect": ratio_value,
                "seed": int(seed_value),
                "true_cfg_scale": float(cfg_value),
            },
            config.PROMPT_DIR,
        )
        return gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=name), f"Пресет «{name}» сохранён"

    def load(name):
        if not name:
            return (gr.update(),) * 7 + ("Пресет не выбран",)
        # Пресет мог быть удалён или испорчен мимо приложения: файлы лежат в
        # user/prompts/ и правятся чем угодно. load_prompt() сообщает об этом
        # исключением (FileNotFoundError либо ValueError), и выбор строки в
        # выпадающем списке не должен превращаться в тост с трейсбеком. То же
        # снисхождение, что и у tab_gallery.restore_fields к чужому PNG.
        try:
            payload = library.load_prompt(name, config.PROMPT_DIR)
        except (FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Пресет «%s» не загружен: %s", name, error)
            return (gr.update(),) * 7 + (f"Пресет «{name}» не загружен: {error}",)
        return (
            payload.get("prompt", ""),
            payload.get("negative_prompt", ""),
            payload.get("styles", []),
            payload.get("preset", presets.DEFAULT),
            payload.get("aspect", "1:1"),
            payload.get("seed", -1),
            payload.get("true_cfg_scale", 1.0),
            f"Пресет «{name}» загружен",
        )

    def delete(name):
        try:
            removed = library.delete_prompt(name, config.PROMPT_DIR)
        except OSError as error:
            # Файл может быть открыт другой программой или лежать на томе,
            # доступном только на чтение: сказать об этом строкой состояния.
            LOGGER.warning("Пресет «%s» не удалён: %s", name, error)
            return gr.update(), f"Пресет «{name}» не удалён: {error}"
        message = f"Пресет «{name}» удалён" if removed else "Пресет не найден"
        return gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=None), message

    reference_upload.change(
        add_references, reference_upload, [references, reference_gallery, status]
    )
    reference_clear.click(
        clear_references, None, [references, reference_gallery, reference_upload, status]
    )
    boost_now.click(rewrite, [prompt, references, ratio], [boosted, ratio, status])

    run_button.click(
        run,
        [prompt, boosted, boost_enabled, references, quality, ratio, image_number,
         styles, negative, cfg, seed, kv_cache],
        [result, status],
        # Общая с вкладкой редактирования группа очереди: без неё предел
        # concurrency в единицу действовал бы только внутри этого обработчика,
        # а «Сгенерировать» и «Применить правку» стартовали бы одновременно.
        concurrency_id=GPU_CONCURRENCY_ID,
    )
    # Кнопка остановки должна срабатывать, пока генерация занимает очередь.
    stop_button.click(stop, None, status, queue=False)

    save_button.click(
        save, [preset_name, prompt, negative, styles, quality, ratio, seed, cfg], [saved, status]
    )
    saved.change(load, saved, [prompt, negative, styles, quality, ratio, seed, cfg, status])
    delete_button.click(delete, saved, [saved, status])

    return {
        "prompt": prompt,
        "boosted": boosted,
        "negative": negative,
        "styles": styles,
        "quality": quality,
        "ratio": ratio,
        "seed": seed,
        "cfg": cfg,
        "status": status,
        "result": result,
        "references": references,
        "reference_gallery": reference_gallery,
        "advanced": advanced,
    }
