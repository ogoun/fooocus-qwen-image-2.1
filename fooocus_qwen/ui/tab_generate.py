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
from ..engine.generator import (
    GenerationRequest,
    condition_slots,
    resolve_reference_scale,
)
from ..imaging import aspect, metadata
from ..prompting import boost as boost_module
from ..prompting import library
from ..storage import gallery
from . import layout
from .i18n import Localizer, pick, say
from .state import GPU_CONCURRENCY_ID, describe_failure

LOGGER = logging.getLogger(__name__)

MAX_REFERENCES = 10


def _ratio_choices(lang: str) -> list[tuple[str, str]]:
    """Пары (подпись, значение) для выпадающего списка соотношений.

    Семь числовых соотношений ("16:9" и т.п.) — не текст интерфейса, а
    идентификаторы: они пишутся в пресеты промтов и в метаданные PNG как
    есть, поэтому подписью служит само значение. Единственное исключение —
    ``aspect.FOLLOW_REFERENCE``: это предложение на языке интерфейса
    («от референса»), и без перевода английская сборка вкладки заканчивала
    список русской строкой (тот же класс дефекта, что уже дважды правился —
    у радиокнопки режима области и у селектора системного промта). Значение
    сентинела при этом не меняется: ``engine.generator.resolve_size`` сравнивает
    с ним, а PNG прежних генераций хранят его как есть.
    """
    return [
        (pick("aspect_follow_reference", lang) if value == aspect.FOLLOW_REFERENCE else value, value)
        for value in aspect.ASPECT_RATIOS
    ]


def _captions(images, lang: str) -> list[str]:
    """Подписи миниатюр: ровно те теги, которыми промт адресует изображения.

    Считаются из ``condition_slots()`` — той же функции, что задаёт порядок
    условных изображений для самой модели. Своя формула «i-й референс —
    ``<imageI>``» здесь была бы вторым описанием того же правила и разошлась
    бы с первым в тот день, когда рядом с референсами появится исходное
    изображение: тогда ``<image1>`` принадлежит ему, а первому референсу
    достаётся ``<image2>``.

    При единственном условном изображении теги запрещены спецификацией Qwen,
    и ``condition_slots()`` отдаёт пустой тег; подпись говорит об этом прямо
    и на языке интерфейса, а не показывает несуществующий «<image1>».
    """
    return [
        slot.tag or say("caption_untagged", lang)
        for slot in condition_slots(references=tuple(images))
        if slot.role == "reference"
    ]


def build(studio, localizer: Localizer, language=None) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком (выпадающий список из шапки).
    Он приходит извне и добавляется последним входом к каждому обработчику,
    который что-то сообщает пользователю: язык меняется во время работы, и
    захваченный в замыкание ``lang`` навсегда остался бы языком запуска.
    Если компонент не передан (так вкладку собирают тесты), заводится
    ``gr.State`` с языком запуска — обработчики от этого не меняются.
    """
    lang = studio.config.lang
    catalogue_names = sorted(studio.catalogue)
    if language is None:
        language = gr.State(lang)

    references = gr.State([])
    # Исходный текст, по которому составлен переписанный промт. Нужен, чтобы
    # заметить, что промт с тех пор правили, а в модель по-прежнему уйдёт
    # старая переписка.
    boost_source = gr.State("")

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.CANVAS_COL]):
            result = localizer.bind(
                gr.Gallery(
                    label=pick("result", lang),
                    show_label=True,
                    columns=2,
                    height=layout.CANVAS_HEIGHT,
                    elem_classes=[layout.BOARD],
                    object_fit="contain",
                    format="png",
                    # Крупный просмотр с лентой миниатюр под ним. Без него
                    # сетка в две колонки отдавала единственному
                    # изображению — а это значение по умолчанию — половину
                    # ширины холста, и вторая половина стояла пустой.
                    preview=True,
                ),
                label=("Результат", "Result"),
            )

            with gr.Row(elem_classes=[layout.PROMPT_BAR]):
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

            # Референсы обязаны быть видны в главном окне вместе с тегами,
            # которыми на них ссылаются из промта, — это требование
            # заказчика. Но пустая лента с областью загрузки отнимала треть
            # высоты у результата в каждой генерации, а референсы нужны
            # далеко не в каждой. Поэтому секция закрыта, пока референсов
            # нет, и раскрывается сама, как только первый добавлен: видно
            # ровно тогда, когда есть что показывать.
            reference_box = localizer.bind(
                gr.Accordion(pick("references", lang), open=False),
                label=("Референсы", "References"),
            )
            with reference_box:
                reference_gallery = localizer.bind(
                    gr.Gallery(
                        label=pick("references", lang),
                        columns=5,
                        height=layout.STRIP_HEIGHT,
                        object_fit="contain",
                        show_label=False,
                    ),
                    label=("Референсы", "References"),
                )
                with gr.Row():
                    reference_upload = localizer.bind(
                        gr.File(
                            label=pick("reference_add", lang),
                            file_count="multiple",
                            file_types=["image"],
                            elem_classes=[layout.DROP_ZONE],
                        ),
                        label=("Добавить референсы", "Add references"),
                    )
                    reference_clear = localizer.bind(
                        gr.Button(pick("reference_clear", lang)),
                        value=("Очистить референсы", "Clear references"),
                    )

        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.SIDE_COL]):
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
                    choices=_ratio_choices(lang),
                    value="1:1",
                    label=pick("aspect", lang),
                ),
                label=("Соотношение сторон", "Aspect ratio"),
                # Подписи вариантов — переводимый текст, а не служебные
                # идентификаторы (см. докстринг _ratio_choices), поэтому, как и
                # у mask_mode на вкладке редактирования, переводится и label, и
                # choices.
                choices=(_ratio_choices("ru"), _ratio_choices("en")),
            )
            image_number = localizer.bind(
                gr.Slider(1, 8, value=1, step=1, label=pick("image_number", lang)),
                label=("Количество изображений", "Image number"),
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
                # Выпадающий список, а не ползунок: осмысленных значений
                # немного, они привязаны к измеренным режимам
                # (docs/BENCHMARK.md), и промежуточные числа обещали бы
                # плавность, которой здесь нет.
                reference_scale = localizer.bind(
                    gr.Dropdown(
                        choices=[(pick("reference_scale_auto", lang), 0),
                                 ("512", 512), ("768", 768), ("1024", 1024),
                                 ("1536", 1536), ("2048", 2048)],
                        value=0,
                        label=pick("reference_scale", lang),
                        info=pick("reference_scale_info", lang),
                    ),
                    label=("Детальность референсов", "Reference detail"),
                    info=(
                        "Разрешение, к которому приводятся референсы и исходник правки. "
                        "Размер кадра от этого не меняется. «Авто» урезает масштаб при "
                        "двух и более референсах: на полном пять референсов считаются "
                        "тринадцать минут, на 512 — сорок секунд.",
                        "The resolution condition images are scaled to. It does not change "
                        "the frame size. “Auto” reduces the scale from two "
                        "references up: at full scale five references take thirteen minutes "
                        "per frame, at 512 — forty seconds.",
                    ),
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

    def add_references(files, lang):
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
        captioned = list(zip(images, _captions(images, lang)))
        # Секция раскрывается сама, когда есть что показать: подписи с
        # тегами `<imageN>` бесполезны, если до них надо доклацаться.
        return (
            images,
            captioned,
            gr.update(open=bool(images)),
            say("references_counted", lang, count=len(images), total=MAX_REFERENCES),
        )

    def clear_references(lang):
        # Сбрасываем и сам виджет: иначе следующее добавление файла принесёт
        # с собой прежний набор, который виджет продолжает хранить внутри себя.
        # Секция закрывается: показывать в ней больше нечего.
        return [], [], None, gr.update(open=False), say("references_cleared", lang)

    def rewrite(prompt_text, current_references, ratio_value, lang):
        """Переписывает промт и сразу включает буст.

        Галочка включается потому, что нажатие этой кнопки и есть заявление
        «хочу переписанный промт». Без этого кнопка работала независимо от
        галочки, а в модель переписанный текст уходит только при включённой:
        нажать, увидеть текст и получить картинку по старому промту было
        проще простого, и ничто об этом не сообщало.
        """
        mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
        text, wh_ratio, message = studio.boost_prompt(
            prompt_text, mode, lang, current_references or None
        )
        chosen = wh_ratio if wh_ratio in aspect.ASPECT_RATIOS else ratio_value
        # Источник запоминается только при удачной переписке: иначе отказ
        # сервера пометил бы прежний текст как свежий.
        source = prompt_text if text else gr.update()
        return text, chosen, gr.update(value=bool(text)), source, message

    def run(
        prompt_text, boosted_text, boost_source_text, use_boost, current_references,
        quality_name, ratio_value, count, style_names, negative_text, cfg_value, seed_value,
        kv_value, scale_value, lang,
        progress=gr.Progress(),
    ):
        # Обработчик целиком под try: отсутствующие веса, испорченный
        # model_index.json, нехватка видеопамяти и отказ записи PNG — всё это
        # обязано становиться строкой состояния, а не сырым трейсбеком в тосте.
        try:
            effective = (boosted_text or "").strip() if use_boost else ""
            message = ""

            # Оба предупреждения — про молчаливое расхождение между тем, что
            # пользователь видит в поле, и тем, что уходит в модель.
            if not use_boost and (boosted_text or "").strip():
                message = say("boost_ignored", lang)
            elif effective and (boost_source_text or "") != (prompt_text or ""):
                message = say("boost_stale", lang)

            if use_boost and not effective:
                mode = boost_module.MODE_EDIT if current_references else boost_module.MODE_T2I
                effective, wh_ratio, message = studio.boost_prompt(
                    prompt_text, mode, lang, current_references or None
                )  # сюда попадаем только при пустом поле, предупреждать не о чем
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
                reference_scale=int(scale_value or 0),
            )

            # Автоматический выбор масштаба обязан быть виден: пользователь,
            # подавший пять референсов, получит их в 512, и молчать об этом —
            # значит оставить необъяснимую потерю детальности.
            chosen = resolve_reference_scale(request)
            if not scale_value and chosen != request.preset.output_resolution:
                message = f"{message} {say('reference_scale_chosen', lang, scale=chosen)}".strip()

            def report(index: int, step: int, total: int) -> None:
                progress(
                    (step, total),
                    desc=say("progress_image", lang, index=index + 1, total=int(count)),
                )

            # Стадия до первого шага: прогресс из пайплайна приходит только
            # после шага, а загрузка модели и кодирование промта идут раньше
            # и молча. Отличить работу от зависания пользователь не мог.
            progress(0, desc=say(
                "stage_loading" if not studio.model_loaded else "stage_preparing", lang
            ))
            produced, failure = studio.run_generation(request, lang, progress=report)
            if failure is not None:
                return [], f"{message} {failure}".strip()
            if not produced:
                return [], f"{message} {say('generation_interrupted', lang)}".strip()

            paths = []
            for item in produced:
                destination = gallery.next_path(config.OUTPUT_DIR)
                metadata.save_png(item.image, destination, item.parameters)
                paths.append(str(destination))

            seeds = ", ".join(str(item.seed) for item in produced)
            report_line = say(
                "generation_done", lang, seeds=seeds, memory=studio.memory_report(lang)
            )
            return paths, f"{message} {report_line}".strip()
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Обработчик генерации не выполнен")
            return [], describe_failure(error, lang)

    def stop(lang):
        if studio.model_loaded:
            studio.generator.interrupt()
        return say("stopping", lang)

    def save(
        name, prompt_text, negative_text, style_names, quality_name, ratio_value,
        seed_value, cfg_value, lang,
    ):
        if not (name or "").strip():
            return gr.update(), say("preset_name_required", lang)
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
        return (
            gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=name),
            say("preset_saved", lang, name=name),
        )

    def load(name, lang):
        if not name:
            return (gr.update(),) * 7 + (say("preset_not_selected", lang),)
        # Пресет мог быть удалён или испорчен мимо приложения: файлы лежат в
        # user/prompts/ и правятся чем угодно. load_prompt() сообщает об этом
        # исключением (FileNotFoundError либо ValueError), и выбор строки в
        # выпадающем списке не должен превращаться в тост с трейсбеком. То же
        # снисхождение, что и у tab_gallery.restore_fields к чужому PNG.
        try:
            payload = library.load_prompt(name, config.PROMPT_DIR)
        except (FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Пресет «%s» не загружен: %s", name, error)
            return (gr.update(),) * 7 + (say("preset_load_failed", lang, name=name, error=error),)
        return (
            payload.get("prompt", ""),
            payload.get("negative_prompt", ""),
            payload.get("styles", []),
            payload.get("preset", presets.DEFAULT),
            payload.get("aspect", "1:1"),
            payload.get("seed", -1),
            payload.get("true_cfg_scale", 1.0),
            say("preset_loaded", lang, name=name),
        )

    def delete(name, lang):
        try:
            removed = library.delete_prompt(name, config.PROMPT_DIR)
        except OSError as error:
            # Файл может быть открыт другой программой или лежать на томе,
            # доступном только на чтение: сказать об этом строкой состояния.
            LOGGER.warning("Пресет «%s» не удалён: %s", name, error)
            return gr.update(), say("preset_delete_failed", lang, name=name, error=error)
        message = (
            say("preset_deleted", lang, name=name) if removed else say("preset_missing", lang)
        )
        return gr.update(choices=library.list_prompts(config.PROMPT_DIR), value=None), message

    reference_upload.change(
        add_references,
        [reference_upload, language],
        [references, reference_gallery, reference_box, status],
    )
    reference_clear.click(
        clear_references,
        language,
        [references, reference_gallery, reference_upload, reference_box, status],
    )
    boost_now.click(
        rewrite,
        [prompt, references, ratio, language],
        [boosted, ratio, boost_enabled, boost_source, status],
    )

    run_button.click(
        run,
        [prompt, boosted, boost_source, boost_enabled, references, quality, ratio,
         image_number, styles, negative, cfg, seed, kv_cache, reference_scale, language],
        [result, status],
        # Общая с вкладкой редактирования группа очереди: без неё предел
        # concurrency в единицу действовал бы только внутри этого обработчика,
        # а «Сгенерировать» и «Применить правку» стартовали бы одновременно.
        concurrency_id=GPU_CONCURRENCY_ID,
    )
    # Кнопка остановки должна срабатывать, пока генерация занимает очередь.
    stop_button.click(stop, language, status, queue=False)

    save_button.click(
        save,
        [preset_name, prompt, negative, styles, quality, ratio, seed, cfg, language],
        [saved, status],
    )
    saved.change(load, [saved, language], [prompt, negative, styles, quality, ratio, seed, cfg, status])
    delete_button.click(delete, [saved, language], [saved, status])

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
        "reference_box": reference_box,
        "advanced": advanced,
    }
