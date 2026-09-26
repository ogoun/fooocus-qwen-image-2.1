"""Вкладка генерации: референсы, результат, промт, качество.

Компоновка повторяет Fooocus: наверху результат, под ним строка промта и одна
кнопка, а всё остальное спрятано за переключателем «Продвинутое». Референсы —
сетка из десяти слотов слева от результата, два ряда по пять: картинку кладут
в слот кликом или перетаскиванием, и заполненный слот подписан тегом, которым
на него ссылаются из промта.
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
from .i18n import Localizer, pick, say, sentences
from .state import GPU_CONCURRENCY_ID, describe_failure
from .tab_edit import chosen_path, remember_selection

LOGGER = logging.getLogger(__name__)

# Сетка референсов слева от результата: два ряда по пять. Десять — предел
# самой модели (карточка Qwen-Image-2.1), и сетка ровно его покрывает.
REFERENCE_ROWS = 2
REFERENCE_COLUMNS = 5
MAX_REFERENCES = REFERENCE_ROWS * REFERENCE_COLUMNS


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


def _grid(references) -> list:
    """Список ровно из ``MAX_REFERENCES`` позиций: картинка или ``None``.

    Состояние вкладки поначалу пустой список — слоты ещё никто не трогал, —
    а обработчикам удобнее всегда видеть все десять позиций.
    """
    grid = list(references or [])[:MAX_REFERENCES]
    return grid + [None] * (MAX_REFERENCES - len(grid))


def filled(references) -> list:
    """Заполненные слоты по порядку — ровно то, что уходит в модель.

    Слоты позиционные и бывают с дырами: человек мог положить картинки в
    первый, третий и седьмой. Модели дыры не нужны и не передаются.
    """
    return [image for image in (references or []) if image is not None]


def slot_labels(references, lang: str) -> list:
    """Подписи всех слотов: тег у заполненных, ничего у пустых.

    Тег считается по порядку **заполненных** слотов, а не по позиции слота:
    если заполнены первый, третий и седьмой, они — ``<image1>``,
    ``<image2>`` и ``<image3>``. Подпись по позиции («<image7>») ссылалась бы
    на изображение, которого у модели нет, — пустые слоты в неё не уходят.
    Сами теги берутся из ``_captions``, то есть из той же функции, что строит
    вход модели.

    Обновляется только подпись, картинка не трогается: иначе каждый ответ
    заново слал бы изображение в браузер.
    """
    grid = _grid(references)
    captions = iter(_captions(filled(grid), lang))
    return [
        gr.update(label=next(captions), show_label=True) if image is not None
        else gr.update(label="", show_label=False)
        for image in grid
    ]


def send_to_references(produced, selected, references, lang: str) -> tuple:
    """«Отправить в референсы»: выбранная картинка результата — в первый свободный слот.

    Выходы: состояние референсов, десять слотов, строка состояния вкладки
    генерации. Кнопка есть и на правке (связывается в app.py) — тогда к
    выходам добавляется переход на эту вкладку.
    """
    from PIL import Image as PILImage

    unchanged = (gr.update(),) * (1 + MAX_REFERENCES)
    path = chosen_path(produced, selected)
    if path is None:
        return (*unchanged, say("nothing_to_send", lang))

    grid = _grid(references)
    free = next((index for index, image in enumerate(grid) if image is None), None)
    if free is None:
        return (*unchanged, say("references_full", lang, total=MAX_REFERENCES))

    with PILImage.open(path) as opened:
        grid[free] = opened.convert("RGB")

    slots = slot_labels(grid, lang)
    slots[free] = gr.update(value=grid[free], **{k: v for k, v in slots[free].items() if k != "__type__"})
    count = len(filled(grid))
    return (grid, *slots, say("reference_sent", lang, count=count, total=MAX_REFERENCES))


def open_generate_tab():
    """Переход на вкладку генерации — первый шаг отправки в референсы с правки.

    Переход и сама отправка — два события подряд, а не одно, и это не
    стилистика. В Gradio 6.5.1 новое значение поля изображения, пришедшее в
    том же ответе, что и переключение на его вкладку, не отрисовывается:
    подпись слота менялась на ``<image2>``, а картинки в нём не было —
    вместо неё оставался «+» пустой ячейки (найдено браузерной проверкой
    ``tools/ui_check.py``). Когда вкладка уже открыта, значение ложится как
    положено — так же, как при отправке прямо с генерации.

    Переход делается всегда, в том числе при отказе («нечего отправлять»,
    «референсов уже десять»): сообщение пишется в строку состояния
    генерации, и без перехода человек его бы не увидел.
    """
    return gr.Tabs(selected=layout.TAB_GENERATE)


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

    # Референсы — по позициям слотов, с дырами: картинка или None на каждый
    # из десяти. В модель уходят только заполненные (``filled``). Поначалу
    # пустой список: слоты ещё никто не трогал.
    references = gr.State([])
    # Исходный текст, по которому составлен переписанный промт. Нужен, чтобы
    # заметить, что промт с тех пор правили, а в модель по-прежнему уйдёт
    # старая переписка.
    boost_source = gr.State("")

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        # Сетка референсов — слева от результата, два ряда по пять слотов.
        # Каждый слот — отдельное поле изображения: картинку кладут кликом или
        # перетаскиванием, убирают его собственным крестиком. Слот принимает
        # только загрузку: веб-камера и буфер обмена в ячейке в пару
        # сантиметров добавили бы панель переключения источников крупнее
        # самой ячейки.
        with gr.Column(min_width=0, elem_classes=[layout.REFS_COL]):
            localizer.bind(
                gr.Markdown(pick("references_hint", lang)),
                value=(
                    "**Референсы** — нажмите на ячейку или перетащите в неё картинку",
                    "**References** — click a cell or drop an image onto it",
                ),
            )
            reference_slots: list[gr.Image] = []
            for _row in range(REFERENCE_ROWS):
                with gr.Row(equal_height=True):
                    for _column in range(REFERENCE_COLUMNS):
                        reference_slots.append(
                            gr.Image(
                                type="pil",
                                image_mode="RGB",
                                sources=["upload"],
                                label="",
                                show_label=False,
                                buttons=[],
                                placeholder="# +",
                                min_width=0,
                                scale=1,
                                elem_classes=[layout.REF_SLOT],
                            )
                        )
            reference_clear = localizer.bind(
                gr.Button(pick("reference_clear", lang), size="sm"),
                value=("Очистить референсы", "Clear references"),
            )

        with gr.Column(min_width=layout.CANVAS_MIN_WIDTH, elem_classes=[layout.CANVAS_COL]):
            result = localizer.bind(
                gr.Gallery(
                    label=pick("result", lang),
                    show_label=True,
                    columns=2,
                    elem_classes=[layout.BOARD],
                    object_fit="contain",
                    format="png",
                    # Только для чтения: поле результата — выход, и зазывать
                    # «перетащить файл сюда» ему незачем.
                    interactive=False,
                    # Крупный просмотр с лентой миниатюр под ним. Без него
                    # сетка в две колонки отдавала единственному
                    # изображению — а это значение по умолчанию — половину
                    # ширины холста, и вторая половина стояла пустой.
                    preview=True,
                    buttons=layout.GALLERY_BUTTONS,
                ),
                label=("Результат", "Result"),
            )
            # Результат — в кисть правки, с переходом на её вкладку. Связывается
            # в app.py: вкладка правки собирается после этой.
            with gr.Row():
                send_to_edit = localizer.bind(
                    gr.Button(pick("send_to_edit", lang)),
                    value=("Отправить в редактор", "Send to editor"),
                )
                send_to_refs = localizer.bind(
                    gr.Button(pick("send_to_references", lang)),
                    value=("Отправить в референсы", "Send to references"),
                )
            selected = gr.State(None)

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
                    label=("Имя для сохранения", "Name to save as"),
                )
                saved = localizer.bind(
                    gr.Dropdown(
                        choices=library.list_prompts(config.PROMPT_DIR),
                        label=pick("load_prompt", lang),
                    ),
                    label=("Сохранённые промты", "Saved prompts"),
                )
                with gr.Row():
                    save_button = localizer.bind(
                        gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
                    )
                    delete_button = localizer.bind(
                        gr.Button(pick("delete_prompt", lang)), value=("Удалить промт", "Delete prompt")
                    )

    # --- обработчики ---

    def slots_changed(*values):
        """Пересобирает референсы из всех десяти слотов разом.

        Источник истины — то, что сейчас стоит в слотах, а не история
        изменений: слот меняют в любом порядке и чистят крестиком, и
        дописывать к накопленному списку значило бы рано или поздно
        разойтись с экраном. Вызывается событием ``input``, то есть только
        на действие человека: программная запись в слот («Отправить в
        референсы») его не порождает, и ответ обработчика не зацикливается.
        """
        *images, lang = values
        grid = _grid(images)
        return (
            grid,
            *slot_labels(grid, lang),
            say("references_counted", lang, count=len(filled(grid)), total=MAX_REFERENCES),
        )

    def clear_references(lang):
        grid = _grid([])
        return (
            grid,
            *(gr.update(value=None, label="", show_label=False) for _ in grid),
            say("references_cleared", lang),
        )

    def rewrite(prompt_text, current_references, ratio_value, lang):
        """Переписывает промт и сразу включает буст.

        Галочка включается потому, что нажатие этой кнопки и есть заявление
        «хочу переписанный промт». Без этого кнопка работала независимо от
        галочки, а в модель переписанный текст уходит только при включённой:
        нажать, увидеть текст и получить картинку по старому промту было
        проще простого, и ничто об этом не сообщало.
        """
        images = filled(current_references)
        mode = boost_module.MODE_EDIT if images else boost_module.MODE_T2I
        text, wh_ratio, message = studio.boost_prompt(prompt_text, mode, lang, images or None)
        chosen = wh_ratio if wh_ratio in aspect.ASPECT_RATIOS else ratio_value
        # Источник запоминается только при удачной переписке: иначе отказ
        # сервера пометил бы прежний текст как свежий.
        source = prompt_text if text else gr.update()
        return text, chosen, gr.update(value=bool(text)), source, message

    def run(
        prompt_text, boosted_text, boost_source_text, use_boost, current_references,
        quality_name, ratio_value, count, style_names, negative_text, cfg_value, seed_value,
        kv_value, scale_value, lang,
        progress=gr.Progress(track_tqdm=True),
    ):
        # Обработчик целиком под try: отсутствующие веса, испорченный
        # model_index.json, нехватка видеопамяти и отказ записи PNG — всё это
        # обязано становиться строкой состояния, а не сырым трейсбеком в тосте.
        try:
            # Пустые слоты в модель не уходят; порядок заполненных сохраняется.
            current_references = filled(current_references)
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
                message = sentences(message, say("reference_scale_chosen", lang, scale=chosen))

            def report(index: int, step: int, total: int) -> None:
                progress(
                    (step, total),
                    desc=say("progress_image", lang, index=index + 1, total=int(count)),
                )

            # Пресету Turbo нужен адаптер; при первом выборе он качается здесь.
            missing = studio.weights_for(request.preset, lang, progress)
            if missing:
                return [], sentences(message, missing)

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
                return [], sentences(message, say("generation_interrupted", lang))

            paths = []
            for item in produced:
                destination = gallery.next_path(config.OUTPUT_DIR)
                metadata.save_png(item.image, destination, item.parameters)
                paths.append(str(destination))

            seeds = ", ".join(str(item.seed) for item in produced)
            report_line = say(
                "generation_done", lang, seeds=seeds, memory=studio.memory_report(lang)
            )
            # Результат открывается крупно, а не сеткой. ``preview=True`` у галереи
            # действует только при первой загрузке страницы: при новом значении Gradio
            # возвращается к сетке, и единственная картинка становилась квадратной
            # миниатюрой выше окна — видна была средняя полоса кадра (найдено на снимках
            # для README). ``selected_index=0`` открывает первую картинку в просмотре.
            return gr.Gallery(value=paths, selected_index=0), sentences(message, report_line)
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

    # Все слоты — один обработчик на событие каждого: список собирается из
    # десяти значений разом (см. ``slots_changed``).
    #
    # Индикатор выполнения скрыт: все десять слотов — выходы обработчика, и
    # на время запроса Gradio накрывает свои выходы индикатором — десять
    # крутилок ради одной ячейки. Итог и так виден в подписях и в строке
    # состояния. Замечание для проверяющего: текст «0.0s» в ``innerText``
    # ячейки есть всегда — индикатор лежит в разметке с нулевой
    # прозрачностью, — и видимым таймером это не является.
    reference_targets = [references, *reference_slots, status]
    for slot in reference_slots:
        slot.input(
            slots_changed, [*reference_slots, language], reference_targets,
            queue=False, show_progress="hidden",
        )
    reference_clear.click(
        clear_references, language, reference_targets, queue=False, show_progress="hidden",
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

    result.select(remember_selection, result, selected, queue=False)
    send_to_refs.click(
        send_to_references,
        [result, selected, references, language],
        reference_targets,
        show_progress="hidden",
    )

    return {
        "reference_targets": reference_targets,
        "reference_slots": reference_slots,
        "send_to_edit": send_to_edit,
        "selected": selected,
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
        "advanced": advanced,
    }
