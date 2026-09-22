"""Двуязычные подписи и сообщения интерфейса.

Здесь две таблицы, и разница между ними существенная.

``T`` — **подписи компонентов**: label, placeholder, info, choices, текст
кнопок. Они задаются в Gradio при сборке, поэтому смена языка — это массовое
обновление уже созданных компонентов. ``Localizer`` запоминает, какие поля
какого компонента переводимы, и по запросу отдаёт список обновлений в том же
порядке, в каком компоненты регистрировались.

``MESSAGES`` — **сообщения, которые пользователь читает по ходу работы**:
строка состояния, подписи миниатюр, текст прогресса, сообщения об ошибках.
Их нельзя разложить по компонентам заранее, потому что они возникают внутри
обработчика и часто содержат подставляемые значения. Поэтому они не
привязываются к компонентам, а собираются в момент ответа функцией ``say()``,
которой текущий язык передаётся аргументом. Именно аргументом, а не
захваченной при сборке переменной: язык меняется во время работы, и
обработчик, захвативший ``lang`` в замыкание, навсегда остался бы на языке
запуска — это и был исходный дефект, из-за которого вокруг английских
подписей появлялось «Референсов: 3 из 10».

Граница перевода проведена так: интерфейс переводит то, что пишет сам, и
цитирует техническую причину как есть. Тексты исключений приходят из torch,
diffusers, urllib и операционной системы и всегда английские; наши
собственные исключения (``llm/``, ``prompting/``) написаны по-русски.
Переводить и их значило бы затащить эту таблицу в слои, которые о
существовании интерфейса ничего не знают, — ровно ту границу, ради которой
они и выделены (см. «Слои и их границы» в ``docs/ARCHITECTURE.md``). Поэтому
в сообщении вида «No connection: <причина>» переводится рамка, а причина
остаётся на языке того слоя, который её породил.

Промты к модели тоже остаются английскими: переписыватель Qwen отвечает
по-английски независимо от языка запроса, и переводить его ответ обратно было бы
потерей качества.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

LANGUAGES = ("ru", "en")

T: dict[str, tuple[str, str]] = {
    "app_title": ("Qwen-Image-2.1 — студия", "Qwen-Image-2.1 Studio"),
    "language": ("Язык", "Language"),
    "tab_generate": ("Генерация", "Generate"),
    "tab_edit": ("Редактирование", "Edit"),
    "tab_gallery": ("Галерея", "Gallery"),
    "tab_settings": ("Настройки", "Settings"),
    "prompt": ("Промт", "Prompt"),
    "prompt_placeholder": ("Опишите изображение…", "Describe the image…"),
    "generate": ("Сгенерировать", "Generate"),
    "stop": ("Прервать", "Stop"),
    "advanced": ("Продвинутое", "Advanced"),
    "quality": ("Качество", "Quality"),
    "aspect": ("Соотношение сторон", "Aspect ratio"),
    # Единственный пункт выпадающего списка соотношений, который на самом деле
    # текст интерфейса, а не идентификатор: остальные семь — числовые
    # соотношения вроде "16:9", их незачем переводить. Значение-сентинел
    # ``aspect.FOLLOW_REFERENCE`` при этом не меняется — под него заведён этот
    # отдельный ключ подписи, а не наоборот.
    "aspect_follow_reference": ("от референса", "from reference"),
    "image_number": ("Количество изображений", "Image number"),
    "seed": ("Сид", "Seed"),
    "seed_info": ("−1 — выбрать случайно", "−1 picks a random one"),
    "styles": ("Стили", "Styles"),
    "negative": ("Негативный промт", "Negative prompt"),
    "cfg": ("Сила guidance (true_cfg_scale)", "Guidance strength (true_cfg_scale)"),
    "cfg_info": (
        "Модель рассчитана на работу без guidance. При значении 1.0 негативный промт "
        "и негативные части стилей в модель не попадают вовсе.",
        "The model is meant to be sampled without guidance. At 1.0 the negative prompt "
        "and the negative half of every style are not sent to the model at all.",
    ),
    "reference_scale": ("Детальность референсов", "Reference detail"),
    "reference_scale_info": (
        "Разрешение, к которому приводятся референсы и исходник правки. Размер "
        "кадра от этого не меняется. «Авто» урезает масштаб при двух и более "
        "референсах: на полном пять референсов считаются тринадцать минут, на 512 — "
        "сорок секунд.",
        "The resolution condition images are scaled to. It does not change the frame "
        "size. “Auto” reduces the scale from two references up: at full "
        "scale five references take thirteen minutes per frame, at 512 — forty seconds.",
    ),
    "reference_scale_auto": ("Авто", "Auto"),
    "kv_cache": ("Кэш ключей и значений", "KV cache"),
    "kv_cache_info": (
        "Ускоряет генерацию. Переключение меняет результат при том же сиде.",
        "Speeds generation up. Toggling it changes the result for the same seed.",
    ),
    "references": ("Референсы", "References"),
    "reference_add": ("Добавить референсы", "Add references"),
    "reference_clear": ("Очистить референсы", "Clear references"),
    "boost": ("AI буст", "AI boost"),
    "boost_info": (
        "Переписать промт внешней языковой моделью перед генерацией.",
        "Rewrite the prompt with an external language model before generating.",
    ),
    "boost_result": ("Переписанный промт", "Rewritten prompt"),
    "boost_now": ("Переписать сейчас", "Rewrite now"),
    "describe": ("Описать изображение", "Describe image"),
    "result": ("Результат", "Result"),
    "save": ("Сохранить", "Save"),
    "save_prompt": ("Сохранить промт", "Save prompt"),
    "load_prompt": ("Загрузить промт", "Load prompt"),
    "delete_prompt": ("Удалить промт", "Delete prompt"),
    "preset_name": ("Название пресета", "Preset name"),
    "status": ("Состояние", "Status"),
    "source_image": ("Исходное изображение", "Source image"),
    "mask_mode": ("Режим области", "Region mode"),
    "mask_mode_none": ("Без области — править весь кадр", "No region — edit the whole frame"),
    "mask_mode_mask": ("Маска", "Mask"),
    "mask_mode_annotation": ("Аннотация", "Annotation"),
    "mask_mode_region": ("Точная область", "Exact region"),
    "mask_grow": ("Запас маски, пикселей", "Mask grow, pixels"),
    "mask_feather": ("Растушёвка, пикселей", "Feather, pixels"),
    "keep_outside": ("Сохранять кадр вне маски", "Keep pixels outside the mask"),
    "keep_outside_info": (
        "Склеивает результат с оригиналом: вне маски пиксели остаются исходными.",
        "Blends the result with the original so pixels outside the mask stay untouched.",
    ),
    "outpaint": ("Расширить холст", "Outpaint"),
    "outpaint_sides": ("Стороны", "Sides"),
    "outpaint_amount": ("Насколько расширить", "How far to expand"),
    "apply_edit": ("Применить правку", "Apply edit"),
    "send_to_edit": ("Отправить в редактор", "Send to editor"),
    "llm_endpoint": ("Адрес языковой модели", "Language model endpoint"),
    "llm_check": ("Проверить связь", "Check connection"),
    "refresh": ("Обновить", "Refresh"),
    "open_folder": ("Открыть папку", "Open folder"),
    "restore_params": ("Восстановить параметры из PNG", "Restore parameters from PNG"),
    "llm_endpoint_placeholder": (
        "имя бэкенда, адрес хоста:порт, token=…",
        "backend name, host:port, token=…",
    ),
    "sysprompt_select": ("Системный промт", "System prompt"),
    "sysprompt_t2i": ("Текст → изображение", "Text to image"),
    "sysprompt_edit": ("Редактирование", "Editing"),
    "sysprompt_describe": ("Описание изображения", "Describe image"),
    "memory": ("Память видеокарты", "GPU memory"),
}


# Сообщения, которые пользователь читает по ходу работы. Формат подстановок —
# ``str.format`` с ИМЕНОВАННЫМИ полями: позиционные ``{}`` в переводе легко
# переставить местами, а именованные переживают любой порядок слов.
MESSAGES: dict[str, tuple[str, str]] = {
    # вкладка «Генерация»
    "caption_untagged": ("без тега — одно изображение", "no tag — a single image"),
    "references_counted": ("Референсов: {count} из {total}", "References: {count} of {total}"),
    # Две ловушки вокруг «Переписанного промта», обе молчаливые.
    #
    # Первая: кнопка «Переписать сейчас» работает независимо от галочки «AI
    # буст», а в модель переписанный текст уходит только при включённой
    # галочке. Нажать кнопку, увидеть текст и получить картинку по старому
    # промту было проще простого, и ничто об этом не сообщало.
    #
    # Вторая: переписанный текст составлен по конкретному исходному промту.
    # Поправив исходный, пользователь ожидает, что правка учтётся, — а в
    # модель по-прежнему уходит то, что лежит в поле.
    "boost_ignored": (
        "Поле «Переписанный промт» заполнено, но «AI буст» выключен — в модель ушёл "
        "исходный промт. Включите галочку, чтобы использовать переписанный.",
        "The “Rewritten prompt” field is filled but “AI boost” is off — "
        "the original prompt was sent. Tick the box to use the rewritten one.",
    ),
    "boost_stale": (
        "Переписанный промт составлен по другому тексту: в модель ушёл он, а не то, "
        "что сейчас в поле «Промт». Нажмите «Переписать сейчас» или очистите поле.",
        "The rewritten prompt was made from different text: it was sent, not what is now "
        "in the “Prompt” field. Press “Rewrite now” or clear the field.",
    ),
    # Стадии до первого шага денойзинга. Прогресс приходит из обратного
    # вызова пайплайна, то есть **после** шага, а до него успевают пройти
    # загрузка модели (около пятидесяти секунд на первом запуске) и
    # кодирование промта с условными изображениями. Всё это время интерфейс
    # молчал, и отличить работу от зависания было нельзя.
    "stage_loading": (
        "Загружаю модель в память — это разовая минута на первом запуске…",
        "Loading the model into memory — a one-off minute on first run…",
    ),
    "stage_preparing": (
        "Готовлю: кодирую промт и условные изображения. Первый шаг дольше остальных…",
        "Preparing: encoding the prompt and condition images. The first step takes longest…",
    ),
    "references_cleared": ("Референсы очищены", "References cleared"),
    "reference_scale_chosen": (
        "детальность референсов {scale} (выбрано автоматически)",
        "reference detail {scale} (chosen automatically)",
    ),
    "generation_interrupted": ("Генерация прервана", "Generation interrupted"),
    "generation_done": ("Готово. Сиды: {seeds}. {memory}", "Done. Seeds: {seeds}. {memory}"),
    "progress_image": ("изображение {index}/{total}", "image {index}/{total}"),
    "preset_name_required": ("Укажите название пресета", "Enter a preset name"),
    "preset_saved": ("Пресет «{name}» сохранён", "Preset “{name}” saved"),
    "preset_not_selected": ("Пресет не выбран", "No preset selected"),
    "preset_load_failed": (
        "Пресет «{name}» не загружен: {error}",
        "Preset “{name}” could not be loaded: {error}",
    ),
    "preset_loaded": ("Пресет «{name}» загружен", "Preset “{name}” loaded"),
    "preset_deleted": ("Пресет «{name}» удалён", "Preset “{name}” deleted"),
    "preset_missing": ("Пресет не найден", "Preset not found"),
    "preset_delete_failed": (
        "Пресет «{name}» не удалён: {error}",
        "Preset “{name}” could not be deleted: {error}",
    ),
    "stopping": ("Останавливаю…", "Stopping…"),
    # вкладка «Редактирование»
    "upload_first": ("Сначала загрузите изображение", "Upload an image first"),
    "choose_a_side": ("Выберите хотя бы одну сторону", "Choose at least one side"),
    # Подсказка в этом сообщении — не украшение. Измерения показали, что при
    # дорисовке полей промт, описывающий операцию («продолжи сцену»,
    # «расширь фон»), уводит модель в вырезание наклейки: новая площадь
    # выходит прозрачной целиком. Описание желаемой сцены то же самое
    # расширение делает безупречно — непрозрачность 100 % против нуля.
    # См. docs/research/2026-09-22-maska-kak-alfa.md.
    # Предупреждение о шве. Появляется ровно тогда, когда случай наступил:
    # просьба оказалась глобальной по смыслу, модель перекрасила кадр
    # целиком и связно, а склейка обрезала её работу по границе маски.
    # Порог в четверть кадра выбран по наблюдению: правка платиновых волос
    # по эллипсу на макушке задевает около 47 % кадра вне маски и даёт
    # отчётливую границу, а точечная правка — единицы процентов.
    "edit_clipped": (
        "Модель изменила {share:.0f} % кадра вне маски, и склейка это обрезала — "
        "на границе маски будет виден шов. Снимите «Сохранять кадр вне маски», "
        "чтобы принять правку целиком, либо расширьте маску.",
        "The model changed {share:.0f} % of the frame outside the mask and the blend "
        "clipped it — the mask edge will show a seam. Clear “Keep pixels outside "
        "the mask” to accept the edit whole, or widen the mask.",
    ),
    "canvas_expanded": (
        "Холст расширен до {width}×{height}. В промте опишите всю желаемую картину "
        "целиком, а не действие: «продолжи сцену» даст прозрачную заливку. "
        "Кнопка «Описать изображение» составит описание за вас.",
        "Canvas expanded to {width}×{height}. In the prompt, describe the whole "
        "picture you want, not the action: “continue the scene” yields a "
        "transparent fill. The “Describe image” button will write the "
        "description for you.",
    ),
    "edit_interrupted": ("Правка прервана", "Edit interrupted"),
    "edit_done": ("Готово. {memory}", "Done. {memory}"),
    "nothing_to_send": ("Нечего отправлять", "Nothing to send"),
    "sent_to_editor": ("Результат перенесён в редактор", "Result moved to the editor"),
    "progress_edit": ("правка", "editing"),
    # вкладка «Галерея»: ключи словаря gr.JSON тоже видны пользователю
    "field_message": ("сообщение", "message"),
    "field_opened_folder": ("открыт каталог", "opened folder"),
    "png_without_parameters": (
        "в этом PNG нет наших параметров",
        "this PNG carries none of our parameters",
    ),
    "pick_or_drop_png": (
        "выберите изображение в галерее или перетащите PNG",
        "select an image in the gallery or drop a PNG here",
    ),
    "parameters_not_found": ("параметры не найдены", "parameters not found"),
    # вкладка «Настройки»
    "endpoint_file_missing": (
        "# Файл не найден. Укажите бэкенд, адрес и token=…\n",
        "# File not found. Specify the backend, the address and token=…\n",
    ),
    "prompt_file_missing": (
        "# Файл {name} не найден. Запустите tools/fetch_system_prompts.py\n",
        "# File {name} not found. Run tools/fetch_system_prompts.py\n",
    ),
    "endpoint_saved": ("Адрес сохранён", "Endpoint saved"),
    "llm_no_connection": ("Нет связи: {error}", "No connection: {error}"),
    "llm_connected": (
        "Связь есть. Выбранная модель: {model}",
        "Connected. Selected model: {model}",
    ),
    "llm_no_model_named": ("сервер не назвал ни одной", "the server named none"),
    "prompt_file_saved": ("Файл {name} сохранён", "File {name} saved"),
    # состояние приложения
    "model_not_loaded": ("модель ещё не загружена", "the model is not loaded yet"),
    "memory_report": (
        "видеопамять: {allocated:.1f} ГиБ занято, {reserved:.1f} ГиБ зарезервировано; "
        "перестановок энкодера: {swaps}; кэш промтов: {hits} попаданий / {misses} промахов",
        "GPU memory: {allocated:.1f} GiB allocated, {reserved:.1f} GiB reserved; "
        "encoder swaps: {swaps}; prompt cache: {hits} hits / {misses} misses",
    ),
    "boost_failed": ("AI буст не выполнен: {error}", "AI boost failed: {error}"),
    "boost_done": ("AI буст выполнен", "AI boost done"),
    "describe_failed": ("Описание не выполнено: {error}", "Describing failed: {error}"),
    "describe_done": ("Описание готово", "Description ready"),
    # классификация сбоя генерации
    "failure_out_of_memory": (
        "Не хватило видеопамяти. Попробуйте пресет качества пониже, меньше изображений "
        "за раз или меньше референсов. {error}",
        "Out of GPU memory. Try a lower quality preset, fewer images at once or fewer "
        "references. {error}",
    ),
    "failure_file_not_found": (
        "Файл не найден: {error}. Проверьте, что веса модели на месте.",
        "File not found: {error}. Check that the model weights are in place.",
    ),
    "failure_io": ("Ошибка ввода-вывода: {error}", "Input/output error: {error}"),
    "failure_other": ("Сбой: {kind}: {error}", "Failure: {kind}: {error}"),
}


def pick(key: str, lang: str) -> str:
    """Подпись компонента на нужном языке."""
    entry = T.get(key)
    if entry is None:
        return key
    return entry[0] if lang == "ru" else entry[1]


def say(key: str, lang: str, **values: Any) -> str:
    """Сообщение пользователю на нужном языке, с подстановками.

    Язык — аргумент, а не захваченная при сборке переменная: переключатель
    RU/EN работает во время работы приложения, и обработчик, запомнивший язык
    запуска, показывал бы русский текст вокруг английских подписей.

    Неизвестный ключ возвращается как есть — по тому же правилу, что и в
    ``pick``: пропущенное сообщение не должно ронять обработчик.
    """
    entry = MESSAGES.get(key)
    if entry is None:
        return key
    template = entry[0] if lang == "ru" else entry[1]
    return template.format(**values)


class Localizer:
    """Запоминает переводимые поля компонентов и обновляет их разом."""

    def __init__(self, default: str = "ru") -> None:
        self._default = default
        self._entries: list[tuple[Any, dict[str, tuple[str, str]]]] = []

    def bind(self, component: Any, **fields: tuple[str, str]) -> Any:
        self._entries.append((component, fields))
        return component

    @property
    def default(self) -> str:
        """Язык, на котором интерфейс собирается при старте."""
        return self._default

    @property
    def components(self) -> list[Any]:
        return [component for component, _ in self._entries]

    @property
    def entries(self) -> list[tuple[Any, dict[str, tuple[str, str]]]]:
        """Пары (компонент, переводимые поля) для проверки согласованности.

        Нужна тестам, которые сверяют текущее значение поля компонента с тем,
        что зарегистрировано для языка запуска: без этой пары такую сверку
        нельзя сделать иначе как через приватный ``_entries``.
        """
        return list(self._entries)

    def updates(self, lang: str) -> list[Any]:
        index = 0 if lang == "ru" else 1
        return [
            gr.update(**{name: values[index] for name, values in fields.items()})
            for _, fields in self._entries
        ]
