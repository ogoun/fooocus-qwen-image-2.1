"""Вкладка настроек: адрес языковой модели, системные промты, память видеокарты.

Токен языковой модели в браузер не отправляется никогда. Раньше адрес
правился одним текстовым полем с содержимым файла целиком — вместе с
токеном открытым текстом, а интерфейс слушает ``0.0.0.0``: токен читал
любой в локальной сети, открывший вкладку. Теперь адрес — обычное поле,
токен — поле пароля, которое сервер не заполняет; пустое при сохранении
значит «оставить прежний», для удаления есть отдельная кнопка. Файл пишет
тот же ``llm.setup.render``, что и установка, — формат один на оба пути.

Системные промты правятся прямо здесь и перечитываются заново при каждом
обращении к языковой модели (см. ``prompting.boost``), а адрес — при каждом
``Studio.llm_client()``: оболочку не нужно перезапускать, чтобы подхватить
правку.

Отсутствие настроенной языковой модели — обычное состояние свежей установки,
а не ошибка: ``check_connection`` обязана сообщить об этом читаемым текстом,
а не уронить вкладку исключением.
"""

from __future__ import annotations

import gradio as gr

from .. import config
from .. import settings as settings_module
from ..engine import attention, hardware
from ..llm import LlmError, load_endpoint, parse_endpoint_file
from ..llm import setup as llm_setup
from . import layout
from .i18n import MESSAGES, Localizer, T, pick, say, sentences
from .state import GPU_CONCURRENCY_ID


def precision_choices(lang: str, vram_gib: float | None) -> list[tuple[str, str]]:
    """Варианты точности: что это, сколько видеопамяти, для каких карт.

    Подписи — из общего каталога (``settings.PRECISION_INFO``), того же, что
    у установки; подходящий к этой карте вариант помечен.
    """
    recommended = settings_module.recommended_precision(vram_gib)
    return [
        (settings_module.precision_label(name, lang, recommended=name == recommended), name)
        for name in settings_module.PRECISIONS
    ]


# Профиль: (для каких карт, как раскладываются веса) — (ru, en).
_PROFILE_CARDS = {
    settings_module.MEMORY_HIGH: ("Для карт от 20 ГБ", "For 20 GB cards and up"),
    settings_module.MEMORY_LOW: ("Для карт 6–16 ГБ", "For 6–16 GB cards"),
}
_PROFILE_HOW = {
    settings_module.MEMORY_HIGH: ("модели по очереди целиком на карте", "models take turns on the card whole"),
    settings_module.MEMORY_LOW: (
        "модель на карте, энкодер по слоям, кэш в ОЗУ",
        "model on the card, encoder layer by layer, cache in RAM",
    ),
}


def profile_choices(lang: str, vram_gib: float | None) -> list[tuple[str, str]]:
    """Профили памяти словами; у «Авто» — во что он превратится на этой карте."""
    index = 0 if lang == "ru" else 1
    cards = _PROFILE_CARDS[settings_module.resolve_profile(settings_module.MEMORY_AUTO, vram_gib)][index]
    resolved = cards[0].lower() + cards[1:]
    auto = f"Авто — по видеокарте (сейчас: {resolved})" if lang == "ru" else f"Auto — by the video card (now: {resolved})"
    return [(auto, settings_module.MEMORY_AUTO)] + [
        (f"{_PROFILE_CARDS[name][index]} — {_PROFILE_HOW[name][index]}", name)
        for name in (settings_module.MEMORY_HIGH, settings_module.MEMORY_LOW)
    ]


_PROMPT_FILES: tuple[str, ...] = (
    "system_prompt_t2i.txt",
    "system_prompt_edit.txt",
    "system_prompt_describe.txt",
)

# Дисплей-имена файлов промтов — переводимый текст, а не служебные
# идентификаторы, поэтому у выпадающего списка ниже, как и у mask_mode во
# вкладке редактирования, переводятся и label, и choices.
_PROMPT_KEYS: dict[str, str] = {
    "system_prompt_t2i.txt": "sysprompt_t2i",
    "system_prompt_edit.txt": "sysprompt_edit",
    "system_prompt_describe.txt": "sysprompt_describe",
}


def _current_endpoint():
    """Настроенный адрес или ``None``, если настройки нет или она не читается."""
    try:
        return load_endpoint(config.ENDPOINT_FILE)
    except (OSError, ValueError):
        return None


def has_token() -> bool:
    """Сохранён ли токен: от этого зависит, показывать ли «Убрать токен»."""
    endpoint = _current_endpoint()
    return bool(endpoint and endpoint.token)


def describe_endpoint(lang: str) -> str:
    """Что настроено сейчас — без токена: о нём говорится только, есть ли он."""
    endpoint = _current_endpoint()
    if endpoint is None:
        return say("endpoint_not_configured", lang)
    token = say("token_set" if endpoint.token else "token_unset", lang)
    return say("endpoint_current", lang, url=endpoint.base_url, token=token)


def describe_performance(studio, lang: str) -> str:
    """Что выбрано сейчас: точность, механизм внимания, готовность turbo."""
    chosen = settings_module.load()
    if chosen.sage_attention and attention.sage_available():
        attention_text = "SageAttention"  # имя собственное, не переводится
    elif chosen.sage_attention:
        attention_text = say("perf_attention_missing", lang)
    else:
        attention_text = say("perf_attention_native", lang)
    turbo = say("perf_turbo_ready" if studio.turbo_weights_present() else "perf_turbo_later", lang)
    return say(
        "perf_current", lang, precision=chosen.precision.upper(), attention=attention_text, turbo=turbo,
        profile=studio.memory_profile(),
    )


def _read_prompt(name: str, lang: str) -> str:
    try:
        return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return say("prompt_file_missing", lang, name=name)


def build(studio, localizer: Localizer, language=None) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком; см. докстринг
    ``tab_generate.build``.
    """
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            current = _current_endpoint()
            address = localizer.bind(
                gr.Textbox(
                    label=pick("llm_address", lang),
                    placeholder=pick("llm_address_placeholder", lang),
                    value=current.base_url if current else "",
                ),
                label=("Адрес сервера языковой модели", "Language model server address"),
                placeholder=("192.168.1.10:8000 или https://…", "192.168.1.10:8000 or https://…"),
            )
            # Поле пароля и всегда пустое: сервер токен не показывает — ни
            # здесь, ни где-либо ещё в интерфейсе.
            token = localizer.bind(
                gr.Textbox(
                    label=pick("llm_token", lang),
                    placeholder=pick("llm_token_placeholder", lang),
                    type="password",
                    value="",
                ),
                label=("Токен", "Token"),
                placeholder=("Пусто — оставить прежний", "Empty keeps the current one"),
            )
            with gr.Row():
                save_endpoint = localizer.bind(
                    gr.Button(pick("save", lang)), value=("Сохранить", "Save")
                )
                check = localizer.bind(
                    gr.Button(pick("llm_check", lang), variant="primary"),
                    value=("Проверить связь", "Check connection"),
                )
            # Только когда токен есть: без него кнопке нечего убирать, а
            # нажатие лишь повторяло строку «токен не задан».
            forget = localizer.bind(
                gr.Button(pick("llm_forget_token", lang), size="sm", visible=has_token()),
                value=("Убрать токен", "Remove the token"),
            )
            endpoint_status = localizer.bind(
                gr.Textbox(
                    label=pick("status", lang), interactive=False, lines=2,
                    value=describe_endpoint(lang),
                ),
                label=("Состояние", "Status"),
            )

        # Системный промт — единственное место вкладки, где ширина идёт в
        # дело: это абзацы текста, который правят руками.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.TEXT_COL]):
            ru_choices = [(pick(_PROMPT_KEYS[name], "ru"), name) for name in _PROMPT_FILES]
            en_choices = [(pick(_PROMPT_KEYS[name], "en"), name) for name in _PROMPT_FILES]
            chosen_file = localizer.bind(
                gr.Dropdown(
                    choices=ru_choices if lang == "ru" else en_choices,
                    value=_PROMPT_FILES[0],
                    label=pick("sysprompt_select", lang),
                ),
                label=("Системный промт", "System prompt"),
                choices=(ru_choices, en_choices),
            )
            prompt_text = gr.Textbox(
                lines=18,
                value=_read_prompt(_PROMPT_FILES[0], lang),
                show_label=False,
                elem_classes=[layout.LONG_TEXT],
            )
            save_prompt_button = localizer.bind(
                gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
            )
            prompt_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

        # Память стояла отдельной строкой под обеими колонками и занимала
        # её целиком ради двух строк текста. Третьей колонкой она и читается
        # лучше (рядом с тем, что тоже про состояние), и не отнимает высоту
        # у того, что под ней.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            # Производительность: точность трансформера и механизм внимания.
            # Смена точности — это докачка весов и перезагрузка модели, поэтому
            # отдельная кнопка, а не мгновенная реакция на выбор; внимание
            # переключается на лету.
            chosen = settings_module.load()
            vram = hardware.vram_gib()
            # Столбцом, а не выпадающим списком: восемь вариантов различаются
            # ценой, и сравнивать их удобнее, когда видны все сразу.
            precision = localizer.bind(
                gr.Radio(
                    choices=precision_choices(lang, vram),
                    value=chosen.precision,
                    label=pick("perf_precision", lang),
                    info=pick("perf_precision_info", lang),
                    elem_classes=[layout.CHOICES],
                ),
                label=T["perf_precision"],
                info=T["perf_precision_info"],
                choices=(precision_choices("ru", vram), precision_choices("en", vram)),
            )
            apply_precision_button = localizer.bind(
                gr.Button(pick("perf_apply", lang)), value=("Применить точность", "Apply precision")
            )
            # Профиль памяти — раскладка весов (settings.MEMORY_PROFILES). Как
            # и точность, меняется перезагрузкой модели, поэтому — кнопкой.
            memory_profile = localizer.bind(
                gr.Radio(
                    choices=profile_choices(lang, vram),
                    value=chosen.memory_profile,
                    label=pick("perf_profile", lang),
                    elem_classes=[layout.CHOICES],
                ),
                label=T["perf_profile"],
                choices=(profile_choices("ru", vram), profile_choices("en", vram)),
            )
            apply_profile_button = localizer.bind(
                gr.Button(pick("perf_profile_apply", lang)), value=("Применить профиль", "Apply profile")
            )
            sage = localizer.bind(
                gr.Checkbox(
                    label=pick("perf_sage", lang),
                    value=chosen.sage_attention,
                    # Без пакета галочка ничего бы не включила — и не обещает.
                    interactive=attention.sage_available(),
                ),
                label=("SageAttention — быстрое внимание", "SageAttention — fast attention"),
            )
            performance_status = localizer.bind(
                gr.Textbox(
                    label=pick("status", lang), interactive=False, lines=2,
                    value=describe_performance(studio, lang),
                ),
                label=("Состояние", "Status"),
            )

            memory = localizer.bind(
                gr.Textbox(
                    label=pick("memory", lang),
                    interactive=False,
                    lines=2,
                    value=say("model_not_loaded", lang),
                ),
                label=("Память видеокарты", "GPU memory"),
                # Значение поля тоже переводимо: это заглушка «ещё не
                # загружена», а не результат измерения. Переключение языка
                # сбрасывает поле к ней — для снимка, который обновляют
                # кнопкой, это верное поведение, а вот русская строка вокруг
                # английских подписей — нет.
                value=MESSAGES["model_not_loaded"],
            )
            memory_refresh = localizer.bind(
                gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh")
            )

    def store_endpoint(address_text, token_text, lang):
        address_text = (address_text or "").strip()
        if not address_text:
            return "", say("endpoint_need_address", lang), gr.update()
        current = _current_endpoint()
        # Пустое поле токена — «оставить прежний»: сервер токен не показывает,
        # и заставлять вводить его заново при каждой правке адреса нельзя.
        new_token = (token_text or "").strip() or (current.token if current else None)
        text = llm_setup.render(address_text, new_token, current.backend if current else "")
        try:
            parse_endpoint_file(text)
        except ValueError as error:
            return "", say("endpoint_bad_address", lang, error=error), gr.update()
        config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
        return "", sentences(say("endpoint_saved", lang), describe_endpoint(lang)), gr.update(visible=has_token())

    def forget_token(lang):
        current = _current_endpoint()
        if current is None or not current.token:
            return describe_endpoint(lang), gr.update(visible=False)
        config.ENDPOINT_FILE.write_text(
            llm_setup.render(current.base_url, None, current.backend), encoding="utf-8"
        )
        return sentences(say("token_forgotten", lang), describe_endpoint(lang)), gr.update(visible=False)

    def check_connection(lang):
        # load_endpoint бросает FileNotFoundError (файла нет — обычное дело
        # для свежей установки) или ValueError (в файле нет ни одного хоста),
        # ping() — LlmError (сервер недоступен или ответил ошибкой). Все три
        # случая — читаемое сообщение, а не падение вкладки.
        try:
            client = studio.llm_client()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            return say("llm_no_connection", lang, error=error)
        return say(
            "llm_connected", lang, model=client.model or say("llm_no_model_named", lang)
        )

    def load_prompt_file(name, lang):
        if name not in _PROMPT_FILES:
            return say("prompt_file_missing", lang, name=name)
        return _read_prompt(name, lang)

    def store_prompt_file(name, text, lang):
        # Имя пришло из браузера и ложится в путь записи. Gradio и сам сверяет
        # значение списка с вариантами, но граница, на которой стоит запись
        # файла, не должна держаться на чужой проверке.
        if name not in _PROMPT_FILES:
            return say("prompt_file_missing", lang, name=name)
        (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")
        return say("prompt_file_saved", lang, name=name)

    def memory_report(lang):
        return studio.memory_report(lang)

    def apply_precision(choice, lang, progress=gr.Progress(track_tqdm=True)):
        if choice not in settings_module.PRECISIONS:
            return describe_performance(studio, lang)
        if choice == settings_module.load().precision:
            return sentences(say("precision_unchanged", lang), describe_performance(studio, lang))
        progress(0, desc=say("precision_downloading", lang))
        try:
            studio.ensure_precision_weights(choice)
        except Exception as error:  # noqa: BLE001 — сеть и диск: строка состояния, не падение
            return say("precision_download_failed", lang, error=error)
        studio.switch_precision(choice)
        if studio.config.preload:
            studio.preload_in_background()
        return sentences(say("precision_switched", lang, precision=choice.upper()), describe_performance(studio, lang))

    def apply_memory_profile(choice, lang):
        if choice not in settings_module.MEMORY_PROFILES:
            return describe_performance(studio, lang)
        if choice == settings_module.load().memory_profile:
            return sentences(say("profile_unchanged", lang), describe_performance(studio, lang))
        studio.switch_memory_profile(choice)
        if studio.config.preload:
            studio.preload_in_background()
        return sentences(say("profile_switched", lang, profile=choice), describe_performance(studio, lang))

    def toggle_sage(enabled, lang):
        studio.set_sage_attention(bool(enabled))
        return describe_performance(studio, lang)

    # Смена точности выгружает модель — в одной очереди с генерацией, чтобы не
    # вклиниться в чужую работу видеокарты (замок генератора её всё равно
    # дождётся, но очередь не выпустит генерацию навстречу перезагрузке).
    apply_precision_button.click(
        apply_precision, [precision, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID
    )
    apply_profile_button.click(
        apply_memory_profile, [memory_profile, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID
    )
    # input, а не change: change срабатывает и от программного обновления при
    # открытии вкладки, и каждое открытие «переключало» бы внимание.
    sage.input(toggle_sage, [sage, language], performance_status, concurrency_id=GPU_CONCURRENCY_ID)
    save_endpoint.click(store_endpoint, [address, token, language], [token, endpoint_status, forget])
    forget.click(forget_token, language, [endpoint_status, forget])
    check.click(check_connection, language, endpoint_status)
    chosen_file.change(load_prompt_file, [chosen_file, language], prompt_text)
    save_prompt_button.click(
        store_prompt_file, [chosen_file, prompt_text, language], prompt_status
    )
    memory_refresh.click(memory_report, language, memory)

    def refresh(lang):
        # Вкладка перечитывает настройку при каждом открытии: иначе она
        # показывала бы состояние на момент запуска процесса — и отставала,
        # если адрес поменяли из другого окна или руками в файле.
        current = _current_endpoint()
        chosen = settings_module.load()
        return (
            (current.base_url if current else ""),
            describe_endpoint(lang),
            studio.memory_report(lang),
            chosen.precision,
            gr.update(value=chosen.sage_attention, interactive=attention.sage_available()),
            describe_performance(studio, lang),
            chosen.memory_profile,
            gr.update(visible=bool(current and current.token)),
        )

    return {
        "memory": memory,
        "endpoint_status": endpoint_status,
        "prompt_status": prompt_status,
        "address": address,
        "precision": precision,
        "sage": sage,
        "performance_status": performance_status,
        "memory_profile": memory_profile,
        "forget": forget,
        "refresh": refresh,
    }
