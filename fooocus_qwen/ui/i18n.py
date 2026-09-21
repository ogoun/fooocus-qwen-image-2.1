"""Двуязычные подписи интерфейса.

Подписи в Gradio задаются при сборке, поэтому смена языка — это массовое
обновление уже созданных компонентов. Локализатор запоминает, какие поля какого
компонента переводимы, и по запросу отдаёт список обновлений в том же порядке,
в каком компоненты регистрировались.

Промты к модели при этом остаются английскими: переписыватель Qwen отвечает
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


def pick(key: str, lang: str) -> str:
    entry = T.get(key)
    if entry is None:
        return key
    return entry[0] if lang == "ru" else entry[1]


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
