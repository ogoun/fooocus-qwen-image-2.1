"""Ячейки LoRA на вкладках генерации и правки — как у Fooocus: «вкл. · файл · вес».

Пять ячеек (``lora.SLOTS``), в каждой — галочка, выбор файла из каталога LoRA
(``AppConfig.lora_dir``) и вес от −2 до 2. Под ячейкой — что известно о
файле по его заголовку (как в InvokeAI): ранг, число слоёв, слова-триггеры,
а если файл не для Qwen-Image-2.1 — для чего он и почему не подключится.
Проверка — по заголовку, без чтения весов, поэтому мгновенная.

Значения ячеек сводятся в одно состояние (список словарей): его берут
генерация, сохранённые промты и метаданные PNG. Обратная дорога —
``values_for``: по записи из PNG или промта ячейки заполняются снова, а
переименованный файл находится по отпечатку (как recall у InvokeAI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from ..engine import lora
from . import layout
from .i18n import Localizer, T, pick, say

LOGGER = logging.getLogger(__name__)

NONE = ""
# Чем назвать архитектуру чужой LoRA.
_FAMILY_KEYS = {
    "qwen-image": "lora_family_qwen1",
    "flux": "lora_family_flux",
    "sdxl": "lora_family_sd",
    "unknown": "lora_family_unknown",
}
_PROBLEM_KEYS = {"empty": "lora_empty", "unreadable": "lora_unreadable", "outside": "lora_outside"}
_KIND_NAMES = {"lokr": "LoKr", "loha": "LoHa", "dora": "DoRA", "full": "full weight diffs"}


@dataclass
class LoraSlots:
    """Компоненты ячеек и их общее состояние."""

    enabled: list = field(default_factory=list)
    names: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    state: object = None
    refresh: object = None
    open_folder: object = None
    accordion: object = None
    folder_note: object = None

    @property
    def components(self) -> list:
        """Ячейки подряд: вкл., файл, вес — в порядке ``values_for``."""
        return [item for slot in zip(self.enabled, self.names, self.weights) for item in slot]


def _choices(directory: Path, lang: str) -> list[tuple[str, str]]:
    return [(pick("lora_none", lang), NONE), *[(name, name) for name in lora.list_names(directory)]]


def note(directory: Path, name: str, lang: str) -> str:
    """Строка под ячейкой: что за файл, или почему он не подключится."""
    if not name:
        return ""
    try:
        path = lora.path_for(directory, name)
    except lora.LoraError:
        return say("lora_missing", lang, name=name)
    info = lora.inspect(path, name)
    return describe(info, lang)


def describe(info: lora.LoraInfo, lang: str) -> str:
    if info.problem == "other_model":
        return say("lora_other_model", lang, family=say(_FAMILY_KEYS.get(info.family, "lora_family_unknown"), lang))
    if info.problem == "unsupported":
        return say("lora_unsupported", lang, kind=_KIND_NAMES.get(info.family, info.family))
    if info.problem is not None:
        return say(_PROBLEM_KEYS.get(info.problem, "lora_unreadable"), lang)
    text = say("lora_ok", lang, rank=info.rank or "?", layers=info.layers)
    if info.triggers:
        text += say("lora_triggers", lang, words=", ".join(f"`{word}`" for word in info.triggers))
    return text


def collect(*values) -> list[dict]:
    """Значения ячеек подряд (вкл., файл, вес …) → список выбранных LoRA."""
    chosen = []
    for index in range(0, len(values), 3):
        enabled, name, weight = values[index:index + 3]
        if enabled and name:
            chosen.append({"name": str(name), "weight": round(float(weight or 0), 3)})
    return chosen


def choices_of(state) -> list[lora.LoraChoice]:
    return [lora.LoraChoice(item["name"], float(item["weight"])) for item in (state or []) if item.get("name")]


def resolve(directory: Path, state, lang: str) -> tuple[tuple[lora.ResolvedLora, ...], str]:
    """Состояние ячеек → проверенные LoRA и строка о пропущенных (пустая — все годны)."""
    resolved, problems = lora.resolve(directory, choices_of(state))
    lines = []
    for problem in problems:
        if problem.code == "missing":
            reason = say("lora_missing", lang, name=problem.name)
        else:
            info = lora.LoraInfo(problem.name, problem=problem.code, family=problem.detail)
            reason = describe(info, lang)
        lines.append(say("lora_skipped", lang, name=problem.name, reason=reason))
    return tuple(resolved), " ".join(lines)


def values_for(directory: Path, saved) -> tuple:
    """Значения ячеек по записи из PNG или промта: вкл., файл, вес — на все ячейки.

    Файл ищется по имени, а не найдя — по отпечатку (``hash``): переименованная
    LoRA узнаётся. Не найденная никак остаётся в ячейке под своим именем —
    строка под ячейкой скажет, что файла нет.
    """
    items = [item for item in (saved or []) if isinstance(item, dict) and item.get("name")][: lora.SLOTS]
    names = lora.list_names(directory)
    by_hash: dict[str, str] | None = None
    values: list = []
    for item in items:
        name = str(item["name"])
        if name not in names and item.get("hash"):
            if by_hash is None:
                by_hash = _hashes(directory, names)
            name = by_hash.get(str(item["hash"]), name)
        try:
            weight = float(item.get("weight", lora.WEIGHT_DEFAULT))
        except (TypeError, ValueError):
            weight = lora.WEIGHT_DEFAULT
        values += [True, name, max(lora.WEIGHT_MIN, min(lora.WEIGHT_MAX, weight))]
    for _ in range(lora.SLOTS - len(items)):
        values += [False, NONE, lora.WEIGHT_DEFAULT]
    return tuple(values)


def _hashes(directory: Path, names: list[str]) -> dict[str, str]:
    found = {}
    for name in names:
        try:
            found[lora.short_hash(lora.path_for(directory, name))] = name
        except (OSError, lora.LoraError):
            continue
    return found


def shown_path(directory: Path) -> str:
    """Каталог для подписи: внутри проекта — от его корня (``loras``), иначе целиком."""
    from .. import config

    try:
        return Path(directory).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(directory)


def build(localizer: Localizer, lang: str, language, directory: Path) -> LoraSlots:
    """Собирает аккордеон «LoRA» в текущей колонке и связывает его ячейки."""
    slots = LoraSlots()
    slots.accordion = localizer.bind(gr.Accordion(pick("lora", lang), open=False), label=T["lora"])
    with slots.accordion:
        for number in range(1, lora.SLOTS + 1):
            # Класс — на колонке: у gr.Group Gradio 6.5.1 его не выводит.
            with gr.Group(), gr.Column(elem_classes=[layout.LORA_SLOT]):
                with gr.Row():
                    slots.enabled.append(localizer.bind(
                        gr.Checkbox(value=False, label=pick("lora_enabled", lang), scale=0, min_width=80),
                        label=T["lora_enabled"],
                    ))
                    slots.names.append(localizer.bind(
                        gr.Dropdown(
                            choices=_choices(directory, lang), value=NONE, label=f"LoRA {number}",
                            scale=4, allow_custom_value=True,
                        ),
                        choices=(_choices(directory, "ru"), _choices(directory, "en")),
                    ))
                slots.weights.append(localizer.bind(
                    gr.Slider(
                        lora.WEIGHT_MIN, lora.WEIGHT_MAX, value=lora.WEIGHT_DEFAULT, step=0.05,
                        label=pick("lora_weight", lang),
                    ),
                    label=T["lora_weight"],
                ))
                slots.notes.append(gr.Markdown(elem_classes=[layout.LORA_NOTE]))
        with gr.Row():
            slots.refresh = localizer.bind(gr.Button(pick("lora_refresh", lang), size="sm"), value=T["lora_refresh"])
            slots.open_folder = localizer.bind(gr.Button(pick("lora_open", lang), size="sm"), value=T["lora_open"])
        slots.folder_note = localizer.bind(
            gr.Markdown(say("lora_folder", lang, path=shown_path(directory))),
            value=(say("lora_folder", "ru", path=shown_path(directory)), say("lora_folder", "en", path=shown_path(directory))),
        )
    slots.state = gr.State([])

    def update_state(*values):
        return collect(*values)

    for component in slots.components:
        component.change(update_state, slots.components, slots.state, queue=False, show_progress="hidden")

    def explain(name, lang_value):
        return note(directory, name, lang_value)

    for name, note_box in zip(slots.names, slots.notes):
        # Выбор файла включает ячейку: так ведут себя ячейки Fooocus.
        name.input(lambda value: bool(value), name, slots.enabled[slots.names.index(name)], queue=False)
        name.change(explain, [name, language], note_box, queue=False, show_progress="hidden")

    def refresh(lang_value, *names):
        choices = _choices(directory, lang_value)
        updates = [gr.Dropdown(choices=choices) for _ in names]
        notes = [note(directory, value, lang_value) for value in names]
        count = len(choices) - 1
        return (*updates, *notes, say("lora_found", lang_value, count=count, path=shown_path(directory)))

    slots.refresh.click(
        refresh, [language, *slots.names], [*slots.names, *slots.notes, slots.folder_note], queue=False,
    )

    def open_folder(lang_value):
        from .tab_gallery import _open_folder

        directory.mkdir(parents=True, exist_ok=True)
        _open_folder(directory)
        return say("lora_folder", lang_value, path=shown_path(directory))

    slots.open_folder.click(open_folder, language, slots.folder_note, queue=False)
    return slots
