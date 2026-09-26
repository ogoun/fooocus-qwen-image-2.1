"""Сетка референсов: десять позиционных ячеек, их теги и запись в ячейку.

Сетка одна и та же на двух вкладках — генерации (слева от результата) и
правки (слева от кисти); у каждой вкладки своя. Здесь — построение сетки
(``build_grid``), связывание её событий (``wire``), подписи и запись в
ячейку (``place``), общие для вкладок и окон инструментов ячейки
(``reference_tools``: поза, эскиз). Отдельным модулем — потому что окна
живут в своём модуле, а вкладки их встраивают, и держать сетку во вкладке
значило бы замкнуть импорты по кругу.

**Режим области** (``mode``) — единственное, чем сетки вкладок отличаются.
На генерации его нет (``None``): референсы — все условные изображения, и
первый — ``<image1>``. На правке перед ними стоит исходник (``<image1>``),
а в режимах «маска» и «точная область» ещё и маска (``<image2>``), так что
первый референс — ``<image3>`` или ``<image2>``. Подписи считает та же
``condition_slots()``, что строит вход модели.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from PIL import Image

from ..engine.generator import MASK_MASK, MASK_NONE, MASK_REGION, condition_slots
from . import layout
from .i18n import Localizer, pick, say

# Сетка референсов слева от результата: два столбца по пять ячеек, в рост
# поля результата. Десять — предел самой модели (карточка Qwen-Image-2.1), и
# сетка ровно его покрывает. Нумерация — по рядам: первый ряд — ячейки 1 и 2.
REFERENCE_ROWS = 5
REFERENCE_COLUMNS = 2
MAX_REFERENCES = REFERENCE_ROWS * REFERENCE_COLUMNS


def _captions(images, lang: str, mode: str | None = None) -> list[str]:
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

    ``mode`` — режим области вкладки правки (см. докстринг модуля): исходник
    и маска подставляются заглушками, важно только их место в очереди.
    """
    source = mask = None
    if mode is not None:
        source = _PLACEHOLDER
        if mode in (MASK_MASK, MASK_REGION):
            mask = _PLACEHOLDER
    slots = condition_slots(source=source, mask=mask, mask_mode=mode or MASK_NONE, references=tuple(images))
    return [slot.tag or say("caption_untagged", lang) for slot in slots if slot.role == "reference"]


_PLACEHOLDER = Image.new("L", (1, 1))


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


# Пустая строка тега — неразрывный пробел, а не "": пустое поле схлопнулось
# бы по высоте, и ряды ячеек с подписью и без поехали бы относительно друг
# друга.
_NO_TAG = chr(0xA0)


def slot_tags(references, lang: str, mode: str | None = None) -> list:
    """Строки тегов под ячейками: тег у заполненных, пусто у пустых.

    Тег считается по порядку **заполненных** ячеек, а не по номеру ячейки:
    если заполнены первая, третья и седьмая, они — ``<image1>``,
    ``<image2>`` и ``<image3>``. Подпись по номеру («<image7>») ссылалась бы
    на изображение, которого у модели нет, — пустые ячейки в неё не уходят.
    Сами теги берутся из ``_captions``, то есть из той же функции, что строит
    вход модели.

    Тег стоит отдельной строкой под ячейкой, а не подписью самого поля
    изображения. Подпись Gradio делит верх ячейки со значком и крестиком, и в
    ячейке в семьдесят пикселей от ``<image1>`` оставалось «<im» — видно на
    снимке, хотя текстом подпись была на месте. Тег обёрнут в код: иначе
    Markdown принял бы ``<image1>`` за тег разметки и не показал бы вовсе.
    """
    grid = _grid(references)
    captions = iter(_captions(filled(grid), lang, mode))
    return [
        gr.update(value=_tag_markdown(next(captions)) if image is not None else _NO_TAG)
        for image in grid
    ]


def _tag_markdown(caption: str) -> str:
    return f"`{caption}`" if caption.startswith("<image") else caption


def place(references, index: int, image, lang: str, message: str, mode: str | None = None) -> tuple:
    """Кладёт картинку в ячейку ``index``; выходы — как у ``reference_targets``.

    Выходы: состояние референсов, десять ячеек, десять тегов, строка
    состояния. Картинка уходит только в свою ячейку: остальные не трогаются,
    иначе каждая запись заново слала бы в браузер все десять изображений.
    """
    grid = _grid(references)
    grid[index] = image
    slots = [gr.update(value=image) if position == index else gr.update() for position in range(MAX_REFERENCES)]
    return (grid, *slots, *slot_tags(grid, lang, mode), message)


# --- построение и события --------------------------------------------------


@dataclass
class Grid:
    """Компоненты одной сетки: состояние, ячейки, теги, значки, кнопка очистки."""

    state: gr.State
    slots: list
    tags: list
    pose_buttons: list
    sketch_buttons: list
    clear: gr.Button

    def targets(self, status) -> list:
        """Выходы записи в сетку — в порядке ``place``."""
        return [self.state, *self.slots, *self.tags, status]


def build_grid(localizer: Localizer, lang: str) -> Grid:
    """Колонка сетки: заголовок, пять рядов по две ячейки, «Очистить референсы».

    Строится внутри строки вкладки (``layout.RESULT_ROW``) рядом с полем, в
    рост которого она встаёт (см. style.css).
    """
    state = gr.State([])
    slots: list[gr.Image] = []
    tags: list[gr.Markdown] = []
    pose_buttons: list[gr.Button] = []
    sketch_buttons: list[gr.Button] = []
    # Каждая ячейка — отдельное поле изображения: картинку кладут кликом или
    # перетаскиванием, убирают её собственным крестиком. Ячейка принимает
    # только загрузку: веб-камера и буфер обмена в поле в пару сантиметров
    # добавили бы панель переключения источников крупнее самой ячейки.
    with gr.Column(min_width=0, elem_classes=[layout.REFS_COL]):
        localizer.bind(gr.Markdown(pick("references", lang)), value=("Референсы", "References"))
        for _row in range(REFERENCE_ROWS):
            with gr.Row(equal_height=True, elem_classes=[layout.REF_ROW]):
                for _column in range(REFERENCE_COLUMNS):
                    # Ячейка — картинка и под ней строка с тегом (см. slot_tags).
                    with gr.Column(min_width=0, elem_classes=[layout.REF_CELL]):
                        slots.append(gr.Image(
                            type="pil",
                            image_mode="RGB",
                            sources=["upload"],
                            label="",
                            show_label=False,
                            buttons=[],
                            # Заглушка — пробел нулевой ширины. Пустую строку и
                            # обычный пробел Gradio считает «не задано» и пишет
                            # свою «Перетащите изображение сюда - или - Нажмите
                            # для загрузки», которая в ячейку не влезает и
                            # обрезается (проверено снимком). U+200B пробельным не
                            # считается ни в Python, ни в JS: заглушка задана, но
                            # невидима, и в ячейке остаётся один значок загрузки.
                            placeholder=chr(0x200B),
                            elem_classes=[layout.REF_SLOT],
                        ))
                        # Значки «поза» и «эскиз» — поверх нижних углов
                        # картинки (CSS): под ячейкой места нет, в ней стоит
                        # тег. Окна — в reference_tools.
                        with gr.Row(elem_classes=[layout.REF_TOOLS]):
                            pose_buttons.append(gr.Button(
                                "🧍", size="sm", min_width=0, elem_classes=[layout.REF_POSE],
                            ))
                            sketch_buttons.append(gr.Button(
                                "✏️", size="sm", min_width=0, elem_classes=[layout.REF_SKETCH],
                            ))
                        tags.append(gr.Markdown(_NO_TAG, elem_classes=[layout.REF_TAG]))
        clear = localizer.bind(
            gr.Button(pick("reference_clear", lang), size="sm"),
            value=("Очистить референсы", "Clear references"),
        )
    return Grid(state, slots, tags, pose_buttons, sketch_buttons, clear)


def wire(grid: Grid, language, status, mode=None) -> None:
    """События сетки: ячейка изменена, «Очистить референсы», смена режима области.

    ``mode`` — компонент режима области (вкладка правки) или ``None``. Если он
    есть, он идёт входом перед языком и подписи пересчитываются при его смене.

    Индикатор выполнения скрыт: все десять ячеек — выходы обработчика, и на
    время запроса Gradio накрывает свои выходы индикатором — десять крутилок
    ради одной ячейки. Итог и так виден в подписях и в строке состояния.
    Замечание для проверяющего: текст «0.0s» в ``innerText`` ячейки есть
    всегда — индикатор лежит в разметке с нулевой прозрачностью, — и видимым
    таймером это не является.
    """
    context = [mode] if mode is not None else []
    targets = grid.targets(status)

    def slots_changed(*values):
        """Пересобирает референсы из всех десяти ячеек разом.

        Источник истины — то, что сейчас стоит в ячейках, а не история
        изменений: ячейку меняют в любом порядке и чистят крестиком, и
        дописывать к накопленному списку значило бы рано или поздно
        разойтись с экраном. Вызывается событием ``input``, то есть только
        на действие человека: программная запись в ячейку («Отправить в
        референсы», поза, эскиз) его не порождает, и ответ обработчика не
        зацикливается.
        """
        images, rest = values[:MAX_REFERENCES], values[MAX_REFERENCES:]
        *ctx, lang = rest
        cells = _grid(images)
        return (
            cells,
            # Сами ячейки не трогаются: картинка в них уже стоит, её туда
            # положил человек, а переслать её обратно — лишний круг в браузер.
            *(gr.update() for _ in cells),
            *slot_tags(cells, lang, ctx[0] if ctx else None),
            say("references_counted", lang, count=len(filled(cells)), total=MAX_REFERENCES),
        )

    def clear_references(*values):
        *ctx, lang = values
        cells = _grid([])
        return (
            cells,
            *(gr.update(value=None) for _ in cells),
            *slot_tags(cells, lang, ctx[0] if ctx else None),
            say("references_cleared", lang),
        )

    def retag(references, mode_value, lang):
        """Режим области сменился — теги референсов сдвигаются вместе с маской."""
        return slot_tags(references, lang, mode_value)

    for slot in grid.slots:
        slot.input(
            slots_changed, [*grid.slots, *context, language], targets,
            queue=False, show_progress="hidden",
        )
    grid.clear.click(
        clear_references, [*context, language], targets, queue=False, show_progress="hidden",
    )
    if mode is not None:
        mode.change(retag, [grid.state, mode, language], grid.tags, queue=False, show_progress="hidden")
