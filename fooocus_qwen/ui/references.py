"""Сетка референсов: десять позиционных ячеек, их теги и запись в ячейку.

Общее для вкладки генерации (ячейки, «Отправить в референсы») и окон
инструментов ячейки (``reference_tools``: поза, эскиз) — поэтому отдельным
модулем: окна живут в своём модуле, а вкладка их встраивает, и держать
сетку во вкладке значило бы замкнуть импорты по кругу.
"""

from __future__ import annotations

import gradio as gr

from ..engine.generator import condition_slots
from .i18n import say

# Сетка референсов слева от результата: два столбца по пять ячеек, в рост
# поля результата. Десять — предел самой модели (карточка Qwen-Image-2.1), и
# сетка ровно его покрывает. Нумерация — по рядам: первый ряд — ячейки 1 и 2.
REFERENCE_ROWS = 5
REFERENCE_COLUMNS = 2
MAX_REFERENCES = REFERENCE_ROWS * REFERENCE_COLUMNS


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


# Пустая строка тега — неразрывный пробел, а не "": пустое поле схлопнулось
# бы по высоте, и ряды ячеек с подписью и без поехали бы относительно друг
# друга.
_NO_TAG = chr(0xA0)


def slot_tags(references, lang: str) -> list:
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
    captions = iter(_captions(filled(grid), lang))
    return [
        gr.update(value=_tag_markdown(next(captions)) if image is not None else _NO_TAG)
        for image in grid
    ]


def _tag_markdown(caption: str) -> str:
    return f"`{caption}`" if caption.startswith("<image") else caption


def place(references, index: int, image, lang: str, message: str) -> tuple:
    """Кладёт картинку в ячейку ``index``; выходы — как у ``reference_targets``.

    Выходы: состояние референсов, десять ячеек, десять тегов, строка
    состояния. Картинка уходит только в свою ячейку: остальные не трогаются,
    иначе каждая запись заново слала бы в браузер все десять изображений.
    """
    grid = _grid(references)
    grid[index] = image
    slots = [gr.update(value=image) if position == index else gr.update() for position in range(MAX_REFERENCES)]
    return (grid, *slots, *slot_tags(grid, lang), message)
