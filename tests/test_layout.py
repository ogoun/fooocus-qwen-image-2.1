"""Раскладка: связь Python и CSS не должна рваться молча.

Имена классов живут в двух местах сразу — в вызовах компонентов и в
``ui/style.css``. Опечатка в любом из них ничего не сломает на запуске:
Gradio повесит класс, которого нет в стилях, браузер пропустит правило,
которому не на чем сработать, и интерфейс просто поедет. Такое находят
глазами через неделю, поэтому здесь это проверяется сразу.

Сама геометрия целиком в стиле, и проверяется тоже здесь — не пиксели (их
считает браузер), а устройство правил: что холст зависит и от окна, и от
своей ширины, что ширина достаётся холсту, а не форме. Перепутать эти
правила местами легко, а увидеть последствия — только на большом мониторе
с загруженной картинкой.
"""

from __future__ import annotations

import re

import pytest

from fooocus_qwen import config
from fooocus_qwen.ui import layout

CSS = (config.PROJECT_ROOT / "fooocus_qwen" / "ui" / "style.css").read_text(encoding="utf-8")
SOURCES = sorted((config.PROJECT_ROOT / "fooocus_qwen" / "ui").glob("*.py"))

# Имена собираются из модуля, а не перечисляются здесь: список, который ведут
# руками, отстаёт от кода ровно тогда, когда проверка нужнее всего — при
# добавлении класса. Соглашение простое: все классы раскладки начинаются с
# ``qs-``, и этого достаточно, чтобы узнать их в модуле.
CLASS_NAMES = {
    name: value
    for name, value in vars(layout).items()
    if isinstance(value, str) and value.startswith("qs-")
}


def rule(selector: str) -> str:
    """Тело правила для селектора — то, что между фигурными скобками."""
    found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", CSS, re.S)
    assert found, f"правило {selector} пропало из стиля"
    return found.group(1)


@pytest.mark.parametrize("name,value", sorted(CLASS_NAMES.items()))
def test_every_layout_class_has_a_rule_in_the_stylesheet(name, value):
    assert f".{value}" in CSS, f"{name} = {value!r}: класс есть в Python, правила в CSS нет"


@pytest.mark.parametrize("name,value", sorted(CLASS_NAMES.items()))
def test_every_layout_class_is_actually_applied_somewhere(name, value):
    used = [
        p.name
        for p in SOURCES
        if f"layout.{name}" in p.read_text(encoding="utf-8") and p.name != "layout.py"
    ]
    assert used, f"{name}: правило в CSS есть, но класс никому не повешен"


def test_the_stylesheet_invents_no_classes_of_its_own():
    """Правило под класс, которого в Python нет, не сработает никогда."""
    in_css = set(re.findall(r"\.(qs-[a-z-]+)", CSS))
    assert in_css <= set(CLASS_NAMES.values()), (
        f"в CSS есть классы мимо layout.py: {sorted(in_css - set(CLASS_NAMES.values()))}"
    )


def test_no_sizes_are_left_in_python():
    """Геометрия живёт в стиле, и только там.

    Пока высоты задавались параметром ``height`` компонента, выразить ими
    можно было лишь зависимость от окна. Зависимость от ширины колонки —
    главную — выразить было нечем, и в узкой колонке картинка оставалась
    плавать в пустом поле.
    """
    for source in SOURCES:
        text = source.read_text(encoding="utf-8")
        assert "height=layout." not in text, f"{source.name}: высота снова задаётся из Python"
    assert not [name for name in vars(layout) if name.endswith("_HEIGHT")], (
        "в layout.py вернулись размеры — их место в style.css"
    )


def test_the_board_depends_on_the_window_and_on_its_own_width():
    """Обе зависимости сразу — иначе холст снова начнёт врать по высоте.

    Только окно — и в узкой колонке остаётся высокое пустое поле вокруг
    сжавшейся картинки. Только пропорция — и на широком мониторе холст
    вылезет за пределы экрана.
    """
    board = rule(".qs-board")
    assert "aspect-ratio" in board, "пропорция пропала: высота перестала зависеть от ширины"
    assert "max-height" in board, "потолок по окну пропал: холст вылезет за экран"
    assert "min-height" in board, "нижняя граница пропала: пустой холст схлопнется"
    assert "height: auto" in board, (
        "без height: auto инлайновая высота Gradio перебивает пропорцию"
    )


def test_the_canvas_grows_and_the_side_panel_does_not():
    """Суть раскладки: ширина достаётся холсту, а не форме."""
    assert "flex: 1 1 0" in rule(".qs-canvas"), "холст перестал тянуться"
    assert "flex: 0 0" in rule(".qs-side"), "панель снова растягивается вместе с холстом"


def test_the_browse_grid_keeps_the_window_height():
    """Сетка галереи — список, а не кадр: пропорция ей ни к чему."""
    browse = rule(".qs-browse")
    assert "--qs-browse-height" in browse
    assert "aspect-ratio" not in browse, "сетке миниатюр навязали форму одного кадра"


def test_every_variable_is_declared_and_used():
    """Переменная без объявления молча превращается в пустоту."""
    declared = set(re.findall(r"(--qs-[a-z-]+)\s*:", CSS))
    used = set(re.findall(r"var\((--qs-[a-z-]+)", CSS))
    assert used <= declared, f"используются необъявленные переменные: {sorted(used - declared)}"
    unused = declared - used
    assert not unused, f"объявлены, но не используются: {sorted(unused)}"


def test_the_two_boards_split_when_they_stop_fitting():
    """Ниже порога холсты обязаны вставать друг под друга.

    Число держится в одном месте — в самом правиле: `@media` переменных не
    понимает, и вынести порог в `:root` значило бы завести второе место, где
    его можно забыть поправить.
    """
    media = re.search(r"@media \(max-width: (\d+)px\)\s*\{\s*\.qs-boards", CSS)
    assert media, "правило переноса двух холстов пропало"
    assert 1200 <= int(media.group(1)) <= 2200, (
        f"порог {media.group(1)} px выглядит случайным: рядом холсты помещаются "
        "примерно от 1750, и сильно уходить от этого незачем"
    )
    assert ".qs-preview.qs-board" in CSS, "под редактором результат должен занимать меньше высоты"


def test_no_variables_are_redefined_inside_media_queries():
    """Внутри ``@media`` переменные в ``:root`` не работают — и молча.

    Gradio переписывает наши селекторы, приписывая к ним
    ``.gradio-container … .contain``, и внутри ``@media`` в браузер попадает
    только переписанная копия. Для обычного класса приписка безвредна, а
    ``:root`` превращается в ``.contain :root`` — селектор, которому нечему
    соответствовать. Проверено на живом приложении: панель оставалась 380 px
    там, где правило обещало 440.
    """
    for block in re.findall(r"@media[^{]*\{(.*?)\n\}", CSS, re.S):
        assert ":root" not in block, (
            "внутри @media переопределяется :root — Gradio это правило потеряет; "
            "переопределяйте сам класс"
        )
