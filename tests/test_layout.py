"""Раскладка: связь Python и CSS не должна рваться молча.

Классы из ``ui/layout.py`` живут в двух местах сразу — в вызовах
компонентов и в ``ui/style.css``. Опечатка в любом из них ничего не
сломает на запуске: Gradio повесит класс, которого нет в стилях, браузер
пропустит правило, которому не на чем сработать, и интерфейс просто
поедет. Такое находят глазами через неделю, поэтому здесь это проверяется
сразу.

Высоты сверяются на осмысленность как выражения CSS: строка в ``height``
уходит в стиль как есть, и опечатка вроде ``100vw`` вместо ``100vh`` дала
бы холст шириной с экран по вертикали.
"""

from __future__ import annotations

import re

import pytest

from fooocus_qwen import config
from fooocus_qwen.ui import layout

CSS = (config.PROJECT_ROOT / "fooocus_qwen" / "ui" / "style.css").read_text(encoding="utf-8")
SOURCES = sorted((config.PROJECT_ROOT / "fooocus_qwen" / "ui").glob("*.py"))

# Имена собираются из модуля, а не перечисляются здесь: список, который
# ведут руками, отстаёт от кода ровно тогда, когда проверка нужнее всего —
# при добавлении класса. Соглашение простое: все классы раскладки начинаются
# с ``qs-``, и этого достаточно, чтобы узнать их в модуле.
CLASS_NAMES = {
    name: value
    for name, value in vars(layout).items()
    if isinstance(value, str) and value.startswith("qs-")
}
HEIGHTS = {
    name: value
    for name, value in vars(layout).items()
    if name.endswith("_HEIGHT") and isinstance(value, str)
}


@pytest.mark.parametrize("name,value", sorted(CLASS_NAMES.items()))
def test_every_layout_class_has_a_rule_in_the_stylesheet(name, value):
    assert f".{value}" in CSS, f"{name} = {value!r}: класс есть в Python, правила в CSS нет"


@pytest.mark.parametrize("name,value", sorted(CLASS_NAMES.items()))
def test_every_layout_class_is_actually_applied_somewhere(name, value):
    used = [p.name for p in SOURCES if f"layout.{name}" in p.read_text(encoding="utf-8")
            and p.name != "layout.py"]
    assert used, f"{name}: правило в CSS есть, но класс никому не повешен"


def test_the_stylesheet_invents_no_classes_of_its_own():
    """Каждый селектор ``.qs-*`` в CSS обязан приходить из ``layout.py``.

    Обратная сторона той же проверки: правило, написанное под класс,
    которого в Python нет, не сработает никогда и будет тихо вводить в
    заблуждение того, кто станет править вёрстку.
    """
    in_css = set(re.findall(r"\.(qs-[a-z-]+)", CSS))
    declared = set(CLASS_NAMES.values())
    assert in_css <= declared, f"в CSS есть классы мимо layout.py: {sorted(in_css - declared)}"


@pytest.mark.parametrize("name,value", sorted(HEIGHTS.items()))
def test_every_height_is_a_plausible_css_length(name, value):
    # Строка уходит в атрибут стиля как есть (см. докстроку height у
    # gr.Gallery), поэтому проверяется именно она, а не число.
    assert isinstance(value, str), f"{name}: высота должна быть выражением CSS, а не {type(value)}"
    assert value.startswith("clamp("), f"{name}: без clamp высота не имеет ни пола, ни потолка"
    assert "vh" in value, f"{name} = {value!r}: высота не зависит от окна — это и было неадаптивно"
    assert "vw" not in value, f"{name} = {value!r}: высота по ширине окна — почти наверняка опечатка"
    assert value.count("(") == value.count(")"), f"{name}: скобки не сходятся"


def test_no_tab_pins_a_pixel_height_any_more():
    """Числовые высоты вернули бы ровно ту неадаптивность, ради которой всё.

    Проверяются вызовы вида ``height=620`` — именно они стояли в трёх
    вкладках и не давали холсту считаться с размером окна.
    """
    offenders = []
    for path in SOURCES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"\bheight\s*=\s*\d", line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "высота в пикселях вместо выражения CSS:\n" + "\n".join(offenders)


def test_the_container_is_centred():
    """Ради чего правка и начиналась: max-width без margin прижимал всё влево."""
    container = re.search(r"\.gradio-container\s*\{([^}]*)\}", CSS)
    assert container, "правила для .gradio-container нет вовсе"
    body = container.group(1)
    assert "margin-left" in body and "auto" in body, "контейнер не центрирован"
    assert "max-width" in body, "ширина контейнера не ограничена"


def test_the_layout_reflows_on_narrow_windows():
    assert "@media" in CSS, "переносов по ширине окна нет — интерфейс остался неадаптивным"
    assert f".{layout.WORK_ROW}" in CSS.split("@media", 1)[1], (
        "рабочая строка не участвует в переносе, а именно она и ломается на узком окне"
    )


def test_the_stylesheet_gets_every_height_substituted():
    """Высоты живут в ``layout.py`` и подставляются в стиль при запуске.

    Они нужны в двух местах сразу: параметром ``height`` компонента и нижней
    границей в CSS, без которой пустой холст схлопывается в полоску. Держать
    одно и то же число в Python и в стиле — значит однажды поправить одно и
    забыть другое, поэтому стиль получает значение подстановкой. Незаменённый
    плейсхолдер браузер молча пропустит, и холст потеряет высоту.
    """
    gr = pytest.importorskip("gradio")  # noqa: F841 — app тянет gradio
    from fooocus_qwen.ui import app

    rendered = app.stylesheet()
    assert "{{" not in rendered, "в стиле остался незаменённый плейсхолдер"
    for name, value in HEIGHTS.items():
        if "{{" + name + "}}" in CSS:
            assert value in rendered, f"{name} не подставлено в стиль"


def test_every_height_placeholder_in_css_comes_from_layout():
    """Обратная сторона: плейсхолдер под несуществующее имя не заменится."""
    placeholders = set(re.findall(r"\{\{([A-Z_]+)\}\}", CSS))
    assert placeholders <= set(HEIGHTS), (
        f"в стиле есть плейсхолдеры мимо layout.py: {sorted(placeholders - set(HEIGHTS))}"
    )


def test_the_canvas_grows_and_the_side_panel_does_not():
    """Суть раскладки: ширина достаётся холсту, а не форме.

    Проверяется по самому стилю — правило можно случайно поменять местами,
    и интерфейс снова начнёт растягивать панель настроек на весь монитор.
    """
    canvas = re.search(r"\.qs-canvas\s*\{[^}]*\}", CSS, re.S)
    side = re.search(r"\.qs-side\s*\{[^}]*\}", CSS, re.S)
    assert canvas and side, "правила колонок пропали из стиля"
    assert "flex: 1 1 0" in canvas.group(0), "холст перестал тянуться"
    assert "flex: 0 0" in side.group(0), "панель снова растягивается вместе с холстом"
