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

CLASS_NAMES = {
    name: getattr(layout, name)
    for name in ("LANG", "WORK_ROW", "PROMPT_BAR", "DROP_ZONE", "STATUS")
}
HEIGHTS = {
    name: getattr(layout, name)
    for name in ("CANVAS_HEIGHT", "BROWSE_HEIGHT", "PREVIEW_HEIGHT", "STRIP_HEIGHT")
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
