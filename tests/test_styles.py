"""Загрузка каталога стилей и подстановка промта."""

from fooocus_qwen import config
from fooocus_qwen.prompting import styles as st


def test_catalogue_contains_all_downloaded_styles():
    catalogue = st.load_styles(config.STYLES_DIR)
    assert len(catalogue) == 277
    assert "sai-anime" in catalogue
    assert "Fooocus Sharp" in catalogue


def test_single_style_substitutes_prompt():
    catalogue = {"s": st.Style(name="s", prompt="anime artwork {prompt} . vibrant", negative_prompt="photo")}
    prompt, negative = st.apply_styles("кот", "", ["s"], catalogue)
    assert prompt == "anime artwork кот . vibrant"
    assert negative == "photo"


def test_two_styles_are_joined_and_negatives_accumulate():
    catalogue = {
        "a": st.Style(name="a", prompt="A {prompt}", negative_prompt="na"),
        "b": st.Style(name="b", prompt="B {prompt}", negative_prompt="nb"),
    }
    prompt, negative = st.apply_styles("кот", "", ["a", "b"], catalogue)
    assert prompt == "A кот, B кот"
    assert negative == "na, nb"


def test_style_without_template_only_adds_negative():
    # Три стиля Fooocus содержат пустой prompt и существуют ради негатива.
    catalogue = {"n": st.Style(name="n", prompt="", negative_prompt="deformed")}
    prompt, negative = st.apply_styles("кот", "мутный", ["n"], catalogue)
    assert prompt == "кот"
    assert negative == "мутный, deformed"


def test_user_negative_comes_first():
    catalogue = {"s": st.Style(name="s", prompt="S {prompt}", negative_prompt="ns")}
    _, negative = st.apply_styles("кот", "свой негатив", ["s"], catalogue)
    assert negative == "свой негатив, ns"


def test_unknown_style_is_ignored_not_fatal():
    # Пресет промта мог быть сохранён на другой сборке с иным набором стилей.
    catalogue = {"s": st.Style(name="s", prompt="S {prompt}", negative_prompt="")}
    prompt, _ = st.apply_styles("кот", "", ["s", "нет такого"], catalogue)
    assert prompt == "S кот"


def test_empty_selection_returns_input_unchanged():
    assert st.apply_styles("кот", "мутный", [], {}) == ("кот", "мутный")
