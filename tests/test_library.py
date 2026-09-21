"""Именованные пресеты промтов на диске."""

import pytest

from fooocus_qwen.prompting import library

PAYLOAD = {"prompt": "кот", "negative_prompt": "", "styles": ["sai-anime"], "seed": 7}


def test_saved_prompt_is_read_back(tmp_path):
    library.save_prompt("Мой кот", PAYLOAD, tmp_path)
    assert library.load_prompt("Мой кот", tmp_path) == PAYLOAD


def test_listing_is_sorted_and_shows_display_names(tmp_path):
    library.save_prompt("бета", PAYLOAD, tmp_path)
    library.save_prompt("альфа", PAYLOAD, tmp_path)
    assert library.list_prompts(tmp_path) == ["альфа", "бета"]


def test_saving_twice_overwrites(tmp_path):
    library.save_prompt("имя", PAYLOAD, tmp_path)
    library.save_prompt("имя", dict(PAYLOAD, seed=9), tmp_path)
    assert library.load_prompt("имя", tmp_path)["seed"] == 9
    assert library.list_prompts(tmp_path) == ["имя"]


def test_unsafe_characters_do_not_escape_the_directory(tmp_path):
    library.save_prompt("../../побег", PAYLOAD, tmp_path)
    assert list(tmp_path.glob("*.json"))
    assert not (tmp_path.parent.parent / "побег.json").exists()


def test_delete_removes_the_preset(tmp_path):
    library.save_prompt("имя", PAYLOAD, tmp_path)
    assert library.delete_prompt("имя", tmp_path) is True
    assert library.list_prompts(tmp_path) == []
    assert library.delete_prompt("имя", tmp_path) is False


def test_missing_preset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        library.load_prompt("нет такого", tmp_path)


def test_empty_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        library.save_prompt("   ", PAYLOAD, tmp_path)
