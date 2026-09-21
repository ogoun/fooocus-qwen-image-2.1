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


def test_list_prompts_skips_corrupt_json_file(tmp_path):
    """list_prompts должна пропустить повреждённый JSON и продолжить со следующего файла."""
    library.save_prompt("хороший", PAYLOAD, tmp_path)
    # Создаём файл с невалидным JSON
    (tmp_path / "плохой.json").write_text("{это не json", encoding="utf-8")

    # list_prompts должна вернуть только хороший пресет
    assert library.list_prompts(tmp_path) == ["хороший"]


def test_list_prompts_skips_non_dict_json_file(tmp_path):
    """list_prompts должна пропустить JSON, который не является словарём, и продолжить."""
    library.save_prompt("хороший", PAYLOAD, tmp_path)
    # Создаём файл с валидным JSON, но это список, не словарь
    (tmp_path / "массив.json").write_text("[1, 2, 3]", encoding="utf-8")

    # list_prompts должна вернуть только хороший пресет
    assert library.list_prompts(tmp_path) == ["хороший"]


def test_load_prompt_raises_on_non_dict_file(tmp_path):
    """load_prompt должна вызвать ошибку с понятным сообщением о повреждённом пресете."""
    # Создаём файл с валидным JSON, но это строка, не словарь
    (tmp_path / "bad.json").write_text('"просто строка"', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        library.load_prompt("bad", tmp_path)

    # Сообщение должно содержать путь файла
    assert "bad.json" in str(exc_info.value)


def test_collision_different_display_names_stored_separately(tmp_path):
    """Два пресета с разными дисплей-именами, но одинаковым sanitised именем, сохраняются отдельно."""
    # '.' и '..' обе обезвреживаются в 'preset'
    library.save_prompt(".", PAYLOAD, tmp_path)
    library.save_prompt("..", dict(PAYLOAD, seed=99), tmp_path)

    # Оба должны быть в списке с их оригинальными дисплей-именами
    names = library.list_prompts(tmp_path)
    assert "." in names
    assert ".." in names
    assert len(names) == 2

    # Оба должны загружаться и иметь правильные параметры
    assert library.load_prompt(".", tmp_path)["seed"] == 7
    assert library.load_prompt("..", tmp_path)["seed"] == 99
