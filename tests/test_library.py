"""Именованные пресеты промтов на диске."""

import json

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
    (tmp_path / "плохой.json").write_text("{это не json", encoding="utf-8")
    assert library.list_prompts(tmp_path) == ["хороший"]


def test_list_prompts_skips_non_dict_json_file(tmp_path):
    """list_prompts должна пропустить JSON, который не является словарём, и продолжить."""
    library.save_prompt("хороший", PAYLOAD, tmp_path)
    (tmp_path / "массив.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert library.list_prompts(tmp_path) == ["хороший"]


def test_load_prompt_raises_on_non_dict_file(tmp_path):
    """load_prompt должна вызвать ошибку с понятным сообщением о повреждённом пресете."""
    (tmp_path / "bad.json").write_text('"просто строка"', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        library.load_prompt("bad", tmp_path)
    assert "bad.json" in str(exc_info.value)


def test_collision_different_display_names_stored_separately(tmp_path):
    """Два пресета с разными дисплей-именами, но одинаковым sanitised именем, сохраняются отдельно."""
    library.save_prompt(".", PAYLOAD, tmp_path)
    library.save_prompt("..", dict(PAYLOAD, seed=99), tmp_path)
    names = library.list_prompts(tmp_path)
    assert "." in names
    assert ".." in names
    assert len(names) == 2
    assert library.load_prompt(".", tmp_path)["seed"] == 7
    assert library.load_prompt("..", tmp_path)["seed"] == 99


def test_delete_removes_correct_preset_in_collision(tmp_path):
    """delete_prompt должна удалить ровно тот пресет, который попросили, несмотря на коллизию."""
    library.save_prompt("Портрет/студия", {"prompt": "первый"}, tmp_path)
    library.save_prompt("Портрет:студия", {"prompt": "второй"}, tmp_path)
    assert library.delete_prompt("Портрет:студия", tmp_path) is True
    remaining = library.list_prompts(tmp_path)
    assert remaining == ["Портрет/студия"]
    loaded = library.load_prompt("Портрет/студия", tmp_path)
    assert loaded["prompt"] == "первый"
    with pytest.raises(FileNotFoundError):
        library.load_prompt("Портрет:студия", tmp_path)


def test_load_prompt_corrupted_json_raises_damaged_not_notfound(tmp_path):
    """load_prompt должна сообщить 'повреждён', а не 'не найден', для испорченного JSON."""
    (tmp_path / "test.json").write_text('{"__name__": "test", "data": {это не json', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        library.load_prompt("test", tmp_path)
    error_msg = str(exc_info.value).lower()
    assert "повреждён" in error_msg or "повреж" in error_msg


def test_load_prompt_non_dict_json_raises_damaged_not_notfound(tmp_path):
    """load_prompt должна сообщить 'повреждён', а не 'не найден', для не-словаря в JSON."""
    (tmp_path / "test.json").write_text('["не", "словарь"]', encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        library.load_prompt("test", tmp_path)
    error_msg = str(exc_info.value).lower()
    assert "повреждён" in error_msg or "повреж" in error_msg


def test_load_prompt_genuinely_missing_raises_notfound(tmp_path):
    """load_prompt должна сообщить 'не найден' для полностью отсутствующего пресета."""
    with pytest.raises(FileNotFoundError) as exc_info:
        library.load_prompt("совсем_нет", tmp_path)
    assert "не найден" in str(exc_info.value).lower()


def test_save_prompt_does_not_duplicate_when_corrupted_file_exists(tmp_path):
    """save_prompt должна обновить существующий пресет, а не создать дубликат.

    Сценарий: preset.json повреждён, preset_2.json содержит ".".
    Вызываем save_prompt(".", {...}), должна перезаписать preset_2.json.
    """
    # Создаём повреждённый preset.json
    (tmp_path / "preset.json").write_text("{corrupted", encoding="utf-8")

    # Создаём валидный preset_2.json с display-имнем "."
    preset_2_content = {"__name__": ".", "prompt": "старая версия"}
    (tmp_path / "preset_2.json").write_text(json.dumps(preset_2_content), encoding="utf-8")

    # Сохраняем новый пресет с именем "."
    new_payload = {"prompt": "новая версия", "seed": 42}
    result_path = library.save_prompt(".", new_payload, tmp_path)

    # Должна перезаписать preset_2.json
    assert result_path == tmp_path / "preset_2.json"

    # list_prompts должна вернуть ровно один "."
    names = library.list_prompts(tmp_path)
    assert names == ["."]

    # load_prompt должна вернуть новое содержимое
    loaded = library.load_prompt(".", tmp_path)
    assert loaded["prompt"] == "новая версия"
    assert loaded["seed"] == 42

    # Повреждённый файл должен остаться нетронутым
    assert (tmp_path / "preset.json").read_text(encoding="utf-8") == "{corrupted"


def _write_duplicate_display_name(tmp_path):
    """Два файла на диске с одинаковым __name__ — состояние, которое больше не
    может возникнуть через публичный API, но могло появиться до раунда 4
    (например, было создано вручную или осталось от старой версии кода).
    """
    (tmp_path / "preset.json").write_text(
        json.dumps({"__name__": ".", "v": "ПЕРВЫЙ"}), encoding="utf-8"
    )
    (tmp_path / "preset_2.json").write_text(
        json.dumps({"__name__": ".", "v": "ВТОРОЙ"}), encoding="utf-8"
    )


def test_load_prompt_returns_first_slot_on_duplicate_display_name(tmp_path):
    """При двух файлах с одинаковым display-именем побеждает первый слот по порядку кандидатов."""
    _write_duplicate_display_name(tmp_path)
    assert library.load_prompt(".", tmp_path)["v"] == "ПЕРВЫЙ"


def test_delete_prompt_removes_first_slot_on_duplicate_display_name(tmp_path):
    """delete_prompt должна удалить первый слот, а не последний, при коллизии display-имён."""
    _write_duplicate_display_name(tmp_path)
    assert library.delete_prompt(".", tmp_path) is True
    assert not (tmp_path / "preset.json").exists()
    assert (tmp_path / "preset_2.json").exists()


def test_save_prompt_overwrites_first_slot_on_duplicate_display_name(tmp_path):
    """save_prompt должна перезаписать первый слот и не создавать новый файл."""
    _write_duplicate_display_name(tmp_path)
    library.save_prompt(".", {"v": "НОВЫЙ"}, tmp_path)

    assert json.loads((tmp_path / "preset.json").read_text(encoding="utf-8"))["v"] == "НОВЫЙ"
    assert json.loads((tmp_path / "preset_2.json").read_text(encoding="utf-8"))["v"] == "ВТОРОЙ"
    assert not (tmp_path / "preset_3.json").exists()


def test_save_prompt_avoids_corrupted_slot(tmp_path):
    """save_prompt должна пропустить повреждённый слот и использовать первый свободный.

    Сценарий: preset.json повреждён, preset_2.json имеет другое display-имя.
    Сохраняем новый пресет ".", должна использовать первый свободный слот (preset_3.json).
    """
    # Создаём повреждённый preset.json
    corrupted_content = "{bad json"
    (tmp_path / "preset.json").write_text(corrupted_content, encoding="utf-8")

    # Создаём валидный файл с другим display-имнем
    (tmp_path / "preset_2.json").write_text(
        json.dumps({"__name__": "что-то", "prompt": "первое"}),
        encoding="utf-8"
    )

    # Сохраняем новый пресет "." (которая обезвреживается в "preset")
    new_payload = {"prompt": "новый пресет"}
    result_path = library.save_prompt(".", new_payload, tmp_path)

    # Должна использовать первый свободный слот preset_3.json
    assert result_path == tmp_path / "preset_3.json"

    # Повреждённый файл должен остаться нетронутым
    assert (tmp_path / "preset.json").read_text(encoding="utf-8") == corrupted_content

    # Загружаем новый пресет - должен работать
    loaded = library.load_prompt(".", tmp_path)
    assert loaded["prompt"] == "новый пресет"
