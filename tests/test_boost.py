"""Разбор ответа переписывателя промтов и сборка сообщения пользователя."""

import pytest
from PIL import Image

from fooocus_qwen.prompting import boost


def test_plain_json_is_parsed():
    result = boost.parse_response('{"rewritten_prompt": "a cat", "wh_ratio": "3:2"}')
    assert result.prompt == "a cat"
    assert result.wh_ratio == "3:2"
    assert result.ratio_follow is None


def test_edit_response_carries_ratio_follow():
    text = '{"rewritten_prompt": "swap", "wh_ratio": "", "ratio_follow": "<image1>"}'
    result = boost.parse_response(text)
    assert result.ratio_follow == "<image1>"
    assert result.wh_ratio is None


def test_json_wrapped_in_markdown_fence_is_parsed():
    # Обычные модели любят оборачивать ответ в блок кода вопреки инструкции.
    text = 'Вот результат:\n```json\n{"rewritten_prompt": "a dog", "wh_ratio": "1:1"}\n```\n'
    result = boost.parse_response(text)
    assert result.prompt == "a dog"
    assert result.wh_ratio == "1:1"


def test_non_json_falls_back_to_raw_text():
    result = boost.parse_response("просто текст без json")
    assert result.prompt == "просто текст без json"
    assert result.wh_ratio is None


def test_empty_answer_yields_empty_prompt():
    result = boost.parse_response("   ")
    assert result.prompt == ""


def test_json_without_rewritten_prompt_falls_back_to_raw():
    result = boost.parse_response('{"wh_ratio": "3:2"}')
    assert result.prompt == '{"wh_ratio": "3:2"}'


def test_user_message_without_references_has_no_tags():
    message = boost.build_user_message("кот в шляпе", reference_count=1)
    assert "<image1>" not in message
    assert "кот в шляпе" in message


def test_user_message_lists_tags_for_several_references():
    message = boost.build_user_message("объедини их", reference_count=3)
    assert "<image1>" in message and "<image2>" in message and "<image3>" in message
    assert "<image4>" not in message


class RecordingClient:
    """Подставной клиент: запоминает, с чем его позвали."""

    def __init__(self, answer='{"rewritten_prompt": "x", "wh_ratio": "1:1"}'):
        self.answer = answer
        self.calls = []

    def complete(self, system, user, images=None, temperature=0.3, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "images": images})
        return self.answer


def test_boost_sends_references_only_in_edit_mode(tmp_path):
    """Референсы отправляются модели только в режиме EDIT, не в T2I."""
    # Подготовка системных промтов
    t2i_prompt = tmp_path / "system_prompt_t2i.txt"
    t2i_prompt.write_text("system t2i", encoding="utf-8")
    edit_prompt = tmp_path / "system_prompt_edit.txt"
    edit_prompt.write_text("system edit", encoding="utf-8")

    # Создаем тестовые изображения (1x1 пиксель)
    image = Image.new("RGB", (1, 1))
    references = [image, image]

    # Тест T2I режима: должен отправить images=None
    client_t2i = RecordingClient()
    result_t2i = boost.boost(
        client_t2i,
        "a cat",
        mode=boost.MODE_T2I,
        prompt_dir=tmp_path,
        references=references,
    )
    assert len(client_t2i.calls) == 1
    assert client_t2i.calls[0]["images"] is None
    assert result_t2i.prompt == "x"

    # Тест EDIT режима: должен отправить ровно те же изображения
    client_edit = RecordingClient()
    result_edit = boost.boost(
        client_edit,
        "a cat",
        mode=boost.MODE_EDIT,
        prompt_dir=tmp_path,
        references=references,
    )
    assert len(client_edit.calls) == 1
    assert client_edit.calls[0]["images"] == references
    assert result_edit.prompt == "x"


def test_boost_rereads_system_prompt_from_disk(tmp_path):
    """Системный промт перечитывается с диска при каждом вызове, не кэшируется."""
    prompt_file = tmp_path / "system_prompt_t2i.txt"
    prompt_file.write_text("first version", encoding="utf-8")

    client = RecordingClient()

    # Первый вызов
    boost.boost(client, "a cat", mode=boost.MODE_T2I, prompt_dir=tmp_path)
    assert client.calls[0]["system"] == "first version"

    # Перезаписываем файл
    prompt_file.write_text("second version", encoding="utf-8")
    client.calls.clear()

    # Второй вызов должен прочитать новый текст
    boost.boost(client, "a dog", mode=boost.MODE_T2I, prompt_dir=tmp_path)
    assert client.calls[0]["system"] == "second version"


def test_boost_raises_file_not_found_for_missing_prompt(tmp_path):
    """При отсутствии файла системного промта выбрасывается FileNotFoundError."""
    client = RecordingClient()

    # tmp_path пуст, системные промты не существуют
    with pytest.raises(FileNotFoundError, match="system_prompt_t2i"):
        boost.boost(client, "a cat", mode=boost.MODE_T2I, prompt_dir=tmp_path)
