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


def test_json_without_rewritten_prompt_is_an_empty_answer_not_prose():
    """JSON без текста промтом не становится: раньше промтом уходил сам JSON."""
    for text in ('{"wh_ratio": "3:2"}', '{"rewritten_prompt": "", "ratio_follow": "<image1>"}'):
        result = boost.parse_response(text)
        assert result.prompt == ""
    assert boost.parse_response('{"rewritten_prompt": "", "wh_ratio": "3:2"}').wh_ratio == "3:2"


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


# --- язык описания: правило (A) системных промтов ---------------------------


class SequenceClient(RecordingClient):
    """Отвечает по очереди из списка — для проверки повтора."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def complete(self, system, user, images=None, temperature=0.3, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "images": images})
        return self.answers.pop(0)


def _prompts(tmp_path):
    for name in ("system_prompt_t2i.txt", "system_prompt_edit.txt"):
        (tmp_path / name).write_text("rewrite", encoding="utf-8")
    return tmp_path


def test_chinese_prose_for_an_english_instruction_is_off_language():
    assert boost.off_language("replace the sky with a sunset", "将图像中的天空区域替换为日落景象")


def test_a_chinese_instruction_may_get_a_chinese_description():
    assert not boost.off_language("把天空换成日落", "将图像中的天空区域替换为日落景象")


def test_quoted_text_is_rendered_text_not_the_description_language():
    """Надпись в кавычках — для картинки; её язык решает правило (B), не (A)."""
    assert not boost.off_language("add a sign saying sale", 'Add a shop sign reading "特价" above the door')


def test_russian_prose_for_a_russian_instruction_is_off_language():
    """Описание — по-английски для любой инструкции, кроме китайской (правило A)."""
    assert boost.off_language("сделай небо закатным", "Заменить текущее небо на закатное")
    assert not boost.off_language("сделай небо закатным", "Replace the sky with a warm sunset")


def test_russian_text_in_quotes_is_rendered_text():
    assert not boost.off_language("добавь вывеску «Распродажа»", 'Add a shop sign reading "Распродажа"')
    assert not boost.off_language("добавь вывеску", "Add a shop sign reading «Распродажа»")


def test_latin_accents_count_as_english_script():
    """Заимствования вроде «café» и «naïve» — латиница, не чужой язык."""
    assert not boost.off_language("кафе", "A cozy café with a naïve mural, 25 °C")


def test_the_blind_note_keeps_the_english_rule():
    """Пометка напоминает правило (A), а не велит писать на языке инструкции:
    прежняя редакция давала 75 % русских ответов (boost_language.py)."""
    assert "language of the instruction" not in boost.BLIND_NOTE
    assert "English unless the instruction is in Chinese" in boost.BLIND_NOTE


def test_an_empty_answer_is_asked_again(tmp_path):
    client = SequenceClient([
        '{"rewritten_prompt": "", "ratio_follow": "<image1>"}',
        '{"rewritten_prompt": "A cat in glasses reads a newspaper"}',
    ])
    result = boost.boost(client, "кот в очках", mode=boost.MODE_T2I, prompt_dir=_prompts(tmp_path))
    assert result.prompt == "A cat in glasses reads a newspaper" and len(client.calls) == 2


def test_two_empty_answers_keep_the_original_prompt(tmp_path):
    client = SequenceClient(['{"rewritten_prompt": ""}', "   "])
    result = boost.boost(client, "кот в очках", mode=boost.MODE_T2I, prompt_dir=_prompts(tmp_path))
    assert result.prompt == "кот в очках" and len(client.calls) == 2


def test_an_off_language_answer_is_asked_again(tmp_path):
    client = SequenceClient([
        '{"rewritten_prompt": "将天空替换为日落"}',
        '{"rewritten_prompt": "Replace the sky with a sunset"}',
    ])
    result = boost.boost(client, "replace the sky with a sunset", mode=boost.MODE_T2I, prompt_dir=_prompts(tmp_path))
    assert result.prompt == "Replace the sky with a sunset"
    assert len(client.calls) == 2


def test_the_second_answer_is_taken_whatever_it_is(tmp_path):
    """Повтор один: задерживать генерацию бесконечными попытками хуже."""
    client = SequenceClient(['{"rewritten_prompt": "日落"}', '{"rewritten_prompt": "日落二"}'])
    result = boost.boost(client, "sunset", mode=boost.MODE_T2I, prompt_dir=_prompts(tmp_path))
    assert result.prompt == "日落二" and len(client.calls) == 2


def test_the_blind_note_goes_only_to_a_request_without_its_images(tmp_path):
    client = RecordingClient()
    picture = [Image.new("RGB", (8, 8))]
    boost.boost(client, "make it red", mode=boost.MODE_EDIT, prompt_dir=_prompts(tmp_path), references=picture)
    boost.boost(
        client, "make it red", mode=boost.MODE_EDIT, prompt_dir=_prompts(tmp_path),
        references=picture, send_images=False,
    )
    boost.boost(client, "make it red", mode=boost.MODE_T2I, prompt_dir=_prompts(tmp_path), references=picture)
    with_images, without_images, t2i = client.calls
    assert with_images["images"] and boost.BLIND_NOTE not in with_images["user"]
    assert not without_images["images"] and without_images["user"].endswith(boost.BLIND_NOTE)
    assert not t2i["images"] and boost.BLIND_NOTE not in t2i["user"], "переписывателю T2I картинки и не нужны"


def test_reasoning_before_the_answer_is_ignored():
    """Официальные переписыватели PE думают в <think> перед JSON — и в
    рассуждениях бывают фигурные скобки, которые путали бы поиск ответа."""
    text = (
        '<think>The user wants {a cat}; ratio maybe {"wh_ratio": "1:1"}?</think>\n'
        '{"rewritten_prompt": "a ginger cat on a windowsill", "wh_ratio": "3:2"}'
    )
    result = boost.parse_response(text)
    assert result.prompt == "a ginger cat on a windowsill"
    assert result.wh_ratio == "3:2"

