"""Разбор ответа переписывателя промтов и сборка сообщения пользователя."""

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
