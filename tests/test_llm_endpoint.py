"""Разбор llm_endpoint.txt во всех вариантах свободного формата."""

import pytest

from fooocus_qwen.llm import endpoint as ep


def test_plain_host_gets_scheme_and_default_port():
    result = ep.parse_endpoint_file("llama.cpp\n192.0.2.10\ntoken=abc\n")
    assert result.backend == "llama.cpp"
    assert result.base_url == "http://192.0.2.10:8000"
    assert result.token == "abc"


def test_explicit_port_is_kept():
    result = ep.parse_endpoint_file("192.0.2.10:1234\n")
    assert result.base_url == "http://192.0.2.10:1234"


def test_explicit_scheme_is_kept_and_https_port_not_forced():
    result = ep.parse_endpoint_file("https://api.example.com\n")
    assert result.base_url == "https://api.example.com"


def test_keyword_form_and_arbitrary_line_order():
    text = "token: SECRET\n# комментарий\nurl = http://10.0.0.5:9000\nvllm\n"
    result = ep.parse_endpoint_file(text)
    assert result.base_url == "http://10.0.0.5:9000"
    assert result.token == "SECRET"
    assert result.backend == "vllm"


def test_api_key_alias_is_accepted():
    assert ep.parse_endpoint_file("host\napi_key=k1\n").token == "k1"


def test_missing_host_is_an_error():
    with pytest.raises(ValueError):
        ep.parse_endpoint_file("token=only\n")


def test_urls_are_built_without_double_slashes():
    result = ep.parse_endpoint_file("http://host:8000/\n")
    assert result.chat_url == "http://host:8000/v1/chat/completions"
    assert result.models_url == "http://host:8000/v1/models"


def test_the_token_never_appears_in_repr():
    """Объект адреса попадает в журналы — токен не должен уехать вместе с ним."""
    endpoint = ep.parse_endpoint_file("192.0.2.10:8000\ntoken=repr-secret\n")
    assert endpoint.token == "repr-secret"
    assert "repr-secret" not in repr(endpoint) and "repr-secret" not in str(endpoint)
