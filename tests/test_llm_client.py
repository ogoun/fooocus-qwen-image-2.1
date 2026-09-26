"""Клиент проверяется на настоящем HTTP-сервере в потоке, без заглушек сети."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from fooocus_qwen.llm.client import LlmClient, LlmError, LlmImagesRejected
from fooocus_qwen.llm.endpoint import LlmEndpoint
from fooocus_qwen.prompting import boost

RECEIVED: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # тишина в выводе тестов
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply({"data": [{"id": "qwen3"}, {"id": "llama"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        RECEIVED.append({"body": body, "auth": self.headers.get("Authorization")})
        self._reply({"choices": [{"message": {"content": "rewritten prompt"}}]})

    def _reply(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    RECEIVED.clear()
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    # shutdown() останавливает цикл serve_forever, но не закрывает слушающий
    # сокет — без этого он утекает и порождает ResourceWarning на выходе.
    httpd.server_close()
    thread.join(timeout=5)


def test_ping_returns_model_names(server):
    client = LlmClient(LlmEndpoint(base_url=server))
    assert client.ping() == ["qwen3", "llama"]


def test_complete_sends_system_and_user_messages(server):
    client = LlmClient(LlmEndpoint(base_url=server, token="SECRET"), model="qwen3")
    answer = client.complete("системный", "пользовательский")

    assert answer == "rewritten prompt"
    body = RECEIVED[0]["body"]
    assert body["model"] == "qwen3"
    assert body["messages"][0] == {"role": "system", "content": "системный"}
    assert body["messages"][1]["content"] == "пользовательский"
    assert RECEIVED[0]["auth"] == "Bearer SECRET"


def test_images_are_sent_as_data_urls(server):
    client = LlmClient(LlmEndpoint(base_url=server))
    client.complete("س", "опиши", images=[Image.new("RGB", (8, 8), "red")])

    content = RECEIVED[0]["body"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_thinking_is_disabled():
    # Рассуждения вслух добавляют секунды и не нужны для переписывания промта.
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1"))
    body = client._build_body("s", "u", None, 0.3, 100)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_unreachable_server_raises_llm_error():
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1"), timeout=0.5)
    with pytest.raises(LlmError):
        client.ping()


# --- токен вне latin-1 (например, неотредактированный кириллический
# плейсхолдер из шаблона llm_endpoint.txt) обязан давать читаемую ошибку, а не
# голый UnicodeEncodeError из http.client ---


def test_non_latin1_token_raises_llm_error_from_ping_with_a_readable_message():
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1", token="ЗАМЕНИТЕ_НА_СВОЙ_ТОКЕН"))
    with pytest.raises(LlmError, match="[Тт]окен"):
        client.ping()


def test_non_latin1_token_raises_llm_error_from_complete_the_same_way():
    # ping() и complete() обязаны пропускать одни и те же ошибки — иначе
    # AI-буст (через complete) и «Проверить связь» (через ping) расходились бы
    # в поведении на одном и том же битом токене.
    client = LlmClient(LlmEndpoint(base_url="http://127.0.0.1:1", token="ЗАМЕНИТЕ_НА_СВОЙ_ТОКЕН"))
    with pytest.raises(LlmError, match="[Тт]окен"):
        client.complete("системный", "пользовательский")


# --- текстовая модель отвечает на картинку пятисотой ошибкой: по одному её
# номеру человеку не догадаться, что дело в отсутствии зрения у модели.
# Обнаружено на живом сервере во время дымового прогона ---


def _raising_urlopen(error):
    def fake(*_args, **_kwargs):
        raise error

    return fake


def test_server_error_on_a_request_with_an_image_hints_at_missing_vision(monkeypatch):
    import urllib.error
    import urllib.request

    from PIL import Image

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _raising_urlopen(urllib.error.HTTPError("u", 500, "Internal Server Error", {}, None)),
    )
    client = LlmClient(LlmEndpoint(base_url="http://example.invalid"))
    with pytest.raises(LlmError, match="не умеет их читать"):
        client.complete("s", "u", images=[Image.new("RGB", (8, 8), "red")])


def test_the_same_server_error_without_an_image_does_not_mention_vision(monkeypatch):
    import urllib.error
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _raising_urlopen(urllib.error.HTTPError("u", 500, "Internal Server Error", {}, None)),
    )
    client = LlmClient(LlmEndpoint(base_url="http://example.invalid"))
    with pytest.raises(LlmError) as info:
        client.complete("s", "u")
    assert "зрени" not in str(info.value)


def test_a_network_failure_with_an_image_does_not_blame_vision(monkeypatch):
    # Недоступный хост — не повод рассуждать о зрении модели: подсказка
    # обязана относиться только к ответам самого сервера.
    import urllib.error
    import urllib.request

    from PIL import Image

    monkeypatch.setattr(urllib.request, "urlopen", _raising_urlopen(urllib.error.URLError("down")))
    client = LlmClient(LlmEndpoint(base_url="http://example.invalid"))
    with pytest.raises(LlmError) as info:
        client.complete("s", "u", images=[Image.new("RGB", (8, 8), "red")])
    assert "зрени" not in str(info.value)


# --- подсказка про модель без зрения -----------------------------------


class FailingHandler(Handler):
    """Сервер, который давится изображением ровно так, как настоящий.

    Проверено на живой llama.cpp с текстовой моделью: запрос без картинки
    проходит, запрос с картинкой возвращает 500. По одному номеру ошибки
    человеку не догадаться, что дело не в сети и не в токене.
    """

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        RECEIVED.append({"body": body, "auth": self.headers.get("Authorization")})
        if isinstance(body["messages"][1]["content"], list):
            self.send_error(500, "Internal Server Error")
        else:
            self._reply({"choices": [{"message": {"content": "done"}}]})


@pytest.fixture
def blind_server():
    RECEIVED.clear()
    httpd = HTTPServer(("127.0.0.1", 0), FailingHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_a_text_only_model_is_named_as_the_likely_cause(blind_server):
    client = LlmClient(LlmEndpoint(base_url=blind_server))
    with pytest.raises(LlmError) as failure:
        client.complete("s", "опиши", images=[Image.new("RGB", (8, 8), "red")])

    text = str(failure.value)
    assert "500" in text, "техническая причина обязана остаться в сообщении"
    assert "не умеет их читать" in text, text
    # Оба пути сюда ведут: кнопка описания и AI буст при правке с
    # референсами. Называть только кнопку значило бы вводить в заблуждение
    # того, кто пришёл через буст.
    assert "Описать изображение" in text
    assert "буст" in text



# --- AI буст с текстовой моделью -----------------------------------------
#
# Промт обязан переписываться и моделью без зрения: изображения переписывателю
# правки — подспорье, а не условие. Раньше отказ сервера от картинки ронял буст
# целиком, и с текстовой моделью AI буст при правке не работал вовсе.


def _prompt_dir(tmp_path):
    for name in ("system_prompt_t2i.txt", "system_prompt_edit.txt"):
        (tmp_path / name).write_text("rewrite", encoding="utf-8")
    return tmp_path


def _posts():
    return [isinstance(item["body"]["messages"][1]["content"], list) for item in RECEIVED]


def test_a_refused_image_is_its_own_error_type(blind_server):
    client = LlmClient(LlmEndpoint(base_url=blind_server))
    with pytest.raises(LlmImagesRejected):
        client.complete("s", "u", images=[Image.new("RGB", (8, 8))])


def test_an_access_error_is_not_blamed_on_the_image(monkeypatch):
    """401 с картинкой в запросе — это токен, а не зрение: повторять без картинки незачем."""
    import urllib.error
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlopen",
        _raising_urlopen(urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)),
    )
    client = LlmClient(LlmEndpoint(base_url="http://192.0.2.1:8000"))
    with pytest.raises(LlmError) as failure:
        client.complete("s", "u", images=[Image.new("RGB", (8, 8))])
    assert not isinstance(failure.value, LlmImagesRejected)


def test_boost_rewrites_by_text_when_the_model_refuses_images(blind_server, tmp_path):
    client = LlmClient(LlmEndpoint(base_url=blind_server))
    result = boost.boost(
        client, "make it red", mode=boost.MODE_EDIT, prompt_dir=_prompt_dir(tmp_path),
        references=[Image.new("RGB", (8, 8))],
    )
    assert result.prompt == "done"
    assert result.images_skipped
    assert _posts() == [True, False], "сначала с картинкой, затем тот же запрос одним текстом"


def test_boost_skips_images_for_a_known_text_model(blind_server, tmp_path):
    client = LlmClient(LlmEndpoint(base_url=blind_server))
    result = boost.boost(
        client, "make it red", mode=boost.MODE_EDIT, prompt_dir=_prompt_dir(tmp_path),
        references=[Image.new("RGB", (8, 8))], send_images=False,
    )
    assert result.prompt == "done" and result.images_skipped
    assert _posts() == [False]


def test_boost_with_a_vision_model_is_unchanged(server, tmp_path):
    client = LlmClient(LlmEndpoint(base_url=server))
    result = boost.boost(
        client, "make it red", mode=boost.MODE_EDIT, prompt_dir=_prompt_dir(tmp_path),
        references=[Image.new("RGB", (8, 8))],
    )
    assert not result.images_skipped
    assert _posts() == [True]


def test_studio_remembers_a_text_model_and_says_so(blind_server, tmp_path, monkeypatch):
    from fooocus_qwen import config
    from fooocus_qwen.ui.state import Studio

    endpoint_file = tmp_path / "llm_endpoint.txt"
    endpoint_file.write_text(blind_server + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENDPOINT_FILE", endpoint_file)
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", _prompt_dir(tmp_path))
    studio = Studio(config.AppConfig())
    picture = [Image.new("RGB", (8, 8))]

    text, _, message = studio.boost_prompt("make it red", boost.MODE_EDIT, "ru", picture)
    assert text == "done"
    assert "по тексту" in message and "не выполнен" not in message
    assert _posts() == [True, False]

    RECEIVED.clear()
    text, _, message = studio.boost_prompt("make it blue", boost.MODE_EDIT, "en", picture)
    assert text == "done" and "text alone" in message
    assert _posts() == [False], "известная текстовая модель — без заведомо неудачного запроса"
