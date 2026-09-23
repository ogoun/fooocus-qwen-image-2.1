"""Установка спрашивает адрес и токен языковой модели и пишет их в файл.

Файл `llm_endpoint.txt` — единственное место в проекте, где лежит секрет, и
в репозиторий он не попадает (`.gitignore`). До этой правки человек, забравший
проект с GitHub, должен был узнать о файле из документации и составить его
руками; теперь о нём спрашивает установка.

Отказ отвечать — полноправный ответ: AI-буст промтов необязателен, без
внешней модели работает всё остальное. Поэтому пустой ввод означает «пропустить»,
а не «записать пустоту».
"""

from __future__ import annotations

from fooocus_qwen.llm import endpoint as ep
from fooocus_qwen.llm import setup


class _Dialog:
    """Изображает человека за консолью: отвечает заготовленным списком."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []
        self.printed: list[str] = []

    def ask(self, question: str) -> str:
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else ""

    def out(self, line: str = "") -> None:
        self.printed.append(line)


def test_address_and_token_are_written_and_read_back(tmp_path):
    """Круг замкнулся: что установка записала, то разбор и прочитал."""
    path = tmp_path / "llm_endpoint.txt"
    dialog = _Dialog("192.0.2.10:8000", "s3cret")

    assert setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None) is True

    result = ep.load_endpoint(path)
    assert result.base_url == "http://192.0.2.10:8000"
    assert result.token == "s3cret"


def test_an_empty_address_skips_the_whole_thing(tmp_path):
    path = tmp_path / "llm_endpoint.txt"
    dialog = _Dialog("")

    assert setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None) is False
    assert not path.exists(), "пустой ввод не должен создавать файл"


def test_a_server_without_a_token_is_normal(tmp_path):
    """llama.cpp в локальной сети обычно ничего не спрашивает."""
    path = tmp_path / "llm_endpoint.txt"
    dialog = _Dialog("192.0.2.10:8000", "")

    setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None)

    assert ep.load_endpoint(path).token is None
    assert "token" not in path.read_text(encoding="utf-8").lower().split("#")[-1]


def test_an_existing_file_is_kept_on_an_empty_answer(tmp_path):
    path = tmp_path / "llm_endpoint.txt"
    path.write_text("llama.cpp\n192.0.2.99:1234\ntoken=old\n", encoding="utf-8")
    dialog = _Dialog("")

    assert setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None) is False
    assert ep.load_endpoint(path).token == "old", "настройку нельзя терять по недосмотру"


def test_an_existing_token_is_never_printed(tmp_path):
    """Показать секрет на экране — значит отдать его тому, кто смотрит через плечо."""
    path = tmp_path / "llm_endpoint.txt"
    path.write_text("192.0.2.99:1234\ntoken=SUPERSECRET\n", encoding="utf-8")
    dialog = _Dialog("")

    setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None)

    assert "SUPERSECRET" not in "\n".join(dialog.printed + dialog.asked)


def test_an_existing_file_is_replaced_when_a_new_address_is_given(tmp_path):
    path = tmp_path / "llm_endpoint.txt"
    path.write_text("192.0.2.99:1234\ntoken=old\n", encoding="utf-8")
    dialog = _Dialog("192.0.2.10:8000", "new")

    assert setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=None) is True

    result = ep.load_endpoint(path)
    assert result.base_url == "http://192.0.2.10:8000"
    assert result.token == "new"


def test_an_unreachable_server_does_not_lose_the_answers(tmp_path):
    """Сервер может быть просто выключен — это не повод терять введённое."""
    path = tmp_path / "llm_endpoint.txt"
    dialog = _Dialog("192.0.2.10:8000", "")

    def probe(_endpoint):
        raise OSError("connection refused")

    assert setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=probe) is True
    assert path.is_file()
    assert any("192.0.2.10" in line or "не отклик" in line.lower() for line in dialog.printed)


def test_a_reachable_server_reports_its_models(tmp_path):
    path = tmp_path / "llm_endpoint.txt"
    dialog = _Dialog("192.0.2.10:8000", "")

    setup.configure(path, ask=dialog.ask, ask_secret=dialog.ask, out=dialog.out, probe=lambda _e: ["qwen3-vl"])

    assert any("qwen3-vl" in line for line in dialog.printed)


def test_the_written_file_is_a_template_a_human_can_edit(tmp_path):
    """Файл правят руками и из вкладки «Настройки» — значит он с пояснением."""
    path = tmp_path / "llm_endpoint.txt"
    setup.configure(path, ask=_Dialog("192.0.2.10:8000", "t").ask, ask_secret=lambda _q: "t", out=lambda _l="": None, probe=None)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("#"), "без шапки файл выглядит случайным набором строк"


def test_render_is_parseable_for_any_reasonable_answer():
    """Свойство, а не пример: что бы ни ввели, разбор обязан это принять."""
    for host, token in [
        ("192.0.2.10", None),
        ("192.0.2.10:1234", "abc"),
        ("http://model.example.com:8080", "abc"),
        ("https://api.example.com", "abc"),
    ]:
        text = setup.render(host, token)
        result = ep.parse_endpoint_file(text)
        assert result.token == token
        assert host.split("://")[-1].split(":")[0] in result.base_url


def test_the_end_of_input_means_skip_not_a_crash(tmp_path):
    """Установку запускают и сценарием — там на вопрос отвечать некому.

    Проверять `sys.stdin.isatty()` заранее мало: в Git Bash под Windows
    перенаправление из `/dev/null` даёт `isatty() == True`, и установка
    падала с `EOFError` после того, как зависимости уже поставлены.
    """
    path = tmp_path / "llm_endpoint.txt"

    def eof(_question: str) -> str:
        raise EOFError("некому отвечать")

    assert setup.configure(path, ask=eof, ask_secret=eof, out=lambda _l="": None, probe=None) is False
    assert not path.exists()


def test_ctrl_c_during_the_question_is_a_skip_too(tmp_path):
    path = tmp_path / "llm_endpoint.txt"

    def interrupt(_question: str) -> str:
        raise KeyboardInterrupt

    assert setup.configure(path, ask=interrupt, ask_secret=interrupt, out=lambda _l="": None, probe=None) is False
