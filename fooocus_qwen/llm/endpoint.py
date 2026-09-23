"""Разбор файла с адресом внешней языковой модели.

Формат свободный и построчный: имя бэкенда, адрес хоста (возможно со схемой и
портом), строка вида ``token=...``. Порядок строк не важен — файл пишет человек,
а не программа, и требовать от него порядка значит собирать ошибки на ровном месте.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN_LINE = re.compile(r"^(token|api_key|key)\s*[=:]\s*(\S+)$", re.IGNORECASE)
_URL_LINE = re.compile(r"^(?:url|endpoint|host)\s*[=:]\s*(\S+)$", re.IGNORECASE)
# Порт: двоеточие, за которым идут цифры (а не произвольный текст вида "host: заметка").
_HAS_PORT = re.compile(r":\d+")
# Голый IPv4 без схемы и порта — единственный случай, когда точки в строке
# однозначно указывают на адрес, а не на имя бэкенда вроде "llama.cpp".
_BARE_IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


@dataclass(frozen=True)
class LlmEndpoint:
    """Реквизиты доступа к OpenAI-совместимому серверу."""

    base_url: str
    # Вне repr: объект адреса попадает в журналы и сообщения об ошибках,
    # а токен не должен оказаться ни там, ни там.
    token: str | None = field(default=None, repr=False)
    backend: str = ""

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"


def _normalize(candidate: str, default_port: int) -> str:
    """Доводит строку до полного URL.

    Порт дописывается только к голому http-адресу: у https порт по умолчанию
    свой, и навязывать ему 8000 значит ломать обращение к облачным сервисам.
    """
    url = candidate.strip().rstrip("/")
    has_scheme = url.startswith(("http://", "https://"))
    if not has_scheme:
        url = f"http://{url}"

    without_scheme = url.split("://", 1)[1]
    host_part = without_scheme.split("/", 1)[0]
    if ":" not in host_part and url.startswith("http://"):
        url = url.replace(host_part, f"{host_part}:{default_port}", 1)
    return url


def _looks_like_address(line: str) -> bool:
    """Отличает адрес от имени бэкенда.

    Точка в строке сама по себе ничего не значит: "llama.cpp" — это имя
    бэкенда, а не хост, хотя точка в нём есть. Однозначные признаки адреса —
    явная схема, явный порт (двоеточие с цифрами) или голый IPv4.
    """
    return (
        line.startswith(("http://", "https://"))
        or bool(_HAS_PORT.search(line))
        or bool(_BARE_IPV4.match(line))
    )


def parse_endpoint_file(text: str, default_port: int = 8000) -> LlmEndpoint:
    backend = ""
    token: str | None = None
    candidates: list[str] = []
    ambiguous: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        token_match = _TOKEN_LINE.match(line)
        if token_match:
            token = token_match.group(2)
            continue

        url_match = _URL_LINE.match(line)
        if url_match:
            candidates.append(url_match.group(1))
            continue

        if _looks_like_address(line):
            candidates.append(line)
        else:
            ambiguous.append(line)

    if candidates:
        # Адрес уже есть — первая неоднозначная строка тогда точно имя бэкенда.
        if ambiguous:
            backend = ambiguous[0]
    elif ambiguous:
        # Адреса не нашлось вовсе — тогда первая неоднозначная строка и есть адрес,
        # а имя бэкенда остаётся пустым (человек его просто не указал).
        candidates.append(ambiguous[0])

    if not candidates:
        raise ValueError("В файле адреса языковой модели не найдено ни одного хоста")

    return LlmEndpoint(base_url=_normalize(candidates[0], default_port), token=token, backend=backend)


def load_endpoint(path: Path, default_port: int = 8000) -> LlmEndpoint:
    return parse_endpoint_file(path.read_text(encoding="utf-8"), default_port)
