"""Клиент OpenAI-совместимого сервера языковой модели.

Только стандартная библиотека: единственное, что нужно оболочке от LLM, — один
запрос «перепиши промт», и тащить ради него клиентскую библиотеку с её версиями
зависимостей не за что.

Потоковая выдача не используется намеренно: ответ переписывателя — это JSON,
который всё равно разбирается целиком, и показывать его по кусочкам нечего.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from .endpoint import LlmEndpoint

LOGGER = logging.getLogger(__name__)


class LlmError(RuntimeError):
    """Сервер языковой модели недоступен или ответил ошибкой."""


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class LlmClient:
    """Один запрос — один ответ. Историю диалога оболочка не ведёт."""

    def __init__(self, endpoint: LlmEndpoint, model: str = "", timeout: float = 180.0) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self._endpoint.token:
            headers["Authorization"] = f"Bearer {self._endpoint.token}"
        return headers

    def ping(self) -> list[str]:
        """Возвращает список моделей на сервере. Пустой список — сервер жив, но пуст."""
        request = urllib.request.Request(self._endpoint.models_url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=min(self._timeout, 15)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise LlmError(f"Сервер языковой модели недоступен: {error}") from error
        return [item.get("id", "") for item in payload.get("data", [])]

    def _build_body(
        self,
        system: str,
        user: str,
        images: list[Image.Image] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if images:
            content: Any = [{"type": "text", "text": user}]
            content.extend({"type": "image_url", "image_url": {"url": _image_to_data_url(i)}} for i in images)
        else:
            content = user

        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Режим размышления съедает секунды, а его текст в ответе не нужен.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def complete(
        self,
        system: str,
        user: str,
        images: list[Image.Image] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        body = self._build_body(system, user, images, temperature, max_tokens)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self._endpoint.chat_url, data=data, headers=self._headers())

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise LlmError(f"Ошибка обращения к языковой модели: {error}") from error

        choices = payload.get("choices") or []
        if not choices:
            raise LlmError("Языковая модель вернула пустой ответ")

        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            # reasoning_content игнорируем намеренно: это внутренние рассуждения.
            raise LlmError("Языковая модель вернула ответ без текста")
        return text
