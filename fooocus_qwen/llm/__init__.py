"""Подключение к внешней языковой модели по OpenAI-совместимому протоколу."""

from .client import LlmClient, LlmError, LlmImagesRejected
from .endpoint import LlmEndpoint, load_endpoint, parse_endpoint_file

__all__ = [
    "LlmClient",
    "LlmEndpoint",
    "LlmError",
    "LlmImagesRejected",
    "load_endpoint",
    "parse_endpoint_file",
]
