"""Подключение к внешней языковой модели по OpenAI-совместимому протоколу."""

from .client import LlmClient, LlmError
from .endpoint import LlmEndpoint, load_endpoint, parse_endpoint_file

__all__ = ["LlmClient", "LlmError", "LlmEndpoint", "load_endpoint", "parse_endpoint_file"]
