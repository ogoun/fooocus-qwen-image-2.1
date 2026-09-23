"""Кисть маски: собственный компонент вместо ``gr.ImageEditor``.

``component`` — компонент Gradio и его клиентская часть (``painter.js``,
``painter.css``); ``payload`` — разбор и сборка значения, которым кисть
обменивается с сервером. Почему своя кисть и на чём она держится — в
докстрингах обоих модулей и в ``docs/research/2026-09-23-kist-maski.md``.
"""

from .component import MaskPainter, client_script, flush_js
from .payload import Canvas, PayloadError, decode, encode

__all__ = ["Canvas", "MaskPainter", "PayloadError", "client_script", "decode", "encode", "flush_js"]
