"""Оболочка сама замечает случай, когда склейка обрежет работу модели.

Случай единственный в своём роде: результат формально безупречен — обещание
о неприкосновенности кадра вне маски выполнено побайтово, — а глазу виден
резкий шов по границе маски. Так выходит, когда просьба глобальна по смыслу
(«сделай волосы платиновыми»): модель перекрашивает все волосы связно и
красиво, а склейка возвращает наружную часть к оригиналу.

Найдено визуальным разбором дымового прогона (`docs/SMOKE.md`), измерено там
же: правка платиновых волос по эллипсу на макушке задевает около 47 % кадра
вне маски. У случая есть штатный выход — снять «Сохранять кадр вне маски», —
и пользователь должен узнать о нём тогда, когда случай наступил.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GeneratedImage  # noqa: E402
from fooocus_qwen.ui import tab_edit  # noqa: E402
from fooocus_qwen.ui.i18n import Localizer  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402

SIZE = (64, 64)


class _Generator:
    """Возвращает готовое изображение и заявленную долю обрезанного."""

    def __init__(self, clipped: float) -> None:
        self._clipped = clipped

    def generate(self, _request, progress=None):
        return [
            GeneratedImage(
                image=Image.new("RGBA", SIZE, "white"),
                seed=1,
                parameters={"clipped_outside_pct": self._clipped},
            )
        ]


def _painter_value(monkeypatch, tmp_path, background, layer=None):
    """Значение кисти, собранное так же, как его собирает сервер.

    Каталог загрузок подменяется временным: разбор принимает пути только
    оттуда, и проверяется настоящий разбор, а не подставной.
    """
    from fooocus_qwen.ui.painter import payload

    monkeypatch.setattr(payload, "upload_root", lambda: tmp_path)
    return payload.encode(background, layer)


def _run(monkeypatch, tmp_path, clipped: float, lang: str = "ru"):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    cfg = config.AppConfig(lang=lang)
    studio = Studio(cfg)
    studio._generator = _Generator(clipped)

    with gr.Blocks() as demo:
        tab_edit.build(studio, Localizer(cfg.lang))
    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)

    painted = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    painted.paste((255, 0, 0, 255), (0, 0, 32, 32))
    value = _painter_value(monkeypatch, tmp_path, Image.new("RGBA", SIZE, "black"), painted)
    return handlers["run"](
        value, "сделать волосы платиновыми", False, "mask", presets.DEFAULT,
        8, 12, True, -1, [], lang, progress=lambda *args, **kwargs: None,
    )


def test_a_large_clipped_edit_is_reported(monkeypatch, tmp_path):
    _images, status = _run(monkeypatch, tmp_path, clipped=47.0)
    assert "47" in status, status
    assert "шов" in status, status


def test_a_local_edit_says_nothing_extra(monkeypatch, tmp_path):
    # Точечная правка задевает единицы процентов; предупреждать не о чем, а
    # предупреждение на каждой правке быстро перестают читать.
    _images, status = _run(monkeypatch, tmp_path, clipped=3.0)
    assert "шов" not in status, status


def test_the_threshold_is_the_only_thing_that_decides(monkeypatch, tmp_path):
    just_below = _run(monkeypatch, tmp_path, clipped=tab_edit.CLIPPED_WARNING_PCT - 0.1)[1]
    just_above = _run(monkeypatch, tmp_path, clipped=tab_edit.CLIPPED_WARNING_PCT)[1]
    assert "шов" not in just_below
    assert "шов" in just_above


def test_the_warning_is_translated(monkeypatch, tmp_path):
    _images, status = _run(monkeypatch, tmp_path, clipped=47.0, lang="en")
    assert "seam" in status, status
    assert "шов" not in status


# --- масштаб условных изображений доходит через обработчик ---


class _Capturing:
    """Запоминает запрос, который собрала вкладка."""

    def __init__(self) -> None:
        self.request = None

    def generate(self, request, progress=None):
        self.request = request
        return [GeneratedImage(image=Image.new("RGBA", SIZE, "white"), seed=1, parameters={})]


def _capture(monkeypatch, tmp_path, mode: str):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)
    engine = _Capturing()
    studio._generator = engine

    with gr.Blocks() as demo:
        tab_edit.build(studio, Localizer(cfg.lang))
    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)

    painted = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    painted.paste((255, 0, 0, 255), (0, 0, 32, 32))
    value = _painter_value(monkeypatch, tmp_path, Image.new("RGBA", SIZE, "black"), painted)
    _images, status = handlers["run"](
        value, "убрать фон", False, mode, "MiddleQuality", 8, 12, True, -1, [], "ru",
        progress=lambda *a, **k: None,
    )
    return engine.request, status


def test_the_edit_tab_caps_the_condition_scale(monkeypatch, tmp_path):
    """Без этого правка на среднем пресете не помещается в карту.

    Замер: полный масштаб — нехватка памяти без маски и 26.8 ГиБ с маской,
    то есть заведомо больше двадцати четырёх на карте
    (docs/research/2026-09-22-pravka-ne-pomeshalas.md).
    """
    from fooocus_qwen.engine.generator import resolve_reference_scale

    request, status = _capture(monkeypatch, tmp_path, "mask")
    assert request is not None
    # Исходник и маска — два условных изображения.
    assert resolve_reference_scale(request) == 768
    assert "детальность референсов" in status.lower(), status


def test_editing_without_a_mask_is_capped_as_well(monkeypatch, tmp_path):
    from fooocus_qwen.engine.generator import resolve_reference_scale

    request, _status = _capture(monkeypatch, tmp_path, "none")
    assert resolve_reference_scale(request) == 1024
