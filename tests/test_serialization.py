"""Одновременных генераций не бывает: ни в движке, ни в очереди интерфейса.

Самый дорогой из возможных дефектов этого проекта — две генерации, стартовавшие
одновременно. Он не проявляется ни в одном модульном тесте и не виден при обычной
работе одним пользователем в одной вкладке: нужно нажать «Сгенерировать» и следом
«Применить правку». Дальше — три разных последствия, все молчаливые:

1. ``Generator.generate`` открывается сбросом ``self._interrupted`` и
   ``pipe._interrupt``, поэтому второй запрос разоружает кнопку «Прервать»
   первого.
2. ``ResidencyManager.text_encoder_resident()`` не реентерабелен: при промахе
   кэша эмбеддингов второй запрос снимает трансформер с видеокарты прямо посреди
   чужого цикла денойзинга, переприсваивая ``tensor.data`` под работающим
   forward.
3. Два цикла денойзинга при 13.3 ГБ резидентного трансформера на 24 ГБ карты не
   помещаются в видеопамять.

Отсюда две проверки: замок в самом генераторе (главная — инвариант держится тем
слоем, который от него зависит) и общая группа очереди Gradio (вспомогательная —
не даёт очереди выпустить работу, которая всё равно упрётся в замок).
"""

from __future__ import annotations

import threading
import types

import pytest
import torch
from PIL import Image

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.engine import presets


def request(**overrides):
    base = dict(prompt="кот", preset=presets.get("LowQuality"))
    base.update(overrides)
    return gen.GenerationRequest(**base)


class BlockingPipeline:
    """Пайплайн, который стоит в вызове, пока его не отпустят.

    Настоящая генерация занимает видеокарту на десятки секунд; здесь тот же
    интервал изображается событием, что делает проверку детерминированной, а не
    зависящей от того, кто кого успел обогнать.
    """

    def __init__(self) -> None:
        self._execution_device = torch.device("cpu")
        self._interrupt = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.inside = 0
        self.max_inside = 0
        self._guard = threading.Lock()

    def __call__(self, **kwargs):
        with self._guard:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        self.entered.set()
        self.release.wait(timeout=10)
        with self._guard:
            self.inside -= 1
        return types.SimpleNamespace(images=[Image.new("RGBA", (64, 64), "green")])


def _generator(pipe):
    return gen.Generator(pipe, residency=None, cache=None, catalogue={})


def test_a_second_generation_waits_for_the_first_to_finish():
    pipe = BlockingPipeline()
    engine = _generator(pipe)
    order: list[str] = []

    def call(name: str):
        engine.generate(request())
        order.append(name)

    first = threading.Thread(target=call, args=("первый",), daemon=True)
    second = threading.Thread(target=call, args=("второй",), daemon=True)

    first.start()
    assert pipe.entered.wait(timeout=5), "первая генерация не дошла до пайплайна"
    second.start()

    # Даём второму потоку заведомо больше времени, чем нужно, чтобы дойти до
    # пайплайна, если бы его никто не держал: проверка «не вошёл» без паузы
    # была бы проверкой того, что поток не успел стартовать.
    second.join(timeout=0.5)
    assert second.is_alive(), "вторая генерация не должна начинаться, пока идёт первая"
    assert pipe.inside == 1, f"внутри пайплайна оказалось {pipe.inside} вызовов вместо одного"

    pipe.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert pipe.max_inside == 1, "две генерации побывали в пайплайне одновременно"
    assert order == ["первый", "второй"]


def test_the_second_request_cannot_disarm_the_first_stop_button():
    # Последствие №1 из докстринга модуля: без замка второй запрос выполнял бы
    # `self._interrupted = False` посреди первого, и нажатая кнопка «Прервать»
    # переставала бы действовать.
    pipe = BlockingPipeline()
    engine = _generator(pipe)

    first = threading.Thread(target=lambda: engine.generate(request(image_number=3)), daemon=True)
    first.start()
    assert pipe.entered.wait(timeout=5)

    second = threading.Thread(target=lambda: engine.generate(request()), daemon=True)
    second.start()
    second.join(timeout=0.3)

    engine.interrupt()
    assert pipe._interrupt is True

    pipe.release.set()
    first.join(timeout=5)
    second.join(timeout=5)


def test_interrupt_is_reachable_while_the_lock_is_held():
    # Замок не должен сделать кнопку «Прервать» недостижимой: interrupt() его не
    # берёт, поэтому вызов проходит из чужого потока, пока генерация идёт.
    pipe = BlockingPipeline()
    engine = _generator(pipe)

    worker = threading.Thread(target=lambda: engine.generate(request(image_number=4)), daemon=True)
    worker.start()
    assert pipe.entered.wait(timeout=5)

    stopped = threading.Event()

    def press_stop():
        engine.interrupt()
        stopped.set()

    presser = threading.Thread(target=press_stop, daemon=True)
    presser.start()
    assert stopped.wait(timeout=2), "interrupt() заблокировался на замке генерации"

    pipe.release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()


# --- вторая линия: очередь Gradio ---

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.ui import tab_edit, tab_generate  # noqa: E402
from fooocus_qwen.ui.i18n import Localizer  # noqa: E402
from fooocus_qwen.ui.state import GPU_CONCURRENCY_ID, Studio  # noqa: E402


def test_both_generate_handlers_share_one_concurrency_group(monkeypatch, tmp_path):
    """``default_concurrency_limit=1`` сам по себе ничего не сериализует.

    Gradio заводит группу очереди на идентичность функции:
    ``BlockFunction.concurrency_id = concurrency_id or str(id(fn))``
    (``gradio/block_function.py``). ``tab_generate.run`` и ``tab_edit.run`` —
    разные объекты, поэтому без явного общего идентификатора каждый ограничен
    единицей только сам против себя.
    """
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        tab_generate.build(studio, localizer)
        tab_edit.build(studio, localizer)

    run_handlers = [block_fn for block_fn in demo.fns.values() if block_fn.fn.__name__ == "run"]
    assert len(run_handlers) == 2, "ожидались обработчики генерации и правки"

    groups = {block_fn.concurrency_id for block_fn in run_handlers}
    assert groups == {GPU_CONCURRENCY_ID}, (
        "обработчики, доходящие до видеокарты, обязаны стоять в одной очереди; "
        f"получено {groups}"
    )

    # Проверка не должна проходить только потому, что Gradio случайно выдала
    # двум функциям одинаковый str(id(fn)): идентификатор обязан быть нашим.
    for block_fn in run_handlers:
        assert block_fn.concurrency_id != str(id(block_fn.fn))
