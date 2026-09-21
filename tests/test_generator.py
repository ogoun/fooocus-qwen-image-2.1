"""Сборка запроса на генерацию: условные изображения, теги, размеры.

Тесты не трогают видеокарту: проверяется подготовка запроса, а она к модели
отношения не имеет. Ниже к этому добавлены поведенческие тесты самого
``Generator.generate`` — они используют поддельный пайплайн (без CUDA и без
настоящих весов), чтобы проверить обещания генератора по существу: разные
сиды для image_number > 1, замену отрицательного сида, композитинг по маске,
режим аннотации и прерывание.
"""

import types

import numpy as np
import torch
from PIL import Image

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.engine import presets
from fooocus_qwen.imaging import aspect


def request(**overrides):
    base = dict(prompt="кот", preset=presets.get("LowQuality"))
    base.update(overrides)
    return gen.GenerationRequest(**base)


def image(size=(64, 64), colour="red"):
    return Image.new("RGBA", size, colour)


def test_text_to_image_has_no_conditions():
    assert gen.build_conditions(request()) == []


def test_single_reference_gets_no_tag():
    # Спецификация Qwen: при одном изображении теги использовать запрещено.
    slots = gen.build_conditions(request(references=(image(),)))
    assert len(slots) == 1
    assert slots[0].tag == ""
    assert slots[0].role == "reference"


def test_several_references_are_tagged_from_one():
    slots = gen.build_conditions(request(references=(image(), image(), image())))
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>", "<image3>"]


def test_mask_edit_puts_source_first_and_mask_second():
    slots = gen.build_conditions(
        request(source=image(), mask=Image.new("L", (64, 64), 255), mask_mode=gen.MASK_MASK)
    )
    assert [slot.role for slot in slots] == ["source", "mask"]
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>"]
    assert slots[1].image.mode == "RGB"  # маска подаётся как изображение


def test_references_are_numbered_after_source_and_mask():
    slots = gen.build_conditions(
        request(
            source=image(),
            mask=Image.new("L", (64, 64), 255),
            mask_mode=gen.MASK_MASK,
            references=(image(), image()),
        )
    )
    assert [slot.tag for slot in slots] == ["<image1>", "<image2>", "<image3>", "<image4>"]
    assert [slot.role for slot in slots] == ["source", "mask", "reference", "reference"]


def test_annotation_mode_sends_one_image_and_no_mask():
    # Пометки нарисованы прямо на изображении, отдельная маска не нужна.
    slots = gen.build_conditions(
        request(source=image(), mask=Image.new("L", (64, 64), 255), mask_mode=gen.MASK_ANNOTATION)
    )
    assert [slot.role for slot in slots] == ["source"]
    assert slots[0].tag == ""


def test_prompt_edit_without_mask_sends_only_the_source():
    slots = gen.build_conditions(request(source=image(), mask_mode=gen.MASK_NONE))
    assert [slot.role for slot in slots] == ["source"]


def test_size_comes_from_the_aspect_and_preset():
    width, height = gen.resolve_size(request(aspect="16:9", preset=presets.get("MaxQuality")))
    assert (width, height) == (2752, 1536)


def test_follow_reference_leaves_size_to_the_pipeline():
    assert gen.resolve_size(request(aspect=aspect.FOLLOW_REFERENCE)) == (None, None)


def test_editing_follows_the_source_size_with_the_explicit_sentinel():
    # Наследование размеров задаётся явным признаком, а не значением "1:1":
    # иначе осознанный выбор квадрата при правке молча игнорировался бы.
    resolved = gen.resolve_size(
        request(source=image((100, 50)), mask_mode=gen.MASK_NONE, aspect=aspect.FOLLOW_REFERENCE)
    )
    assert resolved == (None, None)


def test_explicit_square_during_editing_is_honoured():
    width, height = gen.resolve_size(
        request(source=image((100, 50)), mask_mode=gen.MASK_NONE, aspect="1:1")
    )
    assert width == height and width is not None


def test_seed_minus_one_is_replaced_by_a_random_one():
    resolved = gen.resolve_seed(-1)
    assert 0 <= resolved < 2**31


def test_explicit_seed_is_kept():
    assert gen.resolve_seed(12345) == 12345


# ---------------------------------------------------------------------------
# Поведенческие тесты Generator.generate на поддельном пайплайне.
#
# Настоящую модель в тестах не грузим — она занимает 45 секунд и требует
# видеокарту. Подделка записывает переданные именованные аргументы и отдаёт
# пустые изображения, этого достаточно, чтобы проверить сиды, композитинг и
# прерывание по существу, а не только то, что код не падает.
# ---------------------------------------------------------------------------


class FakePipeline:
    """Записывает вызовы, отдаёт заготовленное изображение вместо настоящей генерации."""

    def __init__(self, image_factory=None, on_call=None):
        self._execution_device = torch.device("cpu")
        self._interrupt = False
        self.calls: list[dict] = []
        self._image_factory = image_factory or (lambda: Image.new("RGBA", (64, 64), "green"))
        self.on_call = on_call

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_call is not None:
            self.on_call(self)
        return types.SimpleNamespace(images=[self._image_factory()])


def make_generator(pipe=None, catalogue=None):
    return gen.Generator(pipe or FakePipeline(), residency=None, cache=None, catalogue=catalogue or {})


def test_image_number_runs_the_pipeline_with_different_seeds():
    pipe = FakePipeline()
    engine = make_generator(pipe)

    results = engine.generate(request(image_number=3, seed=555))

    assert len(pipe.calls) == 3
    seeds_used = [result.seed for result in results]
    assert len(set(seeds_used)) == 3  # три разных сида, а не один повторно
    assert [result.parameters["seed"] for result in results] == seeds_used


def test_negative_seed_is_resolved_and_recorded():
    engine = make_generator()

    results = engine.generate(request(seed=-1))

    assert len(results) == 1
    assert 0 <= results[0].seed < 2**31
    # Значение, которое реально использовалось, обязано попасть в метаданные —
    # иначе интерфейс не сможет предложить «повторить этот сид».
    assert results[0].parameters["seed"] == results[0].seed


def test_mask_mode_with_keep_outside_composites_pixels():
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))  # красный фон
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (16, 16, 48, 48))  # белый квадрат в центре — область правки

    generated_colour = (0, 255, 0, 255)  # зелёный «результат» модели
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    results = engine.generate(
        request(
            source=source,
            mask=mask,
            mask_mode=gen.MASK_MASK,
            mask_grow=0,
            mask_feather=0,
            keep_outside=True,
        )
    )

    assert len(results) == 1
    pixels = np.asarray(results[0].image)
    # Вне маски — пиксель источника без изменений, ровно как и было до правки.
    assert tuple(pixels[0, 0]) == (255, 0, 0, 255)
    assert tuple(pixels[63, 63]) == (255, 0, 0, 255)
    # Внутри маски — результат модели.
    assert tuple(pixels[32, 32]) == generated_colour


def test_keep_outside_false_returns_raw_pipeline_output():
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (16, 16, 48, 48))

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    results = engine.generate(
        request(
            source=source,
            mask=mask,
            mask_mode=gen.MASK_MASK,
            mask_grow=0,
            mask_feather=0,
            keep_outside=False,
        )
    )

    assert len(results) == 1
    pixels = np.asarray(results[0].image.convert("RGBA"))
    # Без keep_outside возвращается сырой вывод пайплайна целиком, даже там,
    # где источник был бы неприкосновенен при включённом keep_outside.
    assert tuple(pixels[0, 0]) == generated_colour
    assert tuple(pixels[32, 32]) == generated_colour


def test_annotation_mode_sends_no_separate_mask_and_skips_compositing():
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = Image.new("L", (64, 64), 255)  # пометки нарисованы поверх source

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    results = engine.generate(
        request(source=source, mask=mask, mask_mode=gen.MASK_ANNOTATION, keep_outside=True)
    )

    assert len(results) == 1
    call = pipe.calls[0]
    assert len(call["image"]) == 1  # маска не подана отдельным условным изображением

    pixels = np.asarray(results[0].image.convert("RGBA"))
    # Результат не склеивается с оригиналом — пометки были частью условного
    # изображения, а не отдельной маской для последующего blend.
    assert tuple(pixels[0, 0]) == generated_colour


def test_interruption_returns_no_images():
    pipe = FakePipeline()
    engine = make_generator(pipe)
    pipe.on_call = lambda _pipe: engine.interrupt()

    results = engine.generate(request(image_number=3))

    # Прерванный цикл денойзинга всё равно декодирует латенты, но это шум,
    # а не картинка, и наружу отдавать его нельзя.
    assert results == []
    # А главное — цикл обязан остановиться после первой же картинки, а не
    # сгенерировать все и потом выбросить: смысл прерывания — не тратить
    # видеопамять и время на кадры, которые всё равно не покажут.
    assert len(pipe.calls) == 1


def test_metadata_records_the_generation_parameters():
    pipe = FakePipeline()
    engine = make_generator(pipe)

    results = engine.generate(
        request(
            seed=42,
            preset=presets.get("MaxQuality"),
            use_kv_cache=False,
            references=(image(), image()),
        )
    )

    parameters = results[0].parameters
    assert parameters["seed"] == 42
    assert parameters["steps"] == presets.get("MaxQuality").num_inference_steps
    assert parameters["preset"] == "MaxQuality"
    assert parameters["use_kv_cache"] is False
    assert parameters["mask_mode"] == gen.MASK_NONE
    assert parameters["references"] == 2


# ---------------------------------------------------------------------------
# MASK_REGION: до этого раунда правок у режима точной области не было ни
# одного теста, хотя это самая структурно сложная ветка файла и именно она
# несла находку 1 (утечка region_box между вызовами).
# ---------------------------------------------------------------------------


def test_region_mode_crops_and_stitches_outside_pixels_untouched():
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (8, 8, 24, 24))  # маленькая область в углу

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    results = engine.generate(
        request(
            source=source,
            mask=mask,
            mask_mode=gen.MASK_REGION,
            mask_grow=0,
            mask_feather=0,
        )
    )

    assert len(results) == 1
    condition_image = pipe.calls[0]["image"][0]
    # Пайплайн получил вырезку вокруг маски, а не полный кадр.
    assert condition_image.size[0] < source.size[0]
    assert condition_image.size[1] < source.size[1]

    result_image = results[0].image
    assert result_image.size == source.size  # результат — в размере исходного кадра

    pixels = np.asarray(result_image)
    # Далеко за пределами и маски, и вырезанного окна пиксели не тронуты.
    assert tuple(pixels[60, 60]) == (255, 0, 0, 255)
    assert tuple(pixels[0, 60]) == (255, 0, 0, 255)


def test_region_mode_with_empty_mask_degrades_to_whole_frame_editing():
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    empty_mask = Image.new("L", (64, 64), 0)

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    # Пустая маска не должна поднимать исключение — режим точной области
    # вырождается в правку целого кадра.
    results = engine.generate(
        request(
            source=source,
            mask=empty_mask,
            mask_mode=gen.MASK_REGION,
            mask_grow=0,
            mask_feather=0,
        )
    )

    assert len(results) == 1
    condition_image = pipe.calls[0]["image"][0]
    assert condition_image.size == source.size  # вырезки не было, кадр целиком


def test_reusing_the_request_after_region_mode_does_not_leak_the_crop_box():
    # Регрессия находки 1: `_prepare` писал region_box в `request.extras`,
    # общий с объектом вызывающей стороны, поэтому повторное использование
    # того же GenerationRequest для другого запуска подставляло старую
    # вырезку в новый — и склейка не меняла ни одного пикселя.
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    region_mask = Image.new("L", (64, 64), 0)
    region_mask.paste(255, (8, 8, 24, 24))  # маленькая область в одном углу

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (64, 64), generated_colour))
    engine = make_generator(pipe)

    shared_request = request(
        source=source,
        mask=region_mask,
        mask_mode=gen.MASK_REGION,
        mask_grow=0,
        mask_feather=0,
    )
    engine.generate(shared_request)

    # Тот же объект переиспользуется для другого запуска: правка по маске в
    # ПРОТИВОПОЛОЖНОМ углу кадра — там, где первая вырезка не была.
    other_mask = Image.new("L", (64, 64), 0)
    other_mask.paste(255, (40, 40, 56, 56))
    shared_request.mask = other_mask
    shared_request.mask_mode = gen.MASK_MASK

    results = engine.generate(shared_request)

    pixels = np.asarray(results[0].image)
    # Пиксель внутри новой маски обязан измениться. Если бы region_box от
    # первого вызова просочился во второй, склейка использовала бы старую
    # (уже неактуальную) вырезку, новая маска не пересекалась бы с ней, и
    # этот пиксель остался бы равен источнику — ноль изменённых пикселей.
    assert tuple(pixels[48, 48]) == generated_colour


def test_region_mode_without_compositing_returns_the_frame_not_the_crop():
    # Незакрытая клетка матрицы «режим × keep_outside». В режиме точной области
    # `_prepare` уже вырезал источник, поэтому `produced` — патч, а не кадр:
    # возврат его «как есть» отдавал пользователю, правившему деталь на
    # большом холсте, крошечную картинку вместо изображения. Склейки по маске
    # при выключенном keep_outside быть не должно, а вклейка на место — должна.
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (8, 8, 24, 24))

    generated_colour = (0, 255, 0, 255)
    pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (32, 32), generated_colour))
    engine = make_generator(pipe)

    results = engine.generate(
        request(
            source=source,
            mask=mask,
            mask_mode=gen.MASK_REGION,
            mask_grow=0,
            mask_feather=0,
            keep_outside=False,
        )
    )

    assert len(results) == 1
    result_image = results[0].image
    assert result_image.size == source.size, "вернулась вырезка вместо кадра"

    pixels = np.asarray(result_image)
    # Внутри вырезанного прямоугольника — целиком вывод модели, без смешивания
    # по маске: keep_outside выключен, и края области размывать нечем.
    assert tuple(pixels[16, 16]) == generated_colour
    # Далеко за пределами прямоугольника кадр остался исходным.
    assert tuple(pixels[60, 60]) == (255, 0, 0, 255)


def test_region_mode_without_compositing_does_not_blend_inside_the_region():
    # Отличие от keep_outside=True: там всё, что вне маски, остаётся исходным,
    # в том числе внутри вырезанного прямоугольника. Здесь склейки нет, поэтому
    # прямоугольник заполняется выводом модели целиком. Без этой проверки
    # предыдущий тест прошёл бы и для обычной склейки.
    source = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (8, 8, 24, 24))
    generated_colour = (0, 255, 0, 255)

    def run(keep_outside: bool):
        pipe = FakePipeline(image_factory=lambda: Image.new("RGBA", (32, 32), generated_colour))
        engine = make_generator(pipe)
        produced = engine.generate(
            request(
                source=source,
                mask=mask,
                mask_mode=gen.MASK_REGION,
                mask_grow=0,
                mask_feather=0,
                keep_outside=keep_outside,
            )
        )
        return np.asarray(produced[0].image), pipe.calls[0]["image"][0].size

    without_blend, crop_size = run(keep_outside=False)
    with_blend, _ = run(keep_outside=True)

    assert crop_size != source.size, "ожидалась именно вырезка, иначе проверка вырождена"

    # Точка внутри прямоугольника вырезки, но вне маски.
    assert tuple(with_blend[28, 28]) == (255, 0, 0, 255), "со склейкой вне маски кадр обязан уцелеть"
    assert tuple(without_blend[28, 28]) == generated_colour, "без склейки патч ложится встык"
