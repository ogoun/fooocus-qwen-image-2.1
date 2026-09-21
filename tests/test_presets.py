"""Пресеты качества."""

import pytest

from fooocus_qwen.engine import presets


def test_three_presets_exist_in_ascending_order():
    assert presets.NAMES == ("LowQuality", "MiddleQuality", "MaxQuality")
    resolutions = [presets.get(name).output_resolution for name in presets.NAMES]
    steps = [presets.get(name).num_inference_steps for name in presets.NAMES]
    assert resolutions == sorted(resolutions)
    assert steps == sorted(steps)


def test_max_quality_matches_the_model_card_defaults():
    preset = presets.get("MaxQuality")
    assert preset.output_resolution == 2048
    assert preset.num_inference_steps == 40


def test_resolutions_are_multiples_of_32():
    for name in presets.NAMES:
        assert presets.get(name).output_resolution % 32 == 0


def test_default_is_the_middle_one():
    assert presets.DEFAULT == "MiddleQuality"


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        presets.get("Ultra")
