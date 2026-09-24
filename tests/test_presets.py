"""Пресеты качества."""

import pytest

from fooocus_qwen.engine import presets


def test_quality_presets_ascend_and_turbo_comes_last():
    assert presets.NAMES == ("LowQuality", "MiddleQuality", "MaxQuality", "Turbo")
    full = [name for name in presets.NAMES if not presets.get(name).turbo]
    resolutions = [presets.get(name).output_resolution for name in full]
    steps = [presets.get(name).num_inference_steps for name in full]
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


def test_turbo_follows_the_distillation_schedule():
    """Шесть шагов — ровно столько узлов в расписании дистиллята."""
    from fooocus_qwen.engine import turbo

    preset = presets.get("Turbo")
    assert preset.turbo
    assert preset.num_inference_steps == len(turbo.SIGMAS) == 6
    assert not any(presets.get(name).turbo for name in ("LowQuality", "MiddleQuality", "MaxQuality"))

