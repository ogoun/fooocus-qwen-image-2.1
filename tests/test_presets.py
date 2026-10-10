"""Пресеты качества."""

import pytest

from fooocus_qwen.engine import presets


def test_quality_presets_ascend_and_turbo_comes_last():
    assert presets.NAMES == (
        "LowQuality", "MiddleQuality", "MaxQuality", "Turbo", "TurboDraft", "Turbo4",
        "QwenTurbo", "QwenTurboHD", "QwenTurbo2K",
    )
    full = [name for name in presets.NAMES if not presets.get(name).turbo and not presets.get(name).transformer]
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



def test_turbo_draft_is_the_same_distillation_on_a_smaller_frame():
    """Черновик: тот же дистиллят и те же шесть шагов, кадр 768² — для перебора."""
    draft, turbo = presets.get("TurboDraft"), presets.get("Turbo")
    assert draft.turbo and draft.num_inference_steps == turbo.num_inference_steps
    assert draft.output_resolution == 768 < turbo.output_resolution


def test_the_command_line_offers_every_preset():
    from fooocus_qwen import config

    assert config.PRESET_NAMES == presets.NAMES


def test_turbo4_is_four_uniform_steps_on_its_own_transformer():
    """Turbo4 — отдельный трансформер (не адаптер) и четыре равномерных узла."""
    from fooocus_qwen.engine import turbo

    preset = presets.get("Turbo4")
    assert preset.transformer == presets.TURBO4 and not preset.turbo
    assert preset.num_inference_steps == len(turbo.SIGMAS4) == 4
    assert preset.output_resolution == 1024
    arguments = turbo.call_arguments({"num_inference_steps": 16, "true_cfg_scale": 4.0, "negative_prompt": "x"}, turbo.SIGMAS4)
    assert arguments["sigmas"] == [1.0, 0.75, 0.5, 0.25]
    assert arguments["true_cfg_scale"] == 1.0 and arguments["negative_prompt"] is None


def test_qwen_turbo_presets_are_the_official_checkpoint_at_three_sizes():
    official = [presets.get(name) for name in ("QwenTurbo", "QwenTurboHD", "QwenTurbo2K")]
    assert [preset.output_resolution for preset in official] == [1024, 1536, 2048]
    assert all(preset.transformer == presets.TURBO_OFFICIAL and preset.num_inference_steps == 8
               and preset.sigmas == presets.SIGMAS_OFFICIAL and not preset.turbo for preset in official)

