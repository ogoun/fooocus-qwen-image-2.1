"""Официальный Qwen-Image-2.1-Turbo: файлы по точности, расписание, подмена трансформера."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fooocus_qwen import config
from fooocus_qwen import settings as settings_module
from fooocus_qwen.engine import fetch, plan, presets
from fooocus_qwen.engine import generator as gen
from tests.test_generator import FakePipeline


def _plan(precision: str) -> plan.WeightsPlan:
    return plan.resolve(settings_module.Settings(precision=precision), vram_gib=24.0)


def test_weight_folders_never_collide_on_a_case_insensitive_disk():
    """Регресс: «Qwen-Image-2.1-Turbo» на Windows — это каталог Viggle «…-turbo»,
    и официальный планировщик перезаписал планировщик адаптера."""
    folders = [value for name, value in vars(config).items()
               if name.endswith("_DIR") and isinstance(value, Path) and value.parent == config.PROJECT_ROOT]
    lowered = [folder.name.lower() for folder in folders]
    assert len(lowered) == len(set(lowered)), sorted(lowered)


@pytest.mark.parametrize(("precision", "variant"), [("bf16", None), ("int8", "Q8_0"), ("Q4_K_M", "Q4_K_M"), ("Q3_K_M", "Q3_K_M")])
def test_the_official_turbo_follows_the_chosen_precision(precision, variant):
    chosen = _plan(precision)
    assert chosen.official_turbo_variant == variant
    if variant is None:
        assert chosen.official_turbo_weights == config.TURBO_OFFICIAL_DIR
    else:
        assert chosen.official_turbo_weights.name == fetch.official_turbo_gguf(variant)
        assert chosen.official_turbo_weights.parent.name == fetch.OFFICIAL_TURBO_GGUF_SUBDIR, \
            "свой подкаталог: имя Q4_K_M совпадает с файлом Turbo4"


def test_every_gguf_precision_has_an_official_turbo_file():
    """Abiray выпустил тот же набор вариантов, что Unsloth для основной модели."""
    assert set(settings_module.GGUF_VARIANTS) == {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M"}


def test_the_schedule_matches_the_checkpoint():
    index = config.TURBO_OFFICIAL_DIR / "model_index.json"
    if not index.is_file():
        pytest.skip("the official Turbo is not downloaded")
    saved = json.loads(index.read_text(encoding="utf-8"))["sample_sigmas"]
    assert list(presets.SIGMAS_OFFICIAL) == saved


def test_missing_and_download_pick_bf16_shards_or_the_gguf_file(tmp_path):
    calls = []

    def downloader(*, repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    gguf = tmp_path / fetch.OFFICIAL_TURBO_GGUF_SUBDIR / fetch.official_turbo_gguf("Q4_K_M")
    assert fetch.official_turbo_missing(gguf, tmp_path)
    assert fetch.ensure_official_turbo(tmp_path, "Q4_K_M", downloader) is True
    assert not fetch.official_turbo_missing(gguf, tmp_path)
    assert (fetch.OFFICIAL_TURBO_GGUF_REPO, fetch.official_turbo_gguf("Q4_K_M")) in calls
    assert not any(name.endswith(".safetensors") for _repo, name in calls), "bf16-шарды при GGUF не качаются"
    assert fetch.official_turbo_missing(tmp_path, tmp_path), "bf16-шардов нет — так и сказано"


class _Swap:
    def __init__(self, name, log):
        self.name, self.log, self.active = name, log, False

    def activate(self, enabled):
        if enabled != self.active:
            self.log.append((self.name, enabled))
        self.active = enabled


class _Adapters:
    def sync(self, items):
        pass

    def activate(self, active):
        pass


def test_the_generator_swaps_to_the_preset_transformer_and_back():
    log = []
    swaps = {presets.TURBO4: _Swap("turbo4", log), presets.TURBO_OFFICIAL: _Swap("official", log)}
    pipe = FakePipeline()
    engine = gen.Generator(pipe, residency=None, cache=None, catalogue={}, adapters=_Adapters(), swaps=swaps)
    fast = presets.QualityPreset("Fast", 1024, 8, transformer=presets.TURBO_OFFICIAL, sigmas=presets.SIGMAS_OFFICIAL)

    engine.generate(gen.GenerationRequest(prompt="cat", preset=fast))
    assert log == [("official", True)]
    assert pipe.calls[-1]["sigmas"] == list(presets.SIGMAS_OFFICIAL) and pipe.calls[-1]["true_cfg_scale"] == 1.0

    engine.generate(gen.GenerationRequest(prompt="cat", preset=presets.get("Turbo4")))
    assert log[-2:] == [("official", False), ("turbo4", True)], "прежний подменный — назад, затем нужный"

    engine.generate(gen.GenerationRequest(prompt="cat", preset=presets.get("LowQuality")))
    assert log[-1] == ("turbo4", False) and "sigmas" not in pipe.calls[-1]


def test_a_preset_without_its_transformer_is_refused():
    engine = gen.Generator(FakePipeline(), residency=None, cache=None, catalogue={}, adapters=_Adapters())
    fast = presets.QualityPreset("Fast", 1024, 8, transformer=presets.TURBO_OFFICIAL)
    with pytest.raises(RuntimeError, match="Fast"):
        engine.generate(gen.GenerationRequest(prompt="cat", preset=fast))
