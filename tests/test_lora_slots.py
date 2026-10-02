"""Ячейки LoRA: состояние, восстановление по имени и отпечатку, сообщения; генератор с LoRA."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.engine import lora, presets
from fooocus_qwen.ui import lora_slots
from tests.test_generator import FakePipeline


def _file(path: Path, layer: str = "transformer.transformer_blocks.0.attn.to_q", width: int = 4096) -> Path:
    save_file({f"{layer}.lora_A.weight": torch.zeros(2, width), f"{layer}.lora_B.weight": torch.zeros(width, 2)},
              str(path))
    return path


def test_collect_keeps_only_ticked_named_slots():
    values = (True, "a", 0.8, False, "b", 1.0, True, "", 1.0, True, "c", -0.5, False, "", 1.0)
    assert lora_slots.collect(*values) == [{"name": "a", "weight": 0.8}, {"name": "c", "weight": -0.5}]


def test_values_for_finds_a_renamed_file_by_its_hash(tmp_path):
    path = _file(tmp_path / "renamed.safetensors")
    saved = [{"name": "old name", "weight": 0.7, "hash": lora.short_hash(path)}, {"name": "gone", "weight": 9}]
    values = lora_slots.values_for(tmp_path, saved)
    assert values[:6] == (True, "renamed", 0.7, True, "gone", lora.WEIGHT_MAX), "вес приводится к пределам ползунка"
    assert values[6:] == (False, "", 1.0) * 3


def test_skipped_loras_are_named_in_the_status_line(tmp_path):
    _file(tmp_path / "good.safetensors")
    _file(tmp_path / "old.safetensors", "transformer.transformer_blocks.0.attn.add_q_proj", 3072)
    resolved, message = lora_slots.resolve(tmp_path, [{"name": "good", "weight": 1}, {"name": "old", "weight": 1}], "en")
    assert [item.name for item in resolved] == ["good"]
    assert "“old” skipped" in message and "Qwen-Image 1" in message


def test_the_note_under_a_slot_says_rank_layers_and_problems(tmp_path):
    _file(tmp_path / "good.safetensors")
    _file(tmp_path / "old.safetensors", "transformer.transformer_blocks.0.attn.add_q_proj", 3072)
    assert lora_slots.note(tmp_path, "good", "ru") == "ранг 2 · слоёв: 1"
    assert lora_slots.note(tmp_path, "old", "en").startswith("⚠ Not for Qwen-Image 2.1")
    assert lora_slots.note(tmp_path, "", "en") == ""


class _Adapters:
    def __init__(self):
        self.synced, self.active = None, None

    def sync(self, items):
        self.synced = list(items)

    def activate(self, active):
        self.active = list(active)


class _Turbo:
    def __init__(self):
        self.enabled = False

    def activate(self, enabled):
        self.enabled = enabled

    def adapter_weights(self):
        return [("turbo", 1.0)] if self.enabled else []


def test_the_generator_switches_turbo_and_user_loras_in_one_list(tmp_path):
    path = _file(tmp_path / "style.safetensors")
    item = lora.ResolvedLora("style", 0.5, path, lora.short_hash(path))
    adapters, turbo = _Adapters(), _Turbo()
    engine = gen.Generator(FakePipeline(), residency=None, cache=None, catalogue={}, turbo=turbo, adapters=adapters)

    results = engine.generate(gen.GenerationRequest(prompt="cat", preset=presets.get("Turbo"), loras=(item,)))
    assert adapters.synced == [item]
    assert adapters.active == [("turbo", 1.0), (item.adapter, 0.5)]
    assert results[0].parameters["loras"] == [{"name": "style", "weight": 0.5, "hash": item.hash}]

    engine.generate(gen.GenerationRequest(prompt="cat", preset=presets.get("LowQuality")))
    assert adapters.synced == [] and adapters.active == [], "без LoRA и без Turbo — адаптеры выключены"
