"""LoRA: форматы ключей, проверка «для этой ли модели», перевод весов и подключение."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fooocus_qwen.engine import lora
from fooocus_qwen.imaging import metadata

R = 4  # ранг тестовых LoRA
H, M = 4096, 12288


def _pair(prefix: str, out_features: int, in_features: int, a="lora_A", b="lora_B", mid=""):
    return {
        f"{prefix}.{a}{mid}.weight": torch.randn(R, in_features, dtype=torch.bfloat16),
        f"{prefix}.{b}{mid}.weight": torch.randn(out_features, R, dtype=torch.bfloat16),
    }


def _save(path, tensors, meta=None):
    save_file(tensors, str(path), metadata=meta)
    return path


def test_layer_names_match_the_real_transformer():
    """Список слоёв 2.1 задан руками — сверяется с самой моделью на ``meta``."""
    pytest.importorskip("accelerate")
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel

    from fooocus_qwen import config

    model_config = QwenImage21Transformer2DModel.load_config(str(config.MODEL_DIR / "transformer")) \
        if (config.MODEL_DIR / "transformer" / "config.json").is_file() else None
    if model_config is None:
        pytest.skip("model config is not available")
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(model_config)
    real = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    assert real == set(lora.layer_names())
    shapes = {name.split(".", 2)[2]: tuple(module.weight.shape)
              for name, module in model.named_modules()
              if name.startswith("transformer_blocks.0.") and isinstance(module, torch.nn.Linear)}
    assert shapes == {key: value for key, value in lora._SHAPES.items() if key != lora.FUSED_GATE_UP}


def test_diffusers_peft_format_is_usable(tmp_path):
    path = _save(tmp_path / "peft.safetensors", {
        **_pair("transformer.transformer_blocks.0.attn.to_q", H, H),
        **_pair("transformer.transformer_blocks.0.img_mlp.gate_layer", M, H),
    })
    info = lora.inspect(path)
    assert info.usable and info.layers == 2 and info.rank == R


def test_kohya_flat_names_keep_gate_layer_whole(tmp_path):
    """diffusers' own converter turns ``gate_layer`` into ``gate.layer`` — ours does not."""
    tensors = _pair("lora_unet_transformer_blocks_3_img_mlp_gate_layer", M, H, "lora_down", "lora_up")
    tensors["lora_unet_transformer_blocks_3_img_mlp_gate_layer.alpha"] = torch.tensor(8.0)
    tensors.update(_pair("lora_unet_txt_in_in_layer", H, H, "lora_down", "lora_up"))
    path = _save(tmp_path / "kohya.safetensors", tensors)
    assert lora.inspect(path).usable
    converted = lora.convert(tensors)
    up = tensors["lora_unet_transformer_blocks_3_img_mlp_gate_layer.lora_up.weight"]
    assert torch.allclose(converted["transformer.transformer_blocks.3.img_mlp.gate_layer.lora_B.weight"].float(),
                          up.float() * (8.0 / R), atol=1e-2), "alpha / rank вложен в lora_B"
    assert "transformer.txt_in.in_layer.lora_A.weight" in converted


def test_comfyui_fused_gate_up_is_split_gate_first(tmp_path):
    tensors = _pair("diffusion_model.transformer_blocks.1.img_mlp.gate_up", 2 * M, H)
    path = _save(tmp_path / "comfy.safetensors", tensors)
    assert lora.inspect(path).usable
    converted = lora.convert(tensors)
    up = tensors["diffusion_model.transformer_blocks.1.img_mlp.gate_up.lora_B.weight"]
    gate = converted["transformer.transformer_blocks.1.img_mlp.gate_layer.lora_B.weight"]
    proj = converted["transformer.transformer_blocks.1.img_mlp.proj.lora_B.weight"]
    assert torch.equal(gate, up[:M]) and torch.equal(proj, up[M:])
    assert torch.equal(converted["transformer.transformer_blocks.1.img_mlp.proj.lora_A.weight"],
                       tensors["diffusion_model.transformer_blocks.1.img_mlp.gate_up.lora_A.weight"])


def test_diffsynth_default_names_and_trigger_word(tmp_path):
    tensors = _pair("transformer_blocks.0.attn.to_k", H, H, mid=".default")
    path = _save(tmp_path / "ds.safetensors", tensors, {"trigger_word": "neutral exposure"})
    info = lora.inspect(path)
    assert info.usable and info.triggers == ("neutral exposure",)
    assert "transformer.transformer_blocks.0.attn.to_k.lora_B.weight" in lora.convert(tensors)


def test_peft_alpha_from_metadata_is_applied():
    """Pruna: ранг 64 при alpha 128 — без метаданных LoRA действовала бы вполсилы."""
    tensors = _pair("transformer.transformer_blocks.0.attn.to_v", H, H)
    meta = {"lora_adapter_metadata": json.dumps({"transformer.lora_alpha": 2 * R, "transformer.r": R})}
    converted = lora.convert(tensors, meta)
    assert torch.allclose(converted["transformer.transformer_blocks.0.attn.to_v.lora_B.weight"].float(),
                          tensors["transformer.transformer_blocks.0.attn.to_v.lora_B.weight"].float() * 2)


@pytest.mark.parametrize(("tensors", "problem", "family"), [
    (_pair("transformer.transformer_blocks.0.attn.add_q_proj", 3072, 3072), "other_model", "qwen-image"),
    (_pair("transformer.transformer_blocks.0.attn.to_q", 3072, 3072), "other_model", "unknown"),
    (_pair("lora_unet_down_blocks_0_attentions_0_proj_in", 320, 320, "lora_down", "lora_up"), "other_model", "sdxl"),
    ({**_pair("transformer.transformer_blocks.0.attn.to_q", H, H),
      "transformer.transformer_blocks.0.attn.to_q.dora_scale": torch.ones(1, H)}, "unsupported", "dora"),
    ({"lycoris_transformer_blocks_0_attn_to_q.lokr_w1": torch.ones(4, 4)}, "unsupported", "lokr"),
])
def test_foreign_files_are_refused_with_a_reason(tmp_path, tensors, problem, family):
    info = lora.inspect(_save(tmp_path / "x.safetensors", tensors))
    assert (info.problem, info.family) == (problem, family)


def test_the_comfyui_prefix_is_not_mistaken_for_a_full_diff(tmp_path):
    """«.diff» — подстрока «model.diffusion_model.»; признак ищется по сегментам имени."""
    tensors = _pair("model.diffusion_model.transformer_blocks.0.attn.to_q", H, H)
    assert lora.inspect(_save(tmp_path / "c.safetensors", tensors)).usable


def test_text_encoder_keys_are_skipped_not_fatal(tmp_path):
    tensors = {**_pair("transformer.transformer_blocks.0.attn.to_q", H, H),
               **_pair("lora_te_text_model_encoder_layers_0_q_proj", 64, 64, "lora_down", "lora_up")}
    info = lora.inspect(_save(tmp_path / "te.safetensors", tensors))
    assert info.usable and info.skipped_text_encoder == 2
    assert all(key.startswith("transformer.transformer_blocks") for key in lora.convert(tensors))


def test_triggers_from_a_civitai_card(tmp_path):
    path = _save(tmp_path / "style.v1.safetensors", _pair("transformer.transformer_blocks.0.attn.to_q", H, H))
    (tmp_path / "style.v1.civitai.info").write_text(json.dumps({"trainedWords": ["zzz style", "cel"]}), encoding="utf-8")
    assert lora.inspect(path).triggers == ("zzz style", "cel")


def test_library_lists_nested_files_and_refuses_paths_outside(tmp_path):
    (tmp_path / "styles").mkdir()
    _save(tmp_path / "styles" / "Anime.safetensors", _pair("transformer.transformer_blocks.0.attn.to_q", H, H))
    _save(tmp_path / "b.safetensors", _pair("transformer.transformer_blocks.0.attn.to_q", H, H))
    assert lora.list_names(tmp_path) == ["b", "styles/Anime"]
    assert lora.path_for(tmp_path, "styles/Anime").name == "Anime.safetensors"
    with pytest.raises(lora.LoraError) as raised:
        lora.path_for(tmp_path / "styles", "../b")
    assert raised.value.code == "outside"


def test_resolve_drops_zero_weights_duplicates_and_reports_problems(tmp_path):
    good = _save(tmp_path / "good.safetensors", _pair("transformer.transformer_blocks.0.attn.to_q", H, H))
    (tmp_path / "copy.safetensors").write_bytes(good.read_bytes())
    _save(tmp_path / "old.safetensors", _pair("transformer.transformer_blocks.0.attn.add_q_proj", 3072, 3072))
    resolved, problems = lora.resolve(tmp_path, [
        lora.LoraChoice("good", 0.7), lora.LoraChoice("copy", 1.0), lora.LoraChoice("good", 0.0),
        lora.LoraChoice("old", 1.0), lora.LoraChoice("absent", 1.0),
    ])
    assert [(item.name, item.weight) for item in resolved] == [("good", 0.7)], "копия того же файла — один адаптер"
    assert [(error.name, error.code) for error in problems] == [("old", "other_model"), ("absent", "missing")]
    assert resolved[0].adapter == lora.USER_PREFIX + lora.short_hash(good)


class _Pipe:
    def __init__(self):
        self.transformer = SimpleNamespace(peft_config={"turbo": {}, "user_old": {}})
        self.events = []

    def delete_adapters(self, names):
        self.events.append(("delete", tuple(names)))
        for name in names:
            self.transformer.peft_config.pop(name)

    def load_lora_weights(self, state, adapter_name):
        self.events.append(("load", adapter_name, len(state)))
        self.transformer.peft_config[adapter_name] = {}

    def enable_lora(self):
        self.events.append("enable")

    def disable_lora(self):
        self.events.append("disable")

    def set_adapters(self, names, weights):
        self.events.append(("set", tuple(names), tuple(weights)))


class _Residency:
    def __init__(self):
        self.restaged = 0

    def restage_transformer(self, change):
        self.restaged += 1
        change(None)


def test_adapters_load_what_is_ticked_drop_the_rest_and_keep_turbo(tmp_path):
    path = _save(tmp_path / "a.safetensors", _pair("transformer.transformer_blocks.0.attn.to_q", H, H))
    item = lora.ResolvedLora("a", 0.6, path, lora.short_hash(path))
    pipe, residency = _Pipe(), _Residency()
    adapters = lora.LoraAdapters(pipe, residency)

    adapters.sync([item])
    assert residency.restaged == 1, "загрузка и выгрузка — одной перестановкой"
    assert pipe.events == [("delete", ("user_old",)), ("load", item.adapter, 2)]
    assert set(pipe.transformer.peft_config) == {"turbo", item.adapter}

    adapters.sync([item])
    assert residency.restaged == 1, "всё на месте — трансформер не трогается"

    adapters.activate([("turbo", 1.0), (item.adapter, 0.6)])
    assert pipe.events[-2:] == ["enable", ("set", ("turbo", item.adapter), (1.0, 0.6))]
    adapters.activate([])
    assert pipe.events[-1] == "disable"


def test_the_readable_line_carries_lora_hashes_like_a1111():
    line = metadata.to_readable({"prompt": "cat", "loras": [{"name": "zzz", "weight": 0.8, "hash": "abcdef0123"}]})
    assert 'Lora hashes: "zzz: abcdef0123"' in line and 'Lora weights: "zzz: 0.8"' in line
