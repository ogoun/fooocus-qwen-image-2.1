"""Производительность: настройки, опрос при установке, веса, внимание, turbo, размещение INT8.

Всё здесь проверяется без модели и без сети: скачивание подменяется, пайплайн
— заглушкой с тем же интерфейсом. Настоящая модель проверяется дымовым
прогоном и опытами (`tools/experiments/turbo.py`, `sage_int8.py`).
"""

from __future__ import annotations

import json

import pytest
import torch

from fooocus_qwen import config
from fooocus_qwen import settings as settings_module
from fooocus_qwen.engine import attention, fetch, presets, turbo
from fooocus_qwen.engine import setup as perf_setup
from fooocus_qwen.engine.residency import StagedModule

# --- настройки ---------------------------------------------------------------


def test_missing_settings_mean_the_old_behaviour():
    """Нет файла — bf16 без SageAttention, то есть как до появления настроек."""
    assert settings_module.load() == settings_module.Settings("bf16", False)


def test_settings_round_trip():
    settings_module.save(settings_module.Settings("int8", True))
    assert settings_module.load() == settings_module.Settings("int8", True)
    assert json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8")) == {
        "precision": "int8", "sage_attention": True,
    }


@pytest.mark.parametrize("text", ["{broken", '"a string"', '{"precision": "fp4", "sage_attention": "yes"}'])
def test_a_damaged_settings_file_falls_back_field_by_field(text):
    config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_FILE.write_text(text, encoding="utf-8")
    assert settings_module.load() == settings_module.Settings()


def test_update_changes_only_what_is_named():
    settings_module.save(settings_module.Settings("int8", False))
    settings_module.update(sage_attention=True)
    assert settings_module.load() == settings_module.Settings("int8", True)


def test_an_unknown_precision_is_never_written():
    with pytest.raises(ValueError):
        settings_module.save(settings_module.Settings("fp4", False))
    assert not config.SETTINGS_FILE.exists()


# --- опрос при установке -------------------------------------------------------


def _answers(*replies):
    queue = list(replies)
    return lambda _question: queue.pop(0)


def test_setup_records_int8_and_sage_when_the_package_installs():
    installed = []
    chosen = perf_setup.configure(
        ask=_answers("2", "да"), out=lambda *_a: None,
        install_sage=lambda out: installed.append(True) or True, sage_available=lambda: False,
    )
    assert chosen == settings_module.Settings("int8", True)
    assert installed == [True]
    assert settings_module.load() == chosen


def test_a_failed_sage_install_is_not_recorded_as_enabled():
    chosen = perf_setup.configure(
        ask=_answers("1", "да"), out=lambda *_a: None,
        install_sage=lambda out: False, sage_available=lambda: False,
    )
    assert chosen.sage_attention is False


def test_empty_answers_keep_the_current_choice():
    settings_module.save(settings_module.Settings("int8", True))
    chosen = perf_setup.configure(
        ask=_answers("", ""), out=lambda *_a: None,
        install_sage=lambda out: pytest.fail("ставить нечего"), sage_available=lambda: True,
    )
    assert chosen == settings_module.Settings("int8", True)


def test_no_console_keeps_the_defaults():
    """Конец ввода (сценарий без консоли) — «оставить как есть», а не падение."""

    def no_console(_question):
        raise EOFError

    chosen = perf_setup.configure(ask=no_console, out=lambda *_a: None, sage_available=lambda: False)
    assert chosen == settings_module.Settings()


@pytest.mark.parametrize(
    ("torch_version", "cuda", "triton"),
    [("2.11.0+cu128", "12.8", "triton-windows>=3.7,<3.8"), ("2.10.1+cu130", "13.0", "triton-windows>=3.6,<3.7")],
)
def test_the_sage_wheel_matches_torch_and_cuda(torch_version, cuda, triton):
    plan = perf_setup.sage_install_plan(torch_version, cuda, platform="win32")
    assert plan[0] == [triton]
    assert plan[1][0].endswith(perf_setup.SAGE_WHEELS["cu" + cuda.replace(".", "")])


@pytest.mark.parametrize(
    ("torch_version", "cuda", "platform"),
    [("2.9.1", "12.8", "win32"), ("2.11.0", "12.6", "win32"), ("2.11.0", "12.8", "linux")],
)
def test_no_matching_sage_build_is_explained_not_guessed(torch_version, cuda, platform):
    plan = perf_setup.sage_install_plan(torch_version, cuda, platform=platform)
    assert isinstance(plan, str) and plan


# --- дополнительные веса -------------------------------------------------------


def test_extra_weights_download_only_what_is_missing(tmp_path):
    (tmp_path / "scheduler").mkdir()
    (tmp_path / fetch.TURBO_SCHEDULER).write_text("{}", encoding="utf-8")
    fetched = []

    def downloader(*, repo_id, filename, local_dir):
        fetched.append(filename)
        (local_dir / filename).write_bytes(b"weights")

    assert fetch.ensure_turbo(tmp_path, downloader=downloader) is True
    assert fetched == [fetch.TURBO_LORA]
    assert fetch.ensure_turbo(tmp_path, downloader=downloader) is False


def test_an_incomplete_download_is_an_error(tmp_path):
    with pytest.raises(fetch.ModelDownloadError):
        fetch.ensure_int8(tmp_path, downloader=lambda **_kwargs: None)


def test_int8_install_does_not_need_the_bf16_transformer(tmp_path):
    """Кто выбрал INT8, bf16-шарды трансформера (14 ГБ) не качает и не ждёт."""
    (tmp_path / fetch.MARKER).write_text("{}", encoding="utf-8")
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "diffusion_pytorch_model-00001.safetensors"}}), encoding="utf-8"
    )
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    assert fetch.missing_files(tmp_path, include_transformer=False) == []
    assert fetch.missing_files(tmp_path) == ["transformer/diffusion_pytorch_model-00001.safetensors"]

    calls = []
    fetch.ensure_model(tmp_path, downloader=lambda **kwargs: calls.append(kwargs), include_transformer=False)
    assert calls == []


# --- turbo ----------------------------------------------------------------------


def test_turbo_arguments_follow_the_authors_rules():
    arguments = turbo.call_arguments(
        {"num_inference_steps": 28, "true_cfg_scale": 4.0, "negative_prompt": "blurry", "prompt": "a cat"}
    )
    assert arguments["num_inference_steps"] == 6
    assert arguments["sigmas"] == [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
    assert arguments["true_cfg_scale"] == 1.0 and arguments["negative_prompt"] is None
    assert arguments["prompt"] == "a cat"


class _Pipe:
    """Ровно то, что трогает адаптер turbo: планировщик, LoRA, включение."""

    def __init__(self):
        self.scheduler = "base"
        self.events = []

    def load_lora_weights(self, path, weight_name, adapter_name):
        self.events.append(("load", weight_name, adapter_name))

    def enable_lora(self):
        self.events.append("enable")

    def disable_lora(self):
        self.events.append("disable")


class _Residency:
    def __init__(self):
        self.restaged = 0

    def restage_transformer(self, change):
        self.restaged += 1
        change(None)


def _turbo_dir(tmp_path):
    (tmp_path / "scheduler").mkdir()
    (tmp_path / fetch.TURBO_LORA).write_bytes(b"lora")
    (tmp_path / fetch.TURBO_SCHEDULER).write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "shift_terminal": None}), encoding="utf-8"
    )
    return tmp_path


def test_turbo_loads_lazily_once_and_restores_the_base_model(tmp_path):
    pipe, residency = _Pipe(), _Residency()
    adapter = turbo.TurboAdapter(pipe, residency, _turbo_dir(tmp_path))
    adapter.activate(False)
    assert residency.restaged == 0 and pipe.events == [], "без запроса Turbo адаптер не подключается"

    adapter.activate(True)
    adapter.activate(True)
    assert residency.restaged == 1, "подключение — однократно, через перерегистрацию трансформера"
    assert pipe.scheduler != "base"

    adapter.activate(False)
    assert pipe.scheduler == "base"
    assert pipe.events == [("load", fetch.TURBO_LORA, "turbo"), "enable", "disable"]


def test_turbo_without_weights_says_so(tmp_path):
    adapter = turbo.TurboAdapter(_Pipe(), _Residency(), tmp_path)
    with pytest.raises(FileNotFoundError):
        adapter.activate(True)


def test_the_generator_refuses_turbo_without_an_adapter():
    from types import SimpleNamespace

    from fooocus_qwen.engine.generator import GenerationRequest, Generator

    engine = Generator(pipe=SimpleNamespace(_interrupt=False), residency=None, cache=None, catalogue={})
    with pytest.raises(RuntimeError, match="Turbo"):
        engine.generate(GenerationRequest(prompt="a cat", preset=presets.get("Turbo")))


# --- внимание ---------------------------------------------------------------------


def test_sage_requested_without_the_package_stays_native(monkeypatch):
    chosen = []

    class Transformer:
        def set_attention_backend(self, name):
            chosen.append(name)

    monkeypatch.setattr(attention, "sage_available", lambda: False)
    assert attention.apply(Transformer(), True) is False
    assert chosen == ["native"]


def test_the_mask_guard_drops_only_masks_that_hide_nothing(monkeypatch):
    from diffusers.models.transformers import transformer_qwenimage21 as module

    seen = []
    monkeypatch.setattr(module, "dispatch_attention_fn",
                        lambda q, k, v, attn_mask=None, backend=None, **kw: seen.append((attn_mask, backend)))
    attention._install_mask_guard()
    full = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    partial = torch.tensor([[[[True, True, False, False]]]])

    module.dispatch_attention_fn(None, None, None, attn_mask=full, backend="sage")
    module.dispatch_attention_fn(None, None, None, attn_mask=partial, backend=None)
    module.dispatch_attention_fn(None, None, None, attn_mask=partial, backend="native")
    assert seen[0] == (None, "sage"), "маска из одних True ничего не отсекает — опускается"
    assert seen[1][1] == "native", "отсекающая маска уходит штатному ядру"
    assert seen[2][0] is partial and seen[2][1] == "native", "в штатном режиме ничего не меняется"


# --- размещение INT8-весов ----------------------------------------------------------


def test_staging_moves_int8_weights_together_with_their_data():
    """``param.data = …`` у ``Int8Tensor`` меняет лишь обёртку: данные оставались
    на старом устройстве, и умножение падало. Такой параметр заменяется целиком."""
    torchao = pytest.importorskip("torchao.quantization")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = torch.nn.Linear(64, 32, dtype=torch.bfloat16)
    torchao.quantize_(layer, torchao.Int8WeightOnlyConfig())
    staged = StagedModule(layer, device, pin_memory=False)

    staged.to_device()
    assert layer.weight.qdata.device.type == device and layer.bias.device.type == device
    x = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    assert layer(x).shape == (2, 32)

    staged.to_host()
    assert layer.weight.qdata.device.type == "cpu"
    assert staged.nbytes < 64 * 32 * 2 + 32 * 2, "INT8 считается по хранимым байтам, а не как bf16"
