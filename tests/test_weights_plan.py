"""План весов: одно решение о файлах и раскладке для загрузки, установки и самопроверки."""

from __future__ import annotations

from pathlib import Path

from fooocus_qwen import config
from fooocus_qwen import settings as settings_module
from fooocus_qwen.engine import fetch, plan, text_encoder


def _plan(precision: str, profile: str = "auto", vram: float | None = 24.0) -> plan.WeightsPlan:
    chosen = settings_module.Settings(precision=precision, memory_profile=profile)
    return plan.resolve(chosen, model_dir=Path("model"), vram_gib=vram)


def test_a_24_gb_card_swaps_whole_models_and_reads_the_chosen_file():
    chosen = _plan("int8")
    assert chosen.profile == "high" and chosen.policy == plan.SWAP and not chosen.low
    assert chosen.int8_file == config.INT8_DIR / fetch.INT8_FILE and chosen.gguf_file is None
    assert chosen.text_encoder_dir is None and chosen.turbo_files == fetch.TURBO_FILES
    assert chosen.loader_arguments()["policy"] == plan.SWAP


def test_an_8_gb_card_streams_the_int8_encoder_and_takes_the_light_turbo():
    chosen = _plan("Q4_K_M", vram=8.0)
    assert chosen.low and chosen.policy == plan.STREAM
    assert chosen.gguf_file == config.GGUF_DIR / fetch.gguf_file("Q4_K_M") and chosen.int8_file is None
    assert chosen.text_encoder_dir == config.TE_INT8_DIR
    assert chosen.turbo_files == fetch.TURBO_FILES_LIGHT


def test_the_profile_set_by_hand_beats_the_card():
    assert _plan("bf16", profile="low", vram=24.0).low
    assert not _plan("Q4_K_M", profile="high", vram=8.0).low


def test_changes_apply_before_saving():
    """Вкладка настроек докачивает веса новой точности раньше, чем её запишет."""
    chosen = plan.resolve(settings_module.Settings(), vram_gib=24.0, precision="Q6_K")
    assert chosen.precision == "Q6_K" and chosen.gguf_file.name == fetch.gguf_file("Q6_K")


def test_switching_to_bf16_does_not_fetch_an_encoder_already_built(monkeypatch):
    """Регресс: возврат к bf16 в профиле «low» качал заново 16 ГБ bf16-энкодера,
    хотя работает собранная INT8-копия."""
    calls = []
    monkeypatch.setattr(text_encoder, "is_current", lambda *_a, **_k: True)
    monkeypatch.setattr(fetch, "ensure_model", lambda *a, **k: calls.append(k) or False)
    assert _plan("bf16", profile="low").ensure_weights() is False
    assert calls == [{"include_transformer": True, "include_text_encoder": False}]


def test_the_encoder_shards_are_needed_until_the_int8_copy_exists(monkeypatch):
    monkeypatch.setattr(text_encoder, "is_current", lambda *_a, **_k: False)
    assert _plan("Q4_K_M", vram=8.0).needs_text_encoder_shards()
    assert _plan("bf16").needs_text_encoder_shards(), "в «high» bf16-энкодер нужен всегда"


def test_gguf_fetches_its_file_and_not_the_bf16_transformer(monkeypatch):
    calls = []
    monkeypatch.setattr(plan.importlib.util, "find_spec", lambda name: object())  # пакет gguf «на месте»
    monkeypatch.setattr(text_encoder, "is_current", lambda *_a, **_k: False)
    monkeypatch.setattr(fetch, "ensure_model", lambda *a, **k: calls.append(("model", k)) or False)
    monkeypatch.setattr(fetch, "ensure_gguf", lambda directory, variant, *a: calls.append(("gguf", variant)) or True)
    assert _plan("Q4_K_M", vram=8.0).ensure_weights() is True
    assert calls == [("model", {"include_transformer": False, "include_text_encoder": True}), ("gguf", "Q4_K_M")]


def test_the_encoder_is_built_only_in_the_low_profile(monkeypatch):
    built = []
    monkeypatch.setattr(text_encoder, "ensure", lambda source, target, **_k: built.append(target) or True)
    assert _plan("bf16").ensure_text_encoder() is False and not built
    assert _plan("Q4_K_M", vram=8.0).ensure_text_encoder() is True and built == [config.TE_INT8_DIR]


def test_gguf_needs_its_package_and_it_is_installed_before_the_weights(monkeypatch):
    """Регресс: окружение без пакета gguf качало веса Q4_K_M и падало на загрузке."""
    calls = []
    monkeypatch.setattr(plan.importlib.util, "find_spec", lambda name: None if not calls else object())
    monkeypatch.setattr(plan.subprocess, "run", lambda args, **_k: calls.append(args[-1]) or
                        type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(fetch, "ensure_model", lambda *a, **k: calls.append("model") or False)
    monkeypatch.setattr(fetch, "ensure_gguf", lambda *a: calls.append("gguf file") or False)
    monkeypatch.setattr(text_encoder, "is_current", lambda *_a, **_k: False)
    chosen = _plan("Q4_K_M", vram=8.0)
    chosen.ensure_weights()
    assert calls == [plan.requirement("gguf"), "model", "gguf file"], "пакет — раньше весов"
    assert plan.requirement("gguf").startswith("gguf>="), "версия — из requirements.txt"


def test_a_failed_package_install_is_reported(monkeypatch):
    monkeypatch.setattr(plan.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(plan.subprocess, "run", lambda *a, **k: type(
        "Failed", (), {"returncode": 1, "stdout": "", "stderr": "no network"})())
    try:
        _plan("Q4_K_M", vram=8.0).ensure_packages()
    except plan.PackageInstallError as error:
        assert "no network" in str(error)
    else:
        raise AssertionError("отказ установки обязан стать ошибкой")


def test_bf16_and_int8_need_no_extra_packages():
    assert _plan("bf16").missing_packages() == [] and _plan("int8").missing_packages() == []

