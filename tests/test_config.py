"""Проверки разбора аргументов и путей приложения."""

from pathlib import Path

from fooocus_qwen import config


def test_defaults_listen_on_all_interfaces():
    # Оболочка должна быть доступна из локальной сети без дополнительных ключей.
    cfg = config.parse_args([])
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 7865
    assert cfg.lang == "ru"
    assert cfg.pin_memory is True
    assert cfg.preset == "MiddleQuality"


def test_arguments_override_defaults():
    cfg = config.parse_args(["--host", "127.0.0.1", "--port", "8000", "--lang", "en", "--no-pin-memory"])
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.lang == "en"
    assert cfg.pin_memory is False


def test_model_dir_points_at_downloaded_weights():
    assert config.MODEL_DIR == config.PROJECT_ROOT / "Qwen-Image-2.1"
    assert (config.MODEL_DIR / "model_index.json").is_file()


def test_user_directories_are_created_on_demand():
    config.ensure_directories()
    for path in (config.OUTPUT_DIR, config.PROMPT_DIR, config.LOG_DIR):
        assert path.is_dir()


def test_preset_name_is_validated():
    import pytest

    with pytest.raises(SystemExit):
        config.parse_args(["--preset", "Ultra"])
