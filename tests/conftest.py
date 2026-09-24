"""Общие предохранители для всего набора тестов.

Здесь живёт защита от действий, которые прорываются за пределы прогона и
затрагивают машину разработчика. Точечная подмена в отдельном тесте такой
защитой не является: её достаточно один раз забыть в новом файле, и набор
снова начнёт хозяйничать на рабочем столе. Именно так и вышло — подмена стояла
в тестах галереи, но не в добавленных позже тестах локализации, и каждый прогон
распахивал окно проводника.
"""

from __future__ import annotations

import pytest

from fooocus_qwen.ui import app, tab_gallery


@pytest.fixture(autouse=True)
def never_open_a_file_manager(monkeypatch):
    """Запрещает тестам вызывать файловый менеджер системы.

    Подменяется на запись вызовов, а не на пустышку: тест, которому нужно
    убедиться, что каталог открывали, по-прежнему может это проверить, но ни
    один прогон не откроет окно.
    """
    opened: list = []
    monkeypatch.setattr(tab_gallery, "_open_folder", opened.append)
    return opened


@pytest.fixture(autouse=True)
def never_open_a_browser(monkeypatch):
    """Запрещает тестам открывать окно браузера.

    Та же защита, что и у файлового менеджера, и по той же причине: запуск
    оболочки умеет открывать страницу сам, и набору тестов достаточно один
    раз позвать ``launch`` без подмены, чтобы прогон распахнул вкладку.
    Записываются адреса — тест, которому нужно убедиться, что страницу
    открывали, проверит их.
    """
    opened: list[str] = []
    monkeypatch.setattr(app.webbrowser, "open", lambda url, *_a, **_k: opened.append(url) or True)
    return opened


@pytest.fixture(autouse=True)
def isolated_settings_and_weights(tmp_path, monkeypatch):
    """Настройки производительности и дополнительные веса — во временном каталоге.

    ``user/settings.json`` человека тесты не читают и не пишут: иначе прогон
    мог бы переключить ему точность весов или SageAttention. Каталоги весов
    INT8 и turbo подменяются тем же ходом — тест, проверяющий «веса
    отсутствуют», не должен зависеть от того, скачаны ли они на этой машине.
    """
    from fooocus_qwen import config

    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "INT8_DIR", tmp_path / "int8")
    monkeypatch.setattr(config, "TURBO_DIR", tmp_path / "turbo")


@pytest.fixture
def painter_value(tmp_path, monkeypatch):
    """Собирает значение кисти так, как его собрал бы сервер.

    Каталог загрузок Gradio подменяется временным: разбор значения принимает
    пути только оттуда, и тест проверяет именно этот, настоящий разбор, а не
    подставной. Файлы ложатся в ``tmp_path`` и исчезают вместе с ним.
    """
    from fooocus_qwen.ui.painter import payload

    monkeypatch.setattr(payload, "upload_root", lambda: tmp_path)

    def make(background, layer=None):
        return payload.encode(background, layer)

    return make
