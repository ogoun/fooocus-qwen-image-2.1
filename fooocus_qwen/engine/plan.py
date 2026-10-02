"""Какие веса и какая раскладка нужны под выбор пользователя — одно решение на всех.

Выбор (``user/settings.json``) — это точность трансформера и профиль памяти,
а следствий у него много: какой файл трансформера читать, нужен ли
INT8-энкодер, как раскладывать веса по памяти, какой адаптер Turbo качать,
нужны ли bf16-шарды энкодера. Раньше каждое следствие выводилось заново там,
где понадобилось, — при загрузке модели, в самопроверке, в установке, во
вкладке настроек, — и вывод в одном месте мог разойтись с другим. Теперь его
делает ``resolve``, а остальные читают готовый ``WeightsPlan``.

Модуль не импортирует torch: самопроверка и установка спрашивают его до
того, как модель понадобится.

План знает и о пакетах, без которых выбор не загрузится: GGUF читает пакет
``gguf``. Окружение, поставленное до появления GGUF, его не имеет, и смена
точности в настройках падала на загрузке с ``No module named 'gguf'`` —
после того как веса уже скачались. Теперь недостающий пакет ставится
(``ensure_packages``) там же, где докачиваются веса, и перед загрузкой.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .. import config
from .. import settings as settings_module
from . import fetch

LOGGER = logging.getLogger(__name__)

REQUIREMENTS = config.PROJECT_ROOT / "requirements.txt"


class PackageInstallError(RuntimeError):
    """Пакет, нужный выбору, не поставился."""


def requirement(name: str) -> str:
    """Строка пакета из ``requirements.txt`` (с версией) — одна правда на установку и докачку."""
    try:
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            spec = line.split("#", 1)[0].strip()
            if spec and spec.replace("=", " ").replace(">", " ").replace("<", " ").split()[0] == name:
                return spec
    except OSError:
        pass
    return name


# Политики размещения (``engine/residency.py``); здесь — строками, чтобы не
# тянуть torch ради двух имён.
SWAP = "swap"
STREAM = "stream"


@dataclass(frozen=True)
class WeightsPlan:
    """Следствия выбора: файлы, раскладка, что качать.

    ``profile`` — уже решённый (``high`` или ``low``), не ``auto``.
    """

    precision: str
    profile: str
    model_dir: Path = config.MODEL_DIR

    @property
    def low(self) -> bool:
        """Профиль «low»: трансформер не покидает карту, энкодер в INT8 — по блоку."""
        return self.profile == settings_module.MEMORY_LOW

    @property
    def policy(self) -> str:
        return STREAM if self.low else SWAP

    @property
    def int8_file(self) -> Path | None:
        if self.precision == settings_module.PRECISION_INT8:
            return config.INT8_DIR / fetch.INT8_FILE
        return None

    @property
    def gguf_file(self) -> Path | None:
        if settings_module.is_gguf(self.precision):
            return config.GGUF_DIR / fetch.gguf_file(self.precision)
        return None

    @property
    def text_encoder_source(self) -> Path:
        """bf16-энкодер в каталоге модели — источник INT8-копии."""
        return Path(self.model_dir) / "text_encoder"

    @property
    def text_encoder_dir(self) -> Path | None:
        """Сжатый энкодер профиля «low» или ``None`` — bf16 из каталога модели."""
        return config.TE_INT8_DIR if self.low else None

    @property
    def turbo_files(self) -> tuple[str, ...]:
        """Файлы адаптера Turbo: в «low» — облегчённый, ранга 128."""
        return fetch.turbo_files(self.low)

    def text_encoder_ready(self) -> bool:
        """INT8-энкодер собран и не устарел (вне «low» он не нужен — и «готов»)."""
        if not self.low:
            return True
        from . import text_encoder

        return text_encoder.is_current(config.TE_INT8_DIR, self.text_encoder_source)

    def needs_text_encoder_shards(self) -> bool:
        """Нужны ли bf16-шарды энкодера (16.3 ГБ): нет, если INT8-копия уже собрана."""
        return not (self.low and self.text_encoder_ready())

    def missing_packages(self) -> list[str]:
        """Пакеты, без которых этот выбор не загрузится (пусто — все на месте)."""
        needed = ["gguf"] if self.gguf_file is not None else []
        return [name for name in needed if importlib.util.find_spec(name) is None]

    def ensure_packages(self, out: Callable[[str], None] | None = None) -> bool:
        """Ставит недостающие пакеты в текущее окружение. ``True`` — что-то ставилось."""
        missing = self.missing_packages()
        if not missing:
            return False
        specs = [requirement(name) for name in missing]
        (out or LOGGER.info)(f"Installing missing packages: {' '.join(specs)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *specs], capture_output=True, text=True, check=False,
        )
        importlib.invalidate_caches()
        still = self.missing_packages()
        if result.returncode != 0 or still:
            tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
            raise PackageInstallError(f"Could not install {' '.join(specs)}: {tail[0]}")
        return True

    def loader_arguments(self) -> dict:
        """Аргументы ``loader.load``, зависящие от выбора."""
        return {
            "int8_file": self.int8_file,
            "gguf_file": self.gguf_file,
            "text_encoder_dir": self.text_encoder_dir,
            "policy": self.policy,
        }

    def missing_transformer(self) -> list[str]:
        """Недостающие файлы трансформера выбранной точности (пусто — на месте)."""
        if self.int8_file is not None:
            return [str(self.int8_file)] if fetch.missing_extra(config.INT8_DIR, (fetch.INT8_FILE,)) else []
        if self.gguf_file is not None:
            missing = fetch.missing_extra(config.GGUF_DIR, (self.gguf_file.name,))
            return [str(self.gguf_file)] if missing else []
        return fetch.missing_files(self.model_dir, include_transformer=True,
                                   include_text_encoder=self.needs_text_encoder_shards())

    def ensure_weights(self) -> bool:
        """Докачивает всё, что нужно этому выбору. ``True`` — что-то качалось.

        Каталог модели — без bf16-шардов трансформера при INT8 и GGUF (их
        место занимает свой файл) и без bf16-шардов энкодера, если его
        INT8-копия уже собрана.
        """
        bf16 = self.int8_file is None and self.gguf_file is None
        # Пакеты — раньше весов: без них скачанное всё равно не загрузится.
        self.ensure_packages()
        downloaded = fetch.ensure_model(
            self.model_dir, include_transformer=bf16, include_text_encoder=self.needs_text_encoder_shards()
        )
        if self.int8_file is not None:
            downloaded = fetch.ensure_int8(config.INT8_DIR) or downloaded
        if self.gguf_file is not None:
            downloaded = fetch.ensure_gguf(config.GGUF_DIR, self.precision) or downloaded
        return downloaded

    def ensure_text_encoder(self, out: Callable[[str], None] | None = None) -> bool:
        """Собирает INT8-энкодер профиля «low», если его нет. ``True`` — собирался."""
        if not self.low:
            return False
        from . import text_encoder

        kwargs = {"out": out} if out is not None else {}
        return text_encoder.ensure(self.text_encoder_source, config.TE_INT8_DIR, **kwargs)


def resolve(
    chosen: settings_module.Settings | None = None,
    model_dir: Path | None = None,
    vram_gib: float | None | object = ...,
    **changes,
) -> WeightsPlan:
    """План для выбора ``chosen`` (по умолчанию — сохранённого) на этой карте.

    ``changes`` — поправки к выбору до сохранения (``precision="Q4_K_M"``):
    вкладка настроек докачивает веса новой точности раньше, чем её запишет.
    ``vram_gib`` — объём видеопамяти; по умолчанию определяется
    (``engine/hardware.py``).
    """
    chosen = chosen or settings_module.load()
    if changes:
        chosen = replace(chosen, **changes)
    if vram_gib is ...:
        from . import hardware

        vram_gib = hardware.vram_gib()
    profile = settings_module.resolve_profile(chosen.memory_profile, vram_gib)
    return WeightsPlan(chosen.precision, profile, Path(model_dir or config.MODEL_DIR))
