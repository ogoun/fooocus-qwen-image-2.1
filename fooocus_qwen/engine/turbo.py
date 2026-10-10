"""Пресет Turbo: дистиллят Qwen-Image-2.1-viggle-turbo, 6 шагов вместо 16–40.

Viggle выпустили к нашей модели LoRA-адаптер, обученный дистилляцией (DMD):
генерация и правка за 6 проходов трансформера без CFG. Чист он только на
площади 1024², на которой учился: выше появляется сетка с периодом 8 px
(рисунок на уровне токенов латента), поэтому пресет Turbo — 1024²; кадр
1280x832 за ~11 с против ~21 у LowQuality
(``docs/research/2026-09-24-uskorenie-turbo-sage-int8.md``).

Правила автора, которым здесь следует код:

* ровно 6 шагов с «сырыми» узлами ``SIGMAS`` — пайплайн сам применяет к ним
  сдвиг по разрешению;
* свой планировщик: у штатного ``shift_terminal: 0.02`` — он портит
  последний шаг;
* ``true_cfg_scale=1`` и без негативного промта;
* адаптер не сливается с весами: слияние в bf16 необратимо портит точность,
  а подключение при загрузке точное.

Адаптер подключается лениво, при первом запросе Turbo: кто пресетом не
пользуется, не платит за него 1.3 ГБ видеопамяти. Подключение идёт через
``ResidencyManager.restage_transformer`` — адаптер добавляет трансформеру
параметры, о которых менеджер размещения иначе не знал бы. В остальных
пресетах адаптер выключен, и модель считает ровно как без него.

В профиле памяти «low» берётся тот же дистиллят ранга 128 (``light=True``,
0.63 ГБ вместо 1.3): на карте в 8 ГБ разница — это запас на активации.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from . import fetch, presets

LOGGER = logging.getLogger(__name__)

SIGMAS: tuple[float, ...] = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
ADAPTER = "turbo"


class TurboAdapter:
    """Подключает, включает и выключает адаптер turbo у пайплайна."""

    def __init__(self, pipe, residency, weights_dir: Path, light: bool = False) -> None:
        self._pipe = pipe
        self._residency = residency
        self._dir = Path(weights_dir)
        self._files = fetch.turbo_files(light)
        # Штатный планировщик запоминается при подключении адаптера, а не
        # здесь: конструктор не трогает пайплайн, пока turbo не понадобился.
        self._base_scheduler = None
        self._turbo_scheduler = None
        self._active = False

    @property
    def loaded(self) -> bool:
        return self._turbo_scheduler is not None

    def weights_present(self) -> bool:
        return not fetch.missing_extra(self._dir, self._files)

    def _load(self) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler

        missing = fetch.missing_extra(self._dir, self._files)
        if missing:
            raise FileNotFoundError(f"Turbo weights missing in {self._dir}: {', '.join(missing)}")
        LOGGER.info("Attaching turbo adapter from %s", self._dir)
        self._base_scheduler = self._pipe.scheduler
        self._residency.restage_transformer(self._attach)
        self._turbo_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self._dir), subfolder="scheduler")

    def _attach(self, _transformer) -> None:
        """Подключает адаптер к трансформеру, не сливая его с весами.

        На INT8-трансформере peft предупреждает, что ``merge()``/``unmerge()``
        с такими слоями невозможны. Нам они и не нужны: адаптер намеренно
        держится отдельно и включается и выключается на лету (слияние в bf16
        к тому же необратимо портит точность). Предупреждение заглушается
        точечно — только это сообщение и только на время подключения, —
        иначе оно пугало бы в консоли при каждом первом выборе Turbo.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"TorchaoLoraLinear was instantiated without")
            self._pipe.load_lora_weights(str(self._dir), weight_name=self._files[0], adapter_name=ADAPTER)

    def reset(self) -> None:
        """Забывает подключённый адаптер: основной трансформер прочитан заново.

        В профиле «low» уход на Turbo4 отпускает основной трансформер, а
        возврат читает его с диска — без адаптера. Следующий выбор Turbo
        подключит адаптер снова.
        """
        if self._active and self._base_scheduler is not None:
            self._pipe.scheduler = self._base_scheduler
        self._turbo_scheduler = None
        self._base_scheduler = None
        self._active = False

    def activate(self, enabled: bool) -> None:
        """Готовит turbo к следующему вызову пайплайна или убирает: адаптер и планировщик.

        Включает сам адаптер не он, а генератор — одним ``set_adapters``
        вместе с пользовательскими LoRA (``lora.LoraAdapters.activate``): у
        peft один список активных адаптеров на всех. Здесь — подключение при
        первом запросе и смена планировщика.
        """
        if enabled and not self.loaded:
            self._load()
        if enabled == self._active:
            return
        self._pipe.scheduler = self._turbo_scheduler if enabled else self._base_scheduler
        self._active = enabled

    def adapter_weights(self) -> list[tuple[str, float]]:
        """Адаптер turbo с весом 1, если он сейчас нужен, — для общего ``set_adapters``."""
        return [(ADAPTER, 1.0)] if self._active else []


def call_arguments(arguments: dict, sigmas: tuple[float, ...] = SIGMAS) -> dict:
    """Аргументы вызова пайплайна для дистиллята: его узлы, без CFG и негатива.

    ``sigmas`` — узлы Turbo (``SIGMAS``) или Turbo4 (``SIGMAS4``).
    """
    changed = dict(arguments)
    changed["num_inference_steps"] = len(sigmas)
    changed["sigmas"] = list(sigmas)
    changed["true_cfg_scale"] = 1.0
    changed["negative_prompt"] = None
    return changed


# --- пресеты на отдельном трансформере: Turbo4 и официальный Turbo -----------------

# Узлы Turbo4 (равномерные, «simple» из карточки Abiray) и официального
# Turbo (``sample_sigmas`` его ``model_index.json``: наш diffusers их оттуда
# не читает — это PR #14950 — и они передаются явно) живут в пресетах.
SIGMAS4 = presets.SIGMAS4
SIGMAS_OFFICIAL = presets.SIGMAS_OFFICIAL


class SwapTransformer:
    """Пресет на отдельном трансформере: подменяет основной и возвращает обратно.

    Дистиллят, влитый в веса, — это не адаптер, а другой трансформер.
    Подмена — ``ResidencyManager.replace_transformer``: на 24 ГБ основной
    откладывается на хост и возвращается без диска, на 8 ГБ отпускается и
    читается заново (``base_loader``, несколько секунд). ``prepare`` вешает
    на поставленный трансформер механизм внимания и KV-кэш в ОЗУ — то же, что
    получил основной при загрузке; ``turbo`` — адаптер Turbo, который надо
    забыть, если основной прочитан заново. ``load(device)`` строит
    трансформер пресета, ``scheduler()`` — его планировщик, ``missing()`` —
    недостающие файлы.
    """

    name = "swap"

    def __init__(self, pipe, residency, load, scheduler, missing, base_loader, prepare,
                 turbo: TurboAdapter | None = None) -> None:
        self._pipe = pipe
        self._residency = residency
        self._load = load
        self._scheduler = scheduler
        self._missing = missing
        self._base_loader = base_loader
        self._prepare = prepare
        self._turbo = turbo
        self._parked = None
        self._base_scheduler = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def weights_present(self) -> bool:
        return not self._missing()

    def activate(self, enabled: bool) -> None:
        if enabled == self._active:
            return
        if enabled:
            missing = self._missing()
            if missing:
                raise FileNotFoundError(f"{self.name} weights missing: {', '.join(missing)}")
            if self._turbo is not None:
                self._turbo.activate(False)
            LOGGER.info("Switching to the %s transformer", self.name)
            scheduler = self._scheduler()
            self._parked = self._residency.replace_transformer(self._load)
            self._prepare(self._pipe.transformer)
            self._base_scheduler = self._pipe.scheduler
            self._pipe.scheduler = scheduler
            self._active = True
            return

        LOGGER.info("Switching back to the main transformer")
        parked, self._parked = self._parked, None
        self._residency.replace_transformer(self._base_loader, parked=parked)
        # И отложенный готовится заново: SageAttention могли переключить, пока
        # работал подменный. KV-хук при этом не двоится — отложенный бывает
        # только при SWAP, где хука нет.
        self._prepare(self._pipe.transformer)
        if parked is None and self._turbo is not None:
            # Основной прочитан заново: адаптер Turbo к нему не подключён.
            self._turbo.reset()
        self._pipe.scheduler = self._base_scheduler
        self._active = False


def _scheduler_from(directory: Path):
    def build():
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler.from_pretrained(str(directory), subfolder="scheduler")
    return build


class Turbo4Transformer(SwapTransformer):
    """Пресет Turbo4: 4-шаговый дистиллят Viggle, GGUF Q4_K_M от Abiray.

    Замер на RTX 4060 Laptop 8 ГБ (``tools/experiments/lowvram_turbo4.py``):
    19.3 с на кадр 1024² против 28.7 у Turbo и 62 у LowQuality; пик 5.7 ГиБ.
    Надписи держит не хуже Turbo, правку — заметно хуже: кадр уходит от
    исходника (``docs/research/2026-10-02-8-gb.md``).
    """

    name = "Turbo4"

    def __init__(self, pipe, residency, gguf_file: Path, turbo_dir: Path, model_dir: Path,
                 base_loader, prepare, turbo: TurboAdapter | None = None) -> None:
        gguf_file, turbo_dir, model_dir = Path(gguf_file), Path(turbo_dir), Path(model_dir)

        def load(device):
            from . import gguf

            return gguf.load_transformer(model_dir, gguf_file, device=device)

        super().__init__(
            pipe, residency, load, _scheduler_from(turbo_dir),
            lambda: fetch.turbo4_missing(gguf_file.parent, turbo_dir), base_loader, prepare, turbo,
        )


class OfficialTurboTransformer(SwapTransformer):
    """Пресеты официального Qwen-Image-2.1-Turbo: 8 шагов, CFG 1, до 2K.

    Трансформер — по выбранной точности основной модели: bf16 — из
    чекпоинта Qwen (``Qwen-Image-2.1-Turbo/transformer``), иначе — GGUF того
    же уровня (``Abiray/Qwen-Image-2.1-Turbo-GGUF``; для INT8 — Q8_0, почти
    без потерь, 7 ГБ). Энкодер и VAE у чекпоинта те же, что у основной
    модели (сверено потензорно), поэтому подменяется только трансформер.
    Планировщик — из чекпоинта (``shift 1``, без динамического сдвига).
    """

    name = "Turbo 2.1"

    def __init__(self, pipe, residency, weights: Path, official_dir: Path, model_dir: Path,
                 base_loader, prepare, turbo: TurboAdapter | None = None) -> None:
        weights, official_dir, model_dir = Path(weights), Path(official_dir), Path(model_dir)

        def load(device):
            if weights.suffix == ".gguf":
                from . import gguf

                return gguf.load_transformer(model_dir, weights, device=device)
            import torch
            from diffusers import QwenImage21Transformer2DModel

            module = QwenImage21Transformer2DModel.from_pretrained(
                str(weights), subfolder="transformer", torch_dtype=torch.bfloat16
            ).eval()
            return module.to(device) if device is not None else module

        super().__init__(
            pipe, residency, load, _scheduler_from(official_dir),
            lambda: fetch.official_turbo_missing(weights, official_dir), base_loader, prepare, turbo,
        )
