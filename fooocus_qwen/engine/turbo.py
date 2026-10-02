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

from . import fetch

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


# --- Turbo4: 4-шаговый дистиллят на отдельном трансформере ---------------------------

# Равномерные узлы — расписание «simple» из карточки Abiray; сдвиг по
# разрешению пайплайн применяет к ним сам, как и к узлам Turbo.
SIGMAS4: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)


class Turbo4Transformer:
    """Пресет Turbo4: подменяет трансформер на 4-шаговый дистиллят и возвращает обратно.

    Дистиллят влит в веса (``fetch.TURBO4_FILE``), поэтому это не адаптер, а
    другой трансформер. Подмена — ``ResidencyManager.replace_transformer``:
    на 24 ГБ основной откладывается на хост и возвращается без диска, на 8 ГБ
    отпускается и читается заново (``base_loader``, несколько секунд).
    ``prepare`` вешает на поставленный трансформер механизм внимания и
    KV-кэш в ОЗУ — то же, что получил основной при загрузке; ``turbo`` —
    адаптер Turbo, который надо забыть, если основной прочитан заново.

    Замер на RTX 4060 Laptop 8 ГБ (``tools/experiments/lowvram_turbo4.py``):
    19.3 с на кадр 1024² против 28.7 у Turbo и 62 у LowQuality; пик 5.7 ГиБ.
    Надписи держит не хуже Turbo, правку — заметно хуже: кадр уходит от
    исходника (``docs/research/2026-10-02-8-gb.md``).
    """

    def __init__(self, pipe, residency, gguf_file: Path, turbo_dir: Path, model_dir: Path,
                 base_loader, prepare, turbo: TurboAdapter | None = None) -> None:
        self._pipe = pipe
        self._residency = residency
        self._file = Path(gguf_file)
        self._turbo_dir = Path(turbo_dir)
        self._model_dir = Path(model_dir)
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
        return not fetch.turbo4_missing(self._file.parent, self._turbo_dir)

    def activate(self, enabled: bool) -> None:
        if enabled == self._active:
            return
        if enabled:
            from diffusers import FlowMatchEulerDiscreteScheduler

            from . import gguf

            missing = fetch.turbo4_missing(self._file.parent, self._turbo_dir)
            if missing:
                raise FileNotFoundError(f"Turbo4 weights missing: {', '.join(missing)}")
            if self._turbo is not None:
                self._turbo.activate(False)
            LOGGER.info("Switching to the Turbo4 transformer %s", self._file.name)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self._turbo_dir), subfolder="scheduler")
            self._parked = self._residency.replace_transformer(
                lambda device: gguf.load_transformer(self._model_dir, self._file, device=device)
            )
            self._prepare(self._pipe.transformer)
            self._base_scheduler = self._pipe.scheduler
            self._pipe.scheduler = scheduler
            self._active = True
            return

        LOGGER.info("Switching back to the main transformer")
        parked, self._parked = self._parked, None
        self._residency.replace_transformer(self._base_loader, parked=parked)
        # И отложенный готовится заново: SageAttention могли переключить, пока
        # работал Turbo4. KV-хук при этом не двоится — отложенный бывает
        # только при SWAP, где хука нет.
        self._prepare(self._pipe.transformer)
        if parked is None and self._turbo is not None:
            # Основной прочитан заново: адаптер Turbo к нему не подключён.
            self._turbo.reset()
        self._pipe.scheduler = self._base_scheduler
        self._active = False

