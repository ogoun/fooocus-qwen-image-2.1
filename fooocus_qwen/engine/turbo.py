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

    def __init__(self, pipe, residency, weights_dir: Path) -> None:
        self._pipe = pipe
        self._residency = residency
        self._dir = Path(weights_dir)
        # Штатный планировщик запоминается при подключении адаптера, а не
        # здесь: конструктор не трогает пайплайн, пока turbo не понадобился.
        self._base_scheduler = None
        self._turbo_scheduler = None
        self._active = False

    @property
    def loaded(self) -> bool:
        return self._turbo_scheduler is not None

    def weights_present(self) -> bool:
        return not fetch.missing_extra(self._dir, fetch.TURBO_FILES)

    def _load(self) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler

        missing = fetch.missing_extra(self._dir, fetch.TURBO_FILES)
        if missing:
            raise FileNotFoundError(f"Нет весов turbo в {self._dir}: {', '.join(missing)}")
        LOGGER.info("Подключаю адаптер turbo из %s", self._dir)
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
            self._pipe.load_lora_weights(str(self._dir), weight_name=fetch.TURBO_LORA, adapter_name=ADAPTER)

    def activate(self, enabled: bool) -> None:
        """Включает turbo для следующего вызова пайплайна или выключает его."""
        if enabled and not self.loaded:
            self._load()
        if enabled == self._active:
            return
        if enabled:
            self._pipe.enable_lora()
            self._pipe.scheduler = self._turbo_scheduler
        else:
            self._pipe.disable_lora()
            self._pipe.scheduler = self._base_scheduler
        self._active = enabled


def call_arguments(arguments: dict) -> dict:
    """Аргументы вызова пайплайна для turbo: шаги, узлы, без CFG и негатива."""
    changed = dict(arguments)
    changed["num_inference_steps"] = len(SIGMAS)
    changed["sigmas"] = list(SIGMAS)
    changed["true_cfg_scale"] = 1.0
    changed["negative_prompt"] = None
    return changed
