"""Размещение весов между хостом и видеопамятью.

Задача: 33 ГБ весов против 24 ГБ видеопамяти. Штатный
``enable_model_cpu_offload`` гоняет по шине всё и на каждую генерацию; при
быстром пресете это треть времени.

Принятая политика: трансформер и VAE резидентны, текстовый энкодер живёт на
хосте и поднимается только при промахе кэша эмбеддингов. Тогда перебор сида и
шагов не создаёт трафика по шине вовсе, а перебор разрешения — только пока в
запросе нет условных изображений (см. докстринг ``embeds_cache``: пайплайн
масштабирует их по ``output_resolution`` до кодирования, и отпечаток меняется
вместе с пресетом).

Канонической копией весов считается копия на хосте, а не в модуле. Благодаря
этому закрепление памяти делается один раз: возврат «на хост» — это возврат
ссылки на уже закреплённый тензор, а не новое копирование.

Это политика ``SWAP`` — для карт на 24 ГБ. Для карт на 6–12 ГБ есть
``STREAM``: трансформер (GGUF) не покидает карту вовсе, а энкодер живёт на
хосте и поднимается по блоку (``engine/streaming.py``). Политику выбирает
профиль памяти (``settings.memory_profile``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from .faults import is_gpu_fault

# Имена политик — из плана весов (``engine/plan.py``): там решается, какая
# нужна, и там же они объявлены, без torch.
from .plan import STREAM, SWAP

LOGGER = logging.getLogger(__name__)


def named_tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Параметры и буферы одним потоком.

    Буферы нельзя пропускать: у трансформера в них лежат таблицы поворотных
    вложений, и модуль без них на видеокарте не считается.
    """
    for name, parameter in module.named_parameters(recurse=True):
        yield f"p:{name}", parameter
    for name, buffer in module.named_buffers(recurse=True):
        yield f"b:{name}", buffer


def place_tensor(module: torch.nn.Module, key: str, tensor: torch.Tensor, value: torch.Tensor) -> None:
    """Ставит на место тензора ``value`` — копию того же тензора на другом устройстве.

    Для обычного тензора это ``tensor.data = value``: объект параметра
    остаётся прежним, и все, кто держит на него ссылку, видят новое место.

    Для подклассов тензора (INT8-веса torchao, ``Int8Tensor``) так нельзя:
    присваивание ``.data`` меняет только обёртку — устройство она показывает
    новое, а сами данные (``qdata``, ``scale``) остаются на старом, и первое
    же умножение падает на «тензоры на разных устройствах». Такой параметр
    заменяется новым объектом у модуля-владельца. Ссылок на объекты
    параметров трансформера никто, кроме самого модуля, не держит: адаптеры
    LoRA ссылаются на слой, а не на его вес.
    """
    if type(tensor) in (torch.Tensor, torch.nn.Parameter):
        tensor.data = value
        return
    kind, name = key.split(":", 1)
    owner_name, _, attribute = name.rpartition(".")
    owner = module.get_submodule(owner_name) if owner_name else module
    if kind == "p":
        owner._parameters[attribute] = torch.nn.Parameter(value, requires_grad=False)
    else:
        owner._buffers[attribute] = value


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Объём тензора в байтах, в том числе у подкласса с внутренними тензорами.

    У ``Int8Tensor`` ``numel() * element_size()`` считает логический bf16-вес,
    а хранит он int8 и масштабы — вдвое меньше; сводка памяти врала бы.
    """
    inner = getattr(tensor, "__tensor_flatten__", None)
    if inner is not None and type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        names, _context = inner()
        return sum(tensor_nbytes(getattr(tensor, part)) for part in names)
    return tensor.numel() * tensor.element_size()


# Три состояния размещения вместо булева «резидентен».
#
# Булев флаг не мог описать середину переезда, и это делало сбой необратимым:
# ``to_device()`` выставлял ``_resident = True`` ПОСЛЕ цикла, поэтому падение на
# середине (нехватка видеопамяти на очередном тензоре) оставляло модуль
# разложенным между двумя устройствами при ``_resident == False``, а
# ``to_host()`` на этом флаге делал ранний выход и ничего не чинил. Состояние
# «часть здесь, часть там» обязано быть выразимым, иначе из него нет выхода.
_HOST = "host"
_DEVICE = "device"
_MIXED = "mixed"


class StagedModule:
    """Модуль, чьи веса хранятся на хосте и по требованию поднимаются на устройство."""

    def __init__(self, module: torch.nn.Module, device: str | torch.device, pin_memory: bool = True) -> None:
        self.module = module
        self._device = torch.device(device)
        self._placement = _HOST
        self._host: dict[str, torch.Tensor] = {}
        self._nbytes = 0

        pin_failed = False
        for name, tensor in named_tensors(module):
            host = tensor.detach().to("cpu")
            if pin_memory and not pin_failed:
                try:
                    host = host.pin_memory()
                except RuntimeError as error:
                    # Нехватка памяти под закрепление — повод работать без него:
                    # медленнее, но полностью корректно. Любая другая ошибка CUDA
                    # здесь значит, что контекст процесса сломан (``faults.py``),
                    # и «продолжить без закрепления» было бы неправдой: первое же
                    # ядро упадёт уже с невнятной ошибкой.
                    if is_gpu_fault(error):
                        raise
                    LOGGER.warning("Failed to pin memory, continuing without it: %s", error)
                    pin_failed = True
            self._host[name] = host
            self._nbytes += tensor_nbytes(host)
            place_tensor(module, name, tensor, host)

    @property
    def resident(self) -> bool:
        """Модуль целиком на устройстве. Середина переезда — не «резидентен»."""
        return self._placement == _DEVICE

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def to_device(self) -> None:
        """Поднимает веса на устройство; при сбое откатывает их обратно на хост.

        Откат не косметика: единственный реальный повод упасть здесь — нехватка
        видеопамяти, и тогда недоехавшие веса и бесполезны, и занимают ровно то,
        чего не хватило. Исходное исключение до вызывающей стороны доходит в
        любом случае.
        """
        if self._placement == _DEVICE:
            return

        # Признак «переезд начат» ставится ДО первого присваивания: иначе
        # падение на середине цикла оставило бы половину тензоров на
        # устройстве при флаге «на хосте».
        self._placement = _MIXED
        try:
            for name, tensor in named_tensors(self.module):
                place_tensor(self.module, name, tensor, self._host[name].to(self._device, non_blocking=True))
            if self._device.type == "cuda":
                # Копирование из закреплённой памяти асинхронное: без синхронизации
                # первый же вызов модуля прочитал бы наполовину заполненные веса.
                torch.cuda.synchronize(self._device)
        except BaseException:
            self._rollback_to_host()
            raise
        self._placement = _DEVICE

    def to_host(self) -> None:
        """Возвращает веса на хост. Вызывается и из середины неудавшегося переезда."""
        if self._placement == _HOST:
            return

        self._placement = _MIXED
        for name, tensor in named_tensors(self.module):
            place_tensor(self.module, name, tensor, self._host[name])
        self._placement = _HOST
        if self._device.type == "cuda":
            # Кеширующий аллокатор не возвращает освобождённые блоки драйверу
            # сам по себе — он держит их про запас. На бюджете в 24 ГБ входящий
            # модуль (например, поднимаемый следом текстовый энкодер) может в
            # эти блоки просто не поместиться, если не освободить их явно.
            torch.cuda.empty_cache()

    def _rollback_to_host(self) -> None:
        """Откат после неудавшегося переезда, не подменяющий исходную ошибку.

        Если не удался и он, размещение остаётся ``_MIXED`` — и это правильный
        итог: следующий ``to_host()`` не сделает ранний выход, а повторит
        попытку. Важнее сообщить настоящую причину сбоя, чем причину неудачного
        отката.
        """
        try:
            self.to_host()
        except BaseException:
            LOGGER.exception("Failed to roll weights back to the host; the module is split across devices")


class DeviceModule:
    """Модуль, который живёт на устройстве и не покидает его.

    Трансформер политики ``STREAM``: копия на хосте ему не нужна — он никуда
    не переезжает, а на машине с 16–24 ГБ оперативной памяти четыре лишних
    гигабайта закреплённой копии заметны. Интерфейс — как у ``StagedModule``,
    чтобы менеджер не различал их там, где различать незачем.
    """

    def __init__(self, module: torch.nn.Module, device: str | torch.device) -> None:
        self.module = module
        self._device = torch.device(device)
        self._nbytes = sum(tensor_nbytes(tensor) for _name, tensor in named_tensors(module))

    @property
    def resident(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def to_device(self) -> None:
        # ``Module.to`` присваивает ``.data`` — для GGUFParameter (обычный
        # подкласс, без внутренних тензоров) этого достаточно; уже лежащее на
        # устройстве не копируется.
        self.module.to(self._device)

    def to_host(self) -> None:
        """Ничего не делает: этот модуль с устройства не уходит."""


POLICIES = (SWAP, STREAM)


class ResidencyManager:
    """Владеет размещением трёх моделей пайплайна."""

    def __init__(
        self,
        pipe,
        device: str | torch.device = "cuda",
        pin_memory: bool = True,
        policy: str = SWAP,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown residency policy: {policy!r}")
        self._pipe = pipe
        self._device = torch.device(device)
        self._pin_memory = pin_memory
        self._policy = policy
        self._transformer: StagedModule | DeviceModule | None = None
        self._text_encoder = None
        self._vae: StagedModule | None = None
        self._swaps = 0

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def device(self) -> torch.device:
        return self._device

    def start(self) -> None:
        """Раскладывает модели по местам. Вызывается один раз после загрузки."""
        if self._policy == STREAM:
            from . import streaming

            self._transformer = DeviceModule(self._pipe.transformer, self._device)
            encoder = self._pipe.text_encoder
            self._text_encoder = streaming.StreamedModule(
                encoder, streaming.text_encoder_blocks(encoder), self._device,
                host_modules=streaming.text_encoder_host_modules(encoder),
            )
            # VAE на время кодирования уходит с карты (0.63 ГиБ — запас для
            # энкодера по блоку), поэтому у него копия на хосте. Не закреплённая:
            # переезд один на промах кэша, а закреплённая память на машине с
            # 16–24 ГБ дороже десятых долей секунды.
            self._vae = StagedModule(self._pipe.vae, self._device, pin_memory=False)
            self._vae.to_device()
            self._transformer.to_device()
            LOGGER.info(
                "Resident: transformer %.1f GiB, VAE on device; text encoder %.1f GiB streamed by block",
                self._transformer.nbytes / 2**30,
                self._text_encoder.nbytes / 2**30,
            )
            return

        LOGGER.info("Preparing host copies of weights (pinned: %s)", "yes" if self._pin_memory else "no")
        self._transformer = StagedModule(self._pipe.transformer, self._device, self._pin_memory)
        self._text_encoder = StagedModule(self._pipe.text_encoder, self._device, self._pin_memory)

        # VAE не переставляется никогда, поэтому копия на хосте ему не нужна:
        # это сэкономленные 1.35 ГБ закреплённой памяти.
        self._pipe.vae.to(self._device)
        self._transformer.to_device()

        LOGGER.info(
            "Resident: transformer %.1f GiB, VAE on device; on host: text encoder %.1f GiB",
            self._transformer.nbytes / 2**30,
            self._text_encoder.nbytes / 2**30,
        )

    @contextmanager
    def text_encoder_resident(self) -> Iterator[None]:
        """Поднимает энкодер, вытеснив трансформер, и возвращает всё обратно.

        Вместе они не помещаются: 16.3 плюс 13.3 гигабайта против 24 доступных.
        """
        if self._text_encoder is None or self._transformer is None:
            raise RuntimeError("ResidencyManager.start() was not called")

        self._swaps += 1
        # Обе перестановки — внутри try. Раньше они стояли до него, и сбой
        # ``text_encoder.to_device()`` (поднять 16.3 ГБ на карту в 24 ГБ — это
        # риск нехватки памяти из раздела 13 спецификации) означал, что finally
        # не выполнится и трансформер останется на хосте навсегда: залечить это
        # мог бы только повторный вход сюда, а он бывает лишь при промахе кэша
        # эмбеддингов — пользователь же повторяет тот же промт, попадает в кэш,
        # сюда не заходит, и цикл денойзинга идёт по весам, лежащим на хосте.
        try:
            # При STREAM трансформер остаётся на месте: энкодеру по блоку
            # хватает того, что трансформер оставил свободным.
            if self._policy == SWAP:
                self._transformer.to_host()
            elif self._vae is not None:
                self._vae.to_host()
            self._text_encoder.to_device()
            yield
        finally:
            self._restore_placement()

    def restage_transformer(self, change: Callable[[torch.nn.Module], None]) -> None:
        """Меняет состав параметров трансформера и заново снимает с него копии.

        Перестановка идёт по списку параметров, снятому при ``start()``.
        Подключение адаптера LoRA добавляет новые (``…base_layer``,
        ``…lora_A``), и старый список о них не знает: первая же перестановка
        упала бы на ``KeyError``. Поэтому изменение делается на хосте, а
        трансформер после него регистрируется заново — уже закреплённые
        тензоры повторно не копируются (``pin_memory`` у них — тот же тензор).
        """
        if self._transformer is None:
            raise RuntimeError("ResidencyManager.start() was not called")
        if self._policy == STREAM:
            # Копии на хосте нет, и снимать нечего: изменение делается прямо
            # на устройстве (peft кладёт новые веса туда же, где лежит слой).
            change(self._pipe.transformer)
            self._transformer = DeviceModule(self._pipe.transformer, self._device)
            self._transformer.to_device()
            return
        self._transformer.to_host()
        try:
            change(self._pipe.transformer)
            self._transformer = StagedModule(self._pipe.transformer, self._device, self._pin_memory)
        finally:
            self._transformer.to_device()

    def replace_transformer(
        self,
        load: Callable[[torch.device | None], torch.nn.Module],
        parked: StagedModule | None = None,
    ) -> StagedModule | None:
        """Ставит на место трансформера другой. Возвращает отложенный прежний или ``None``.

        Нужен пресетам на отдельном трансформере (Turbo4 — дистиллят, влитый
        в веса). Прежний сначала освобождает видеопамять, и только потом
        грузится новый: на 8 ГБ два трансформера по 4 ГБ вместе не помещаются.

        * ``SWAP``: у прежнего уже есть копия на хосте — он откладывается туда
          целиком и возвращается обратно без чтения с диска (``parked``).
        * ``STREAM``: копии на хосте нет и держать её негде, прежний
          отпускается; обратно он читается с диска через ``load``.

        ``load(device)`` строит новый трансформер: при ``STREAM`` — прямо на
        устройстве, при ``SWAP`` — на хосте (``device=None``).
        """
        import gc

        if self._transformer is None:
            raise RuntimeError("ResidencyManager.start() was not called")
        kept: StagedModule | None = None
        if self._policy == SWAP and isinstance(self._transformer, StagedModule):
            self._transformer.to_host()
            kept = self._transformer
        self._transformer = None
        self._pipe.transformer = None
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        if parked is not None:
            parked.to_device()
            self._transformer = parked
        else:
            module = load(self._device if self._policy == STREAM else None)
            if self._policy == STREAM:
                self._transformer = DeviceModule(module, self._device)
            else:
                self._transformer = StagedModule(module, self._device, self._pin_memory)
            self._transformer.to_device()
        self._pipe.transformer = self._transformer.module
        return kept

    def restore(self) -> None:
        """Возвращает штатное размещение: трансформер на устройстве, энкодер на хосте.

        Публичная точка восстановления после сбоя, который мог оборвать
        перестановку где угодно, — в том числе после нехватки видеопамяти в
        самом цикле денойзинга. Обе операции идемпотентны, поэтому вызов при
        уже правильном размещении ничего не стоит и ничего не портит; до
        ``start()`` он просто ничего не делает.
        """
        self._restore_placement()

    def _restore_placement(self) -> None:
        if self._text_encoder is None or self._transformer is None:
            return
        # Энкодер снимается первым и под защитой: освобождённая им видеопамять
        # нужна трансформеру, а возврат трансформера — то единственное, без
        # чего приложение перестаёт работать совсем.
        try:
            self._text_encoder.to_host()
        except BaseException:
            LOGGER.exception("Failed to offload the text encoder from the device")
        self._transformer.to_device()
        if self._vae is not None:
            self._vae.to_device()

    def stats(self) -> dict[str, float]:
        allocated = torch.cuda.memory_allocated(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        reserved = torch.cuda.memory_reserved(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        return {"allocated_gib": allocated, "reserved_gib": reserved, "swaps": float(self._swaps)}
