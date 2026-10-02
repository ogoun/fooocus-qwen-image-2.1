"""Пользовательские LoRA: библиотека файлов, проверка, перевод ключей и подключение.

Как у Fooocus: файлы ``*.safetensors`` лежат в одном каталоге
(``config.LORA_DIR``, ключ ``--lora-dir`` — можно указать каталог ComfyUI),
в интерфейсе — несколько ячеек «вкл. · файл · вес». Как у InvokeAI: файл
проверяется до подключения — для этой ли он модели, — и слова-триггеры
показываются рядом.

**Для какой модели файл.** Qwen-Image-2.1 устроена иначе, чем Qwen-Image
1.x / 2512 / Edit: 32 однопоточных блока, в каждом ``attn.to_q/k/v/to_out.0``
и SwiGLU ``img_mlp.proj/gate_layer/out``; у прежних моделей — двухпоточные
блоки (``add_q_proj``, ``txt_mlp``, ``img_mlp.net.0.proj``). Совпадение имён
слоёв и решает: LoRA подходит, если все её слои есть у 2.1. Чужая LoRA
не подключается — peft молча пропустил бы несовпавшие слои, и человек
получил бы картинку без эффекта и без объяснения.

**Форматы ключей**, которые переводятся в формат diffusers/peft
(``transformer.<слой>.lora_A.weight``):

* diffusers/peft — ``transformer.`` и ``lora_A``/``lora_B``;
* peft с именем адаптера — ``….lora_A.default.weight``;
* ComfyUI и musubi-tuner — префикс ``diffusion_model.`` (или
  ``model.diffusion_model.``) и вход SwiGLU одним слоем ``img_mlp.gate_up``;
* kohya — ``lora_unet_<слой с подчёркиваниями>.lora_down/lora_up`` и
  ``.alpha``.

Свой перевод, а не ``_convert_non_diffusers_qwen_lora_to_diffusers`` из
diffusers: тот написан для Qwen-Image 1 и в именах kohya восстанавливает
точки эвристикой по известным словам — ``gate_layer`` 2.1 он превратил бы в
``gate.layer``. Здесь плоское имя сверяется со списком настоящих слоёв 2.1,
угадывать нечего.

``alpha`` вкладывается в ``lora_B`` (вес умножается на ``alpha / rank``):
peft без него считает масштаб единичным. Берётся он из ключа ``.alpha``
(kohya) или из метаданных ``lora_adapter_metadata`` (peft: ``lora_alpha`` и
``alpha_pattern``) — у Pruna, например, ранг 64 при alpha 128, и без
метаданных LoRA действовала бы вполсилы. Слитый ``gate_up`` режется по
строкам ``lora_B`` пополам — первая половина ``gate_layer``, вторая ``proj``
(тот же порядок, что у весов GGUF, ``engine/gguf.py``); ``lora_A`` общий.

LoKr, LoHa, DoRA и полные разности весов не поддерживаются: diffusers их не
подключает, а перевод их в обычную LoRA — приближение, о котором человек бы
не узнал.

**Подключение** — адаптерами peft, не сливая с весами: вес меняется на
лету без перезагрузки, а слияние в bf16 необратимо огрубляет веса (то же
решение, что у Turbo, ``engine/turbo.py``). Истина о подключённых адаптерах
— ``peft_config`` самого трансформера, а не своя запись: трансформер бывает
подменён (Turbo4) и прочитан заново, и своя запись тогда врала бы.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SUFFIX = ".safetensors"
# Ячеек в интерфейсе — как у Fooocus.
SLOTS = 5
WEIGHT_MIN, WEIGHT_MAX, WEIGHT_DEFAULT = -2.0, 2.0, 1.0
# Адаптер Turbo живёт среди тех же адаптеров peft (engine/turbo.py).
TURBO_ADAPTER = "turbo"
USER_PREFIX = "user_"

NUM_LAYERS = 32
_BLOCK_LAYERS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.out", "img_mlp.gate_layer",
)
_TOP_LAYERS = (
    "time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2",
    "txt_in.in_layer", "txt_in.out_layer", "img_in", "modulation.1", "norm_out.linear", "proj_out",
)
FUSED_GATE_UP = "img_mlp.gate_up"
# Формы слоёв блока: (выход, вход). Совпадение имени ещё не значит, что
# слой тот же: у Qwen-Image 1 есть ``attn.to_q`` шириной 3072, а не 4096.
_HIDDEN, _MLP = 4096, 12288
_SHAPES = {
    "attn.to_q": (_HIDDEN, _HIDDEN), "attn.to_k": (_HIDDEN, _HIDDEN), "attn.to_v": (_HIDDEN, _HIDDEN),
    "attn.to_out.0": (_HIDDEN, _HIDDEN), "img_mlp.proj": (_MLP, _HIDDEN), "img_mlp.gate_layer": (_MLP, _HIDDEN),
    "img_mlp.out": (_HIDDEN, _MLP), FUSED_GATE_UP: (2 * _MLP, _HIDDEN),
}

# Признаки чужих моделей — чтобы сказать человеку, для чего его файл.
_FAMILIES = (
    ("qwen-image", ("add_q_proj", "txt_mlp", "img_mod", "txt_mod", "img_mlp.net", "img_mlp_net", "add_k_proj")),
    ("flux", ("double_blocks", "single_blocks", "single_transformer_blocks")),
    ("sdxl", ("down_blocks", "up_blocks", "input_blocks", "output_blocks", "mid_block", "lora_te")),
)
_PREFIXES = ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "lora_unet_")
_TEXT_ENCODER = ("lora_te", "text_encoder", "te.", "te1.", "te2.")
# Признаки других видов адаптеров — по сегментам имени ключа, не подстрокам
# («.diff» встречается и в «model.diffusion_model.»).
_UNSUPPORTED = (
    ("lokr", ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w2_a")),
    ("loha", ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")),
    ("dora", ("dora_scale", "lora_magnitude_vector")),
    ("full", ("diff", "diff_b")),
)
_SUFFIXES = (
    (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".alpha", "alpha"),
)


class LoraError(ValueError):
    """Файл LoRA не подключить. ``code`` — причина для перевода в интерфейсе."""

    def __init__(self, code: str, name: str, detail: str = "") -> None:
        super().__init__(f"LoRA {name!r}: {code}" + (f" ({detail})" if detail else ""))
        self.code = code
        self.name = name
        self.detail = detail


@dataclass(frozen=True)
class LoraChoice:
    """Выбор в ячейке: имя файла (путь от каталога без расширения) и вес."""

    name: str
    weight: float = WEIGHT_DEFAULT


@dataclass(frozen=True)
class LoraInfo:
    """Что известно о файле по его заголовку, без чтения весов.

    ``problem`` — код, по которому файл не подключить (``None`` — подходит):
    ``other_model`` (``family`` — для какой модели), ``unsupported``
    (``family`` — lokr/loha/dora/full), ``empty``, ``unreadable``.
    """

    name: str
    layers: int = 0
    rank: int | None = None
    triggers: tuple[str, ...] = ()
    problem: str | None = None
    family: str = ""
    skipped_text_encoder: int = 0
    unknown: tuple[str, ...] = field(default=())

    @property
    def usable(self) -> bool:
        return self.problem is None


# --- библиотека файлов ------------------------------------------------------------


def list_names(directory: Path) -> list[str]:
    """Имена LoRA в каталоге (вложенные папки — через ``/``), по алфавиту."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    names = [path.relative_to(directory).with_suffix("").as_posix() for path in directory.rglob(f"*{SUFFIX}")]
    return sorted(names, key=str.lower)


def path_for(directory: Path, name: str) -> Path:
    """Файл по имени — только внутри каталога: имя приходит из браузера."""
    directory = Path(directory).resolve()
    path = (directory / f"{name}{SUFFIX}").resolve()
    if directory not in path.parents:
        raise LoraError("outside", name)
    if not path.is_file():
        raise LoraError("missing", name)
    return path


# --- заголовок safetensors и разбор ключей --------------------------------------------


def read_header(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Тензоры (имя → dtype и форма) и метаданные — без чтения самих весов."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100 * 2**20:
            raise ValueError("header is implausibly large")
        header = json.loads(handle.read(size))
    metadata = header.pop("__metadata__", None) or {}
    return header, {str(key): str(value) for key, value in metadata.items()}


@cache
def layer_names(num_layers: int = NUM_LAYERS) -> frozenset[str]:
    """Линейные слои трансформера 2.1 — сверено тестом с самой моделью."""
    blocks = (f"transformer_blocks.{index}.{layer}" for index in range(num_layers) for layer in _BLOCK_LAYERS)
    return frozenset((*blocks, *_TOP_LAYERS))


@cache
def _flat_index(num_layers: int = NUM_LAYERS) -> dict[str, str]:
    """Плоское имя kohya → настоящее: ``transformer_blocks_0_attn_to_q`` → ``…0.attn.to_q``."""
    names = set(layer_names(num_layers)) | {
        f"transformer_blocks.{index}.{FUSED_GATE_UP}" for index in range(num_layers)
    }
    return {name.replace(".", "_"): name for name in names}


def _split_key(key: str) -> tuple[str, str] | None:
    """``(слой, часть)``: часть — ``down``, ``up`` или ``alpha``; ``None`` — не LoRA."""
    for suffix, part in _SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


def _layer_name(raw: str) -> str:
    """Имя слоя без префиксов форматов; плоское имя kohya — в настоящее."""
    name = raw
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return _flat_index().get(name, name)


def _shape_fits(layer: str, part: str, shape: Sequence[int]) -> bool:
    """Подходят ли формы ``lora_A`` (вход) и ``lora_B`` (выход) к слою блока 2.1."""
    _head, _sep, tail = layer.partition(".")
    tail = tail.partition(".")[2] if layer.startswith("transformer_blocks.") else ""
    expected = _SHAPES.get(tail)
    if expected is None or len(shape) != 2:
        return True
    if part == "down":
        return shape[1] == expected[1]
    if part == "up":
        return shape[0] == expected[0]
    return True


def _peft_alpha(metadata: dict[str, str]) -> tuple[float | None, dict[str, float]]:
    """``lora_alpha`` и ``alpha_pattern`` из ``lora_adapter_metadata`` peft (или ``None``)."""
    raw = metadata.get("lora_adapter_metadata")
    if not raw:
        return None, {}
    try:
        # diffusers пишет поля с префиксом компонента: «transformer.lora_alpha».
        config = {key.removeprefix("transformer."): value for key, value in json.loads(raw).items()}
        alpha = config.get("lora_alpha")
        pattern = {str(key): float(value) for key, value in (config.get("alpha_pattern") or {}).items()}
        return (float(alpha) if alpha is not None else None), pattern
    except (ValueError, TypeError, AttributeError):
        return None, {}


def _is_text_encoder(key: str) -> bool:
    return key.startswith(_TEXT_ENCODER)


def _family(keys: Iterable[str]) -> str:
    text = " ".join(keys)
    for family, markers in _FAMILIES:
        if any(marker in text for marker in markers):
            return family
    return "unknown"


def _triggers(metadata: dict[str, str], path: Path) -> tuple[str, ...]:
    """Слова-триггеры: из метаданных тренеров или из карточки Civitai рядом с файлом.

    kohya пишет ``ss_tag_frequency`` (частоты тегов набора — берутся три
    самых частых), ai-toolkit и modelspec — явную фразу. Карточку
    ``<имя>.civitai.info`` кладут менеджеры моделей (поле ``trainedWords``).
    """
    found: list[str] = []
    for key in ("modelspec.trigger_phrase", "trigger_phrase", "trigger_word", "trigger_words", "ss_trigger_words"):
        if metadata.get(key):
            found += [part.strip() for part in re.split(r"[,;\n]", metadata[key]) if part.strip()]
    if not found and metadata.get("ss_tag_frequency"):
        try:
            frequency: dict[str, int] = {}
            for tags in json.loads(metadata["ss_tag_frequency"]).values():
                for tag, count in tags.items():
                    frequency[tag.strip()] = frequency.get(tag.strip(), 0) + int(count)
            found = [tag for tag, _count in sorted(frequency.items(), key=lambda item: -item[1])[:3] if tag]
        except (ValueError, AttributeError, TypeError):
            pass
    if not found:
        found = _sidecar_triggers(path)
    unique: list[str] = []
    for word in found:
        if word not in unique:
            unique.append(word)
    return tuple(unique[:8])


def _sidecar_triggers(path: Path) -> list[str]:
    """Карточка рядом с файлом: Civitai (``<имя>.civitai.info``, ``trainedWords``) или
    sd-webui/InvokeAI (``<имя>.json``, «activation text» через запятую)."""
    cards = ((path.with_name(path.stem + ".civitai.info"), "trainedWords"), (path.with_suffix(".json"), "activation text"))
    for card_path, field_name in cards:
        if not card_path.is_file():
            continue
        try:
            value = json.loads(card_path.read_text(encoding="utf-8")).get(field_name)
        except (OSError, ValueError, AttributeError):
            continue
        words = value if isinstance(value, list) else str(value or "").split(",")
        found = [str(word).strip() for word in words if str(word).strip()]
        if found:
            return found
    return []


def inspect(path: Path, name: str = "") -> LoraInfo:
    """Подходит ли файл к Qwen-Image-2.1 — по именам и формам из заголовка."""
    path = Path(path)
    name = name or path.stem
    try:
        header, metadata = read_header(path)
    except (OSError, ValueError, struct.error) as error:
        LOGGER.warning("Cannot read LoRA %s: %s", path, error)
        return LoraInfo(name, problem="unreadable")
    triggers = _triggers(metadata, path)
    keys = list(header)
    segments = {segment for key in keys for segment in key.split(".")}
    for family, markers in _UNSUPPORTED:
        if segments.intersection(markers):
            return LoraInfo(name, triggers=triggers, problem="unsupported", family=family)

    known = layer_names() | {f"transformer_blocks.{index}.{FUSED_GATE_UP}" for index in range(NUM_LAYERS)}
    layers: set[str] = set()
    unknown: set[str] = set()
    skipped = 0
    rank = None
    for key in keys:
        if _is_text_encoder(key):
            skipped += 1
            continue
        split = _split_key(key)
        if split is None:
            unknown.add(key)
            continue
        layer = _layer_name(split[0])
        if layer in known and _shape_fits(layer, split[1], header[key].get("shape", ())):
            layers.add(layer)
            if split[1] == "down" and rank is None:
                rank = int(header[key]["shape"][0])
        else:
            unknown.add(layer)
    if not layers and not unknown:
        return LoraInfo(name, triggers=triggers, problem="empty", skipped_text_encoder=skipped)
    if unknown:
        # Хоть один чужой слой — значит, файл учили на другой архитектуре:
        # подключить половину LoRA хуже, чем не подключать вовсе.
        return LoraInfo(
            name, layers=len(layers), rank=rank, triggers=triggers, problem="other_model",
            family=_family(unknown), skipped_text_encoder=skipped, unknown=tuple(sorted(unknown)[:5]),
        )
    return LoraInfo(name, layers=len(layers), rank=rank, triggers=triggers, skipped_text_encoder=skipped)


# --- перевод весов в формат diffusers ---------------------------------------------------


def convert(state: dict, metadata: dict[str, str] | None = None) -> dict:
    """Ключи любого из поддержанных форматов → ``transformer.<слой>.lora_A/B.weight``.

    ``alpha`` (ключ ``.alpha`` или метаданные peft) вкладывается в
    ``lora_B``, слитый ``gate_up`` режется пополам. ``metadata`` — метаданные
    файла safetensors.
    """
    peft_alpha, alpha_pattern = _peft_alpha(metadata or {})
    parts: dict[str, dict[str, object]] = {}
    for key, tensor in state.items():
        if _is_text_encoder(key):
            continue
        split = _split_key(key)
        if split is None:
            raise LoraError("other_model", "", key)
        layer = _layer_name(split[0])
        parts.setdefault(layer, {})[split[1]] = tensor

    converted: dict = {}
    for layer, found in parts.items():
        if "down" not in found or "up" not in found:
            raise LoraError("other_model", "", f"{layer}: incomplete pair")
        down, up = found["down"], found["up"]
        rank = down.shape[0]
        alpha = None
        if "alpha" in found:
            alpha = float(found["alpha"].item())
        elif peft_alpha is not None or alpha_pattern:
            alpha = next((value for pattern, value in alpha_pattern.items() if layer.endswith(pattern)), peft_alpha)
        if alpha is not None and alpha / rank != 1.0:
            up = (up.float() * (alpha / rank)).to(up.dtype)
        if layer.endswith(FUSED_GATE_UP):
            stem = layer[: -len(FUSED_GATE_UP)]
            half = up.shape[0] // 2
            pieces = ((stem + "img_mlp.gate_layer", up[:half]), (stem + "img_mlp.proj", up[half:]))
            for target, piece in pieces:
                converted[f"transformer.{target}.lora_A.weight"] = down.clone()
                converted[f"transformer.{target}.lora_B.weight"] = piece.contiguous()
            continue
        converted[f"transformer.{layer}.lora_A.weight"] = down
        converted[f"transformer.{layer}.lora_B.weight"] = up
    return converted


# --- отпечаток файла для метаданных -------------------------------------------------------

_HASHES: dict[tuple[str, int, int], str] = {}


def short_hash(path: Path) -> str:
    """Первые 10 знаков SHA-256 файла — «AutoV2», по нему Civitai узнаёт модель.

    Считается один раз на файл (ключ — путь, размер и время изменения).
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 2**20), b""):
                digest.update(chunk)
        _HASHES[key] = digest.hexdigest()[:10]
    return _HASHES[key]


def adapter_name(path: Path) -> str:
    """Имя адаптера peft: по содержимому файла, без точек (peft их не терпит)."""
    return USER_PREFIX + short_hash(path)


# --- подключение к пайплайну --------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedLora:
    """Выбор, проверенный и привязанный к файлу: что подключать и что записать в PNG."""

    name: str
    weight: float
    path: Path
    hash: str

    @property
    def adapter(self) -> str:
        return USER_PREFIX + self.hash


def resolve(directory: Path, choices: Sequence[LoraChoice]) -> tuple[list[ResolvedLora], list[LoraError]]:
    """Выборы из ячеек → подходящие файлы и отказы с причинами.

    Нулевой вес и повтор того же файла отбрасываются молча: первое — «не
    действует», второе peft всё равно подключил бы одним адаптером.
    """
    resolved: list[ResolvedLora] = []
    problems: list[LoraError] = []
    seen: set[str] = set()
    for choice in choices:
        if not choice.name or choice.weight == 0:
            continue
        try:
            path = path_for(directory, choice.name)
        except LoraError as error:
            problems.append(error)
            continue
        info = inspect(path, choice.name)
        if not info.usable:
            problems.append(LoraError(info.problem, choice.name, info.family))
            continue
        digest = short_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        resolved.append(ResolvedLora(choice.name, float(choice.weight), path, digest))
    return resolved, problems


class LoraAdapters:
    """Держит на трансформере ровно нужные адаптеры и включает их с весами.

    Подключение и отключение меняют состав параметров трансформера, поэтому
    идут через ``ResidencyManager.restage_transformer`` — одной перестановкой
    на все изменения сразу. Неотмеченные пользовательские адаптеры
    отключаются и выгружаются: на карте в 8 ГБ лишние сотни мегабайт — это
    чужая память. Адаптер Turbo не трогается — им ведает ``TurboAdapter``.
    """

    def __init__(self, pipe, residency) -> None:
        self._pipe = pipe
        self._residency = residency

    def loaded(self) -> set[str]:
        config = getattr(getattr(self._pipe, "transformer", None), "peft_config", None) or {}
        return set(config)

    def sync(self, wanted: Sequence[ResolvedLora]) -> None:
        """Подключает недостающие и выгружает лишние пользовательские адаптеры."""
        loaded = self.loaded()
        names = {item.adapter for item in wanted}
        to_drop = sorted(name for name in loaded if name.startswith(USER_PREFIX) and name not in names)
        to_load = [item for item in wanted if item.adapter not in loaded]
        if not to_drop and not to_load:
            return
        states = {item.adapter: (item, self._read(item)) for item in to_load}

        def change(_transformer) -> None:
            if to_drop:
                LOGGER.info("Unloading LoRA adapters: %s", ", ".join(to_drop))
                self._pipe.delete_adapters(to_drop)
            for adapter, (item, state) in states.items():
                LOGGER.info("Loading LoRA %s (%s)", item.name, adapter)
                self._load(state, adapter)

        self._residency.restage_transformer(change)

    def _read(self, item: ResolvedLora) -> dict:
        from safetensors.torch import load_file

        _header, metadata = read_header(item.path)
        try:
            return convert(load_file(str(item.path)), metadata)
        except LoraError as error:
            raise LoraError(error.code, item.name, error.detail) from error

    def _load(self, state: dict, adapter: str) -> None:
        import warnings

        with warnings.catch_warnings():
            # INT8- и GGUF-слои peft оборачивает без merge(); он и не нужен.
            warnings.filterwarnings("ignore", message=r".*LoraLinear was instantiated without")
            # Несколько адаптеров на одном трансформере — штатный режим здесь.
            warnings.filterwarnings("ignore", message=r"Already found a `peft_config` attribute")
            self._pipe.load_lora_weights(state, adapter_name=adapter)

    def activate(self, active: Sequence[tuple[str, float]]) -> None:
        """Включает ровно эти адаптеры с этими весами; пустой список — LoRA выключены."""
        if not self.loaded():
            return
        if not active:
            self._pipe.disable_lora()
            return
        self._pipe.enable_lora()
        self._pipe.set_adapters([name for name, _weight in active], [weight for _name, weight in active])
