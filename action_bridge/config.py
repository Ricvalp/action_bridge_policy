"""ml_collections config loading with small Hydra-style CLI overrides."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ml_collections import ConfigDict


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_PACKAGE = "action_bridge.configs"
_NON_CONFIG_MODULES = {"__init__", "base"}


def available_config_names() -> List[str]:
    return sorted(
        path.stem
        for path in CONFIG_DIR.glob("*.py")
        if path.stem not in _NON_CONFIG_MODULES and not path.stem.startswith("_")
    )


def _canonical_name(config_name: str) -> str:
    name = config_name
    for suffix in [".py", ".yaml", ".yml"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, ConfigDict):
        value = value.to_dict()
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_dict(item) for item in value)
    return value


def to_config_dict(value: Any) -> ConfigDict:
    if isinstance(value, ConfigDict):
        return copy.deepcopy(value)
    return ConfigDict(to_plain_dict(value))


def load_config(config_name: str) -> ConfigDict:
    name = _canonical_name(config_name)
    try:
        module = importlib.import_module(f"{CONFIG_PACKAGE}.{name}")
    except ModuleNotFoundError as exc:
        available = ", ".join(available_config_names())
        raise FileNotFoundError(f"Unknown config {config_name!r}. Available configs: {available}") from exc
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{CONFIG_PACKAGE}.{name} must define get_config().")
    config = to_config_dict(module.get_config())
    if "config_name" not in config:
        config.config_name = name
    return config


def parse_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def set_nested(config: ConfigDict, dotted_key: str, value: Any) -> None:
    cursor: Any = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, (ConfigDict, dict)):
            existing = ConfigDict()
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def apply_overrides(config: Any, overrides: Iterable[str]) -> ConfigDict:
    result = to_config_dict(config)
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override {item!r} must have KEY=VALUE form.")
        key, value = item.split("=", 1)
        set_nested(result, key, parse_value(value))
    return result


def save_config(config: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_plain_dict(config), f, indent=2, sort_keys=False)


def flatten_dict(data: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in to_plain_dict(data).items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, next_key))
        else:
            flat[next_key] = value
    return flat


def override_args(argv: List[str]) -> List[str]:
    """Return unknown CLI tokens that look like Hydra overrides."""

    return [arg for arg in argv if "=" in arg and not arg.startswith("--")]
