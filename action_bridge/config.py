"""Plain YAML config loading with small Hydra-style CLI overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(config_name: str) -> Dict[str, Any]:
    name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown config {config_name!r}; expected {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("config_name", name)
    return data


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


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override {item!r} must have KEY=VALUE form.")
        key, value = item.split("=", 1)
        set_nested(result, key, parse_value(value))
    return result


def save_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, next_key))
        else:
            flat[next_key] = value
    return flat


def override_args(argv: List[str]) -> List[str]:
    """Return unknown CLI tokens that look like Hydra overrides."""

    return [arg for arg in argv if "=" in arg and not arg.startswith("--")]
