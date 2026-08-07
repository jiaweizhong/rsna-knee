from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries; lists and scalars are replaced."""
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged = deep_merge(result[key], value)
            # Registry specs are atomic at the params level. Without this rule,
            # switching tiny_cnn -> timm would retain incompatible tiny_cnn args.
            if "name" in value and "params" in value:
                merged["params"] = copy.deepcopy(value["params"])
            result[key] = merged
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_dot_path(config: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set {path!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Top-level YAML value in {path} must be a mapping")
        config = deep_merge(config, loaded)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form: {override!r}")
        key, raw_value = override.split("=", 1)
        set_dot_path(config, key, _parse_scalar(raw_value))
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML file. Repeat to apply overlays from left to right.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dot-path override, e.g. model.selector.params.k=10",
    )
