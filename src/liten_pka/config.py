from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dirs(config: dict[str, Any]) -> None:
    for section, keys in {
        "data": ["processed_dir", "conformer_dir"],
        "train": ["checkpoint_dir", "report_dir"],
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value:
                Path(value).mkdir(parents=True, exist_ok=True)

