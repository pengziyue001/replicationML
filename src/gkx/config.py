"""Load the YAML configuration."""

from __future__ import annotations

from functools import lru_cache

import yaml

from gkx.paths import CONFIG_PATH


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    cfg_path = CONFIG_PATH if path is None else CONFIG_PATH.parent / path
    with open(cfg_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
