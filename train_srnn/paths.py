"""Where datasets, run outputs, and analysis caches live.

Everything sits under ``$SRNN_HOME`` (default ``~/srnn``) unless a more
specific variable overrides it:

    SRNN_DATA_DIR      datasets, one folder per task      ($SRNN_HOME/data)
    SRNN_RESULTS_DIR   training runs                      ($SRNN_HOME/results)
    SRNN_CACHE_DIR     downloads made by analysis scripts ($SRNN_HOME/cache)

Archived run summaries go under ``$SRNN_RESULTS_DIR/archive``. The cloud
startup script sets ``SRNN_HOME`` on the VM; locally, export it in your shell.
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf


def home() -> Path:
    return Path(os.environ.get("SRNN_HOME", "~/srnn")).expanduser()


def data_dir() -> Path:
    return Path(os.environ.get("SRNN_DATA_DIR", home() / "data")).expanduser()


def results_dir() -> Path:
    return Path(os.environ.get("SRNN_RESULTS_DIR", home() / "results")).expanduser()


def archive_dir() -> Path:
    return results_dir() / "archive"


def cache_dir() -> Path:
    return Path(os.environ.get("SRNN_CACHE_DIR", home() / "cache")).expanduser()


_KINDS = {"data": data_dir, "results": results_dir, "cache": cache_dir}


def register_resolvers() -> None:
    """Expose the directories to Hydra configs as ``${srnn_path:data}`` etc."""
    if not OmegaConf.has_resolver("srnn_path"):
        OmegaConf.register_new_resolver("srnn_path", lambda kind: str(_KINDS[kind]()))
