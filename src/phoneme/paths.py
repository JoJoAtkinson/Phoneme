"""Filesystem paths for models, cache, and user config."""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

_dirs = PlatformDirs(appname="Phoneme", appauthor=False, roaming=False)


def user_data_dir() -> Path:
    p = Path(_dirs.user_data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_config_dir() -> Path:
    p = Path(_dirs.user_config_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_cache_dir() -> Path:
    p = Path(_dirs.user_cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = user_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_file() -> Path:
    return user_config_dir() / "settings.yaml"


def bundled_resources_dir() -> Path | None:
    """When launched from a py2app bundle, return the Resources dir."""
    import sys

    if getattr(sys, "frozen", False):
        resources = Path(sys.executable).resolve().parent.parent / "Resources"
        if resources.exists():
            return resources
    return None
