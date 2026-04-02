#!/usr/bin/env python3
"""Utility that guarantees a ``language`` entry in ~/.llauncher/config.json.

If the key does not exist we read the current locale from the X11/Wayland
environment (``LANG`` or ``LC_ALL``), turn it into a two‑letter ISO‑639‑1
code (e.g. ``de`` or ``en``) and write it back to the config file.
The function returns the chosen language code so the caller can hand it to
the I18n manager.
"""

import os
from pathlib import Path

from storage import load_config, save_config

DEFAULT_LANG = "en"

def ensure_language() -> str:
    """Make sure a ``language`` key exists and return its value.

    Returns
    -------
    str
        The language code (e.g. ``de`` or ``en``) that will be passed to
        ``I18nManager``.
    """
    cfg_path = Path.home() / ".llauncher" / "config.json"
    config = load_config()

    # -----------------------------------------------------------------
    # Auto‑detect only when the key is missing – we keep the original
    # behaviour (default to German) untouched.
    # -----------------------------------------------------------------
    if "language" not in config:
        env = os.getenv("LANG") or os.getenv("LC_ALL") or DEFAULT_LANG
        # Strip optional “.UTF-8” and keep only the language part before “_”.
        env = env.split(".")[0]          # e.g. "de_DE.UTF-8" → "de_DE"
        lang = env.split("_")[0][:2].lower()
        config["language"] = lang
        save_config(config)

    return config.get("language", DEFAULT_LANG)