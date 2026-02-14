# settings.py
# Runtime-writable user settings, with config.py as defaults.

import json
import logging
from pathlib import Path
from typing import List

import config

SETTINGS_FILE = Path(__file__).parent / "settings.json"

def _defaults() -> dict:
    """Return default settings derived from config.py."""
    return {
        "directories": getattr(config, "DOCUMENT_DIRECTORIES", []),
        "exclusions": getattr(config, "IGNORED_DIRECTORIES", []),
        "extensions": getattr(config, "SUPPORTED_EXTENSIONS", [".md", ".txt"]),
    }


def load_settings() -> dict:
    """Load settings from settings.json, falling back to config.py defaults."""
    defaults = _defaults()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                overrides = json.load(f)
            # Merge: overrides win, but only for keys we recognize
            for key in defaults:
                if key in overrides:
                    defaults[key] = overrides[key]
        except Exception as e:
            logging.warning(f"Failed to load {SETTINGS_FILE}: {e}")
    return defaults


def save_settings(settings: dict):
    """Write settings to settings.json."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    logging.info(f"Settings saved to {SETTINGS_FILE}")


# --- Convenience accessors ---

def get_directories() -> List[str]:
    return load_settings()["directories"]

def get_exclusions() -> List[str]:
    return load_settings()["exclusions"]

def get_extensions() -> List[str]:
    return load_settings()["extensions"]


# --- Mutators ---

def add_directory(path: str) -> List[str]:
    s = load_settings()
    if path not in s["directories"]:
        s["directories"].append(path)
        save_settings(s)
    return s["directories"]

def remove_directory(path: str) -> List[str]:
    s = load_settings()
    s["directories"] = [d for d in s["directories"] if d != path]
    save_settings(s)
    return s["directories"]

def add_exclusion(pattern: str) -> List[str]:
    s = load_settings()
    if pattern not in s["exclusions"]:
        s["exclusions"].append(pattern)
        save_settings(s)
    return s["exclusions"]

def remove_exclusion(pattern: str) -> List[str]:
    s = load_settings()
    s["exclusions"] = [e for e in s["exclusions"] if e != pattern]
    save_settings(s)
    return s["exclusions"]

def add_extension(ext: str) -> List[str]:
    # Normalize: ensure leading dot
    if not ext.startswith("."):
        ext = f".{ext}"
    s = load_settings()
    if ext not in s["extensions"]:
        s["extensions"].append(ext)
        save_settings(s)
    return s["extensions"]

def remove_extension(ext: str) -> List[str]:
    if not ext.startswith("."):
        ext = f".{ext}"
    s = load_settings()
    s["extensions"] = [e for e in s["extensions"] if e != ext]
    save_settings(s)
    return s["extensions"]
