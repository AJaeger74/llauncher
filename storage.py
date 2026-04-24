#!/usr/bin/env python3
"""
llauncher – Storage Utilities
JSON I/O für Config, Presets und Benchmarks.
Unabhängig von PyQt6, kann unit-getestet werden.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt


# Konstanten für Dateipfade (müssen mit llauncher.py übereinstimmen)
CONFIG_DIR = Path.home() / ".llauncher"
CONFIG_FILE = CONFIG_DIR / "config.json"
PRESETS_FILE = CONFIG_DIR / "presets.json"
BENCHMARKS_FILE = CONFIG_DIR / "benchmarks.json"


def ensure_config_dir():
    """Stellt sicher, dass das Konfigurationsverzeichnis existiert."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# Config (config.json)
# ────────────────────────────────────────────────────────────────

def save_config(config_updates: dict) -> None:
    """Partiell Config aktualisieren (bestehende Werte erhalten)."""
    ensure_config_dir()
    
    # Bestehende Config laden oder leeres Dict verwenden
    existing_config = load_config()
    
    # Updates anwenden
    existing_config.update(config_updates)
    
    # Gespeicherte Config schreiben
    with open(CONFIG_FILE, "w") as f:
        json.dump(existing_config, f, indent=2)


def load_config() -> dict:
    """Config aus config.json laden, leeres Dict wenn nicht vorhanden."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────
# Presets (presets.json)
# ────────────────────────────────────────────────────────────────

def save_presets(presets: dict) -> None:
    """Presets-Dict in presets.json schreiben."""
    ensure_config_dir()
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def load_presets() -> dict:
    """Presets aus presets.json laden, leeres Dict wenn nicht vorhanden.
    
    Unterstützt zwei Formate:
    1. {"presets": [...]} - Liste in "presets" Key
    2. {"name": {...}, ...} - Dict direkt
    """
    if not PRESETS_FILE.exists():
        return {}
    
    try:
        with open(PRESETS_FILE, "r") as f:
            data = json.load(f)
        
        # Format 1: {"presets": [...]}
        if "presets" in data and isinstance(data["presets"], list):
            # Konvertiere Liste in Dict mit Namen als Key
            return {p["name"]: p for p in data["presets"] if "name" in p}
        
        # Format 2: {"name1": {...}, "name2": {...}}
        if isinstance(data, dict):
            return data
        
        # Fallback
        return {}
    except (json.JSONDecodeError, KeyError):
        return {}


# ────────────────────────────────────────────────────────────────
# Benchmarks (benchmarks.json)
# ────────────────────────────────────────────────────────────────

def save_benchmarks(benchmarks: list) -> None:
    """Benchmarks-Liste in benchmarks.json schreiben."""
    ensure_config_dir()
    with open(BENCHMARKS_FILE, "w") as f:
        json.dump(benchmarks, f, indent=2)


def load_benchmarks() -> list:
    """Benchmarks aus benchmarks.json laden, leere Liste wenn nicht vorhanden."""
    if not BENCHMARKS_FILE.exists():
        return []
    with open(BENCHMARKS_FILE, "r") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────
# Utility: Preset anwenden (benötigt llauncher-Instanz)
# ────────────────────────────────────────────────────────────────

def apply_preset(window, preset: dict):
    """
    Ein Preset auf eine llauncher-Instanz anwenden.
    
    Args:
        window: llauncher Instanz
        preset: Dict mit Preset-Daten (llama_cpp_path, params, etc.)
    """
    from gguf_utils import read_gguf_context_length
    
    # Pfade setzen (volle Pfade!)
    llama_path = preset.get("llama_cpp_path", str(Path.home() / "llama.cpp"))
    
    window.llama_cpp_path = llama_path
    
    # UI-Textfelder aktualisieren
    if Path(llama_path).exists():
        window.exe_line.setText(llama_path)
    
    # model_directory kommt immer noch aus config.json (load_config())
    # Preset referenziert eventuell ein anderes Verzeichnis - nur für UI-Anzeige
    model_dir = preset.get("model_directory", str(Path.home() / "models"))
    if Path(model_dir).exists():
        window.model_line.setText(model_dir)

    # Dropdowns neu füllen
    window.find_executables()
    window.update_model_dropdown()

    # Executable auswählen (voller Pfad oder nur Name)
    selected_exe = preset.get("selected_exe")
    if selected_exe:
        exe_name = Path(selected_exe).name
        idx = window.exe_combo.findText(exe_name)
        if idx >= 0:
            window.exe_combo.setCurrentIndex(idx)

     # Modell auswählen (voller Pfad)
    selected_model = preset.get("selected_model")
    if selected_model and Path(selected_model).exists():
        model_name = Path(selected_model).name
        
        # Find by UserRole (clean filename), not display text (which includes size)
        idx = -1
        for i in range(window.model_combo.count()):
            if window.model_combo.itemData(i, role=Qt.ItemDataRole.UserRole) == model_name:
                idx = i
                break
        
        if idx >= 0:
            # GGUF-Kontextlänge lesen und Slider-Maximum VOR dem Parameter-Loop setzen
            # (verhindert dass Qt den Wert clampt wenn setValue() aufgerufen wird)
            ctx_length = read_gguf_context_length(selected_model)
            if ctx_length and ctx_length > 0 and "-c" in window.param_sliders:
                slider_data = window.param_sliders["-c"]
                slider_data["slider"].setMaximum(ctx_length)
            
            window.model_combo.blockSignals(True)
            try:
                window.model_combo.setCurrentIndex(idx)
                window.selected_model = selected_model
            finally:
                window.model_combo.blockSignals(False)

    # mmproj setzen (voller Pfad! - immer setzen, nicht nur wenn existiert!)
    mmproj_path = preset.get("mmproj_path", "")
    if mmproj_path is not None:
        window.mmproj_line.setText(mmproj_path)

    # Wenn ein Modell ausgewählt ist, ctx_size Slider Maximum aktualisieren
    # (MUSS vor dem params-Loop passieren, damit Slider-Wert nicht geclampt wird)
    if preset.get("selected_model") and Path(preset["selected_model"]).exists():
        ctx_length = read_gguf_context_length(preset["selected_model"])
        sys.stderr.write(f"[preset] loaded model context_length: {ctx_length}\n")
        if ctx_length and ctx_length > 0:
            slider_data = window.param_sliders.get("-c")
            if slider_data:
                slider = slider_data["slider"] if isinstance(slider_data, dict) else slider_data
                edit = slider_data["edit"] if isinstance(slider_data, dict) else None
                
                old_max = slider.maximum()
                sys.stderr.write(f"[preset] ctx_size slider: maximum {old_max} → {ctx_length}\n")
                slider.setMaximum(ctx_length)

                # Edit-Widget Breite aktualisieren für neue maximale Zahl
                if edit:
                    max_width = len(str(ctx_length)) * 9 + 15
                    edit.setMinimumWidth(max_width)
                    edit.setMaximumWidth(max_width)

    # Parameter-Slider setzen (NACH Maximum-Update für -c, damit Werte nicht geclampt werden)
    for param_key, value in preset.get("params", {}).items():
        # benchmark_file_path NICHT überschreiben (rein user-spezifisch)
        if param_key == "benchmark_file_path":
            continue
            
        if param_key not in window.param_sliders:
            continue
            
        config = window.PARAM_DEFINITIONS[param_key]
        
        if config.get("type") == "float_slider":
            # Float-Slider: Wert aus Edit-Widget setzen
            slider_data = window.param_sliders[param_key]
            value_edit = slider_data["edit"]
            value_edit.setText(f"{value:.2f}")
        elif config.get("type") == "combo":
            # ComboBox: Wert als String setzen
            slider_data = window.param_sliders[param_key]
            combo = slider_data["combo"]
            idx = combo.findText(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        elif config.get("type") in ("text_input", "path_input"):
            # Textfeld oder Pfad-Eingabe – Wert als String setzen
            slider_data = window.param_sliders[param_key]
            value_edit = slider_data["edit"]
            value_edit.setText(str(value))
        else:
            # Integer-Slider: Wert direkt setzen (value ist bereits gespeichert)
            slider_data = window.param_sliders[param_key]
            if isinstance(slider_data, dict):
                slider = slider_data["slider"]
                # Sonderfall: -ngl mit String "all" → Edit auf "all", Slider auf 0
                if param_key == "-ngl" and isinstance(value, str) and value == "all":
                    slider.setValue(0)
                    slider_data["edit"].setText("all")
                else:
                    old_val = slider.value()
                    slider.setValue(int(value))
                    sys.stderr.write(f"[preset] {param_key} slider: setValue {old_val} → {int(value)}\n")
            else:
                old_val = slider_data.value()
                slider_data.setValue(int(value))
                sys.stderr.write(f"[preset] {param_key} slider: setValue {old_val} → {int(value)}\n")

        # Sonderfall: -ngl mit "all" Checkbox → Edit auf "all", Slider = 0 (unwichtig)
        if param_key == "-ngl":
            ngl_all_value = preset.get("params", {}).get("-ngl") == "all" or preset.get("ngl_all", False)
            has_checkbox = hasattr(window, "ngl_all_checkbox")
            checkbox_exists = window.ngl_all_checkbox if has_checkbox else None
            if has_checkbox and checkbox_exists:
                window.ngl_all_checkbox.setChecked(ngl_all_value)
                if ngl_all_value and isinstance(slider_data, dict):
                    value_edit = slider_data["edit"]
                    try:
                        value_edit.setText("all")
                    except Exception:
                        pass
                    slider = slider_data["slider"]
                    slider.setValue(0)

    # Debug: log loaded preset name
    if hasattr(window, "debug_text") and window.debug_text:
        preset_name = preset.get("name", "<unnamed>")
        window.debug_text.append(f"Preset loaded: {preset_name}")
    
    # Splitter-Sizes aus Preset laden (falls vorhanden)
    if 'splitter_sizes' in preset:
        try:
            splitter_sizes = preset['splitter_sizes']
            if isinstance(splitter_sizes, list):
                window.splitter.setSizes(splitter_sizes)
        except Exception as e:
            sys.stderr.write(f"Warnung: Konnte Splitter-Sizes nicht aus Preset laden: {e}\n")
