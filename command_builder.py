#!/usr/bin/env python3
"""
command_builder.py - Command building utilities for llauncher

Zentralisiertes Building von llama.cpp Kommandozeilen aus UI-Parametern.
Liefert get_current_args() und build_full_command() als freie Funktionen.
"""

import shlex
from pathlib import Path


def get_current_args(window) -> list:
    """Baut Parameterliste für llama.cpp Prozess aus UI-Werten.
    
    Args:
        window: llauncher main window instance mit param_sliders, PARAM_DEFINITIONS, etc.
        
    Returns:
        list: Kommandozeilen-Parameter als String-Liste
    """
    args = [str(Path(window.llama_cpp_path) / window.exe_combo.currentText())]
    
    # Modell-Pfad (nur einmal!)
    if window.selected_model:
        args.extend(["-m", window.selected_model])
    
    # mmproj für Vision-Modelle
    mmproj_text = window.mmproj_line.text().strip()
    if mmproj_text:
        mmproj_path = Path(mmproj_text)
        if not mmproj_path.is_absolute():
            mmproj_path = Path(window.model_directory) / mmproj_path
        if mmproj_path.exists():
            args.extend(["--mmproj", str(mmproj_path)])
    
    # Parameter aus Slidern (nur wenn vom Default abweichen)
    for param_key, config in window.PARAM_DEFINITIONS.items():
        if param_key not in window.param_sliders:
            continue
        slider = window.param_sliders[param_key]
        
        if config.get("type") == "float_slider":
            # Float-Slider: Wert aus Edit-Widget lesen
            value_edit = slider["edit"]
            try:
                value = float(value_edit.text())
            except ValueError:
                continue
            
            if abs(value - config["default"]) > 0.01:
                args.append(param_key)
                args.append(f"{value:.2f}")
        
        elif config.get("type") == "combo":
            # ComboBox – Wert als String lesen
            combo = slider["combo"]
            value = combo.currentText()
            # cache-type-k/v immer explizit setzen (nicht nur bei Abweichung vom Default)
            if param_key in ("--cache-type-k", "--cache-type-v"):
                args.append(param_key)
                args.append(value)
            elif value != config["default"]:
                args.append(param_key)
                args.append(value)
        
        elif config.get("type") in ("text_input", "path_input", "file_input"):
            # Textfeld, Pfad oder Datei-Eingabe – Wert als String lesen
            # WICHTIG: benchmark_file_path NICHT in Command Line (nur für Benchmark)
            if param_key == "benchmark_file_path":
                continue
            
            text_edit = slider["edit"]
            value = text_edit.text()
            if value and value != config["default"]:
                args.append(param_key)
                args.append(value)
        
        else:
            # Integer-Slider
            if isinstance(slider, dict):
                # Priorität: Wert aus dem Edit-Feld lesen, falls vorhanden
                edit_widget = slider.get("edit")
                slider_widget = slider.get("slider")
                
                if edit_widget:
                    try:
                        value = int(edit_widget.text())
                    except ValueError:
                        # Fallback auf Slider-Widget, falls Edit leer/ungültig
                        value = slider_widget.value() if slider_widget else 0
                elif slider_widget:
                    value = slider_widget.value()
                else:
                    value = 0
            else:
                value = slider.value()
            
            # Sonderfall: -ngl mit "all" Checkbox
            if param_key == "-ngl":
                if hasattr(window, "ngl_all_checkbox") and window.ngl_all_checkbox.isChecked():
                    args.append(param_key)
                    args.append("all")
                elif value != config["default"]:
                    args.append(param_key)
                    args.append(str(value))
            elif value != config["default"]:
                args.append(param_key)
                args.append(str(value))
    
    return args


def _parse_custom_commands_text(text):
    """Parses custom command text field content into a list of command-line arguments.
    
    Handles space-separated 'key value' format and bare flags.
    Also supports '=' as separator (e.g., '--key=value') for manual entries.
    
    Args:
        text: String content from custom_cmd_edit QTextEdit
        
    Returns:
        list: List of command-line argument strings
    """
    if not text or not text.strip():
        return []
    
    args = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
        
        # Check for 'key=value' format (no spaces around '=')
        if '=' in line:
            parts = line.split('=', 1)
            args.append(parts[0].strip())
            args.append(parts[1].strip())
        else:
            # Space-separated: 'key value' or bare flag
            parts = line.split(None, 1)
            if len(parts) == 2:
                args.append(parts[0])
                args.append(parts[1])
            else:
                # Single flag (no value) - just append the key
                args.append(line)
    
    return args


def build_full_command(window, external_args: dict = None) -> str:
    """Vollständige Kommandozeile als String bauen.
    
    Priorität:
    1. Echte Kommandozeile vom externen Runner (falls vorhanden)
    2. Echte Kommandozeile vom ProcessRunner (falls vorhanden)
    3. UI-Werte zusammengesetzt (Fallback)
    
    Args:
        window: llauncher main window instance
        external_args: Dict von externen Parametern (nicht in APP verwaltbar)
        
    Returns:
        str: Vollständige Kommandozeile als String
    """
    # 1. Versuche externe Runner args (für externe Prozesse)
    if hasattr(window, 'external_runner_args') and window.external_runner_args:
        return " ".join(shlex.quote(arg) for arg in window.external_runner_args)
    
    # 2. Versuche ProcessRunner (für interne Prozesse)
    if hasattr(window, 'process_runner') and window.process_runner:
        try:
            real_args = window.process_runner.get_args_from_proc()
            if real_args:
                return " ".join(shlex.quote(arg) for arg in real_args)
        except Exception:
            pass
    
    # 3. Fallback: UI-Werte zusammengesetzt + externe Args (aus Custom Commands Feld)
    args = get_current_args(window)
    
    # Custom Commands Feld auslesen (benutzerdefinierte Kommandozeilen-Argumente)
    if hasattr(window, 'custom_cmd_edit') and window.custom_cmd_edit:
        custom_text = window.custom_cmd_edit.toPlainText()
        custom_args = _parse_custom_commands_text(custom_text)
        args.extend(custom_args)
    
    return " ".join(shlex.quote(arg) for arg in args)


def on_param_changed(window) -> None:
    """Debug-Output live aktualisieren wenn sich ein Parameter ändert.
    
    Args:
        window: llauncher main window instance
    """
    try:
        # Prüfen ob param_sliders initialisiert ist (kann None sein während init_ui)
        if not hasattr(window, 'param_sliders') or window.param_sliders is None:
            return
        
        # Alle Sliders müssen existieren und initialisiert sein
        for param_key in window.PARAM_DEFINITIONS.keys():
            if param_key not in window.param_sliders:
                continue
            slider = window.param_sliders[param_key]
            config = window.PARAM_DEFINITIONS[param_key]
            
            try:
                if config.get("type") == "float_slider":
                    if "edit" not in slider or not slider["edit"]:
                        return
                elif config.get("type") in ("text_input", "path_input"):
                    if "edit" not in slider or not slider["edit"]:
                        return
                else:
                    if isinstance(slider, dict) and "slider" not in slider:
                        return
            except (KeyError, AttributeError):
                return
        
        command = build_full_command(window)
        window.debug_text.setText(command)
    except Exception as e:
        window.debug_text.setText(f"Fehler beim Aktualisieren: {e}")
