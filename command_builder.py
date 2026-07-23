#!/usr/bin/env python3
"""
command_builder.py - Command building utilities for llauncher

Zentralisiertes Building von llama.cpp Kommandozeilen aus UI-Parametern.
Liefert get_current_args() und build_full_command() als freie Funktionen.
"""

from pathlib import Path


def get_current_args(window) -> list:
    """Baut Parameterliste für llama.cpp Prozess aus UI-Werten.
    
    Args:
        window: llauncher main window instance mit param_sliders, PARAM_DEFINITIONS, etc.
        
    Returns:
        list: Kommandozeilen-Parameter als String-Liste
    """
    exe_name = window.exe_combo.currentText()
    args = [str(Path(window.llama_cpp_path) / "build" / "bin" / exe_name)]
    
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
            
            # --slot-save-path IMMER hinzufügen (auch wenn Default), damit llama.cpp
            # den Wert explizit setzen kann (nicht nur wenn != default)
            if param_key == "--slot-save-path":
                if value:
                    args.append(param_key)
                    args.append(value)
            elif value and value != config["default"]:
                args.append(param_key)
                args.append(value)
        
        else:
            # Integer-Slider
            if isinstance(slider, dict):
                # Direkt vom Slider-Widget lesen – das Edit-Feld kann veraltet sein
                # aufgrund von Signal-Timing (textChanged → handler → setValue → valueChanged
                # → sync_from_slider), wo get_current_args() das Edit liest BEVOR sync
                # vom slider.valueChanged-Sync den Text aktualisiert hat.
                edit_widget = slider.get("edit")
                slider_widget = slider.get("slider")
                
                if slider_widget:
                    raw_value = slider_widget.value()
                    multiplier = config.get("multiplier", 1.0)
                    # Durch Multiplikator teilen für user-facing value
                    if multiplier > 1.0:
                        value = raw_value // int(multiplier)
                    else:
                        value = raw_value
                    # Edit-Feld mit Slider-Wert syncen (Display-Update als Side-Effect)
                    if edit_widget:
                        if multiplier > 1.0:
                            edit_widget.setText(f"{value:.2f}")
                        else:
                            edit_widget.setText(str(value))
                else:
                    value = 0
            else:
                value = slider.value()
            
            # -c (context_size): IMMER zum Kommando hinzufügen – der Default
            # ist das Modell's context_length (ctx_length), nicht 4096.
            # Der Wert auf dem Slider ist die gewählte Kontextgröße.
            if param_key == "-c":
                args.append(param_key)
                args.append(str(value))
            
            # Sonderfall: -ngl mit "all" Checkbox
            elif param_key == "-ngl":
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
    Lines starting with '-' that contain '=' are kept as-is (--flag=value as single arg).
    Single-quoted values (e.g., '--key 'value with spaces'') are properly handled.
    
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
        
        # If line starts with a flag (- or --) and contains '=', keep as single arg
        # (--param=value must stay as one token for subprocess, not split into two)
        if '=' in line and line.startswith('-'):
            # Strip surrounding quotes from the whole line if present
            if len(line) >= 2:
                if (line.startswith("'") and line.endswith("'")) or \
                   (line.startswith('"') and line.endswith('"')):
                    line = line[1:-1]
            # Strip quotes around the value portion (e.g., --jinja="value with spaces")
            eq_idx = line.index('=')
            value_part = line[eq_idx + 1:]
            if len(value_part) >= 2:
                if (value_part.startswith("'") and value_part.endswith("'")) or \
                   (value_part.startswith('"') and value_part.endswith('"')):
                    line = line[:eq_idx + 1] + value_part[1:-1]
            args.append(line)
        else:
            # Space-separated: 'key value' or bare flag
            parts = line.split(None, 1)
            if len(parts) == 2:
                key = parts[0]
                value = parts[1]
                # Strip surrounding quotes from value
                if value and len(value) >= 2:
                    if (value.startswith("'") and value.endswith("'")) or \
                       (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                args.append(key)
                args.append(value)
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
        str: Vollständige Kommandozeile als String (ohne shlex.quote für UI-Display)
    """
    # 1. Versuche externe Runner args (für externe Prozesse)
    if hasattr(window, 'external_runner_args') and window.external_runner_args:
        return " ".join(window.external_runner_args)
    
    # 2. Versuche ProcessRunner (für interne Prozesse)
    if hasattr(window, 'process_runner') and window.process_runner:
        try:
            real_args = window.process_runner.get_args_from_proc()
            if real_args:
                return " ".join(real_args)
        except Exception:
            pass
    
    # 3. Fallback: UI-Werte zusammengesetzt + externe Args (aus Custom Commands Feld)
    args = get_current_args(window)
    
    # Custom Commands Feld auslesen (benutzerdefinierte Kommandozeilen-Argumente)
    if hasattr(window, 'custom_cmd_edit') and window.custom_cmd_edit:
        custom_text = window.custom_cmd_edit.toPlainText()
        custom_args = _parse_custom_commands_text(custom_text)
        args.extend(custom_args)
    
    return " ".join(args)


def on_param_changed(window) -> None:
    """Debug-Output live aktualisieren wenn sich ein Parameter ändert.
    
    KV-Cache-Werte und Kommandozeile werden als einzige Zeilen ausgegeben
    (keine Historie, alte Inhalte werden überschrieben).
    
    Args:
        window: llauncher main window instance
    """
    # Guard: Überspringen wenn cache-type Optionen gerade aktualisiert werden
    if getattr(window, '_updating_cache_type', False):
        return
    
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
            elif config.get("type") == "slider":
                if isinstance(slider, dict) and "slider" not in slider:
                    return
            # combo, text_input (path_input handled above), file_input: keine extra Prüfung
        except (KeyError, AttributeError):
            return
    
    # KV-Cache-Werte lesen
    cache_type_k = ""
    cache_type_v = ""
    for combo_key in ("--cache-type-k", "--cache-type-v"):
        if combo_key in window.param_sliders and "combo" in window.param_sliders[combo_key]:
            val = window.param_sliders[combo_key]["combo"].currentText()
            if combo_key == "--cache-type-k":
                cache_type_k = val
            else:
                cache_type_v = val

    # Debug-Feld leeren und neue KV-Zeile + Kommando schreiben
    try:
        window.debug_text.clear()
        window.debug_text.append(f"KV-K: {cache_type_k} | KV-V: {cache_type_v}")
    except Exception:
        pass

    # Command direkt aus UI-Werten bauen — ohne laufenden Prozess zu berücksichtigen
    args = get_current_args(window)

    # Custom Commands Feld auslesen
    if hasattr(window, 'custom_cmd_edit') and window.custom_cmd_edit:
        custom_text = window.custom_cmd_edit.toPlainText()
        custom_args = _parse_custom_commands_text(custom_text)
        args.extend(custom_args)

    window.debug_text.append(" ".join(args))


# The following _append_vram_estimate function was replaced by live GPU stats in
# window.update_gpu_display(). Kept for reference but no longer called.


def _append_vram_estimate(window) -> None:
    """Append a compact VRAM estimation line to debug output.

    Uses the cached _model_info dict set by llauncher.on_model_selected().
    """
    info = getattr(window, '_model_info', None)
    if not info or not info.get('file_size'):
        return

    try:
        from gguf_utils import read_gpu_vram, estimate_vram
    except ImportError:
        return

    # Read current params
    param_sliders = getattr(window, 'param_sliders', {})
    if not param_sliders:
        return

    ctx_size = 4096
    c_slider = param_sliders.get("-c", {})
    if c_slider:
        s = c_slider.get("slider")
        if s:
            ctx_size = s.value()

    ngl = 0
    ngl_slider = param_sliders.get("-ngl", {})
    if ngl_slider:
        s = ngl_slider.get("slider")
        if s:
            ngl = s.value()
        if hasattr(window, "ngl_all_checkbox") and window.ngl_all_checkbox.isChecked():
            ngl = -1

    np_slots = 1
    np_slider = param_sliders.get("-np", {})
    if np_slider:
        s = np_slider.get("slider")
        if s:
            np_slots = s.value()

    k_combo = param_sliders.get("--cache-type-k", {}).get("combo")
    cache_type_k = k_combo.currentText() if k_combo else "f16"
    v_combo = param_sliders.get("--cache-type-v", {}).get("combo")
    cache_type_v = v_combo.currentText() if v_combo else "f16"

    if ngl == 0:
        return  # CPU mode — no VRAM needed

    try:
        vram = estimate_vram(
            model_info=info,
            ngl=ngl,
            ctx_size=ctx_size,
            np_slots=np_slots,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
        )
        total_mb = vram["total_vram_mb"]
        model_mb = vram["model_vram_mb"]
        cache_mb = vram["cache_vram_mb"]
        total_gb = total_mb / 1024

        gpu = read_gpu_vram()
        if gpu:
            free_mb = gpu["free_mb"]
            if total_mb <= free_mb:
                status = "✓"  # fits
            elif total_mb <= gpu["total_mb"]:
                status = "⚠"  # partial
            else:
                status = "✗"  # no fit
            vram_line = f"VRAM: {total_gb:.2f} GB (model {model_mb:.0f} MB + cache {cache_mb:.1f} MB) | GPU free: {free_mb} MB {status}"
        else:
            vram_line = f"VRAM: {total_gb:.2f} GB (model {model_mb:.0f} MB + cache {cache_mb:.1f} MB) | GPU query unavailable"

        ngl_display = "all" if ngl < 0 else str(ngl)
        vram_line += f" [ngl={ngl_display}, ctx={ctx_size}, slots={np_slots}]"
        window.debug_text.append(vram_line)
    except Exception:
        pass  # Silent fail — don't break debug output
