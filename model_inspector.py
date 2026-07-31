"""Model inspection utilities for llauncher."""

import os
from pathlib import Path
from typing import Any, Dict

# Import GGUF utilities
from gguf_utils import (
    get_model_info as gguf_get_model_info,
    format_size,
    read_gguf_context_length,
    check_model_architecture,
    get_binary_path,
    read_gpu_vram,
    estimate_vram,
    suggest_ngl,
)


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted string (e.g., "1.2 GB", "450 MB")
    """
    if size_bytes < 0:
        return "Unknown"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def _get_model_info(model_path: str) -> Dict[str, Any]:
    """Get metadata about a GGUF model file.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Dictionary with:
        - file_size: File size in bytes
        - formatted_size: Human-readable file size
        - exists: Boolean whether file exists
        - is_gguf: Boolean whether file appears to be GGUF
    """
    result = {
        "file_size": 0,
        "formatted_size": "Unknown",
        "exists": False,
        "is_gguf": False,
    }
    
    if not model_path or not os.path.exists(model_path):
        return result
    
    try:
        file_size = os.path.getsize(model_path)
        result["file_size"] = file_size
        result["formatted_size"] = _format_file_size(file_size)
        result["exists"] = True
        
        # Check for GGUF magic number
        with open(model_path, "rb") as f:
            magic = f.read(4)
            # GGUF magic bytes: 0x46554747 = "GGUF"
            if magic == b"GGUF":
                result["is_gguf"] = True
    except (IOError, OSError):
        pass
    
    return result


def _get_ui_params(window) -> Dict[str, Any]:
    """Read current parameter values from the UI for VRAM estimation."""
    params = {}
    param_sliders = getattr(window, 'param_sliders', {})
    param_defs = getattr(window, 'PARAM_DEFINITIONS', {})

    # Context size (-c)
    c_slider = param_sliders.get("-c", {})
    if c_slider:
        slider_widget = c_slider.get("slider")
        params["ctx_size"] = slider_widget.value() if slider_widget else 4096
    else:
        params["ctx_size"] = 4096

    # GPU layers (-ngl)
    ngl_slider = param_sliders.get("-ngl", {})
    if ngl_slider:
        slider_widget = ngl_slider.get("slider")
        ngl_value = slider_widget.value() if slider_widget else 0
        if hasattr(window, "ngl_all_checkbox") and window.ngl_all_checkbox.isChecked():
            ngl_value = -1
        params["ngl"] = ngl_value
    else:
        params["ngl"] = 0

    # Parallel slots (-np)
    np_slider = param_sliders.get("-np", {})
    if np_slider:
        slider_widget = np_slider.get("slider")
        params["np_slots"] = slider_widget.value() if slider_widget else 1
    else:
        params["np_slots"] = 1

    # Cache types
    k_combo = param_sliders.get("--cache-type-k", {}).get("combo")
    params["cache_type_k"] = k_combo.currentText() if k_combo else "f16"

    v_combo = param_sliders.get("--cache-type-v", {}).get("combo")
    params["cache_type_v"] = v_combo.currentText() if v_combo else "f16"

    return params


def _display_vram_estimate(window, model_info: Dict[str, Any]) -> None:
    """Display VRAM estimation in debug output."""
    try:
        from i18n import I18nManager
        gettext = I18nManager.get_instance().gettext
    except Exception:
        def gettext(key):
            return key

    params = _get_ui_params(window)
    ngl = params["ngl"]
    ctx_size = params["ctx_size"]
    np_slots = params["np_slots"]
    cache_type_k = params["cache_type_k"]
    cache_type_v = params["cache_type_v"]

    # Estimate VRAM — pass calibrated V bytes if VBR is active
    vbr_cal = getattr(window, '_calibrated_v_bytes', None)
    vram = estimate_vram(
        model_info=model_info,
        ngl=ngl,
        ctx_size=ctx_size,
        np_slots=np_slots,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        vbr_calibrated_v_bytes=vbr_cal,
    )

    # Guard: skip VRAM estimation if UI sliders not ready yet (e.g. during init)
    param_sliders = getattr(window, 'param_sliders', None)
    if not param_sliders:
        return

    # Get GPU info
    gpu = read_gpu_vram()
    window.debug_text.append("")
    window.debug_text.append(f"  ┃ {gettext('debug_vram_section')}")
    
    if ngl == 0:
        # CPU mode
        window.debug_text.append(f"  ┃ {gettext('debug_vram_cpu_only')}")
    else:
        # Show model weights VRAM
        model_mb = vram["model_vram_mb"]
        cache_mb = vram["cache_vram_mb"]
        total_mb = vram["total_vram_mb"]
        total_gb = total_mb / 1024
        model_gb = model_mb / 1024
        cache_gb = cache_mb / 1024
        overhead_mb = vram.get("overhead_mb", 0)
        mmproj_mb = vram.get("mmproj_vram_mb", 0)
        mmproj_gb = mmproj_mb / 1024

        window.debug_text.append(f"  ┃ {gettext('debug_vram_model')} {model_gb:.2f} GB ({model_mb:.0f} MB)")
        window.debug_text.append(f"  ┃ {gettext('debug_vram_cache')} {cache_gb:.2f} GB ({cache_mb:.0f} MB)")
        if mmproj_mb > 0:
            window.debug_text.append(f"  ┃ mmproj             {mmproj_gb:.2f} GB ({mmproj_mb:.0f} MB)")
        window.debug_text.append(f"  ┃ overhead           {overhead_mb:.0f} MB")
        window.debug_text.append(f"  ┃ {gettext('debug_vram_total')} {total_gb:.2f} GB ({total_mb:.0f} MB)")

        # Compare with GPU free memory
        if gpu:
            free_gb = gpu["free_mb"] / 1024
            window.debug_text.append(f"  ┃ {gettext('debug_vram_free')} {free_gb:.2f} GB ({gpu['free_mb']} MB / {gpu['total_mb']} MB)")

            if total_mb <= gpu["free_mb"]:
                window.debug_text.append(f"  ┃ {gettext('debug_vram_fit')}")
            elif total_mb <= gpu["total_mb"]:
                window.debug_text.append(f"  ┃ {gettext('debug_vram_partial')}")
            else:
                window.debug_text.append(f"  ┃ {gettext('debug_vram_nofit')}")
        else:
            window.debug_text.append(f"  ┃ {gettext('debug_vram_unavailable')}")

        # Show parameter note
        ngl_display = "all" if ngl < 0 else str(ngl)
        cache_display = f"{cache_type_k}/{cache_type_v}"
        note = gettext("debug_vram_ngl_note").format(
            ngl=ngl_display, ctx=ctx_size, slots=np_slots, cache=cache_display
        )
        window.debug_text.append(f"  ┃ {note}")


def on_model_selected(window, model_name: str) -> None:
    """Handle model selection in the model combo box.

    Updates the UI with model metadata and ensures parameter defaults are set.

    Args:
        window: The main llauncher window instance
        model_name: Selected model name from the combo box (filename only, not full path)
    """
    if not model_name:
        return

    # Resolve full path
    model_path = (Path(window.model_directory) / model_name).resolve()
    window.selected_model = str(model_path)

    # Import translation function here to avoid circular imports
    try:
        from i18n import I18nManager
        gettext = I18nManager.get_instance().gettext
    except Exception:
        def gettext(key):
            return key

    # Display GGUF metadata in debug text
    if model_path.exists() and model_path.is_file():
        try:
            info = gguf_get_model_info(str(model_path))

            # Strip leading/trailing whitespace/NULs from strings
            name = (info.get('name') or '').strip('\x00 \n\r\t')
            arch = (info.get('arch') or 'unknown').strip('\x00 ')

            # Debug separator
            window.debug_text.append("\u2500" * 60)
            window.debug_text.append(f"\U0001f4e6 {gettext('msg_model_selected')}: {info['filename']}")
            window.debug_text.append("\u2500" * 60)
            window.debug_text.append(f"  {gettext('debug_model_name')}             {name or gettext('msg_unavailable')}")
            window.debug_text.append(f"  {gettext('debug_model_architecture')}      {arch}")

            if info.get('tags'):
                tags_str = ", ".join(str(t) for t in info['tags'][:5])
                if len(info['tags']) > 5:
                    tags_str += f" (+{len(info['tags']) - 5} more)"
                window.debug_text.append(f"  {gettext('debug_model_tags')}             {tags_str}")

            if info.get('url'):
                short_url = info['url'][:50] + "..." if len(info['url']) > 50 else info['url']
                window.debug_text.append(f"  {gettext('debug_model_url')}              {short_url}")

            window.debug_text.append(f"  {gettext('debug_model_size')}       {format_size(info['file_size'])}")
            window.debug_text.append(f"  {gettext('debug_gguf_version')}     v{info['version']}")
            window.debug_text.append(f"  {gettext('debug_tensor_count')}     {info['tensor_count']:,}")
            window.debug_text.append(f"  {gettext('debug_context_length')}   {info['context_length'] or gettext('msg_not_found')}")

            if info.get('embedding_length'):
                window.debug_text.append(f"  {gettext('debug_embedding_length')}  {info['embedding_length']}")
            window.debug_text.append(f"  {gettext('debug_block_count')}      {info.get('block_count', '')}")

            window.debug_text.append("\u2500" * 60)

            # Cache model info on window for live VRAM updates
            window._model_info = info

            # VRAM estimation
            _display_vram_estimate(window, info)

            window.debug_text.append("")
            window.debug_text.append("\u2500" * 60)

            # Architektur-Prüfung: Nur statische Liste (schnell, kein Binary-Check hier)
            unsupported_arch = check_model_architecture(arch)
            if unsupported_arch:
                arch_msg = gettext("msg_arch_unsupported").format(arch=unsupported_arch)
                window.debug_text.append(f"  \u26d4 {arch_msg}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    window,
                    gettext("msg_arch_unsupported_title"),
                    arch_msg,
                )
        except Exception as e:
            window.debug_text.append(f"\u26a0\ufe0f {gettext('msg_model_info_read_error').format(error=str(e))}")

    # Update context slider from GGUF context length
    if model_path.exists() and model_path.is_file():
        ctx_length = read_gguf_context_length(str(model_path))
        if ctx_length and ctx_length > 0:
            slider_data = window.param_sliders["-c"]
            slider = slider_data["slider"]
            edit = slider_data["edit"]

            # Set slider maximum (no hard cap, but realistic limit)
            slider.setMaximum(ctx_length)

            # Set default value - BUT only if we're not loading from a preset or running process!
            if not getattr(window, 'loading_running_args', False) and not getattr(window, 'loading_preset', False):
                slider.setValue(ctx_length)

            # Update edit widget width for new max number
            # Skip if loading preset — apply_preset() already sets the correct width
            # (based on effective_max which may exceed ctx_length)
            if not getattr(window, 'loading_preset', False) and not getattr(window, 'loading_running_args', False):
                max_width = len(str(ctx_length)) * 9 + 15
                edit.setMinimumWidth(max_width)
                edit.setMaximumWidth(max_width)
