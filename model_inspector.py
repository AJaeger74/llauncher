"""Model inspection utilities for llauncher."""

import os
from pathlib import Path
from typing import Any, Dict

# Import GGUF utilities
from gguf_utils import get_model_info as gguf_get_model_info, format_size, read_gguf_context_length


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
            window.debug_text.append("─" * 60)
            window.debug_text.append(f"📦 {gettext('msg_model_selected')}: {info['filename']}")
            window.debug_text.append("─" * 60)
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
            if info.get('block_count'):
                window.debug_text.append(f"  {gettext('debug_block_count')}      {info['block_count']}")
            
            window.debug_text.append("─" * 60)
        except Exception as e:
            window.debug_text.append(f"⚠️ {gettext('msg_model_info_read_error', error=str(e))}")
    
    # Update context slider from GGUF context length
    if model_path.exists() and model_path.is_file():
        ctx_length = read_gguf_context_length(str(model_path))
        if ctx_length and ctx_length > 0:
            slider_data = window.param_sliders["-c"]
            slider = slider_data["slider"]
            edit = slider_data["edit"]
            
            # Set slider maximum (no hard cap, but realistic limit)
            slider.setMaximum(ctx_length)
            
            # Set default value - BUT only if we're not loading from a running process!
            if not getattr(window, 'loading_running_args', False):
                slider.setValue(ctx_length)
            
            # Update edit widget width for new max number
            max_width = len(str(ctx_length)) * 9 + 15
            edit.setMinimumWidth(max_width)
            edit.setMaximumWidth(max_width)
    
    # Save config using preset manager
    from preset_manager import save_active_preset
    save_active_preset(window)
