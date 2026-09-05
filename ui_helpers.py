"""UI helper functions for file dialogs and text output."""

import os
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QDir


def browse_llama_dir(window) -> None:
    """Open dialog to select llama.cpp directory.
    
    Args:
        window: The main llauncher window instance
    """
    start_dir = os.path.expanduser("~")
    selected_dir = QFileDialog.getExistingDirectory(
        window, "Select llama.cpp directory", start_dir,
        QFileDialog.Option.ShowDirsOnly
    )
    
    if selected_dir:
        window.llama_dir_line.setText(selected_dir)


def browse_model_dir(window) -> None:
    """Open dialog to select model directory.
    
    Args:
        window: The main llauncher window instance
    """
    start_dir = os.path.expanduser("~")
    selected_dir = QFileDialog.getExistingDirectory(
        window, "Select model directory", start_dir,
        QFileDialog.Option.ShowDirsOnly
    )
    
    if selected_dir:
        window.model_dir_line.setText(selected_dir)


def browse_path(window, line_edit, start_dir: str = None, file_filter: str = "") -> None:
    """Open file browser dialog for a generic path.
    
    Args:
        window: The main llauncher window instance
        line_edit: QLineEdit widget to populate with selected path
        start_dir: Starting directory (default: user home)
        file_filter: File type filter (e.g., "GGUF files (*.gguf)")
    """
    if start_dir is None:
        start_dir = os.path.expanduser("~")
    
    selected_file = QFileDialog.getOpenFileName(
        window, "Select file", start_dir, file_filter
    )[0]
    
    if selected_file:
        line_edit.setText(selected_file)


def on_select_benchmark_file(line_edit) -> None:
    """Open dialog to select benchmark CSV/JSON export file.
    
    Args:
        line_edit: QLineEdit widget to populate with selected path
    """
    start_dir = os.path.expanduser("~/llauncher/benchmarks")
    if not os.path.exists(start_dir):
        start_dir = os.path.expanduser("~")
    
    selected_file, _ = QFileDialog.getSaveFileName(
        None, "Select benchmark export file",
        start_dir,
        "CSV files (*.csv);;JSON files (*.json);;All files (*)"
    )
    
    if selected_file:
        line_edit.setText(selected_file)


def on_clear_benchmark_file(line_edit) -> None:
    """Clear the benchmark export file path.
    
    Args:
        line_edit: QLineEdit widget to clear
    """
    line_edit.clear()


def append_text_to_widget(text: str, text_widget) -> None:
    """Append text to QTextEdit widget.
    
    Args:
        text: Text to append
        text_widget: QTextEdit or QPlainTextEdit widget
    """
    text_widget.append(text)


def _append_text_inline(text: str, text_widget) -> None:
    """Append text to QTextEdit without triggering full reflow.
    
    Efficient method for adding small chunks of text to a QTextEdit widget.
    
    Args:
        text: Text to append
        text_widget: QTextEdit or QPlainTextEdit widget
    """
    cursor = text_widget.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    cursor.insertText(text)
    text_widget.setTextCursor(cursor)


def sanitize_host(host_text: str) -> str:
    """Strip trailing port from a host value (e.g. '127.0.0.1:8080' -> '127.0.0.1').

    Also handles IPv6 like '[::1]:8080' -> '[::1]' and plain hostnames."""
    if not host_text:
        return "localhost"
    # Strip trailing port for plain IPv4/host: remove last ':digits' segment
    if host_text.startswith("["):
        # IPv6 bracket notation: [::1]:8080
        bracket_end = host_text.find("]")
        if bracket_end >= 0:
            return host_text[:bracket_end + 1]
        return host_text
    # Only strip if the last colon is followed by digits (port-like)
    last_colon = host_text.rfind(":")
    if last_colon > 0 and host_text[last_colon + 1:].isdigit():
        return host_text[:last_colon]
    return host_text


def host_port(host_text: str, default: str = "8080") -> str:
    """Extract the trailing port from a host value ('127.0.0.1:8090' -> '8090').

    The --host UI field accepts an optional ':port' suffix (sanitize_host
    strips it). Returns `default` when no port is given or the input is
    empty. Handles IPv6 bracket notation ('[::1]:9000' -> '9000')."""
    if not host_text:
        return default
    text = host_text.strip()
    if text.startswith("["):
        bracket_end = text.find("]")
        if bracket_end >= 0:
            rest = text[bracket_end + 1:]
            if rest.startswith(":") and rest[1:].isdigit():
                return rest[1:]
        return default
    last_colon = text.rfind(":")
    if last_colon > 0 and text[last_colon + 1:].isdigit():
        return text[last_colon + 1:]
    return default


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
