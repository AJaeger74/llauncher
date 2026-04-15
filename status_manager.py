"""Status management utilities for llauncher."""

from PyQt6.QtGui import QColor


# Status colors (hex codes)
STATUS_COLORS = {
    "ready": "#4CAF50",      # Green
    "loading": "#2196F3",    # Blue
    "running": "#FF9800",    # Orange
    "error": "#F44336",      # Red
    "idle": "#9E9E9E",       # Gray
}


def update_status(window, state: str) -> None:
    """Update the status label with colored state indicator.
    
    Args:
        window: The main llauncher window instance
        state: Status state string (ready, loading, running, error, idle)
    """
    if not hasattr(window, "status_label"):
        return
    
    color = STATUS_COLORS.get(state, "#9E9E9E")
    style = f"QLabel {{ color: {color}; font-weight: bold; }}"
    window.status_label.setStyleSheet(style)
    
    state_labels = {
        "ready": "Ready to go",
        "loading": "Loading model...",
        "running": "Running",
        "error": "Error",
        "idle": "Idle",
    }
    window.status_label.setText(state_labels.get(state, state))


def handle_process_error(window, exit_code: int) -> None:
    """Handle process termination with error.
    
    Updates status and logs the error.
    
    Args:
        window: The main llauncher window instance
        exit_code: Process exit code
    """
    update_status(window, "error")
    
    # Log error to debug output if available
    if hasattr(window, "debug_text"):
        window.debug_text.append(f"Process exited with code {exit_code}")


def reset_progress_bar(window) -> None:
    """Reset the progress bar to idle state.
    
    Args:
        window: The main llauncher window instance
    """
    if hasattr(window, "progress_bar"):
        window.progress_bar.setValue(0)
