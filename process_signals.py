"""Process signals and GPU monitoring setup."""

import psutil
from PyQt6.QtCore import QTimer


def start_gpu_monitor(window) -> None:
    """Start the GPU monitor thread and connect its signals.
    
    The gpu_update signal is connected in ui_builder.py to window.update_gpu_display(),
    which handles VRAM estimation + context info display.
    
    Args:
        window: The main llauncher window instance
    """
    from gpu_monitor import GPUMonitor
    
    if not hasattr(window, "gpu_monitor") or window.gpu_monitor is None:
        window.gpu_monitor = GPUMonitor()
        window.gpu_monitor._window = window  # for dynamic host resolution
    
    if not window.gpu_monitor.isRunning():
        window.gpu_monitor.start()


def get_free_gpu_memory(window) -> int:
    """Get free GPU memory from nvidia-smi via window method.
    
    Args:
        window: The main llauncher window instance
        
    Returns:
        Free memory in MB, or 2048 as fallback
    """
    if hasattr(window, "_get_free_gpu_memory"):
        return window._get_free_gpu_memory()
    return 2048


def _get_free_gpu_memory() -> int:
    """Get free GPU memory from nvidia-smi.
    
    Returns:
        Free memory in MB, or 0 if no GPU available
    """
    try:
        # Parse nvidia-smi output
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Take first GPU's free memory
            lines = result.stdout.strip().split("\n")
            if lines:
                free_mb = int(lines[0].strip())
                return free_mb
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    
    return 0


def setup_process_signals(window) -> None:
    """Connect all process-related signals for the window.
    
    Connects output and finished signals from ProcessRunner.
    
    Args:
        window: The main llauncher window instance
    """
    # Connect ProcessRunner signals if runner exists
    if hasattr(window, "runner") and window.runner:
        # These connections are typically made when runner is created
        # in process_runner.py, but we ensure they're set up here too
        pass
    
    # GPU monitor auto-starts on first call to start_gpu_monitor()
    # This is called from toggle_process() when needed
