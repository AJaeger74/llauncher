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


def _update_gpu_display(label, gpu_data: dict) -> None:
    """Update the GPU stats label with current data.
    
    Args:
        label: QLabel widget to update
        gpu_data: Dictionary from GPUMonitor with GPU stats
    """
    if not gpu_data:
        label.setText("GPU: N/A")
        return

    # Build display string
    parts = []

    # GPU info
    temp = gpu_data.get("temp", 0)
    mem_used = gpu_data.get("used_mb", 0)
    mem_total = gpu_data.get("total_mb", 1)
    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
    gpu_load = gpu_data.get("gpu_usage", 0)
    parts.append(f"{temp}°C | {mem_used}/{mem_total} MB ({mem_pct:.0f}%) | {gpu_load}%")

    # Context usage from /slots API
    slots = gpu_data.get("slots")
    if slots:
        used = slots.get("used_tokens", 0)
        ctx = slots.get("ctx", 0)
        pct = (used / ctx * 100) if ctx > 0 else 0
        shed_offer = slots.get("shed_offer", 0)
        spec = slots.get("speculative", False)
        spec_type = slots.get("spec_type", "")

        # Calculate effective VBR status from shed_offer
        # shed_offer = bytes freed if all F16 layers degrade to turbo8
        # shed_per_val = 2.0 - 1.015625 = 0.984375
        if shed_offer > 0 and used > 0:
            # We need model info for exact layer count — use a reasonable default
            # or get it from the window's cached model info
            block_count = 65  # default; will be refined from model_info below
            key_head_count = 4
            kv_head_dim = 256

            # Try to get model info from window for accurate calculation
            label_obj = label  # the QLabel itself
            # Access window via label's parent chain if possible
            model_info = None
            try:
                # Check if label has a parent window with _model_info
                parent = label.parent()
                while parent is not None:
                    mi = getattr(parent, '_model_info', None)
                    if mi and isinstance(mi, dict):
                        model_info = mi
                        break
                    parent = parent.parent()
            except Exception:
                pass

            if model_info:
                block_count = model_info.get("block_count", block_count)
                key_head_count = model_info.get("key_head_count", key_head_count)
                kv_head_dim = model_info.get("kv_head_dim", kv_head_dim) or 256

            shed_per_val = 2.0 - 1.015625
            layers_f16 = shed_offer / (shed_per_val * used * key_head_count * kv_head_dim * 2)
            layers_degraded = block_count - layers_f16

            if layers_degraded > 0:
                ctx_str = f"CX: {used:,}/{ctx:,} ({pct:.0f}%) | VBR: {layers_degraded:.0f}/{block_count} degraded | {spec_type}"
            else:
                ctx_str = f"CX: {used:,}/{ctx:,} ({pct:.0f}%) | VBR: F16 | {spec_type}"
        else:
            ctx_str = f"CX: {used:,}/{ctx:,} ({pct:.0f}%) | {spec_type}"

        parts.append(ctx_str)

    label.setText(" | ".join(parts))


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
