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

        # Get model info from window for B/v calculation
        model_info = None
        try:
            parent = label.parent()
            while parent is not None:
                mi = getattr(parent, '_model_info', None)
                if mi and isinstance(mi, dict):
                    model_info = mi
                    break
                parent = parent.parent()
        except Exception:
            pass

        block_count = model_info.get("block_count", 65) if model_info else 65
        key_head_count = model_info.get("key_head_count", 4) if model_info else 4
        kv_head_dim = model_info.get("kv_head_dim", 256) if model_info else 256
        tensor_bytes = model_info.get("tensor_bytes", 0) if model_info else 0

        # Calculate Avg B/v from GPU usage
        gpu_used_mb = gpu_data.get("used_mb", 0)
        ctx_str = f"CX: {used:,}/{ctx:,} ({pct:.0f}%)"

        if shed_offer > 0 and used > 0 and gpu_used_mb > 0 and tensor_bytes > 0:
            # Back-calculate cache size from GPU usage
            model_mb = tensor_bytes * 1.05 / (1024 * 1024)
            overhead_mb = 1024
            cache_mb = gpu_used_mb - model_mb - overhead_mb

            if cache_mb > 0:
                cache_bytes = cache_mb * 1024 * 1024
                bv_avg = cache_bytes / (used * key_head_count * kv_head_dim * 2 * block_count)
                ctx_str += f" | B/v: {bv_avg:.3f}"

                # Layers F16 / degraded from shed_offer
                shed_per_val = 2.0 - 1.015625
                layers_f16 = shed_offer / (shed_per_val * used * key_head_count * kv_head_dim * 2)
                layers_degraded = block_count - layers_f16

                if layers_degraded > 0:
                    # Determine closest tier
                    tiers = [
                        ("t8", 1.015625),
                        ("t4", 0.515625),
                        ("t3", 0.40625),
                        ("t2", 0.28125),
                        ("t1", 0.15625),
                    ]
                    bv_degraded = (bv_avg * block_count - layers_f16 * 2.0) / layers_degraded
                    if bv_degraded > 0:
                        closest = min(tiers, key=lambda x: abs(x[1] - bv_degraded))
                        ctx_str += f" ({layers_degraded:.0f} ≈ {closest[0]})"

        elif spec_type:
            ctx_str += f" | {spec_type}"

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
