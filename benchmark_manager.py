#!/usr/bin/env python3
"""Benchmark orchestration for llauncher - manages benchmark lifecycle."""

import time
from pathlib import Path


class BenchmarkManager:
    """Manages HTTP-based benchmark execution and state."""
    
    def __init__(self, window):
        """Initialize with reference to main window.
        
        Args:
            window: Main llauncher window instance (provides UI access)
        """
        self.window = window
        self.bench_thread = None
        self._last_benchmark_command = None
    
    def run_benchmark_streaming(self):
        """Run HTTP-based benchmark in streaming mode for live display.
        
        Orchestrates benchmark execution without direct UI manipulation.
        Signals are emitted to update UI via connected slots.
        """
        # Build and store command for benchmark completion handler
        self._last_benchmark_command = self.window.build_full_command()
        
        # Enable cancel button during benchmark
        if hasattr(self.window, 'cancel_bench_btn'):
            self.window.cancel_bench_btn.setEnabled(True)
        
        # Ensure we don't have a stale benchmark thread
        bench_thread = getattr(self.window, 'bench_thread', None)
        if bench_thread and bench_thread.isRunning():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self.window, "Fehler", "Ein Benchmark läuft bereits.")
            return
        
        self.window.debug_text.clear()
        self.window.status_label.setText("Benchmark (Live) läuft...")
        
        # Start GPU monitoring during benchmark (if not already running)
        if not hasattr(self.window, 'gpu_monitor') or not self.window.gpu_monitor.isRunning():
            from gpu_monitor import GPUMonitor
            self.window.gpu_monitor = GPUMonitor()
            self.window.gpu_monitor.gpu_update.connect(self.window.update_gpu_display)
            self.window.gpu_monitor.start()
        
        # Get max_tokens from -n slider (default 64)
        n_slider_data = self.window.param_sliders.get("-n")
        if n_slider_data and isinstance(n_slider_data, dict) and "slider" in n_slider_data:
            max_tokens = n_slider_data["slider"].value()
        else:
            max_tokens = 64
        
        # Import benchmark runner and create thread
        from http_benchmark_thread import HTTPBenchmarkRunner
        
        self.bench_thread = HTTPBenchmarkRunner(
            max_tokens=max_tokens,
            server_pid=self.window.external_runner_pid,
            streaming=True,
            model_path=self.window.selected_model
        )
        self.bench_thread.output_signal.connect(self.window.on_benchmark_output)
        self.bench_thread.status_signal.connect(self.window.status_label.setText)
        self.bench_thread.finished_signal.connect(self.window.on_benchmark_finished)
        self.bench_thread.token_update_signal.connect(self.window.on_benchmark_token_update)
        self.bench_thread.start()
    
    def run_benchmark(self):
        """Run HTTP-based benchmark in standard (non-streaming) mode."""
        # Clear debug text to avoid parsing old benchmark data
        self.window.debug_text.clear()
        
        # Build and store command for benchmark completion handler
        self._last_benchmark_command = self.window.build_full_command()
        
        # Enable cancel button during benchmark
        if hasattr(self.window, 'cancel_bench_btn'):
            self.window.cancel_bench_btn.setEnabled(True)
        
        # Get max_tokens from -n slider (default 64)
        n_slider_data = self.window.param_sliders.get("-n")
        if n_slider_data and isinstance(n_slider_data, dict) and "slider" in n_slider_data:
            max_tokens = n_slider_data["slider"].value()
        else:
            max_tokens = 64
        
        # Import benchmark runner and create thread
        from http_benchmark_thread import HTTPBenchmarkRunner
        
        self.bench_thread = HTTPBenchmarkRunner(
            max_tokens=max_tokens,
            server_pid=self.window.external_runner_pid,
            streaming=False,
            model_path=self.window.selected_model
        )
        self.bench_thread.output_signal.connect(self.window.on_benchmark_output)
        self.bench_thread.status_signal.connect(self.window.status_label.setText)
        self.bench_thread.finished_signal.connect(self.window.on_benchmark_finished)
        self.bench_thread.token_update_signal.connect(self.window.on_benchmark_token_update)
        self.bench_thread.start()
    
    def cancel_benchmark(self):
        """Cancel the currently running benchmark."""
        print("[DEBUG] cancel_benchmark() called!")  # Terminal output for debugging
        
        bench_thread = self.bench_thread
        if not bench_thread:
            self.window.debug_text.append("ERROR: No benchmark thread found to cancel!")
            return
        
        # Signal cancellation to the thread - call cancel() method directly
        self.window.debug_text.append(f"Cancelling benchmark... (thread={bench_thread})")
        
        if hasattr(bench_thread, '_cancelled') and hasattr(bench_thread, 'cancel'):
            bench_thread._cancelled = True
            # Call cancel() which writes to pipe AND closes socket
            bench_thread.cancel()
            self.window.debug_text.append("Cancel signal sent!")
        else:
            self.window.debug_text.append(f"WARNING: Thread missing _cancelled or cancel method (has: {dir(bench_thread)})")
        
        if hasattr(self.window, 'cancel_bench_btn'):
            self.window.cancel_bench_btn.setEnabled(False)
    
    def get_last_benchmark_command(self):
        """Return the last benchmark command that was executed."""
        return self._last_benchmark_command
    
    def is_benchmark_running(self):
        """Check if a benchmark is currently running."""
        bench_thread = self.bench_thread
        return bench_thread is not None and bench_thread.isRunning()
