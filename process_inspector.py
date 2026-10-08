#!/usr/bin/env python3
"""Portable inspection of running llama.cpp servers."""

import shlex
import psutil
from process_runner import find_llama_processes


def _gettext(key: str) -> str:
    """Lazy-loaded gettext function - waits for i18n initialization."""
    try:
        from i18n import I18nManager
        return I18nManager.get_instance().gettext(key)
    except Exception:
        return key


def check_existing_process(window):
    """Detect external servers without adopting the launcher's own runner."""
    runner = getattr(window, 'runner', None)
    if runner and runner.isRunning():
        window.external_runner_pid = None
        window.external_runner_args = None
        return

    for pid in find_llama_processes():
        try:
            args = psutil.Process(pid).cmdline()
        except psutil.Error:
            continue
        if not args:
            continue
        if not getattr(window, 'benchmark_running', False):
            window.start_stop_btn.setText(_gettext("btn_stop"))
            window.start_stop_btn.setObjectName("StopButton")
        window.external_runner_pid = pid
        window.external_runner_args = args
        return

    window.external_runner_pid = None
    window.external_runner_args = None
    if not getattr(window, 'benchmark_running', False):
        window.status_label.setText(_gettext("status_ready"))
        window.status_label.setStyleSheet("")
        window.start_stop_btn.setText(_gettext("btn_start"))
        window.start_stop_btn.setObjectName("StartButton")


def get_running_server_command(window):
    """Return the shell-quoted command line of a running server."""
    runner = getattr(window, 'runner', None)
    own_pid = runner.get_pid() if runner and runner.isRunning() else None
    pids = [own_pid] if own_pid else find_llama_processes()
    for pid in pids:
        try:
            args = psutil.Process(pid).cmdline()
            if args:
                return shlex.join(args)
        except psutil.Error:
            continue
    return None
