"""Run with .venv/bin/python -m unittest test_process_portability."""
import os
import shlex
import signal
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
from process_runner import ProcessRunner, find_llama_processes, read_running_llama_args
from process_inspector import check_existing_process, get_running_server_command


class ProcessPortabilityTests(unittest.TestCase):
    def test_real_process_arguments_and_forced_termination(self):
        script = (
            'import signal,time; '
            'signal.signal(signal.SIGINT, signal.SIG_IGN); '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            'print("ready", flush=True); time.sleep(60)'
        )
        child = subprocess.Popen(
            [sys.executable, '-c', script, '/model folder/model name.gguf'],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            runner = ProcessRunner([], os.getcwd())
            runner._process = child
            self.assertEqual(runner.get_args_from_proc()[-1], '/model folder/model name.gguf')
            window = SimpleNamespace(runner=runner)
            with patch.object(runner, 'isRunning', return_value=True):
                self.assertEqual(shlex.split(get_running_server_command(window))[-1],
                                 '/model folder/model name.gguf')
            self.assertTrue(ProcessRunner.terminate_by_pid(child.pid, timeout_sec=0.15))
            self.assertEqual(child.wait(timeout=2), -signal.SIGKILL)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()

    def test_denied_or_unconfirmed_stop_is_failure(self):
        process = Mock()
        process.send_signal.side_effect = psutil.AccessDenied(123)
        with patch('process_runner.psutil.Process', return_value=process):
            self.assertFalse(ProcessRunner.terminate_by_pid(123, 0.1))
        process.send_signal.side_effect = None
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING
        with patch('process_runner.psutil.Process', return_value=process):
            self.assertFalse(ProcessRunner.terminate_by_pid(123, 0.1))
        self.assertEqual(process.send_signal.call_args.args, (signal.SIGKILL,))

    def test_discovery_ignores_shell_arguments(self):
        processes = [SimpleNamespace(info={'pid': pid, 'name': name, 'cmdline': args})
                     for pid, name, args in (
                         (11, 'llama-server', ['/opt/homebrew/bin/llama-server', '-m', 'a b.gguf']),
                         (12, 'zsh', ['zsh', '-c', 'llama-server -m a.gguf']),
                     )]
        with patch('process_runner.psutil.process_iter', return_value=processes):
            self.assertEqual(find_llama_processes(), [11])
        process = Mock()
        process.cmdline.return_value = ['/opt/homebrew/bin/llama-server', '-m', 'a b.gguf', '-ngl', '-1']
        process.exe.return_value = '/opt/homebrew/bin/llama-server'
        with patch('process_runner.find_llama_processes', return_value=[11]), \
                patch('process_runner.psutil.Process', return_value=process):
            params, model, exe, found = read_running_llama_args()
        self.assertTrue(found)
        self.assertEqual(model, 'a b.gguf')
        self.assertEqual(exe, '/opt/homebrew/bin/llama-server')
        self.assertEqual(params['-ngl'], '-1')

    def test_denied_process_enumeration_is_safe(self):
        with patch('process_runner.psutil.process_iter', side_effect=PermissionError('sysctl denied')):
            self.assertEqual(find_llama_processes(), [])

    def test_own_runner_is_not_adopted_as_external(self):
        window = SimpleNamespace(runner=Mock(), external_runner_pid=123, external_runner_args=['old'])
        window.runner.isRunning.return_value = True
        with patch('process_inspector.find_llama_processes') as discover:
            check_existing_process(window)
        discover.assert_not_called()
        self.assertIsNone(window.external_runner_pid)
        self.assertIsNone(window.external_runner_args)


if __name__ == '__main__':
    unittest.main()
