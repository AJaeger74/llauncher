#!/usr/bin/env python3
"""Add completion_tokens update to server log parsing."""
with open('http_benchmark_thread.py', 'r') as f:
    lines = f.readlines()

# Find the line with total_time_ms and add after it
for i, line in enumerate(lines):
    if "self._metrics['total_time'] = self._server_log_metrics['total_time_ms'] / 1000" in line:
        print(f"Found at line {i+1}")
        
        new_lines = [
            "\n",
            "        # Update completion_tokens and total_tokens from server logs\n",
            "        if self._server_log_metrics.get('gen_tokens'):\n",
            '            self._metrics["completion_tokens"] = self._server_log_metrics[\'gen_tokens\']\n',
            "\n",
            "        if self._server_log_metrics.get('total_tokens'):\n",
            '            self._metrics["total_tokens"] = self._server_log_metrics[\'total_tokens\']\n'
        ]
        
        lines[i+1:i+1] = new_lines
        break

with open('http_benchmark_thread.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed!")

import py_compile
try:
    py_compile.compile('http_benchmark_thread.py', doraise=True)
    print("✓ No syntax errors!")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
