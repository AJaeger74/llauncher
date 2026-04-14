#!/usr/bin/env python3
"""Update _metrics from _server_log_metrics parsing."""
with open('http_benchmark_thread.py', 'r') as f:
    lines = f.readlines()

# Find the section after total_time_ms update
for i, line in enumerate(lines):
    if "if self._server_log_metrics.get('total_time_ms'):" in line:
        print(f"Found at line {i+1}")
        # Find the end of this block and insert completion_tokens
        # Look for next non-indented line or different condition
        j = i + 1
        while j < len(lines) and (lines[j].startswith(' ') or lines[j].strip() == ''):
            j += 1
        
        # Insert before the next major block
        new_lines = [
            "            self._metrics[\"total_time\"] = self._server_log_metrics['total_time_ms'] / 1000\n",
            "\n",
            "        # Update completion_tokens and total_tokens from server logs\n",
            "        if self._server_log_metrics.get('gen_tokens'):\n",
            "            self._metrics[\"completion_tokens\"] = self._server_log_metrics['gen_tokens']\n",
            "        \n",
            "        if self._server_log_metrics.get('total_tokens'):\n",
            "            self._metrics[\"total_tokens\"] = self._server_log_metrics['total_tokens']\n"
        ]
        
        # Replace the old total_time block with new version
        lines[j-1:j] = new_lines
        break

# Write back
with open('http_benchmark_thread.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed! _metrics now updated from server log parsing")

# Verify syntax
import py_compile
try:
    py_compile.compile('http_benchmark_thread.py', doraise=True)
    print("✓ No syntax errors!")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
