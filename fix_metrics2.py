#!/usr/bin/env python3
"""Add completion_tokens and total_tokens to _metrics in standard benchmark."""
with open('http_benchmark_thread.py', 'r') as f:
    lines = f.readlines()

# Find line with "tps = completion_tokens" (around line 291)
for i, line in enumerate(lines):
    if 'tps = completion_tokens / (eval_ms' in line:
        print(f"Found at line {i+1}")
        # Insert after this line
        insert_after = i + 1
        
        new_lines = [
            "\n",
            "                # Save to _metrics so UI can access it (FIX: was missing!)\n",
            "                self._metrics[\"completion_tokens\"] = completion_tokens\n",
            "                self._metrics[\"total_tokens\"] = total_tokens\n"
        ]
        
        lines[insert_after:insert_after] = new_lines
        break

# Write back
with open('http_benchmark_thread.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed! _metrics now populated with completion_tokens and total_tokens")

# Verify syntax
import py_compile
try:
    py_compile.compile('http_benchmark_thread.py', doraise=True)
    print("✓ No syntax errors!")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
