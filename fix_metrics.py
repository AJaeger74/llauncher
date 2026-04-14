#!/usr/bin/env python3
"""Add completion_tokens and total_tokens to _metrics in standard benchmark."""
with open('http_benchmark_thread.py', 'r') as f:
    content = f.read()

# Find the section after completion_tokens is calculated and add _metrics updates
old_code = """                # Log values for debugging TPS calculation - now completion_tokens is defined
                self.output_signal.emit(f"DEBUG: Standard benchmark - completion_tokens={completion_tokens}, eval_ms={eval_ms}\\n")
                
                tps = completion_tokens / (eval_ms / 1000) if eval_ms > 0 else 0"""

new_code = """                # Log values for debugging TPS calculation - now completion_tokens is defined
                self.output_signal.emit(f"DEBUG: Standard benchmark - completion_tokens={completion_tokens}, eval_ms={eval_ms}\\n")
                
                # Save to _metrics so UI can access it (FIX: was missing!)
                self._metrics["completion_tokens"] = completion_tokens
                self._metrics["total_tokens"] = total_tokens
                
                tps = completion_tokens / (eval_ms / 1000) if eval_ms > 0 else 0"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('http_benchmark_thread.py', 'w') as f:
        f.write(content)
    print("✓ Fixed! _metrics now populated with completion_tokens and total_tokens")
else:
    print("✗ Could not find code block to replace")

# Verify syntax
import py_compile
try:
    py_compile.compile('http_benchmark_thread.py', doraise=True)
    print("✓ No syntax errors!")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
