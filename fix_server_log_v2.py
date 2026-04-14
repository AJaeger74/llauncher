#!/usr/bin/env python3
"""Fix the server log parsing to also update completion_tokens and total_tokens."""
with open('http_benchmark_thread.py', 'r') as f:
    content = f.read()

# The issue is indentation - my script used spaces but file has tabs in some places
# Let me find the exact block and replace it

old_section = """        if self._server_log_metrics.get('total_time_ms'):
            self._metrics['total_time'] = self._server_log_metrics['total_time_ms'] / 1000
    
    def _extract_pdf_text(self, pdf_path: str):"""

new_section = """        if self._server_log_metrics.get('total_time_ms'):
            self._metrics['total_time'] = self._server_log_metrics['total_time_ms'] / 1000
        
        # Update completion_tokens and total_tokens from server logs
        if self._server_log_metrics.get('gen_tokens'):
            self._metrics["completion_tokens"] = self._server_log_metrics['gen_tokens']
        
        if self._server_log_metrics.get('total_tokens'):
            self._metrics["total_tokens"] = self._server_log_metrics['total_tokens']
    
    def _extract_pdf_text(self, pdf_path: str):"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('http_benchmark_thread.py', 'w') as f:
        f.write(content)
    print("✓ Fixed! Server log parsing now updates completion_tokens and total_tokens")
else:
    print("✗ Could not find the exact block - showing nearby lines:")
    # Find and show what's there
    import re
    match = re.search(r"if self._server_log_metrics\.get\('total_time_ms'\):.*?def _extract_pdf_text", content, re.DOTALL)
    if match:
        print("Found block:")
        print(repr(match.group(0)[:200]))

import py_compile
try:
    py_compile.compile('http_benchmark_thread.py', doraise=True)
    print("✓ No syntax errors!")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
