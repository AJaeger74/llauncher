#!/usr/bin/env python3
"""HTTP-based benchmark thread for llauncher – runs asynchronously with streaming support."""

import json, os, select, socket, threading, time, re
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class HTTPBenchmarkRunner(QThread):
    """Runs a single HTTP benchmark request without blocking the UI.
    
    Supports two modes:
    - Standard: Single HTTP POST request (fast)
    - Streaming: Read output line-by-line until silence (for live display)
    """
    
    output_signal = pyqtSignal(str)  # Lines to append to debug text
    finished_signal = pyqtSignal(float, int)  # TPS and token count
    
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8080
    SERVER_PATH = "/v1/completions"
    
    def __init__(self, max_tokens: int = 64, server_pid: int = None, streaming: bool = False, model_path: str = None):
        super().__init__()
        
        config_path = Path.home() / ".llauncher" / "config.json"
        if not config_path.exists():
            benchmark_cfg = {"prompt": "", "max_tokens": 64, "temperature": 0.8, "top_p": 0.95}
        else:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                benchmark_cfg = config.get("benchmark", {}) if config else {}
            except Exception:
                benchmark_cfg = {"prompt": "", "max_tokens": 256}
        
        self.raw_prompt = benchmark_cfg.get("prompt", "")
        self.max_tokens = benchmark_cfg.get("max_tokens", 256)
        
        # Lade Benchmark-Datei wenn vorhanden (vor der Fragegenerierung)
        benchmark_file_path = benchmark_cfg.get("benchmark_file_path", "")
        self.context_content = ""
        if benchmark_file_path:
            try:
                self.output_signal.emit(f"ℹ Loading file: {benchmark_file_path}")
                
                # Dateiendung prüfen für PDF-Extraktion
                file_ext = Path(benchmark_file_path).suffix.lower()
                
                if file_ext == '.pdf':
                    # PDF-Text extrahieren
                    self.context_content = self._extract_pdf_text(benchmark_file_path)
                    if not self.context_content:
                        raise Exception("PDF-Textextraktion fehlgeschlagen")
                else:
                    # Normale Textdatei
                    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
                        self.context_content = f.read()
                
                self.output_signal.emit(f"✓ Benchmark file loaded: {benchmark_file_path} ({len(self.context_content)} chars)")
            except Exception as e:
                self.output_signal.emit(f"⚠️ Error loading benchmark file: {e}")
                self.context_content = ""
        
        # Prompt mit Kontext kombinieren
        if self.context_content:
            self.raw_prompt = f"{self.context_content}\n\n{self.raw_prompt}"
        
        # Apply chat template if model path is provided
        self.prompt = self._apply_chat_template_to_prompt(self.raw_prompt, model_path)
        
        self.output_signal.emit(f"[DEBUG] Prompt loaded: {len(self.raw_prompt)} chars (template applied: {len(self.prompt)} chars)")
        
        self.server_pid = server_pid
        self.streaming = streaming
        self._cancelled = False
        
        # Create a pipe for cancel signaling - this allows us to interrupt select()
        self._cancel_read, self._cancel_write = os.pipe()
    
    def cancel(self):
        """Cancel the benchmark and close socket to interrupt recv()."""
        if self._cancelled:
            return  # Already cancelled
        
        self._cancelled = True
        
        # Write to cancel pipe FIRST to wake up select() immediately, THEN close socket
        try:
            os.write(self._cancel_write, b"x")
        except Exception:
            pass
        
        if hasattr(self, '_sock') and self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
    
    def _apply_chat_template_to_prompt(self, prompt: str, model_path: str = None) -> str:
        """Apply chat template to prompt based on model family."""
        if not model_path:
            return prompt
        
        try:
            from chat_templates import detect_model_family, apply_chat_template
            model_family = detect_model_family(model_path)
            return apply_chat_template(prompt, model_family)
        except Exception as e:
            self.output_signal.emit(f"[DEBUG] Chat template warning: {e}")
            return prompt
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            text_parts = []
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            doc.close()
            return '\n\n'.join(text_parts)
        except ImportError:
            self.output_signal.emit("⚠️ PyMuPDF (fitz) nicht installiert")
            return ""
        except Exception as e:
            self.output_signal.emit(f"⚠️ PyMuPDF failed: {e}")
            return ""
    
    def _clean_text_for_display(self, text):
        """Clean text for live display - remove thinking blocks and normalize whitespace."""
        # Remove complete think blocks
        cleaned = re.sub(r'</think>.*?</think>', ' ', text, flags=re.DOTALL)
        
        # Also remove orphaned opening/closing tags
        cleaned = re.sub(r'<think>.*?</think>', ' ', cleaned, flags=re.DOTALL)
        
        # Remove orphaned tags
        cleaned = re.sub(r'<\w+>', ' ', cleaned)
        cleaned = re.sub(r'\x1b\[[0-9;]*m', '', cleaned)  # ANSI escape codes
        
        # Normalize whitespace - collapse multiple spaces/newlines to single space
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

    def run(self):
        """Run the benchmark request."""
        import urllib.request, urllib.error
        
        # Build request
        url = f"http://{self.SERVER_HOST}:{self.SERVER_PORT}{self.SERVER_PATH}"
        
        data = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
        }
        
        try:
            if self.streaming:
                self._run_streaming(url, data)
            else:
                self._run_standard(url, data)
        except Exception as e:
            self.output_signal.emit(f"⚠️ Benchmark error: {e}")
            raise
    
    def _run_standard(self, url: str, data: dict):
        """Run standard (non-streaming) benchmark."""
        import urllib.request, urllib.error, json
        
        try:
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
            
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
            latency = time.time() - start_time
            
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('text', '')
                token_count = len(text.split())
                tps = token_count / latency if latency > 0 else 0
                
                self.output_signal.emit(f"✓ Response: {len(text)} chars")
                self.finished_signal.emit(tps, token_count)
            else:
                self.output_signal.emit("⚠️ No choices in response")
                self.finished_signal.emit(0, 0)
                
        except urllib.error.URLError as e:
            self.output_signal.emit(f"⚠️ Network error: {e}")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            self.output_signal.emit(f"⚠️ Error: {e}")
            self.finished_signal.emit(0, 0)
    
