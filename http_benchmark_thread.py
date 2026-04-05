#!/usr/bin/env python3
"""HTTP-based benchmark thread for llauncher - runs asynchronously with streaming support."""

import json, os, select, socket, threading, time, re, traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class HTTPBenchmarkRunner(QThread):
    """Runs a single HTTP benchmark request without blocking the UI."""
    
    output_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(float, int)
    
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 8080
    SERVER_PATH = "/v1/completions"
    
    def __init__(self, max_tokens: int = 64, server_pid: int = None, streaming: bool = False, model_path: str = None):
        super().__init__()
        
        # Load config but DON'T emit signals yet (thread not started)
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
        
        self.benchmark_cfg = benchmark_cfg
        self.raw_prompt = benchmark_cfg.get("prompt", "")
        self.server_pid = server_pid
        self.streaming = streaming
        self.model_path = model_path
        self._cancelled = False
        self._cancel_read, self._cancel_write = os.pipe()
        self._stream_buffer = ""
        self._context_content = ""  # Will be loaded in run()
    
    def cancel(self):
        if self._cancelled:
            return
        self._cancelled = True
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
        if not model_path:
            return prompt
        try:
            from chat_templates import detect_model_family, apply_chat_template
            model_family = detect_model_family(model_path)
            self.status_signal.emit(f"DEBUG: Using template for family: {model_family}")
            return apply_chat_template(prompt, model_family)
        except Exception as e:
            self.output_signal.emit(f"Chat template warning: {e}")
            return prompt
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        try:
            import fitz
            text_parts = []
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            doc.close()
            return '\n\n'.join(text_parts)
        except ImportError:
            self.output_signal.emit("PyMuPDF not installed")
            return ""
        except Exception as e:
            self.output_signal.emit(f"PyMuPDF failed: {e}")
            return ""
    
    def _load_benchmark_file(self):
        """Load benchmark file and return context content."""
        benchmark_file_path = self.benchmark_cfg.get("benchmark_file_path", "")
        self.output_signal.emit(f"DEBUG: benchmark_file_path='{benchmark_file_path}'")
        
        if not benchmark_file_path:
            self.output_signal.emit("WARNING: No benchmark_file_path in config!")
            return ""
        
        try:
            self.output_signal.emit(f"Loading file: {benchmark_file_path}")
            file_ext = Path(benchmark_file_path).suffix.lower()
            self.output_signal.emit(f"DEBUG: file_ext={file_ext}")
            
            if file_ext == '.pdf':
                context = self._extract_pdf_text(benchmark_file_path)
                self.output_signal.emit(f"DEBUG: PDF extraction result: {len(context)} chars")
                if not context:
                    raise Exception("PDF extraction failed")
            else:
                with open(benchmark_file_path, 'r', encoding='utf-8') as f:
                    context = f.read()
                self.output_signal.emit(f"DEBUG: TXT read result: {len(context)} chars")
            
            self.output_signal.emit(f"File loaded: {benchmark_file_path} ({len(context)} chars)")
            return context
            
        except Exception as e:
            self.output_signal.emit(f"Error loading benchmark file: {e}")
            self.output_signal.emit(f"DEBUG: traceback:\n{traceback.format_exc()}")
            return ""
    
    def _clean_text_for_display(self, text):
        # Remove <think> blocks
        cleaned = re.sub(r'<think>.*?</think>', ' ', text, flags=re.DOTALL)
        cleaned = re.sub(r'</think>', ' ', cleaned, flags=re.DOTALL)
        # Remove generic XML tags
        cleaned = re.sub(r'<\w+>', ' ', cleaned)
        # Remove ANSI escape codes (colors)
        cleaned = re.sub(r'\x1b\[[0-9;]*m', ' ', cleaned)
        
       # Remove llama.cpp server internal logs that leak into the text stream
        # We replace with a space ' ' instead of '' to prevent words from merging
        log_patterns = [
            r'\[TPS: [0-9.]+\]',
            r'slot\s+print_timing: .*',
            r'prompt\s+eval\s+time\s+=\s+.*',
            r'eval\s+time\s+=\s+.*',
            r'total\s+time\s+=\s+.*',
            r'slot\s+release: .*',
            r'srv\s+update_slots: .*'
        ]
        for pattern in log_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.MULTILINE)
            
        return cleaned
        return cleaned

    def run(self):
        import urllib.request, urllib.error
        
        self.output_signal.emit("DEBUG: run() started!")
        
        # Load benchmark file in run() where signals work
        self._context_content = self._load_benchmark_file()
        
        # Combine prompt with context
        if self._context_content:
            self.raw_prompt = f"{self._context_content}\n\n{self.raw_prompt}"
            self.output_signal.emit(f"DEBUG: Combined prompt length: {len(self.raw_prompt)} chars")
        
        # Build final prompt
        self.prompt = self._apply_chat_template_to_prompt(self.raw_prompt, self.model_path)
        self.output_signal.emit(f"DEBUG: Final prompt length: {len(self.prompt)} chars")
        
        url = f"http://{self.SERVER_HOST}:{self.SERVER_PORT}{self.SERVER_PATH}"
        
        data = {
            "prompt": self.prompt,
            "max_tokens": self.benchmark_cfg.get("max_tokens", 4096),
            "stream": self.streaming,
        }
        
        try:
            if self.streaming:
                self._run_streaming(url, data)
            else:
                self._run_standard(url, data)
        except Exception as e:
            self.output_signal.emit(f"Benchmark error: {e}")
            raise
    
    def _run_standard(self, url: str, data: dict):
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
                # We use a simple word count as fallback for standard mode
                token_count = len(text.split())
                tps = token_count / latency if latency > 0 else 0
                
                self.output_signal.emit(f"Response: {len(text)} chars")
                self.finished_signal.emit(tps, token_count)
            else:
                self.output_signal.emit("No choices in response")
                self.finished_signal.emit(0, 0)
                
        except urllib.error.URLError as e:
            self.output_signal.emit(f"Network error: {e}")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            self.output_signal.emit(f"Error: {e}")
            self.finished_signal.emit(0, 0)
    
    def _run_streaming(self, url: str, data: dict):
        import urllib.request, json
        
        try:
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
            
            start_time = time.time()
            token_count = 0
            last_token_time = time.time()
            
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if self._cancelled:
                        break
                    
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data:'):
                        continue
                    
                    data_line = line[5:].strip()
                    if data_line == '[DONE]':
                        break
                    
                    try:
                        json_data = json.loads(data_line)
                        text = json_data.get('choices', [{}])[0].get('text', '')
                        
                        if text:
                            token_count += 1
                            cleaned = self._clean_text_for_display(text)
                            self._stream_buffer += cleaned
                            
                            now = time.time()
                            if now - last_token_time >= 0.5:
                                elapsed = now - start_time
                                if elapsed > 0:
                                    tps = token_count / elapsed
                                    if self._stream_buffer:
                                        # Strip only on emit, not before append
                                        self.output_signal.emit(self._stream_buffer.strip())
                                        self._stream_buffer = ""
                                    self.status_signal.emit(f"[TPS: {tps:.2f}]")
                                last_token_time = now
                    except json.JSONDecodeError:
                        continue
            
           # Strip only trailing whitespace from the accumulated buffer
            if self._stream_buffer:
                self.output_signal.emit(self._stream_buffer.strip())
                self._stream_buffer = ""
            
            latency = time.time() - start_time
            tps = token_count / latency if latency > 0 else 0
            self.status_signal.emit(f"Completed: {token_count} tokens, TPS: {tps:.2f}")
            self.finished_signal.emit(tps, token_count)
            
        except urllib.error.URLError as e:
            self.output_signal.emit(f"Network error: {e}")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            if not self._cancelled:
                self.output_signal.emit(f"Error: {e}")
            self.finished_signal.emit(0, 0)
