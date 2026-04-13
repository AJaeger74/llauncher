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
    token_update_signal = pyqtSignal(int)  # Emit current token count
    
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
        
        # Timing metrics for detailed reporting
        self._metrics = {
            "preload_time": None,      # Time until first token (prefill)
            "inference_time": None,    # Total inference time
            "generation_time": None,   # Time from first to last token
            "prefill_tokens": None,    # Input/context tokens (if available)
            "completion_tokens": None, # Generated tokens
            "total_tokens": None,      # Prefill + completion
            "prompt_eval_time": None,  # Server-reported prompt eval time
            "eval_time": None,         # Server-reported evaluation time
        }
    
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
            self.output_signal.emit(f"DEBUG: Using template for family: {model_family}\n")
            return apply_chat_template(prompt, model_family)
        except Exception as e:
            self.output_signal.emit(f"Chat template warning: {e}\n")
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
            self.output_signal.emit("PyMuPDF not installed\n")
            return ""
        except Exception as e:
            self.output_signal.emit(f"PyMuPDF failed: {e}\n")
            return ""
    
    def _load_benchmark_file(self):
        """Load benchmark file and return context content."""
        benchmark_file_path = self.benchmark_cfg.get("benchmark_file_path", "")
        self.output_signal.emit(f"DEBUG: benchmark_file_path='{benchmark_file_path}'\n")
        
        if not benchmark_file_path:
            self.output_signal.emit("WARNING: No benchmark_file_path in config!\n")
            return ""
        
        try:
            self.output_signal.emit(f"Loading file: {benchmark_file_path}")
            file_ext = Path(benchmark_file_path).suffix.lower()
            self.output_signal.emit(f"DEBUG: file_ext={file_ext}")
            
            if file_ext == '.pdf':
                context = self._extract_pdf_text(benchmark_file_path)
                self.output_signal.emit(f"DEBUG: PDF extraction result: {len(context)} chars\n")
                if not context:
                    raise Exception("PDF extraction failed")
            else:
                with open(benchmark_file_path, 'r', encoding='utf-8') as f:
                    context = f.read()
                self.output_signal.emit(f"DEBUG: TXT read result: {len(context)} chars\n")
            
            self.output_signal.emit(f"File loaded: {benchmark_file_path} ({len(context)} chars)\n")
            return context
            
        except Exception as e:
            self.output_signal.emit(f"Error loading benchmark file: {e}\n")
            self.output_signal.emit(f"DEBUG: traceback:\n{traceback.format_exc()}\n")
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
        
        self.output_signal.emit(f"DEBUG: run() started!\n")
        
        # Load benchmark file in run() where signals work
        self._context_content = self._load_benchmark_file()
        
        # Combine prompt with context
        if self._context_content:
            self.raw_prompt = f"{self._context_content}\n\n{self.raw_prompt}"
            self.output_signal.emit(f"DEBUG: Combined prompt length: {len(self.raw_prompt)} chars\n")
        
        # Build final prompt
        self.prompt = self._apply_chat_template_to_prompt(self.raw_prompt, self.model_path)
        self.output_signal.emit(f"DEBUG: Final prompt length: {len(self.prompt)} chars\n")
        
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
            self.output_signal.emit(f"Benchmark error: {e}\n")
            raise
    
    def _run_standard(self, url: str, data: dict):
        import urllib.request, urllib.error, json
        
        try:
            start_time = time.time()
            request_start = start_time
            
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=300) as response:
                # First token arrival (preload/prefill time)
                preload_start = time.time() - request_start
                
                result = json.loads(response.read().decode('utf-8'))
            
            inference_end = time.time()
            total_inference_time = inference_end - request_start
            
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('text', '')
                cleaned_text = self._clean_text_for_display(text)
                
                # Use server-provided token counts when available (from llama.cpp server)
                usage = result.get('usage', {})
                prompt_eval_tokens = usage.get('prompt_tokens', 0)  # Input/context tokens
                completion_tokens = usage.get('completion_tokens', len(text) // 4 if text else 0)  # Generated tokens
                total_tokens = prompt_eval_tokens + completion_tokens
                
                # Server-reported timing (llama.cpp provides this in usage, already in ms)
                prompt_eval_time_ms = usage.get('prompt_eval_time')  # Prefill time
                eval_time_ms = usage.get('eval_time')  # Generation time
                
                # Calculate server-side times (convert ms to seconds)
                server_prefill_time = prompt_eval_time_ms / 1000 if prompt_eval_time_ms else None
                server_gen_time = eval_time_ms / 1000 if eval_time_ms else None
                
                self._metrics.update({
                    # Use server-reported prefill time (time to process prompt before first token)
                    "preload_time": server_prefill_time,
                    # Total generation time from server
                    "generation_time": server_gen_time,
                    # HTTP request latency as fallback for total time
                    "inference_time": total_inference_time if not server_gen_time else (server_prefill_time + server_gen_time),
                    "prefill_tokens": prompt_eval_tokens if prompt_eval_tokens else completion_tokens // 4,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_eval_time": server_prefill_time,
                    "eval_time": server_gen_time,
                })
                
                # Calculate TPS based on generated tokens only
                generation_time = total_inference_time if not self._metrics["generation_time"] else self._metrics["generation_time"]
                tps = completion_tokens / generation_time if generation_time > 0 and completion_tokens > 0 else 0
                
                # Emit detailed metrics to UI
                self.output_signal.emit(f"\n[DETAILED BENCHMARK METRICS]\n")
                self.output_signal.emit(f"✓ Preload time (time to first token): {self._metrics['preload_time']:.3f}s\n")
                if self._metrics["prompt_eval_time"]:
                    self.output_signal.emit(f"  → Server prompt eval: {self._metrics['prompt_eval_time']:.3f}s ({prompt_eval_tokens} tokens)\n")
                else:
                    self.output_signal.emit(f"  → Server prompt eval: not reported, using heuristic ({prompt_eval_tokens if prompt_eval_tokens else 'N/A'} tokens)\n")
                self.output_signal.emit(f"✓ Inference time (total): {self._metrics['inference_time']:.3f}s\n")
                if self._metrics["eval_time"]:
                    self.output_signal.emit(f"  → Server generation: {self._metrics['eval_time']:.3f}s ({completion_tokens} tokens)\n")
                else:
                    self.output_signal.emit(f"  → Server generation: not reported\n")
                self.output_signal.emit(f"✓ Generated tokens: {completion_tokens}\n")
                self.output_signal.emit(f"  → Context/prefill: {self._metrics['prefill_tokens']}\n")
                self.output_signal.emit(f"  → Total tokens: {total_tokens}\n")
                self.output_signal.emit(f"✓ TPS (generated): {tps:.2f}\n")
                
                self.output_signal.emit(cleaned_text)
                self.finished_signal.emit(tps, completion_tokens)
            else:
                self.output_signal.emit("No choices in response\n")
                self.finished_signal.emit(0, 0)
                
        except urllib.error.URLError as e:
            self.output_signal.emit(f"Network error: {e}\n")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            self.output_signal.emit(f"Error: {e}\n")
            self.finished_signal.emit(0, 0)
    
    def _run_streaming(self, url: str, data: dict):
        import urllib.request, json
        
        try:
            request_start = time.time()
            start_time = request_start
            token_count = 0
            last_token_time = request_start
            first_token_time = None
            
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
            
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
                            # First token arrival (preload/prefill time)
                            if first_token_time is None:
                                first_token_time = time.time() - request_start
                            
                            # Accumulate full response text to count tokens correctly at the end
                            self._stream_buffer += text
                            cleaned = self._clean_text_for_display(text)
                            
                            # Track generation time
                            current_time = time.time() - request_start
                            if self._metrics["generation_time"] is None or current_time < self._metrics["generation_time"]:
                                self._metrics["generation_time"] = current_time - first_token_time if first_token_time else current_time
                            
                            # Update token count for progress bar (4 chars ≈ 1 token)
                            estimated_tokens = len(self._stream_buffer) // 4
                            self.token_update_signal.emit(estimated_tokens)
                            
                            now = time.time()
                            if now - last_token_time >= 0.5:
                                elapsed = now - start_time
                                if elapsed > 0:
                                    # We don't calculate TPS here yet; we do it at the end with total count
                                    self.status_signal.emit(f"[Streaming...]")
                                last_token_time = now
                            
                            # Emit cleaned chunks for live viewing
                            if cleaned:
                                self.output_signal.emit(cleaned)
                    except json.JSONDecodeError:
                        continue
            
            # After streaming loop
            inference_end = time.time() - request_start
            if not self._metrics["inference_time"]:
                self._metrics["inference_time"] = inference_end
            else:
                self._metrics["inference_time"] = min(self._metrics["inference_time"], inference_end)  # Keep faster measurement
            
            if self._stream_buffer:
                # Accumulate final full text for counting
                full_text = self._stream_buffer
                self._stream_buffer = ""
            else:
                full_text = ""
            
            # Count tokens from the complete accumulated response
            # Use a heuristic (len // 4) as a fallback if no server-side usage provided
            token_count = len(full_text) // 4 if full_text else 0
            
            # Calculate TPS based on generated tokens and generation time
            generation_time = self._metrics["generation_time"] or inference_end
            tps = token_count / generation_time if generation_time > 0 else 0
            
            # Also check for server-side usage info in SSE stream (llama.cpp may send it)
            # Look for "usage" fields in the last JSON data chunk
            try:
                json_data = json.loads(data_line if 'data_line' in locals() else '{}')
                if 'choices' in json_data and len(json_data['choices']) > 0:
                    choice = json_data['choices'][0]
                    usage = choice.get('usage', {})
                    
                    # Update metrics with server-provided data
                    if usage.get('completion_tokens'):
                        token_count = usage['completion_tokens']
                    
                    if usage.get('prompt_tokens'):
                        self._metrics["prefill_tokens"] = usage['prompt_tokens']
                    
                    if usage.get('prompt_eval_time'):
                        self._metrics["prompt_eval_time"] = usage['prompt_eval_time'] / 1000  # Convert ms to s
                    
                    if usage.get('eval_time'):
                        self._metrics["eval_time"] = usage['eval_time'] / 1000  # Convert ms to s
                    
                    # Recalculate TPS with server-provided tokens
                    generation_time = self._metrics["generation_time"] or inference_end
                    tps = token_count / generation_time if generation_time > 0 else 0
                    
            except (json.JSONDecodeError, KeyError):
                pass  # Continue with local estimation
            
            # Update metrics
            self._metrics.update({
                "preload_time": first_token_time,
                "inference_time": inference_end,
                "generation_time": self._metrics["generation_time"] or inference_end,
                "completion_tokens": token_count,
                "total_tokens": self._metrics["prefill_tokens"] + token_count if self._metrics["prefill_tokens"] else token_count,
            })
            
            self.status_signal.emit(f"Completed: {token_count} tokens, TPS: {tps:.2f}")
            self.finished_signal.emit(tps, token_count)
            
        except urllib.error.URLError as e:
            self.output_signal.emit(f"Network error: {e}\n")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            if not self._cancelled:
                self.output_signal.emit(f"Error: {e}\n")
            self.finished_signal.emit(0, 0)
