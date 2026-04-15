#!/usr/bin/env python3
"""HTTP-based benchmark thread for llauncher - runs asynchronously with streaming support."""

import json, os, select, socket, threading, time, re, traceback
from pathlib import Path


def safe_add_int(a, b):
    """Safely add two values, treating None as 0 and converting strings to int."""
    a = int(a) if a is not None else 0
    b = int(b) if b is not None else 0
    return a + b


from PyQt6.QtCore import QThread, pyqtSignal


class HTTPBenchmarkRunner(QThread):
    """Runs a single HTTP benchmark request without blocking the UI."""
    
    output_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(float, int)
    token_update_signal = pyqtSignal(int)  # Emit current token count
    server_log_signal = pyqtSignal(str)  # Server console log lines for parsing timing metrics
    
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
            "preload_time": None,
            "inference_time": None,
            "generation_time": None,
            "prefill_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "prompt_eval_time": None,
            "eval_time": None,
        }
        
        # Server log metrics (from llama.cpp console output)
        self._server_log_metrics = {}
    
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
    
    def _parse_server_log_for_metrics(self, log_line: str):
        """Parse llama.cpp server console log for timing metrics.
        
        Format from llama.cpp:
        eval time =   14916.36 ms /  2115 tokens (    7.05 ms per token,   141.79 tokens per second)
        prompt eval time =    1834.08 ms /  9945 tokens (    0.18 ms per token,  5422.33 tokens per second)
        total time =   16750.44 ms / 12060 tokens
        """
        import re
        
        # Parse eval time (generation)
        match = re.search(r'eval\s+time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', log_line, re.IGNORECASE)
        if match:
            self._server_log_metrics['eval_time_ms'] = float(match.group(1))
            self._server_log_metrics['gen_tokens'] = int(match.group(2))
        
        # Parse prompt eval time (prefill)
        match = re.search(r'prompt\s+eval\s+time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', log_line, re.IGNORECASE)
        if match:
            self._server_log_metrics['prompt_eval_time_ms'] = float(match.group(1))
            self._server_log_metrics['prefill_tokens'] = int(match.group(2))
        
        # Parse total time
        match = re.search(r'total\s+time\s*=\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', log_line, re.IGNORECASE)
        if match:
            self._server_log_metrics['total_time_ms'] = float(match.group(1))
            self._server_log_metrics['total_tokens'] = int(match.group(2))
        
        # Update _metrics from server logs if available
        if self._server_log_metrics.get('eval_time_ms'):
            self._metrics['eval_time'] = self._server_log_metrics['eval_time_ms'] / 1000
        
        if self._server_log_metrics.get('prompt_eval_time_ms'):
            self._metrics['prompt_eval_time'] = self._server_log_metrics['prompt_eval_time_ms'] / 1000
        
        if self._server_log_metrics.get('prefill_tokens'):
            self._metrics['prefill_tokens'] = self._server_log_metrics['prefill_tokens']
        
        if self._server_log_metrics.get('total_time_ms'):
            self._metrics['total_time'] = self._server_log_metrics['total_time_ms'] / 1000

        # Update completion_tokens and total_tokens from server logs
        if self._server_log_metrics.get('gen_tokens'):
            self._metrics["completion_tokens"] = self._server_log_metrics['gen_tokens']

        if self._server_log_metrics.get('total_tokens'):
            self._metrics["total_tokens"] = self._server_log_metrics['total_tokens']
    
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
        
        # Reset metrics for this benchmark run
        self._metrics = {
            "preload_time": None,
            "inference_time": None,
            "generation_time": None,
            "prefill_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "prompt_eval_time": None,
            "eval_time": None,
        }
        self._server_log_metrics = {}
        
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
            
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data_json, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=300) as response:
                start_response = time.time()
                result = json.loads(response.read().decode('utf-8'))
            
            total_time = time.time() - start_time
            
            # Calculate actual HTTP request timing (includes network latency)
            http_request_time = time.time() - start_response
            
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('text', '')
                cleaned_text = self._clean_text_for_display(text)
                
                # Server-reported timing from llama.cpp's usage field
                usage = result.get('usage', {})
                
                # Try to get timing from JSON response first (llama.cpp may or may not send it)
                prompt_eval_ms = usage.get('prompt_eval_time', 0)
                eval_ms = usage.get('eval_time', 0)
                
                # Determine token count source for logging and metrics
                # First calculate completion_tokens from usage data
                if usage.get('completion_tokens'):
                    completion_tokens = int(usage['completion_tokens'])  # Ensure numeric type
                else:
                    completion_tokens = len(text) // 4 if text else 0

                # Extract total tokens from usage or calculate it
                total_tokens = safe_add_int(usage.get('completion_tokens'), usage.get('prompt_tokens')) if usage else len(text) // 4 if text else 0
                
                # Ensure all timing values are numeric
                prompt_eval_ms = float(prompt_eval_ms or 0)
                eval_ms = float(eval_ms or 0)
                prefill_tokens = int(usage.get('prompt_tokens', 0))
                
                # Calculate generation TPS (completion tokens / eval time)
                # If eval_time not in JSON, server logs will be parsed later and tps updated there
                if eval_ms > 0:
                    tps = completion_tokens / (eval_ms / 1000)
                else:
                    # eval_time missing from JSON - can't calculate accurate generation TPS
                    # The server log parser in on_benchmark_finished will have the real value
                    # For now, emit what we can calculate from available data
                    tps = 0
                
                # Log values for debugging TPS calculation
                self.output_signal.emit(f"DEBUG: Standard benchmark - completion_tokens={completion_tokens}, eval_ms={eval_ms}, tps={tps:.2f}\n")
                
                # Save to _metrics so UI can access it (FIX: was missing!)
                self._metrics["completion_tokens"] = completion_tokens
                self._metrics["total_tokens"] = total_tokens
                self._metrics["eval_time"] = eval_ms / 1000 if eval_ms > 0 else None  # Store as seconds or None
                self._metrics["prompt_eval_time"] = prompt_eval_ms / 1000 if prompt_eval_ms > 0 else None
                
                # Ensure total_tokens is properly converted to int for display
                total_tokens_int = int(total_tokens) if total_tokens else 0
                
                # Emit server log metrics in llama.cpp format ONLY if we have real eval_time from server
                # This allows the UI parser to extract accurate TPS from server logs (when available)
                # If eval_ms is 0, we skip server log emission and let JSON metrics be used instead
                if eval_ms > 0:
                    self.output_signal.emit("\n[SERVER LOG METRICS]\n")
                    
                    if prompt_eval_ms and prefill_tokens:
                        pe_tps = (prefill_tokens / (prompt_eval_ms / 1000)) if prompt_eval_ms > 0 else 0
                        pe_per_token = (prompt_eval_ms / prefill_tokens) if prefill_tokens > 0 else 0
                        self.output_signal.emit(f"prompt eval time = {prompt_eval_ms:.2f} ms / {prefill_tokens} tokens ({pe_per_token:.2f} ms per token, {pe_tps:.2f} tokens per second)\n")
                    
                    gen_per_token = (eval_ms / completion_tokens) if completion_tokens > 0 else 0
                    self.output_signal.emit(f"eval time = {eval_ms:.2f} ms / {completion_tokens} tokens ({gen_per_token:.2f} ms per token, {tps:.2f} tokens per second)\n")
                    
                    if total_time and total_tokens_int:
                        total_ms = http_request_time * 1000  # Use HTTP timing for total
                        total_per_token = (total_ms / total_tokens_int) if total_tokens_int > 0 else 0
                        self.output_signal.emit(f"total time = {total_ms:.2f} ms / {total_tokens_int} tokens\n")
                
                # Emit detailed metrics to UI - format like llama.cpp output
                self.output_signal.emit(f"\n[DETAILED BENCHMARK METRICS]\n")

                if eval_ms and completion_tokens:
                    gen_per_token = (eval_ms / completion_tokens) if completion_tokens > 0 else 0
                    tps = completion_tokens / (eval_ms / 1000) if eval_ms > 0 else 0
                    self.output_signal.emit(f"✓ Generation time:    {eval_ms/1000:.3f}s / {completion_tokens} tokens ({gen_per_token:.2f} ms/token, {tps:.2f} TPS)")
                elif completion_tokens:
                    # eval_ms missing from JSON - use HTTP timing as fallback
                    gen_t = http_request_time * 1000
                    gen_per_token = (gen_t / completion_tokens) if completion_tokens > 0 else 0
                    tps = completion_tokens / gen_t * 1000 if gen_t > 0 else 0
                    self.output_signal.emit(f"✓ Generation time:    {gen_t/1000:.3f}s / {completion_tokens} tokens ({gen_per_token:.2f} ms/token, {tps:.2f} TPS) [estimated]")
                
                if total_tokens_int:
                    avg_time = (http_request_time * 1000) / total_tokens_int if total_tokens_int > 0 else 0
                    tps_total = total_tokens_int / http_request_time if http_request_time > 0 else 0
                    self.output_signal.emit(f"✓ Total time:         {http_request_time:.3f}s / {total_tokens_int} tokens ({avg_time:.2f} ms/token, {tps_total:.2f} TPS)\n")
                
                self.output_signal.emit(cleaned_text)
                # Ensure numeric types before emitting signal
                self.finished_signal.emit(float(tps), int(completion_tokens))
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
           # Also check for server-side usage info in SSE stream (llama.cpp may send it)
            try:
                json_data = json.loads(data_line if 'data_line' in locals() else '{}')
                if 'choices' in json_data and len(json_data['choices']) > 0:
                    choice = json_data['choices'][0]
                    usage = choice.get('usage', {})
                    
                    # Update metrics with server-provided data (ensure numeric types)
                    if usage.get('completion_tokens'):
                        token_count = int(usage['completion_tokens'])
                    
                    if usage.get('prompt_tokens'):
                        self._metrics["prefill_tokens"] = int(usage['prompt_tokens'])
                    
                    if usage.get('prompt_eval_time'):
                        self._metrics["prompt_eval_time"] = float(usage['prompt_eval_time']) / 1000
                    
                    if usage.get('eval_time'):
                        self._metrics["eval_time"] = float(usage['eval_time']) / 1000
                    
                    # Recalculate TPS with server-provided tokens (ensure numeric types)
                    generation_time = self._metrics["generation_time"] or inference_end
                    tps = float(token_count) / float(generation_time) if generation_time > 0 else 0
                    
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
            # Ensure numeric types before emitting signal
            self.finished_signal.emit(float(tps), int(token_count))
            
        except urllib.error.URLError as e:
            self.output_signal.emit(f"Network error: {e}\n")
            self.finished_signal.emit(0, 0)
        except Exception as e:
            if not self._cancelled:
                self.output_signal.emit(f"Error: {e}\n")
            self._metrics["total_time"] = self._server_log_metrics['total_time_ms'] / 1000

        # Update completion_tokens and total_tokens from server logs
        if self._server_log_metrics.get('gen_tokens'):
            self._metrics["completion_tokens"] = self._server_log_metrics['gen_tokens']
        
        if self._server_log_metrics.get('total_tokens'):
            self._metrics["total_tokens"] = self._server_log_metrics['total_tokens']
