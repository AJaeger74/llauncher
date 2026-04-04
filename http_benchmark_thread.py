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

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfplumber or PyMuPDF."""
        # Versuche pdfplumber zuerst (einfacher)
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return '\n\n'.join(text_parts)
        except ImportError:
            self.output_signal.emit("ℹ pdfplumber nicht installiert, versuche PyMuPDF...")
        except Exception as e:
            self.output_signal.emit(f"⚠️ pdfplumber failed: {e}")
        
        # Fallback: PyMuPDF (fitz)
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

            return prompt
    
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
    
    def _is_server_debug_log(self, line: str) -> bool:
        """Check if a line is a llama.cpp server debug log (not user response)."""
        # llama.cpp server debug patterns
        server_patterns = [
            'slot ',           # slot launch_slot_, slot update_slots, slot release
            'srv ',            # srv log_server_r, srv update_slots
            'kv_cache ',       # kv_cache add/remove
            'n_ctx = ',        # context size info
            'n_probs = ',      # probability info
            'time_prompt = ',  # timing info
            't_sample = ',     # sampling timing
            't_eval = ',       # evaluation timing
            't_tokenize = ',   # tokenization timing
        ]
        return any(pattern in line for pattern in server_patterns)
    
    def _emit_with_prefix(self, text: str, prefix: str):
        """Emit text with a prefix to distinguish log types."""
        self.output_signal.emit(f"{prefix}{text}")
    
    def _safe_get_text_from_response(self, data):
        """Safely extract text from SSE response data."""
        try:
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                return None
            
            choice = choices[0]
            if not isinstance(choice, dict):
                return None
                
            return choice.get("text", "")
        except (KeyError, IndexError, AttributeError):
            return None
    
    def _stream_benchmark(self) -> tuple[int, float]:
        """Streaming mode using raw sockets for cancellable reads."""
        
        payload = json.dumps({
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "stream": True,
        }).encode("utf-8")
        
        start_time = time.time()
        sock = None
        
        try:
            # Blocking socket - we use select with multiple file descriptors for cancel support
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(1)  # Blocking mode
            
            self._sock = sock
            
            # Connect to server with timeout using select on both sockets
            connect_start = time.time()
            while True:
                if self._cancelled:
                    return 0, time.time() - start_time
                
                # Use select with cancel pipe for interruptible connection attempt
                try:
                    ready_r, ready_w, _ = select.select([self._cancel_read], [sock], [], 0.1)
                except (ValueError, OSError):
                    # select failed or socket closed, give up
                    return 0, time.time() - start_time
                
                if self._cancelled:
                    return 0, time.time() - start_time
                
                if len(ready_r) > 0:
                    # Cancel was triggered via pipe
                    try:
                        os.read(self._cancel_read, 4096)
                    except Exception:
                        pass
                    return 0, time.time() - start_time
                
                try:
                    sock.connect((self.SERVER_HOST, self.SERVER_PORT))
                    break
                except BlockingIOError:
                    # Connection in progress
                    continue
            
            request_line = f"POST {self.SERVER_PATH} HTTP/1.1\r\n"
            request_line += f"Host: {self.SERVER_HOST}:{self.SERVER_PORT}\r\n"
            request_line += "Content-Type: application/json\r\n"
            request_line += f"Content-Length: {len(payload)}\r\n"
            request_line += "Connection: close\r\n"
            request_line += "\r\n"
            
            sock.sendall(request_line.encode() + payload)
            
            # Read response using select with cancel pipe for cancellable reads
            full_answer_parts = []
            buffer = b""
            last_output_time = start_time
            
            # Buffer für Live-Output (sammelt mehrere Chunks)
            live_buffer = ""
            LIVE_FLUSH_INTERVAL = 1.0  # Sekunden zwischen Flushes
            
            # Read headers first - wait for \r\n\r\n separator
            header_buffer = b""
            while b"\r\n\r\n" not in header_buffer:
                if self._cancelled:
                    return 0, time.time() - start_time
                
                try:
                    ready_r, ready_w, _ = select.select([self._cancel_read], [sock], [], 0.1)
                except (ValueError, OSError):
                    # select failed or socket closed, give up
                    self.output_signal.emit("Streaming error: Header read timeout")
                    return 0, time.time() - start_time
                
                if self._cancelled:
                    return 0, time.time() - start_time
                
                if len(ready_r) > 0:
                    # Cancel was triggered via pipe
                    try:
                        os.read(self._cancel_read, 4096)
                    except Exception:
                        pass
                    return 0, time.time() - start_time
                
                if not ready_w and time.time() - start_time > 5.0:
                    self.output_signal.emit("Streaming error: Header read timeout")
                    return 0, time.time() - start_time
                
                try:
                    chunk = sock.recv(1)
                    if not chunk:
                        break
                    header_buffer += chunk
                except socket.timeout:
                    continue
            
            # Parse headers to find transfer-encoding or content-length
            headers_raw = header_buffer.decode("utf-8", errors="ignore").lower()
            
            # Main read loop - simplified without nested try blocks
            next_flush_time = time.time() + LIVE_FLUSH_INTERVAL
            
            while True:
                if self._cancelled:
                    return 0, time.time() - start_time
                
                try:
                    ready_r, ready_w, _ = select.select([self._cancel_read], [sock], [], 0.1)
                except (ValueError, OSError):
                    # select failed or socket closed, give up
                    if self._cancelled:
                        return 0, time.time() - start_time
                    break
                
                if self._cancelled:
                    return 0, time.time() - start_time
                
                # Check for flush timeout AFTER select (runs every iteration)
                current_time = time.time()
                if current_time >= next_flush_time and live_buffer.strip():
                    paragraphs = live_buffer.split("\n\n")
                    for para in paragraphs:
                        cleaned_para = self._clean_text_for_display(para.strip())
                        if cleaned_para:
                            # Prefix server logs for visual distinction
                            if self._is_server_debug_log(cleaned_para):
                                self._emit_with_prefix(cleaned_para, "🔧 ")
                            else:
                                self.output_signal.emit(cleaned_para + "\n")
                    next_flush_time = current_time + LIVE_FLUSH_INTERVAL
                
                if len(ready_r) > 0:
                    # Cancel was triggered via pipe
                    try:
                        os.read(self._cancel_read, 4096)
                    except Exception:
                        pass
                    
                    return 0, time.time() - start_time
                
                if not ready_w:
                    # No data - check for silence timeout or cancel
                    if self._cancelled:
                        return 0, time.time() - start_time
                    if time.time() - last_output_time > 2.0:
                        # Final flush before exit due to silence timeout
                        if live_buffer.strip():
                            paragraphs = live_buffer.split("\n\n")
                            for para in paragraphs:
                                cleaned_para = self._clean_text_for_display(para.strip())
                                if cleaned_para:
                                    # Prefix server logs for visual distinction
                                    if self._is_server_debug_log(cleaned_para):
                                        self._emit_with_prefix(cleaned_para, "🔧 ")
                                    else:
                                        self.output_signal.emit(cleaned_para + "\n")
                        break
                    
                    continue
                
                chunk = sock.recv(4096)
                
                if not chunk:
                    # Final flush before exit
                    if live_buffer.strip():
                        paragraphs = live_buffer.split("\n\n")
                        for para in paragraphs:
                            cleaned_para = self._clean_text_for_display(para.strip())
                            if cleaned_para:
                                self.output_signal.emit(cleaned_para + "\n")
                    break
                
                last_output_time = time.time()
                buffer += chunk
                
                sse_lines = buffer.split(b"\n")
                
                for line in sse_lines[:-1]:
                    try:
                        line_bytes = line.strip()
                        if not line_bytes or not line_bytes.startswith(b"data: "):
                            continue
                        
                        line_str = line_bytes[6:].decode("utf-8")
                        data = json.loads(line_str)
                        
                        # Use safe helper function to extract text
                        text = self._safe_get_text_from_response(data)
                        if text is None or text == "":
                            continue
                        
                        # Accumulate for final count
                        full_answer_parts.append(text)
                        
                        # Also add to live buffer (preserving original newlines within chunks)
                        live_buffer += text
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                
                buffer = sse_lines[-1] if sse_lines else b""
        
        except Exception as e:
            if self._cancelled:
                self.output_signal.emit("Benchmark cancelled by user.")
                return 0, time.time() - start_time
            self.output_signal.emit(f"Streaming error: {e}")
            import traceback
            self.output_signal.emit(traceback.format_exc())
            return 0, time.time() - start_time
        
        finally:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass
            
            # Clean up pipe file descriptors
            try:
                os.close(self._cancel_read)
                os.close(self._cancel_write)
            except Exception:
                pass
        
        full_answer = "".join(full_answer_parts)
        
        # Clean the complete answer and remove thinking blocks for final display
        full_answer_clean = self._clean_text_for_display(full_answer)
        
        if not full_answer_clean:
            # If cleaned answer is empty, count raw tokens as fallback
            token_count = max(len(full_answer.split()), 1)
        else:
            token_count = len(full_answer_clean.split())
        
        # Final summary output (don't re-emit live content, just show stats)
        self.output_signal.emit(f"\n--- Benchmark Complete ({len(full_answer)} chars, {token_count} tokens) ---\n")
        
        return int(token_count), time.time() - start_time
    
    def run(self):
        """Main entry point for QThread. Must NOT return values."""
        
        # Initialize common variables
        usage = {}
        resp_json = {}
        latency = 0.0
        
        if self.streaming:
            tokens, latency = self._stream_benchmark()
            tps = int(tokens) / max(latency, 1e-3)
            
            # Finalize streaming benchmark - prepare common variables
            usage = {"prompt_tokens": 0, "completion_tokens": int(tokens)}
            resp_json = {}
        else:
            # Use raw socket for cancellable non-streaming benchmark
            sock = None
            start_time = time.time()
            
            try:
                import socket
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30.0)  # Overall timeout
                self._sock = sock
                
                # Connect to server
                sock.connect((self.SERVER_HOST, self.SERVER_PORT))
                
                # Build and send HTTP request manually (same as urllib would do)
                payload = json.dumps({
                    "prompt": self.prompt,
                    "max_tokens": self.max_tokens,
                }).encode("utf-8")
                
                request_line = f"POST {self.SERVER_PATH} HTTP/1.1\r\n"
                request_line += f"Host: {self.SERVER_HOST}:{self.SERVER_PORT}\r\n"
                request_line += "Content-Type: application/json\r\n"
                request_line += f"Content-Length: {len(payload)}\r\n"
                request_line += "Connection: close\r\n"
                request_line += "\r\n"
                
                sock.sendall(request_line.encode() + payload)
                
                # Read response until EOF or timeout (with cancellation check)
                body = b""
                while True:
                    if self._cancelled:
                        self.output_signal.emit("Benchmark cancelled by user.")
                        break
                    
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        body += chunk
                        # Debug: show received bytes for troubleshooting
                        self.output_signal.emit(f"[DEBUG] Received {len(chunk)} bytes, total={len(body)}")
                    except socket.timeout:
                        # Timeout reached
                        self.output_signal.emit(f"[DEBUG] Socket timeout after receiving {len(body)} bytes")
                        break
                
                # Parse JSON response - skip HTTP headers if present
                resp_json = {}
                try:
                    body_str = body.decode("utf-8", errors="ignore")
                    
                    # Check for HTTP headers and extract body after \r\n\r\n
                    if "HTTP/" in body_str:
                        header_end = body_str.find("\r\n\r\n")
                        if header_end != -1:
                            json_body = body_str[header_end + 4:]
                            self.output_signal.emit(f"[DEBUG] Stripped headers, JSON starts after {header_end} chars")
                        else:
                            self.output_signal.emit("[DEBUG] No \\r\\n\\r\\n separator found in HTTP response")
                            json_body = ""
                    elif body_str.startswith("{"):
                        json_body = body_str
                        self.output_signal.emit("[DEBUG] Direct JSON, no headers to strip")
                    else:
                        json_body = ""
                        self.output_signal.emit("[DEBUG] Unexpected format, skipping parse")
                    
                    if json_body.strip():
                        resp_json = json.loads(json_body)
                        self.output_signal.emit("[DEBUG] Parsed JSON successfully")
                except json.JSONDecodeError as e:
                    self.output_signal.emit(f"[DEBUG] JSON decode error after header stripping: {e}")
                    pass
                
                # Extract usage info
                latency = time.time() - start_time
                usage = resp_json.get("usage", {})
            
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
        
        # Common code for both modes - calculate TPS and signal completion
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tokens = prompt_tokens + completion_tokens
        
        if tokens == 0:
            choices = resp_json.get("choices")
            answer_text_raw = ""
            if isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if isinstance(choice, dict):
                    answer_text_raw = choice.get("text", "")
            tokens = max(len(answer_text_raw.split()), 1)
        
        tps = int(tokens) / max(latency, 1e-3)
        
        # Debug: show raw JSON response for troubleshooting
        self.output_signal.emit(f"\n[DEBUG] resp_json keys: {list(resp_json.keys())}")
        if "usage" in resp_json:
            self.output_signal.emit(f"[DEBUG] usage: {resp_json['usage']}")
        
        self.output_signal.emit(f"Prompt:\n{self.prompt}")
        
        # Safely extract answer text for display
        choices = resp_json.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                answer_text_raw = choice.get("text", "")
                # Clean the answer - remove thinking blocks before display
                answer_text = self._clean_text_for_display(answer_text_raw)
        else:
            answer_text = ""
        
        self.output_signal.emit(f"\nAnswer:\n{answer_text}")
        self.output_signal.emit(f"\nTPS: {tps:.2f} (tokens={tokens} latency={latency:.3f}s)")
        
        # Signal completion to UI
        self.finished_signal.emit(tps, tokens)
