#!/usr/bin/env python3
"""
Helper script to run TPS calculation debugging analysis.
Run this from the llauncher directory to diagnose benchmark TPS inconsistencies.
"""

from PyQt6.QtCore import QThread, pyqtSignal


class BenchmarkTPSDebugRunner(QThread):
    """Debugs TPS calculation by comparing token counting methods across modes."""
    
    output_signal = pyqtSignal(str)
    debug_result = pyqtSignal(dict)  # Returns analysis results
    
    def __init__(self, benchmark_data=None):
        super().__init__()
        self.benchmark_data = benchmark_data or {}
        
    def run(self):
        """
        Debug workflow:
        1. Check http_benchmark_thread.py for token counting logic in both modes
        2. Compare fallback methods between _run_standard() and _run_streaming()
        3. Identify if server-provided tokens are being used consistently
        4. Report which method is used in each mode
        
        Returns: {
            "standard_fallback": "method_used",
            "streaming_fallback": "method_used", 
            "server_tokens_available": bool,
            "consistency_issue": bool,
            "recommendation": "fix description"
        }
        """
        self.output_signal.emit("Starting TPS calculation analysis...\n")
        
        # Read http_benchmark_thread.py to examine token counting logic
        try:
            with open("http_benchmark_thread.py", 'r') as f:
                content = f.read()
            
            # Check standard mode fallback (line ~269)
            if "len(text) // 4" in content:
                self.output_signal.emit("[!] Standard mode uses word-count fallback\n")
                standard_fallback = "word_count"
            elif "completion_tokens" in content and "usage.get('completion_tokens'" in content:
                standard_fallback = "server_provided_or_word_count"
            else:
                standard_fallback = "unknown"
            
            # Check streaming mode fallback (lines ~387-412)
            if "token_count = len(full_text) // 4" in content:
                self.output_signal.emit("[!] Streaming mode uses word-count fallback\n")
                streaming_fallback = "word_count"
            elif "'completion_tokens' in json_data" in content:
                streaming_fallback = "server_provided_or_word_count"
            else:
                streaming_fallback = "unknown"
            
            # Check if server tokens are available
            has_server_usage = "'usage'" in content and "completion_tokens" in content
            
            self.output_signal.emit(f"\nAnalysis complete:\n")
            self.output_signal.emit(f"- Standard mode fallback: {standard_fallback}\n")
            self.output_signal.emit(f"- Streaming mode fallback: {streaming_fallback}\n")
            
            # Determine if there's an inconsistency
            consistency_issue = standard_fallback != streaming_fallback
            
            result = {
                "standard_fallback": standard_fallback,
                "streaming_fallback": streaming_fallback,
                "server_tokens_available": has_server_usage,
                "consistency_issue": consistency_issue,
                "recommendation": self._generate_recommendation(standard_fallback, streaming_fallback)
            }
            
            self.debug_result.emit(result)
            
        except FileNotFoundError:
            self.output_signal.emit("ERROR: http_benchmark_thread.py not found\n")
        except Exception as e:
            self.output_signal.emit(f"Debug error: {e}\n")


    def _generate_recommendation(self, standard_fallback, streaming_fallback):
        """Generate fix recommendation based on analysis."""
        
        if standard_fallback == "word_count" and streaming_fallback != "word_count":
            return (
                "Fix: Make both modes use the same token counting fallback.\n"
                "Replace len(text) // 4 with server-provided tokens when available,\n"
                "or implement a consistent heuristic in both _run_standard() and _run_streaming()."
            )
        elif standard_fallback == streaming_fallback:
            return (
                "Both modes use the same fallback method. If TPS is still inconsistent,\n"
                "check if server-provided tokens are being used correctly or if there's\n"
                "a calculation error in how TPS = tokens / time is computed."
            )
        else:
            return (
                "Token counting methods differ between modes.\n"
                "Standardize the fallback approach by using either:\n"
                "- Server-provided tokens when available, or\n"
                "- A consistent local estimation method in both modes."
            )


# Example usage:
if __name__ == "__main__":
    debugger = BenchmarkTPSDebugRunner()
    
    def on_result(result):
        print("\n=== Debug Results ===")
        print(f"Standard fallback: {result['standard_fallback']}")
        print(f"Streaming fallback: {result['streaming_fallback']}")
        print(f"Server tokens available: {result['server_tokens_available']}")
        print(f"Has consistency issue: {result['consistency_issue']}")
        print("\nRecommendation:")
        print(result['recommendation'])
    
    debugger.debug_result.connect(on_result)
    debugger.start()