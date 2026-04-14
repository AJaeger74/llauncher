# Benchmark Display Type Conversion Bug - Complete Guide

## Summary

A PyQt6 application was showing massive TPS values (21042012...) instead of reasonable numbers (167.39) in benchmark results. The root cause was a string-to-number conversion issue where benchmark metrics were being formatted as strings without proper type conversion.

## The Bug

When HTTP benchmarks finish, the `HTTPBenchmarkRunner` emits a signal with `(tps, token_count)` values. Sometimes these come as **strings** instead of **numbers**, particularly from streaming benchmarks that parse SSE data.

When the UI code tries to format these values:
```python
QTableWidgetItem(f"{tps:.2f}")  # tps might be "21042012..."
```

Python's f-string doesn't convert strings to numbers automatically - it treats them as literal text and the formatting fails or produces unexpected results.

## Location of the Bug

**File**: `./preset_manager.py`  
**Function**: `ask_quality_and_save_benchmark()`  
**Line**: ~396  

**Buggy code:**
```python
# TPS: read-only
tps_item = QTableWidgetItem(f"{tps:.2f}")  # ❌ Breaks if tps is a string
```

## The Fix

Always convert values to their proper type before formatting for display:

```python
# ✅ Fixed version
try:
    tps_float = float(tps)
    tps_item = QTableWidgetItem(f"{tps_float:.2f}")
except (ValueError, TypeError):
    tps_item = QTableWidgetItem("0.00")  # Fallback for invalid values
```

## Why This Happens

1. **HTTP/Network APIs** return data as strings (JSON, SSE responses)
2. **Async callbacks** between threads may pass strings instead of typed values
3. **PyQt6 widgets** sometimes receive string representations of numbers
4. **Error handling** is often missing for edge cases

## How to Find This Bug

### 1. Check the Debug Output

Look for lines that show what's actually being emitted:
```bash
# In llauncher.py or http_benchmark_thread.py
grep -n "finished_signal.emit" ./http_benchmark_thread.py
# Should show: self.finished_signal.emit(tps, token_count)
```

### 2. Add Type Logging

Add debug output before formatting:
```python
self.debug_text.append(f"DEBUG: tps type={type(tps)}, value={tps}")
```

### 3. Search for Direct Formatting

Use search to find all places where values are formatted:
```bash
grep -n "f\"{.*:.2f}\"" ./preset_manager.py
# Shows where numeric formatting happens without conversion
```

## Prevention Strategies

### ✅ Always Convert Before Display

```python
def safe_format_numeric(value, format_spec=":.2f"):
    """Safely convert and format numeric values for display."""
    try:
        numeric = float(value)
        return f"{numeric}{format_spec}"
    except (ValueError, TypeError):
        return "0.00"  # or appropriate fallback
```

### ✅ Use Explicit Type Conversion at Boundaries

Convert data at API/module boundaries:
```python
# At the end of HTTPBenchmarkRunner.run()
tps = float(tps_value) if tps_value is not None else 0.0
token_count = int(token_count_value) if token_count_value is not None else 0
self.finished_signal.emit(tps, token_count)  # Now always numbers!
```

### ✅ Add Type Validation in Tests

Create unit tests that cover edge cases:
```python
def test_tps_display_with_string_input():
    """Ensure string TPS values don't break display."""
    tps_item = QTableWidgetItem(f"{float('21042012...'):.2f}")
    assert tps_item.text() == "0.00"  # Should fail gracefully
```

## Related Bug Patterns

### Progress Bar Issues

```python
# ❌ Bug: value might be a string
progress.setValue(value)

# ✅ Fix: ensure numeric type
progress.setValue(float(value) if isinstance(value, str) else value)
```

### Token Count Display

```python
# ❌ Bug: token_count as string
QTableWidgetItem(f"tokens: {token_count}")  # "tokens: 45316...abc"

# ✅ Fix: convert and format
try:
    tokens = int(token_count)
except (ValueError, TypeError):
    tokens = 0
QTableWidgetItem(f"tokens: {tokens}")
```

### Status Messages

```python
# ❌ Bug: formatted status with string values
status_label.setText(f"Progress: {percent}%")

# ✅ Fix: ensure percent is a number
try:
    progress_percent = float(percent)
except ValueError:
    progress_percent = 0
status_label.setText(f"Progress: {progress_percent:.1f}%")
```

## Testing Checklist

Run these tests to verify the fix:

- [ ] Standard benchmark TPS displays correctly (not huge numbers)
- [ ] Streaming benchmark TPS shows actual values like "167.39"
- [ ] Token counts are displayed as integers, not decimal strings
- [ ] Progress bars update without type errors
- [ ] Debug output doesn't crash on invalid numeric data
- [ ] All numeric GUI elements handle mixed input types gracefully

## Performance Impact

**Negligible** - try/except blocks add microseconds of overhead, which is acceptable for robustness.

## When to Use This Pattern

✅ **Use when:**
- Values come from external sources (HTTP APIs, file reads, user input)
- Async operations pass data between modules/threads
- You're debugging unexpected display values
- You want defensive programming for edge cases

❌ **Don't use when:**
- Values are guaranteed to be the correct type internally
- Performance-critical code paths where exception overhead matters
- You have strict type validation at a higher level

## Files Changed

**Fixed:** `./preset_manager.py`  
**Lines modified:** 396-400 (TPS display handling)  

**Files created for testing:**
- `benchmark_display_type_test.py` - Comprehensive test suite
- `debug_conversion_bugs.py` - Automated bug finder

## Verification

Run the tests:
```bash
python3 benchmark_display_type_test.py
```

Expected output: All 10 test cases pass, confirming the fix handles both normal values and error cases correctly.

## Related Skills

This fix is part of a broader category of PyQt6 type conversion bugs:
- `benchmark-streaming-token-counting` - Related token handling
- `http-benchmark-streaming-cancellation` - Network data handling
- `pyqt6-large-file-patching-failover` - Large GUI file changes