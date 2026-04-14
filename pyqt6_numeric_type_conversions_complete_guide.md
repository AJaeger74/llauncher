# Complete Guide: Finding and Fixing PyQt6 Numeric Type Conversion Bugs

## 🎯 What This Skill Covers

This guide provides a systematic approach to identifying and fixing bugs where numeric values (like TPS, scores, percentages) are passed as strings instead of numbers in PyQt6 applications. These bugs cause display errors like showing "21042012..." instead of 167.39.

## 📋 Symptoms You'll See

- Massive numbers displayed where reasonable values should appear
- "Error" or cryptic messages in debug output
- GUI elements that don't update or show "0.00"
- Formatting errors like `"Unknown format code 'f' for object of type 'str'"`

## 🔍 Systematic Debugging Steps

### Step 1: Identify the Symptom

**Check the actual values being passed:**

```python
# Add debug logging right before formatting
self.debug_text.append(f"DEBUG: tps value='{tps}', type={type(tps)}")
```

**Run the test and check output:**
```bash
# If you see this, you have the bug!
DEBUG: tps value='21042012...', type=<class 'str'>
```

### Step 2: Trace Back to Source

**Find where the value originates:**
```bash
grep -rn "finished_signal.emit.*tps" .  # Common signal
grep -rn "tps.*=" .                      # Assignment locations
grep -rn "emit.*token_count" .          # Related values
```

**Look for string sources:**
- HTTP/JSON responses: `json.loads()` returns strings for some fields
- SSE (Server-Sent Events): Data comes as text streams
- User input: QLineEdit, QComboBox return strings
- Network APIs: Often return numeric strings

### Step 3: Find the Bug Location

**Search for formatting without conversion:**
```bash
# Look for direct f-string formatting
grep -rn "f\"{.*:.2f}\"" . 
grep -rn "f\"{.*:.3f}\"" .
grep -rn "QTableWidgetItem.*:" .
```

**Typical bug locations:**
- `QTableWidget` items (like our bug in `preset_manager.py`)
- `QLabel.setText()` calls
- Progress bar values
- Status messages
- Debug/output console text

### Step 4: Verify the Type Issue

**Create a minimal test case:**
```python
# Bug reproduction
tps = "21042012..."  # String instead of number
print(f"{tps:.2f}")  # ❌ ERROR!

# Working version
tps_float = float("21042012...")  # ✅ Converts (or raises exception)
```

### Step 5: Check Related Code

**Look for similar patterns in the same file:**
```bash
# Find all numeric formatting in this file
grep -n "QTableWidgetItem\|QLabel.*setText" ./preset_manager.py
```

**Check if other values have the same issue:**
- `token_count` - might also be a string
- Progress bar values
- Percentages
- Scores/quality ratings

## 🛠️ The Fix Pattern

### Universal Solution (Recommended)

**Always convert before formatting:**
```python
def safe_format_numeric(value, format_spec=":.2f"):
    """Safely convert and format numeric values for display."""
    try:
        numeric = float(value)
        return f"{numeric}{format_spec}"
    except (ValueError, TypeError):
        return "0.00"  # Fallback for invalid values

# Usage:
tps_item = QTableWidgetItem(safe_format_numeric(tps))
```

### Quick Fix (For Individual Cases)

**Direct fix with error handling:**
```python
# ❌ Before (buggy)
tps_item = QTableWidgetItem(f"{tps:.2f}")

# ✅ After (fixed)
try:
    tps_float = float(tps)
    tps_item = QTableWidgetItem(f"{tps_float:.2f}")
except (ValueError, TypeError):
    tps_item = QTableWidgetItem("0.00")
```

## 🧪 Testing Strategy

### Unit Tests

**Test all edge cases:**
```python
def test_numeric_formatting():
    """Ensure numeric values format correctly."""
    
    test_cases = [
        # (input_value, expected_output)
        (167.39, "167.39"),      # Normal float
        (45, "45.00"),            # Integer
        ("21042012...", "0.00"),  # Invalid string (THE BUG)
        ("167.39", "167.39"),     # String number (should work)
        ("not_a_number", "0.00"), # Non-numeric string
    ]
    
    for value, expected in test_cases:
        result = safe_format_numeric(value)
        assert result == expected, f"Failed for {value}: got {result}, expected {expected}"
```

### Integration Tests

**Test actual UI components:**
```python
def test_benchmark_tps_display():
    """Verify TPS displays correctly in benchmark results."""
    
    # Simulate buggy input
    tps = "21042012..."  # String from network response
    
    # Should not crash and should show fallback
    tps_item = QTableWidgetItem(safe_format_numeric(tps))
    assert tps_item.text() == "0.00", "Should handle invalid TPS gracefully"
```

## 🚨 Prevention Strategies

### 1. Type Conversion at Boundaries

**Convert data types when entering/exiting modules:**
```python
# When receiving HTTP data
def process_benchmark_data(raw_data):
    # Convert strings to proper types immediately
    tps = float(raw_data.get('tps', 0)) if raw_data.get('tps') else 0.0
    token_count = int(raw_data['tokens']) if 'tokens' in raw_data else 0
    
    return tps, token_count  # Now always numbers!
```

### 2. Add Type Validation

**Use validation functions for critical data:**
```python
def validate_numeric(value, default=0):
    """Ensure value is numeric with fallback."""
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Usage
tps = validate_numeric(raw_tps_value, default=0.0)
```

### 3. Use Static Analysis Tools

**Find type mismatches early:**
```bash
# Install and run type checkers
mypy ./llauncher.py
pylint ./preset_manager.py --disable=all --enable=unexpected-keyword-arg
```

## 🔧 Related Bug Patterns

### Progress Bar Values

```python
# ❌ Bug: value is a string
progress.setValue(percent)  # percent might be "85" not 85.0

# ✅ Fix: ensure numeric type
progress.setValue(float(percent) if isinstance(percent, str) else percent)
```

### Token Count Display

```python
# ❌ Bug: token_count is a string
QTableWidgetItem(f"tokens: {token_count}")  # "tokens: 45316...abc"

# ✅ Fix: convert to integer
try:
    tokens = int(token_count)
except (ValueError, TypeError):
    tokens = 0
QTableWidgetItem(f"tokens: {tokens}")
```

### Status Messages

```python
# ❌ Bug: formatted with string value
status_label.setText(f"Progress: {percent}%")  # "Progress: 85%..." if percent is string

# ✅ Fix: convert first
try:
    progress_percent = float(percent)
except ValueError:
    progress_percent = 0
status_label.setText(f"Progress: {progress_percent:.1f}%")
```

## 📁 Files Affected (Common Patterns)

Look for these files and patterns:

| File | Typical Bug Location | Fix Needed |
|------|---------------------|------------|
| `preset_manager.py` | `ask_quality_and_save_benchmark()` | ✓ FIXED - TPS display |
| `http_benchmark_thread.py` | Signal emissions | Ensure numeric types |
| `llauncher.py` | UI update handlers | Add type conversion |
| `progress_monitor.py` | Progress bar values | Convert to float |

## 💡 Pro Tips

**Tip 1: Always add debug logging during development**
```python
# In critical code paths
self.debug_text.append(f"DEBUG: value={value!r}, type={type(value)}")
```

**Tip 2: Use type hints for clarity**
```python
def format_tps(tps: float) -> str:
    """Format TPS value for display."""
    return f"{tps:.2f}"

# If someone passes a string, the type checker will warn!
```

**Tip 3: Create helper functions for common conversions**
```python
# Add to utils.py
def safe_float(value, default=0.0):
    """Safely convert to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert to int."""
    try:
        return int(float(value))  # Handles "45.9" → 45
    except (ValueError, TypeError):
        return default
```

**Tip 4: Use pytest fixtures for edge cases**
```python
@pytest.fixture(params=[
    "21042012...",   # The bug case
    "invalid",       # Non-numeric string  
    "167.39",        # Normal number as string
    167.39,          # Actual float
])
def invalid_input(request):
    return request.param
```

## 📚 Resources

### PyQt6 Documentation
- [QTableWidgetItem](https://doc.qt.io/qtforpython-6/QtWidgets/QTableWidgetItem.html) - accepts any type but formats for display
- [Signals and Slots](https://doc.qt.io/qtforpython-6/overviews/signalsandslots.html) - data types must be compatible

### Python Best Practices
- [PEP 8](https://peps.python.org/pep-0008/) - Code readability standards
- [Type Hints](https://docs.python.org/3/library/typing.html) - Static type checking

### Debugging Tools
- [pdb](https://docs.python.org/3/library/pdb.html) - Interactive debugging
- [breakpoint()](https://docs.python.org/3/library/functions.html#breakpoint) - Modern Python debugger (Python 3.7+)