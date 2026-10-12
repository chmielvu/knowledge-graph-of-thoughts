# KGoT Tier 2 Enhancement Implementation Plan

## ✅ STATUS: COMPLETE (2026-03-27)

All 4 enhancements have been implemented and syntax-verified.

## Overview
This plan details the implementation of 4 remaining high-ROI enhancements for KGoT.

---

## Task 1: Migrate Tools to ToolOutput

### Objective
Update all tools in `kgot/tools/tools_v2_3/` to use the standardized `ToolOutput` wrapper.

### Current State
Tools return inconsistent types:
- `PollinationsSearchTool` → string with markdown sections
- `TextInspectorTool` → string (file content or LLM response)
- `WebBrowserTool` → string
- `LLMTool` → string
- `ExtractZipTool` → string
- `FalkorDBSearchTool` → string with markdown

### Target Pattern
```python
from kgot.tools.base import ToolOutput

class SomeTool(BaseTool):
    def _run(self, **kwargs) -> str:
        try:
            result = self._execute(**kwargs)
            return ToolOutput.success_result(
                result=result,
                metadata={"duration": elapsed, "source": "tool_name"}
            ).to_llm_string()
        except ValidationError as e:
            return ToolOutput.error_result(
                error=str(e),
                error_code="VALIDATION_ERROR"
            ).to_llm_string()
        except APITimeoutError as e:
            return ToolOutput.error_result(
                error=str(e),
                error_code="TIMEOUT"
            ).to_llm_string()
        except Exception as e:
            return ToolOutput.error_result(
                error=str(e),
                error_code="API_ERROR"
            ).to_llm_string()
```

### Implementation Steps

#### Step 1.1: Update PollinationsSearchTool
**File:** `kgot/tools/tools_v2_3/PollinationsSearchTool.py`

Changes:
1. Import `ToolOutput` from `kgot.tools.base`
2. Wrap success case with `ToolOutput.success_result()`
3. Add metadata: `model`, `duration`, `tokens`
4. Wrap error cases with `ToolOutput.error_result()`
5. Return `.to_llm_string()`

Error codes to use:
- `VALIDATION_ERROR` - invalid model name
- `API_ERROR` - API request failed
- `NOT_INITIALIZED` - missing API key

#### Step 1.2: Update TextInspectorTool
**File:** `kgot/tools/tools_v2_3/TextInspectorTool.py`

Changes:
1. Import `ToolOutput`
2. Wrap file reading results
3. Add metadata: `file_path`, `file_size`, `duration`
4. Handle: file not found, unsupported format, parsing errors

#### Step 1.3: Update LLMTool
**File:** `kgot/tools/tools_v2_3/LLMTool.py`

Changes:
1. Wrap LLM response
2. Add metadata: `model`, `tokens`, `duration`
3. Handle: API errors, timeout, rate limits

#### Step 1.4: Update ExtractZipTool
**File:** `kgot/tools/tools_v2_3/ExtractZipTool.py`

Changes:
1. Wrap extraction results
2. Add metadata: `files_extracted`, `total_size`
3. Handle: invalid zip, path traversal, extraction errors

#### Step 1.5: Update FalkorDBSearchTool (already has good error handling)
**File:** `kgot/tools/tools_v2_3/FalkorDBSearchTool.py`

Changes:
1. Wrap existing returns with `ToolOutput`
2. Preserve error messages
3. Add metadata: `search_type`, `k`, `duration`

### Files to Modify
| File | Lines to Change | Complexity |
|------|-----------------|------------|
| PollinationsSearchTool.py | ~30 lines | Low |
| TextInspectorTool.py | ~50 lines | Medium |
| LLMTool.py | ~20 lines | Low |
| ExtractZipTool.py | ~25 lines | Low |
| FalkorDBSearchTool.py | ~40 lines | Low |
| WebBrowserTool.py | ~60 lines | Medium |

### Estimated Effort
- Time: 2-3 hours
- Risk: Low (backward compatible string output)

---

## Task 2: Native Tool Calling Migration

### Objective
Replace custom tool call parsing with LangChain's native `bind_tools()` and `response.tool_calls`.

### Current Implementation
```python
# In llm_invocation_base.py lines 103-131
def define_tool_calls_base(llm_execution, ...):
    # LLM is already bound with tools
    chain = llm_execution  # Already has bind_tools applied
    response = invoke_with_retry(chain, completed_prompt)
    return response.tool_calls  # Uses native tool_calls attribute
```

### Analysis
**Good news:** The code already uses native `response.tool_calls`!

The LLM is bound with tools in the controller:
```python
# In controller __init__
self.llm_execution = self.llm_execution.bind_tools(self.tools, tool_choice="required")
```

### What Needs Fixing
The issue is in **prompt design**, not tool calling. The `DEFINE_TOOL_CALLS_PROMPT_TEMPLATE` may still ask the LLM to output tool calls in a specific format, but since we're using `bind_tools()`, the LLM will use native function calling.

### Verification Needed
1. Check if prompts mention "tool_calls" format
2. Verify `response.tool_calls` returns structured data
3. Test that `_invoke_tools_after_llm_response()` handles native format

### Implementation Steps

#### Step 2.1: Review Prompts
**File:** `kgot/prompts/neo4j/base_prompts.py`

Check `DEFINE_TOOL_CALLS_PROMPT_TEMPLATE` - ensure it doesn't ask for custom format.

#### Step 2.2: Verify Tool Call Handling
**File:** `kgot/controller/falkordb/*/controller.py`

The `_invoke_tools_after_llm_response()` method expects:
```python
for tool_call in tool_calls:
    tool_name = tool_call['name'].lower()
    tool_args = tool_call['args']
```

Native `response.tool_calls` returns:
```python
[{"name": "tool_name", "args": {...}, "id": "...", "type": "tool_call"}]
```

This should work, but verify the format matches.

#### Step 2.3: Add Error Handling
If tool calling fails, provide fallback:
```python
def define_tool_calls_base(llm_execution, ...):
    response = invoke_with_retry(llm_execution, completed_prompt)

    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        logger.warning("No tool calls in response")
        return []

    return response.tool_calls
```

### Estimated Effort
- Time: 1-2 hours (mostly verification)
- Risk: Low (already using native pattern)

---

## Task 3: Circuit Breaker Pattern

### Objective
Add circuit breaker to prevent cascading failures when LLM APIs are degraded.

### Design

```python
# New file: kgot/utils/circuit_breaker.py

from datetime import datetime, timedelta
from typing import Callable, Optional
import logging

logger = logging.getLogger("Controller.CircuitBreaker")

class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            if self._should_attempt_recovery():
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
        return self._state

    def _should_attempt_recovery(self) -> bool:
        if self._last_failure_time is None:
            return True
        return datetime.now() - self._last_failure_time > self.recovery_timeout

    def record_success(self):
        """Record a successful call."""
        if self._state == "HALF_OPEN":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._reset()
        elif self._state == "CLOSED":
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            logger.warning("Circuit breaker opened after failure in half-open state")
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def _reset(self):
        """Reset to closed state."""
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker reset to closed state")

    def is_call_allowed(self) -> bool:
        """Check if a call should be allowed."""
        return self.state in ("CLOSED", "HALF_OPEN")

    def __call__(self, func: Callable) -> Callable:
        """Decorator for wrapping functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            if not self.is_call_allowed():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is {self.state}, calls blocked"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking calls."""
    pass
```

### Integration Points

#### Option A: Wrap LLM Utils
```python
# In kgot/utils/llm_utils.py

from kgot.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

# Create circuit breakers for different services
_llm_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

def invoke_with_retry(chain, *args, **kwargs):
    if not _llm_circuit_breaker.is_call_allowed():
        raise CircuitBreakerOpenError("LLM service circuit breaker is open")

    try:
        result = _retry_call(lambda: chain.invoke(*args, **kwargs))
        _llm_circuit_breaker.record_success()
        return result
    except Exception as e:
        _llm_circuit_breaker.record_failure()
        raise
```

#### Option B: Use with Tenacity
```python
from tenacity import retry, stop_after_delay, wait_exponential
from kgot.utils.circuit_breaker import CircuitBreakerOpenError

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_delay(300),
    retry=retry_if_exception_type(InternalServerError) | retry_if_exception_type(APIConnectionError),
    retry_error_callback=lambda _: None  # Return None after all retries
)
def invoke_with_circuit_breaker(chain, prompt):
    if not _llm_circuit_breaker.is_call_allowed():
        raise CircuitBreakerOpenError("Circuit breaker open")
    return invoke_with_retry(chain, prompt)
```

### Files to Create/Modify
| File | Action |
|------|--------|
| `kgot/utils/circuit_breaker.py` | NEW - Circuit breaker implementation |
| `kgot/utils/llm_utils.py` | MODIFY - Integrate circuit breaker |
| `kgot/controller/falkordb/*/controller.py` | MODIFY - Handle CircuitBreakerOpenError |

### Estimated Effort
- Time: 2-3 hours
- Risk: Medium (new error handling path)

---

## Task 4: Query Validation

### Objective
Validate Cypher queries before execution to catch syntax errors and prevent injection.

### Design

```python
# New file: kgot/utils/cypher_validator.py

import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger("Controller.CypherValidator")

class CypherValidationError(Exception):
    """Raised when Cypher query validation fails."""
    def __init__(self, message: str, query: str, position: Optional[int] = None):
        self.query = query
        self.position = position
        super().__init__(message)

class CypherValidator:
    """
    Validates Cypher queries before execution.

    Checks:
    1. Basic syntax patterns
    2. Dangerous patterns (DELETE all, DROP, etc.)
    3. Unbalanced parentheses/brackets
    4. Invalid clause ordering
    """

    # Dangerous patterns that should be blocked or warned
    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+', 'DROP statements are not allowed'),
        (r'\bREMOVE\s+:\w+', 'REMOVE label syntax is invalid'),
        (r'\bDELETE\s+\*', 'DELETE * is dangerous, use specific nodes'),
    ]

    # Required patterns for valid queries
    VALID_START_PATTERNS = [
        r'^\s*MATCH\b',
        r'^\s*CREATE\b',
        r'^\s*MERGE\b',
        r'^\s*WITH\b',
        r'^\s*UNWIND\b',
        r'^\s*CALL\b',
        r'^\s*RETURN\b',
        r'^\s*USING\b',
    ]

    @classmethod
    def validate(cls, query: str, allow_dangerous: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate a Cypher query.

        Args:
            query: The Cypher query to validate
            allow_dangerous: If True, allow dangerous patterns (with warning)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Empty query"

        query = query.strip()

        # Check for valid start
        if not any(re.match(pattern, query, re.IGNORECASE) for pattern in cls.VALID_START_PATTERNS):
            return False, f"Query must start with a valid Cypher clause (MATCH, CREATE, MERGE, etc.)"

        # Check for dangerous patterns
        for pattern, message in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                if not allow_dangerous:
                    return False, f"Potentially dangerous query: {message}"
                logger.warning(f"Dangerous pattern in query: {message}")

        # Check balanced parentheses
        if not cls._check_balanced(query, '(', ')'):
            return False, "Unbalanced parentheses in query"

        # Check balanced brackets
        if not cls._check_balanced(query, '[', ']'):
            return False, "Unbalanced brackets in query"

        # Check balanced braces
        if not cls._check_balanced(query, '{', '}'):
            return False, "Unbalanced braces in query"

        return True, None

    @staticmethod
    def _check_balanced(text: str, open_char: str, close_char: str) -> bool:
        """Check if characters are balanced in text."""
        count = 0
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char in ('"', "'"):
                in_string = not in_string
            if not in_string:
                if char == open_char:
                    count += 1
                elif char == close_char:
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    @classmethod
    def sanitize_for_logging(cls, query: str) -> str:
        """Remove potentially sensitive data from query for logging."""
        # Replace string literals with placeholder
        sanitized = re.sub(r"'[^']*'", "'<STRING>'", query)
        sanitized = re.sub(r'"[^"]*"', '"<STRING>"', sanitized)
        return sanitized
```

### Integration

```python
# In kgot/knowledge_graph/falkordb/main.py

from kgot.utils.cypher_validator import CypherValidator, CypherValidationError

class FalkorDBKnowledgeGraph:
    def get_query(self, query: str, ...):
        # Validate before execution
        is_valid, error = CypherValidator.validate(query)
        if not is_valid:
            logger.error(f"Invalid Cypher query: {error}")
            return None, False, CypherValidationError(error, query)

        try:
            result = self._ro_query_database(query)
            return result, True, None
        except Exception as e:
            return None, False, e

    def write_query(self, query: str, ...):
        # Validate with stricter rules for writes
        is_valid, error = CypherValidator.validate(query, allow_dangerous=False)
        if not is_valid:
            logger.error(f"Invalid write query: {error}")
            return False, CypherValidationError(error, query)

        try:
            self.graph.query(query)
            self._export_db()
            return True, None
        except Exception as e:
            return False, e
```

### Files to Create/Modify
| File | Action |
|------|--------|
| `kgot/utils/cypher_validator.py` | NEW - Cypher validation |
| `kgot/knowledge_graph/falkordb/main.py` | MODIFY - Add validation calls |
| `kgot/knowledge_graph/kg_interface.py` | MODIFY - Document validation behavior |

### Estimated Effort
- Time: 3-4 hours
- Risk: Medium (may reject valid queries)

---

## Execution Order

1. **Task 1: ToolOutput Migration** - Start here, lowest risk
2. **Task 2: Native Tool Calling** - Mostly verification
3. **Task 3: Circuit Breaker** - Add resilience
4. **Task 4: Query Validation** - Final safety layer

---

## Testing Checklist

### Syntax Verification (PASSED)
- [x] All modified files compile without errors
- [x] ToolOutput wrapper functional
- [x] Circuit breaker imports correctly
- [x] Cypher validator imports correctly

### Task 1 Tests
- [x] ToolOutput.success_result() creates valid output
- [x] ToolOutput.error_result() creates valid output
- [x] to_llm_string() formats correctly
- [x] Each migrated tool returns ToolOutput

### Task 2 Tests
- [x] Native tool_calls format works
- [x] Empty tool_calls handled gracefully (added in llm_invocation_base.py)
- [ ] Tool invocation works end-to-end (requires runtime test)

### Task 3 Tests
- [x] Circuit breaker starts CLOSED
- [x] Opens after threshold failures
- [x] Transitions to HALF_OPEN after timeout
- [x] Resets after successful HALF_OPEN calls

### Task 4 Tests
- [x] Valid queries pass validation
- [x] Invalid syntax caught
- [x] Dangerous patterns blocked
- [x] Balanced parentheses/brackets checked

### Integration Tests (TODO)
- [ ] Run GAIA benchmark with new ToolOutput format
- [ ] Verify circuit breaker behavior under API failures
- [ ] Test Cypher validation with real queries