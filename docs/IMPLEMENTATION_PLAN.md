# KGoT Enhancement Implementation Plan

## Overview
Implementation of Tier 1 high-ROI enhancements for KGoT agent architecture.

## Tasks

### 1. ToolOutput Standardization
**Goal:** Create a standardized output wrapper for all tools

**Files to modify:**
- `kgot/tools/base.py` (NEW) - ToolOutput model
- `kgot/tools/tools_v2_3/tool_manager.py` - Update to handle ToolOutput

**Implementation:**
```python
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class ToolOutput(BaseModel):
    """Standardized output wrapper for all KGoT tools.

    Provides consistent structure for LLM parsing, error handling,
    and metadata collection across all tool implementations.
    """
    success: bool = Field(description="Whether the tool execution succeeded")
    result: Optional[Any] = Field(default=None, description="The tool result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    error_code: Optional[str] = Field(default=None, description="Error code: VALIDATION_ERROR, API_ERROR, TIMEOUT, NOT_INITIALIZED")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata: tokens, duration, sources")

    def to_llm_string(self) -> str:
        """Generate LLM-friendly string representation."""
        if self.success:
            result_str = str(self.result)
            if self.metadata:
                return f"Result: {result_str}\nMetadata: {self.metadata}"
            return f"Result: {result_str}"
        return f"Error [{self.error_code}]: {self.error}"

    @classmethod
    def success_result(cls, result: Any, metadata: Optional[Dict] = None) -> "ToolOutput":
        """Create a successful result."""
        return cls(success=True, result=result, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, error_code: str = "UNKNOWN_ERROR") -> "ToolOutput":
        """Create an error result."""
        return cls(success=False, error=error, error_code=error_code)
```

**Tool Update Pattern:**
```python
# Before
def _run(self, query: str) -> str:
    return "some result"

# After
def _run(self, query: str) -> str:
    try:
        result = self._execute(query)
        return ToolOutput.success_result(result, metadata={"source": "tool_name"}).to_llm_string()
    except ValueError as e:
        return ToolOutput.error_result(str(e), "VALIDATION_ERROR").to_llm_string()
    except Exception as e:
        return ToolOutput.error_result(str(e), "API_ERROR").to_llm_string()
```

---

### 2. FalkorDBSearchTool Registration
**Goal:** Register the existing FalkorDBSearchTool in ToolManager

**Files to modify:**
- `kgot/tools/tools_v2_3/tool_manager.py`

**Implementation:**
1. Import FalkorDBSearchTool factory functions
2. Check if knowledge graph is FalkorDB instance
3. Create and register semantic/hybrid/text-to-cypher tools

**Code location:** `tool_manager.py` lines ~100-150 in `__init__`

---

### 3. Langfuse Integration
**Goal:** Add observability tracing via Langfuse callbacks

**Files to modify:**
- `pyproject.toml` - Add langfuse dependency
- `kgot/controller/falkordb/queryRetrieve/controller.py` - Add callback handler
- `kgot/controller/falkordb/directRetrieve/controller.py` - Add callback handler
- `kgot/config_tools.template.json` - Add Langfuse config template

**Implementation:**
```python
from langfuse.langchain import CallbackHandler

# In controller __init__
self.langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# In invoke calls
result = self.llm.invoke(messages, config={"callbacks": [self.langfuse_handler]})
```

**Environment variables needed:**
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional, defaults to cloud)

---

### 4. Native Tool Calling Migration (Partial)
**Goal:** Replace custom tool parsing with LangChain native bind_tools

**Files to modify:**
- `kgot/controller/falkordb/*/llm_invocation_handle.py` - Update tool calling

**Current pattern (custom):**
```python
tool_calls = parse_tool_calls(response.content)
```

**Target pattern (native):**
```python
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
response = llm_with_tools.invoke(messages)
for tool_call in response.tool_calls:
    # tool_call is structured: {"name": str, "args": dict, "id": str}
```

---

## Execution Order

1. **Parallel Phase 1** (independent tasks):
   - Task 1: ToolOutput base class
   - Task 2: FalkorDBSearchTool registration
   - Task 3: Langfuse dependency and config

2. **Sequential Phase 2** (depends on Phase 1):
   - Update tools to use ToolOutput
   - Add Langfuse callbacks to controllers
   - Update native tool calling

3. **Review Phase**:
   - Code review
   - Integration testing
   - Documentation update

---

## Testing Checklist

- [ ] ToolOutput serializes correctly
- [ ] FalkorDBSearchTool appears in tool list
- [ ] Langfuse traces appear in dashboard
- [ ] Native tool calling works with bound tools
- [ ] Existing tests still pass
- [ ] Error handling works for all error types