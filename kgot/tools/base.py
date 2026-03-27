# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main authors: Lorenzo Paleari
#               Andrea Jiang

"""
Base model for standardized tool outputs in KGoT.

This module provides the ToolOutput Pydantic model that all tools should use
to return consistent, structured outputs. This enables better error handling,
LLM-friendly output formatting, and metadata tracking across all tools.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolOutput(BaseModel):
    """
    Standardized output model for all KGoT tools.

    This model provides a consistent structure for tool outputs, enabling
    uniform error handling, metadata tracking, and LLM-friendly output formatting
    across all tools in the system.

    Attributes:
        success: Whether the tool execution was successful
        result: The output data when successful (e.g., computed value, retrieved data)
        error: Human-readable error message when execution fails
        error_code: Categorized error code for programmatic handling
        metadata: Additional context information (tokens, duration, sources, etc.)

    Example:
        >>> # Successful execution
        >>> ToolOutput.success_result(
        ...     result="The answer is 42",
        ...     metadata={"tokens": 100, "duration": 1.5}
        ... )

        >>> # Error case
        >>> ToolOutput.error_result(
        ...     error="Failed to connect to API",
        ...     error_code="API_ERROR"
        ... )
    """

    success: bool = Field(
        ...,
        description="Whether the tool execution completed successfully"
    )
    result: Optional[Any] = Field(
        default=None,
        description="The output data when successful. Can be any type: str, dict, list, etc."
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message describing what went wrong"
    )
    error_code: Optional[str] = Field(
        default=None,
        description=(
            "Categorized error code for programmatic handling. "
            "Common codes: VALIDATION_ERROR, API_ERROR, TIMEOUT, "
            "NOT_INITIALIZED, PERMISSION_ERROR, RESOURCE_NOT_FOUND"
        )
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional context information about the execution. "
            "Common fields: tokens (int), duration (float), sources (list), "
            "tool_name (str), timestamp (str)"
        )
    )

    def to_llm_string(self) -> str:
        """
        Convert the tool output to an LLM-friendly string representation.

        This method formats the output in a way that is easy for LLMs to parse
        and understand, prioritizing success/error status and including relevant
        metadata.

        Returns:
            A formatted string suitable for LLM consumption

        Example:
            >>> output = ToolOutput.success_result(result="Hello, World!")
            >>> output.to_llm_string()
            'SUCCESS: Hello, World!'
        """
        if self.success:
            result_str = str(self.result) if self.result is not None else "No result"
            output = f"SUCCESS: {result_str}"

            if self.metadata:
                metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
                output += f"\nMetadata: {metadata_str}"

            return output
        else:
            error_msg = self.error or "Unknown error"
            error_code_str = f" ({self.error_code})" if self.error_code else ""
            return f"ERROR{error_code_str}: {error_msg}"

    @classmethod
    def success_result(cls, result: Any, metadata: Optional[Dict[str, Any]] = None) -> "ToolOutput":
        """
        Create a successful ToolOutput instance.

        This is a convenience factory method for creating successful tool outputs.

        Args:
            result: The output data from successful execution
            metadata: Optional metadata about the execution (tokens, duration, etc.)

        Returns:
            A ToolOutput instance with success=True

        Example:
            >>> ToolOutput.success_result(
            ...     result={"answer": 42, "confidence": 0.95},
            ...     metadata={"tokens": 150, "duration": 2.3}
            ... )
        """
        return cls(success=True, result=result, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, error_code: str, metadata: Optional[Dict[str, Any]] = None) -> "ToolOutput":
        """
        Create an error ToolOutput instance.

        This is a convenience factory method for creating error outputs with
        appropriate error categorization.

        Args:
            error: Human-readable error message
            error_code: Categorized error code (e.g., "API_ERROR", "TIMEOUT")
            metadata: Optional metadata about the failed execution

        Returns:
            A ToolOutput instance with success=False

        Example:
            >>> ToolOutput.error_result(
            ...     error="Connection timeout after 30s",
            ...     error_code="TIMEOUT",
            ...     metadata={"attempt": 3, "duration": 30.0}
            ... )
        """
        return cls(success=False, error=error, error_code=error_code, metadata=metadata)

    def __str__(self) -> str:
        """
        String representation of the ToolOutput.

        Returns the LLM-friendly string format by default.

        Returns:
            Formatted string representation
        """
        return self.to_llm_string()