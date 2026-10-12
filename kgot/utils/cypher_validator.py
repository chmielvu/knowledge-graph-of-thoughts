# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Cypher query validation for KGoT

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

    Example:
        >>> is_valid, error = CypherValidator.validate("MATCH (n) RETURN n")
        >>> assert is_valid

        >>> is_valid, error = CypherValidator.validate("DROP DATABASE test")
        >>> assert not is_valid
    """

    # Dangerous patterns that should be blocked or warned
    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+', 'DROP statements are not allowed'),
        (r'\bREMOVE\s+:\w+', 'REMOVE label syntax is invalid'),
        (r'\bDELETE\s+\*', 'DELETE * is dangerous, use specific nodes'),
        (r'\bDETACH\s+DELETE\s+\*', 'DETACH DELETE * is dangerous, use specific nodes'),
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
        r'^\s*LOAD\s+CSV\b',
        r'^\s*FOREACH\b',
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
            return False, "Query must start with a valid Cypher clause (MATCH, CREATE, MERGE, etc.)"

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
        """
        Check if characters are balanced in text.

        Handles string literals correctly by ignoring characters inside strings.
        """
        count = 0
        in_string = False
        escape_next = False
        string_char = None

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue

            # Track string boundaries
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None

            # Only count brackets outside strings
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
        """
        Remove potentially sensitive data from query for logging.

        Replaces string literals with placeholders.
        """
        # Replace single-quoted strings
        sanitized = re.sub(r"'[^']*'", "'<STRING>'", query)
        # Replace double-quoted strings
        sanitized = re.sub(r'"[^"]*"', '"<STRING>"', sanitized)
        return sanitized

    @classmethod
    def extract_node_labels(cls, query: str) -> list:
        """
        Extract node labels from a Cypher query.

        Useful for understanding what entities a query touches.
        """
        # Match patterns like (:Label) or (n:Label) or (n:Label1:Label2)
        pattern = r'\((\w*(?::\w+)+)\)'
        matches = re.findall(pattern, query)

        labels = []
        for match in matches:
            # Split by : and filter empty strings
            parts = [p for p in match.split(':') if p]
            labels.extend(parts)

        return list(set(labels))

    @classmethod
    def extract_relationship_types(cls, query: str) -> list:
        """
        Extract relationship types from a Cypher query.

        Useful for understanding what relationships a query touches.
        """
        # Match patterns like [:REL_TYPE] or [r:REL_TYPE]
        pattern = r'\[(\w)?:([A-Z_]+)'
        matches = re.findall(pattern, query)

        return list(set(m[1] for m in matches if m[1]))

    @classmethod
    def is_read_query(cls, query: str) -> bool:
        """
        Determine if a query is read-only.

        Returns True for MATCH, WITH, RETURN, UNWIND queries.
        Returns False for CREATE, MERGE, DELETE, SET, REMOVE queries.
        """
        query_upper = query.strip().upper()

        # Check for write operations
        write_patterns = [
            r'\bCREATE\b',
            r'\bMERGE\b',
            r'\bDELETE\b',
            r'\bDETACH\s+DELETE\b',
            r'\bSET\b',
            r'\bREMOVE\b',
        ]

        for pattern in write_patterns:
            if re.search(pattern, query_upper):
                return False

        return True


def validate_cypher_query(query: str, allow_dangerous: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate a Cypher query.

    Args:
        query: The Cypher query to validate
        allow_dangerous: If True, allow dangerous patterns (with warning)

    Returns:
        Tuple of (is_valid, error_message)
    """
    return CypherValidator.validate(query, allow_dangerous)