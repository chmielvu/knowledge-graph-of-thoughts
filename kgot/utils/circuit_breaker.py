# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Circuit Breaker Pattern for LLM API resilience

from datetime import datetime, timedelta
from typing import Callable, Optional
import logging

logger = logging.getLogger("Controller.CircuitBreaker")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking calls."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered

    Example:
        >>> cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        >>> @cb
        ... def risky_api_call():
        ...     return external_service.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Successful calls needed to close circuit
            name: Identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current state, transitioning from OPEN to HALF_OPEN if timeout elapsed."""
        if self._state == "OPEN":
            if self._should_attempt_recovery():
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
                logger.info(f"[{self.name}] Circuit breaker transitioned to HALF_OPEN")
        return self._state

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return datetime.now() - self._last_failure_time > self.recovery_timeout

    def record_success(self):
        """Record a successful call."""
        if self._state == "HALF_OPEN":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._reset()
                logger.info(f"[{self.name}] Circuit breaker recovered - reset to CLOSED")
        elif self._state == "CLOSED":
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            logger.warning(f"[{self.name}] Circuit breaker reopened after failure in half-open state")
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"[{self.name}] Circuit breaker opened after {self._failure_count} failures"
            )

    def _reset(self):
        """Reset to closed state."""
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = None

    def is_call_allowed(self) -> bool:
        """Check if a call should be allowed."""
        return self.state in ("CLOSED", "HALF_OPEN")

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator for wrapping functions with circuit breaker.

        Example:
            >>> cb = CircuitBreaker()
            >>> @cb
            ... def my_function():
            ...     return risky_operation()
        """
        def wrapper(*args, **kwargs):
            if not self.is_call_allowed():
                raise CircuitBreakerOpenError(
                    f"[{self.name}] Circuit breaker is {self.state}, calls blocked"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        return wrapper

    def wrap_callable(self, func: Callable) -> Callable:
        """
        Wrap a callable with circuit breaker protection.

        Unlike __call__ decorator, this works with functions that are already defined.

        Args:
            func: The function to wrap

        Returns:
            Wrapped function that respects circuit breaker state
        """
        def wrapper(*args, **kwargs):
            if not self.is_call_allowed():
                raise CircuitBreakerOpenError(
                    f"[{self.name}] Circuit breaker is {self.state}, calls blocked"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        return wrapper

    def get_stats(self) -> dict:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "half_open_calls": self._half_open_calls,
        }


# Global circuit breakers for different services
_llm_circuit_breaker: Optional[CircuitBreaker] = None
_math_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60
) -> CircuitBreaker:
    """
    Get or create the global LLM circuit breaker.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery

    Returns:
        CircuitBreaker instance for LLM calls
    """
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name="llm_service"
        )
    return _llm_circuit_breaker


def get_math_circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: int = 30
) -> CircuitBreaker:
    """
    Get or create the global math executor circuit breaker.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery

    Returns:
        CircuitBreaker instance for math executor calls
    """
    global _math_circuit_breaker
    if _math_circuit_breaker is None:
        _math_circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name="math_executor"
        )
    return _math_circuit_breaker


def reset_all_circuit_breakers():
    """Reset all global circuit breakers (useful for testing)."""
    global _llm_circuit_breaker, _math_circuit_breaker
    _llm_circuit_breaker = None
    _math_circuit_breaker = None