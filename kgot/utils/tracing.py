# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Langfuse tracing utilities for KGoT observability

import os
import logging
from typing import Optional

logger = logging.getLogger("Controller.Tracing")

# Global Langfuse handler (initialized once)
_langfuse_handler = None


def get_langfuse_handler() -> Optional["CallbackHandler"]:
    """
    Get or initialize the Langfuse callback handler.

    Returns None if Langfuse is not configured (keys not set).
    Uses environment variables:
    - LANGFUSE_PUBLIC_KEY: Langfuse public key (required)
    - LANGFUSE_SECRET_KEY: Langfuse secret key (required)
    - LANGFUSE_HOST: Langfuse host (optional, defaults to cloud)
    - LANGFUSE_TRACING_ENABLED: Enable/disable tracing (optional, defaults to true)

    Returns:
        CallbackHandler instance or None if not configured
    """
    global _langfuse_handler

    if _langfuse_handler is not None:
        return _langfuse_handler

    # Check if tracing is disabled
    tracing_enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower()
    if tracing_enabled not in ("true", "1", "yes"):
        logger.info("Langfuse tracing is disabled via LANGFUSE_TRACING_ENABLED")
        return None

    # Check for required keys
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.info("Langfuse keys not configured. Tracing disabled. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable.")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        _langfuse_handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        logger.info(f"Langfuse tracing initialized. Host: {host}")
        return _langfuse_handler

    except ImportError:
        logger.warning("langfuse package not installed. Install with: pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None


def reset_langfuse_handler() -> None:
    """Reset the global Langfuse handler (useful for testing)."""
    global _langfuse_handler
    _langfuse_handler = None