# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# FalkorDB directRetrieve prompts - reuses Neo4j prompts (OpenCypher compatible)

from kgot.prompts.neo4j.directRetrieve.prompts import (
    DEFINE_FORCED_SOLUTION_TEMPLATE,
    DEFINE_NEXT_STEP_PROMPT_TEMPLATE,
)

__all__ = [
    "DEFINE_FORCED_SOLUTION_TEMPLATE",
    "DEFINE_NEXT_STEP_PROMPT_TEMPLATE",
]
