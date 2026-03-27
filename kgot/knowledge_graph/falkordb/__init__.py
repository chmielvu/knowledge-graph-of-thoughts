# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# FalkorDB backend for KGoT - provides native multi-tenancy and vector search

from .main import KnowledgeGraph

__all__ = ["KnowledgeGraph"]