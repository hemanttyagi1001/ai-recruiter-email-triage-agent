"""
FastAPI dependency injection. The graph, checkpointer, and LLM client are
process-wide singletons built at app startup (see main.py::lifespan) and
retrieved here via Depends().

WHY singletons: building the graph (LLM client init, PostgresSaver.setup(),
profile load) is not free. Per-request rebuild would be silly.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request


def get_graph(request: Request) -> Any:
    return request.app.state.graph


def get_checkpointer(request: Request) -> Any:
    return request.app.state.checkpointer
