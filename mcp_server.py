#!/usr/bin/env python3
"""KB Agent MCP tracer — stateless streamable HTTP (FastMCP 4.x).

Usage:
    python3 mcp_server.py

Exposes the full kb-agent MCP tool surface wired to agent.core. Run alongside run.py
(consumer + Flask dashboard) so submitted tasks are processed into vault notes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.mcp.server import run_stateless_streamable_http


if __name__ == "__main__":
    run_stateless_streamable_http()
