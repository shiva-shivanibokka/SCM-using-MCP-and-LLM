"""
MCP Client for Pet Store SCM Agent
Supports two modes:
  1. HTTP/SSE   — calls the MCP server over HTTP (default)
  2. Direct     — bypasses HTTP, calls server functions in-process (HF Spaces mode)

Set BYPASS_MCP_HTTP=true in .env to use direct mode (no separate server process needed).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")
BYPASS_MCP_HTTP = os.getenv("BYPASS_MCP_HTTP", "true").lower() in ("true", "1", "yes")


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if MCP_AUTH_TOKEN:
        h["Authorization"] = f"Bearer {MCP_AUTH_TOKEN}"
    return h


# HTTP MCP Client


async def _http_call_tool(tool_name: str, arguments: dict) -> str:
    """Call the MCP server tool via HTTP POST to /messages."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{MCP_SERVER_URL}/messages",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"MCP Error: {data['error']['message']}"
        content = data.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", str(content))
        return str(data.get("result", ""))


async def _http_list_tools() -> list[dict]:
    """Fetch the tool list from the MCP server."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{MCP_SERVER_URL}/messages",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get("tools", [])


async def _http_read_resource(uri: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "resources/read",
        "params": {"uri": uri},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{MCP_SERVER_URL}/messages",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        contents = data.get("result", {}).get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""


# Direct (in-process) MCP Client


async def _direct_call_tool(tool_name: str, arguments: dict) -> str:
    """Call the MCP server directly, bypassing HTTP (used on HF Spaces)."""
    from mcp_server.server import call_tool_direct

    return await call_tool_direct(tool_name, arguments)


async def _direct_list_tools() -> list[dict]:
    from mcp_server.server import MCP_TOOLS

    return MCP_TOOLS


async def _direct_read_resource(uri: str) -> str:
    from mcp_server.server import read_resource

    return await read_resource(uri)


# Public API auto-selects HTTP vs direct based on BYPASS_MCP_HTTP


async def call_tool(tool_name: str, arguments: dict) -> str:
    """
    Call an MCP tool by name with given arguments.
    Returns a string result (already formatted for injection into LLM context).
    """
    try:
        if BYPASS_MCP_HTTP:
            return await _direct_call_tool(tool_name, arguments)
        else:
            return await _http_call_tool(tool_name, arguments)
    except Exception as exc:
        return f"MCP tool call failed ({tool_name}): {exc}"


async def list_tools() -> list[dict]:
    """Return all available MCP tools.
    BUG-007 fix: log the exception before returning [] so failures are diagnosable.
    """
    try:
        if BYPASS_MCP_HTTP:
            return await _direct_list_tools()
        else:
            return await _http_list_tools()
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).error(
            f"[MCP] list_tools() failed — agent will have no tools: {exc}",
            exc_info=True,
        )
        return []


async def read_resource(uri: str) -> str:
    """Read an MCP resource by URI."""
    try:
        if BYPASS_MCP_HTTP:
            return await _direct_read_resource(uri)
        else:
            return await _http_read_resource(uri)
    except Exception as exc:
        return f"MCP resource read failed ({uri}): {exc}"


async def health_check() -> dict:
    """Check if the MCP server is reachable."""
    if BYPASS_MCP_HTTP:
        return {"mode": "direct (in-process)", "status": "ok"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MCP_SERVER_URL}/health")
            return resp.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


async def test_db_connection(db_type: str, creds: dict) -> dict:
    """
    Test a database connection through the MCP server.

    db_type: "mysql" or "postgres"
    creds:   {host, port, user, password, db}

    Returns {ok: bool, message: str, details: dict}
    """
    tool_name = f"test_{db_type}_connection"
    try:
        if BYPASS_MCP_HTTP:
            # Call the server helper directly for a richer result dict
            from mcp_server.server import (
                test_mysql_connection,
                test_postgres_connection,
            )

            if db_type == "mysql":
                return await test_mysql_connection(creds)
            else:
                return await test_postgres_connection(creds)
        else:
            # Over HTTP the result is a plain string wrap it in a dict
            text = await _http_call_tool(tool_name, {"creds": creds})
            ok = not text.lower().startswith(
                (
                    "mysql connection failed",
                    "postgresql connection failed",
                    "mcp error",
                    "error",
                )
            )
            return {"ok": ok, "message": text, "details": {}}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "details": {}}
