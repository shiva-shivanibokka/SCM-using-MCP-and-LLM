"""Intelligence engines — stockout, anomaly, what-if, and NL→SQL.

Pure-compute modules ported from the pre-revamp HUFT app and adapted to this
project's data model. Each function takes DataFrames (dependency injection) so
both the FastAPI backend (backend/api/routes) and the MCP agent tools
(mcp_server/server.py) can reuse the same logic without cross-layer coupling.
"""
