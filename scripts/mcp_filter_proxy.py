"""Stdio MCP server that re-exposes only an allow-listed subset of AppWorld's MCP tools.

AppWorld tasks allow all 9 apps (~450 APIs, ~84k tokens of tool schemas), which no small-context
model can take up front. AppWorld's own tool-calling agents first predict <= 20 relevant APIs with
its API predictor; this proxy applies that list for external harnesses that talk MCP.

    ALLOWED_TOOLS=supervisor__complete_task,spotify__login \
        python mcp_filter_proxy.py <appworld mcp command...>

Written for the MCP Python SDK 2.x handler API (same pattern as AppWorld's own MCP server).
"""

import asyncio
import os
import sys

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server


async def main() -> None:
    allowed = {name for name in os.environ["ALLOWED_TOOLS"].split(",") if name}
    upstream = StdioServerParameters(command=sys.argv[1], args=sys.argv[2:], env=dict(os.environ))
    async with stdio_client(upstream) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        # Content-only results: drop output schemas so clients never expect structured output.
        tools = [t.model_copy(update={"output_schema": None})
                 for t in (await session.list_tools()).tools if t.name in allowed]

        async def list_tools(ctx, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=tools)

        async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
            if params.name not in allowed:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Tool {params.name} is not available.")],
                    is_error=True)
            result = await session.call_tool(params.name, params.arguments or {})
            return types.CallToolResult(content=result.content, is_error=result.is_error)

        server = Server("appworld", on_list_tools=list_tools, on_call_tool=call_tool)
        async with stdio_server() as (server_read, server_write):
            await server.run(server_read, server_write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
