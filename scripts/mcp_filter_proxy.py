"""Stdio MCP server that re-exposes AppWorld's MCP tools to external harnesses.

AppWorld tasks allow all 9 apps (~450 APIs, ~84k tokens of tool schemas), which no small-context
model can take up front. AppWorld's own tool-calling agents first predict <= 20 relevant APIs with
its API predictor; this proxy exposes that list as direct tools.

With ONDEMAND_APIS=1 it also exposes three small meta-tools so the agent can reach *every* API the
task allows, the way code agents read API docs on demand (AppWorld's api_docs app):
  api_docs__show_api_descriptions(app_name)       -> names + descriptions of an app's APIs
  api_docs__show_api_doc(app_name, api_name)      -> one API's description and parameters
  call_api(app_name, api_name, arguments)         -> call any API of the task's apps

    ALLOWED_TOOLS=supervisor__complete_task,spotify__login ONDEMAND_APIS=1 \
        python mcp_filter_proxy.py <appworld mcp command...>

Written for the MCP Python SDK 2.x handler API (same pattern as AppWorld's own MCP server).
"""

import asyncio
import json
import os
import sys

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server

META_TOOLS = [
    types.Tool(
        name="api_docs__show_api_descriptions",
        description="List the APIs of an app (name and one-line description). Use it to find APIs "
                    "that are not in your tool list.",
        input_schema={"type": "object", "properties": {"app_name": {"type": "string"}},
                      "required": ["app_name"]}),
    types.Tool(
        name="api_docs__show_api_doc",
        description="Show the full documentation (description and parameters) of one API.",
        input_schema={"type": "object", "properties": {"app_name": {"type": "string"},
                                                       "api_name": {"type": "string"}},
                      "required": ["app_name", "api_name"]}),
    types.Tool(
        name="call_api",
        description="Call any API of any app, including ones not in your tool list. "
                    "`arguments` is an object with the API's parameters.",
        input_schema={"type": "object", "properties": {"app_name": {"type": "string"},
                                                       "api_name": {"type": "string"},
                                                       "arguments": {"type": "object"}},
                      "required": ["app_name", "api_name"]}),
]


def text_result(payload, is_error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=1)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)


async def main() -> None:
    allowed = {name for name in os.environ["ALLOWED_TOOLS"].split(",") if name}
    ondemand = os.environ.get("ONDEMAND_APIS") == "1"
    upstream = StdioServerParameters(command=sys.argv[1], args=sys.argv[2:], env=dict(os.environ))
    async with stdio_client(upstream) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        all_tools = {t.name: t for t in (await session.list_tools()).tools}
        # Content-only results: drop output schemas so clients never expect structured output.
        direct = [t.model_copy(update={"output_schema": None}) for name, t in all_tools.items() if name in allowed]
        exposed = direct + (META_TOOLS if ondemand else [])

        async def list_tools(ctx, params) -> types.ListToolsResult:
            return types.ListToolsResult(tools=exposed)

        async def forward(name: str, arguments: dict) -> types.CallToolResult:
            result = await session.call_tool(name, arguments)
            return types.CallToolResult(content=result.content, is_error=result.is_error)

        async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
            name, args = params.name, params.arguments or {}
            if name in allowed and name in all_tools:
                return await forward(name, args)
            if ondemand and name == "api_docs__show_api_descriptions":
                app = str(args.get("app_name", ""))
                apis = [{"name": n.split("__", 1)[1], "description": t.description}
                        for n, t in all_tools.items() if n.startswith(app + "__")]
                return text_result(apis or f"Unknown app: {app}", is_error=not apis)
            if ondemand and name == "api_docs__show_api_doc":
                full = f"{args.get('app_name', '')}__{args.get('api_name', '')}"
                if full not in all_tools:
                    return text_result(f"Unknown API: {full}", is_error=True)
                t = all_tools[full]
                return text_result({"app_name": args["app_name"], "api_name": args["api_name"],
                                    "description": t.description, "parameters": t.input_schema})
            if ondemand and name == "call_api":
                full = f"{args.get('app_name', '')}__{args.get('api_name', '')}"
                if full not in all_tools:
                    return text_result(f"Unknown API: {full}. Use api_docs__show_api_descriptions.", is_error=True)
                arguments = args.get("arguments") or {}
                if isinstance(arguments, str):  # some models send the object as a JSON string
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return text_result("`arguments` must be a JSON object.", is_error=True)
                return await forward(full, arguments)
            return text_result(f"Tool {name} is not available.", is_error=True)

        server = Server("appworld", on_list_tools=list_tools, on_call_tool=call_tool)
        async with stdio_server() as (server_read, server_write):
            await server.run(server_read, server_write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
