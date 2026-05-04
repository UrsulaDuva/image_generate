"""MCP 桥（streamable_http）

使用 mcp 官方 SDK 连接远程 MCP server，把工具登记到统一的工具表里。
不缓存连接（每次工具调用都开短连接），保证简单可靠。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

# 服务器配置
MCP_SERVERS: Dict[str, Dict[str, str]] = {
    "bing-cn": {
        "url": "https://mcp.api-inference.modelscope.net/ef8a83e8e8a140/mcp",
    },
}


async def _list_tools(url: str) -> List[Dict[str, Any]]:
    """连一次 MCP server 拿工具元数据。"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema or {},
                })
            return tools


async def _call_tool(url: str, name: str, args: Dict[str, Any]) -> str:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args or {})
            # 拼接所有返回内容
            chunks = []
            for c in result.content:
                if hasattr(c, "text") and c.text:
                    chunks.append(c.text)
                elif hasattr(c, "data"):
                    chunks.append(str(c.data)[:2000])
            text = "\n".join(chunks).strip()
            if result.isError:
                return f"[MCP 工具错误] {text[:1000]}"
            return text or "[MCP 工具无内容返回]"


# ---------- 异步入口（在 FastAPI 事件循环中直接 await） ----------

async def list_all_mcp_tools_async() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for server_id, cfg in MCP_SERVERS.items():
        try:
            tools = await _list_tools(cfg["url"])
            for t in tools:
                t["_server"] = server_id
                out.append(t)
        except Exception as e:
            import traceback
            print(f"[mcp_bridge] {server_id} 列出工具失败: {e}")
            traceback.print_exc()
    return out


async def call_mcp_tool_async(server_id: str, name: str, args: Dict[str, Any]) -> str:
    cfg = MCP_SERVERS.get(server_id)
    if not cfg:
        return f"[MCP] 未知 server: {server_id}"
    try:
        return await _call_tool(cfg["url"], name, args)
    except Exception as e:
        return f"[MCP] 调用 {server_id}.{name} 失败: {e}"


# ---------- 同步包装（独立脚本/CLI 调试用） ----------

def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def list_all_mcp_tools() -> List[Dict[str, Any]]:
    return _run_async(list_all_mcp_tools_async())


def call_mcp_tool(server_id: str, name: str, args: Dict[str, Any]) -> str:
    return _run_async(call_mcp_tool_async(server_id, name, args))
