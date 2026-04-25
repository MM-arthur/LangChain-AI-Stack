# MCP Client - 动态加载 MCP 工具（MultiServerMCPClient）

import json
import os
from typing import Dict, List, Any
from pathlib import Path

try:
    from langchain_mcp_adapters.clients import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


def get_config_path() -> Path:
    """获取 MCP 配置文件的路径"""
    return Path(__file__).parent.parent.parent / "mcp_config.json"


def load_mcp_config() -> Dict[str, Any]:
    """加载 MCP 配置文件"""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def filter_mcp_config(config: Dict[str, Any], include_tools: List[str]) -> Dict[str, Any]:
    """
    从完整配置中过滤出指定工具的配置。

    Args:
        config: 完整的 mcp_config.json 内容
        include_tools: 要加载的工具名列表

    Returns:
        只包含指定工具的配置字典
    """
    filtered = {}
    for key in include_tools:
        if key in config:
            filtered[key] = config[key]
    return filtered


class MCPToolLoader:
    """
    MCP 工具加载器 - 封装 MultiServerMCPClient 的加载逻辑

    使用方式：
        loader = MCPToolLoader()
        tools = loader.load_tools(["tavily_search", "tavily_sources", "get_current_time"])
    """

    def __init__(self):
        self._clients: Dict[str, Any] = {}
        self._tools_cache: Dict[str, List] = {}

    def load_tools(self, tool_names: List[str]) -> List[Any]:
        """
        加载指定的 MCP 工具

        Args:
            tool_names: 工具名列表，如 ["tavily_search", "get_current_time"]

        Returns:
            LangChain Tool 对象列表
        """
        if MultiServerMCPClient is None:
            print("[MCPToolLoader] langchain-mcp-adapters 未安装，无法加载 MCP 工具")
            return []

        # 加载配置并过滤
        full_config = load_mcp_config()
        filtered_config = filter_mcp_config(full_config, tool_names)

        if not filtered_config:
            print(f"[MCPToolLoader] 未找到以下工具的配置: {tool_names}")
            return []

        # 每个工具独立 client（避免跨工具状态污染）
        tools = []
        for tool_name in tool_names:
            if tool_name in self._tools_cache:
                tools.extend(self._tools_cache[tool_name])
                continue

            if tool_name not in filtered_config:
                continue

            try:
                client = MultiServerMCPClient({tool_name: filtered_config[tool_name]})
                fetched_tools = client.get_tools()
                self._clients[tool_name] = client
                self._tools_cache[tool_name] = fetched_tools
                tools.extend(fetched_tools)
                print(f"[MCPToolLoader] ✅ 加载工具: {tool_name} ({len(fetched_tools)} 个)")
            except Exception as e:
                print(f"[MCPToolLoader] ❌ 加载工具失败 {tool_name}: {e}")

        return tools

    def get_all_available_tools(self) -> List[str]:
        """
        返回所有已配置的可加载工具名
        """
        config = load_mcp_config()
        return list(config.keys())


# 全局单例（进程内共享）
_mcp_tool_loader: MCPToolLoader = None


def get_mcp_tool_loader() -> MCPToolLoader:
    global _mcp_tool_loader
    if _mcp_tool_loader is None:
        _mcp_tool_loader = MCPToolLoader()
    return _mcp_tool_loader