# Tavily Search MCP Server - 标准 MCP 协议工具（stdio 通信）

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os

mcp = FastMCP("TavilySearch")


@mcp.tool()
def search(query: str, max_results: int = 5) -> str:
    """
    使用 Tavily 进行网页搜索，获取最新信息。

    Args:
        query: 搜索关键词（中文/英文均可）
        max_results: 返回结果数量，默认5条

    Returns:
        JSON格式的搜索结果，包含标题、URL、内容摘要
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return '{"error": "TAVILY_API_KEY未配置，请在.env中设置TAVILY_API_KEY"}'

    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)

        formatted = []
        for r in results.get("results", []):
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:300]
            })
        import json
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as e:
        return f'{{"error": "搜索失败: {str(e)}"}}'


@mcp.tool()
def get_sources(query: str) -> str:
    """
    获取搜索来源链接列表。

    Args:
        query: 搜索关键词

    Returns:
        格式化的来源列表，每行一个
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY未配置，请在.env中设置TAVILY_API_KEY"

    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=3)
        lines = []
        for r in results.get("results", []):
            lines.append(f"- {r.get('title', '未知')}: {r.get('url', '')}")
        return "\n".join(lines) if lines else "未找到相关来源"
    except Exception as e:
        return f"获取来源失败: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")