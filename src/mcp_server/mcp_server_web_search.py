"""
MCP Web Search Server - 网页搜索工具
使用Tavily API进行网页搜索，供LLM调用
"""

import os
import json
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

mcp = FastMCP("web_search")

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    使用Tavily API进行网页搜索，获取最新信息
    
    Args:
        query: 搜索关键词或问题
        max_results: 返回结果数量，默认5条
    
    Returns:
        搜索结果的JSON字符串，包含标题、链接、内容摘要
    """
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return json.dumps({
                "success": False,
                "error": "TAVILY_API_KEY未配置，请在.env文件中设置"
            }, ensure_ascii=False)
        
        client = TavilyClient(api_key=api_key)
        
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )
        
        results = response.get("results", [])
        
        if not results:
            return json.dumps({
                "success": False,
                "error": "未找到相关搜索结果"
            }, ensure_ascii=False)
        
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append({
                "index": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0)
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }, ensure_ascii=False, indent=2)
        
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "tavily-python未安装，请运行: pip install tavily-python"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"搜索失败: {str(e)}"
        }, ensure_ascii=False)

@mcp.tool()
def get_search_sources(query: str, max_results: int = 3) -> str:
    """
    获取搜索结果的来源链接列表
    
    Args:
        query: 搜索关键词
        max_results: 返回结果数量，默认3条
    
    Returns:
        来源链接列表的JSON字符串
    """
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return json.dumps({
                "success": False,
                "error": "TAVILY_API_KEY未配置"
            }, ensure_ascii=False)
        
        client = TavilyClient(api_key=api_key)
        
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results
        )
        
        results = response.get("results", [])
        
        sources = []
        for r in results:
            sources.append({
                "title": r.get("title", ""),
                "url": r.get("url", "")
            })
        
        return json.dumps({
            "success": True,
            "sources": sources
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
