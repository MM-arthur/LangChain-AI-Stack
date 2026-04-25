# Skill: web_search

## 概述
网页搜索技能。当用户问题属于"最新知识"类型，或 RAG 无结果时降级触发。

## 触发条件
```
question_type: 最新知识
fallback_for: rag_processing (当 RAG 无结果时)
```

## 实现
```
type: mcp
entry: src.mcp_client.get_tavily_search_tool
```

## 描述
通过 Tavily MCP 工具搜索网页，获取最新信息，返回给 LLM 生成回答。

## 输入
- `optimized_text`: 用户问题，作为搜索关键词

## 输出
- `web_search_result`: 搜索结果摘要
- `web_sources`: 来源链接列表

## 配置
```
version: "1.0"
mcp_tool: tavily_search
max_results: 5
fallback: true
```

## 关联
- 依赖 MCP: tavily_search
- 触发 question_type: 最新知识