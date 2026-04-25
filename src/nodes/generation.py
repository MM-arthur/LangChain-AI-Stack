# Generation node: generate_response (handles behavior analysis + normal RAG/web responses)

from typing import Dict, Any
import json
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import init_llm
from src.core.retry import with_retry


def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 统一生成最终回复
    根据上下文类型（行为分析 / RAG / 网页搜索）生成对应回复
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))
    rag_result = state.get("rag_result", "")
    rag_sources = state.get("rag_sources", [])
    web_search_result = state.get("web_search_result", "")
    web_sources = state.get("web_sources", [])
    behavior_result = state.get("behavior_result", {})

    # ── 行为分析回复 ────────────────────────────────────────────────────────
    if behavior_result and behavior_result.get("success", False):
        return _generate_behavior_response(state, behavior_result)

    # ── 普通回复（RAG / 网页搜索）───────────────────────────────────────────
    return _generate_normal_response(state)


def _build_sources_str(sources: list, max_count: int = 3) -> str:
    """构建来源字符串"""
    if not sources:
        return "（无外部数据来源）"
    return "\n".join([f"- {s}" for s in sources[:max_count]])


@with_retry(max_retries=3)
def _call_llm_chain(chain, input_dict):
    """带重试的 LLM 调用"""
    return chain.invoke(input_dict)


def _generate_behavior_response(state: Dict[str, Any], behavior_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成面试官行为分析反馈
    """
    llm = init_llm()
    posture = behavior_result.get("posture", {})
    expression = behavior_result.get("expression", {})
    gaze = behavior_result.get("gaze", {})
    attention = behavior_result.get("attention", {})
    warnings = behavior_result.get("warnings", [])

    behavior_desc = f"""【面试官行为分析】
- 坐姿状态：{posture.get('state', 'unknown')}（置信度: {posture.get('confidence', 0):.0%}）
- 面部表情：{expression.get('state', 'unknown')}（置信度: {expression.get('confidence', 0):.0%}）
- 视线方向：{gaze.get('direction', 'unknown')}（置信度: {gaze.get('confidence', 0):.0%}）
- 注意力水平：{attention.get('level', 'unknown')}（得分: {attention.get('score', 0):.0%}）
"""
    if warnings:
        behavior_desc += f"- 警告信息：{'；'.join(warnings)}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的面试观察助手，帮助面试者分析面试官的行为。

**你的身份**：
- 你在帮助面试者（用户）分析面试官的行为
- 用户不是面试官，而是来参加面试的候选人

**分析维度**：
1. 表情：面试官表情是否严肃/友善/中性？是否在思考？
2. 视线：面试官视线在哪里？是在看你、看屏幕、还是走神？
3. 姿势：面试官坐姿是否放松？身体前倾表示感兴趣？
4. 注意力：面试官是否专注？还是显得不耐烦？

**输出要求**：
1. 先给出一个总体判断（1-2句话）
2. 针对每个维度给出简短分析
3. 根据面试官的状态，给出应对建议
4. 保持积极正面的语气，帮助用户建立信心

**格式**：
【面试官状态】
...（整体评价）

【细节观察】
- 表情：...
- 视线：...
- 姿势：...
- 注意力：...

【应对建议】
...（如何调整自己的回答策略）"""),
        ("human", "{behavior_analysis}")
    ])

    chain = prompt | llm

    try:
        response = _call_llm_chain(chain, {"behavior_analysis": behavior_desc})
        final_response = response.content

        print(f"✅ 行为分析回复生成完成")

        return {
            **state,
            "response": final_response,
            "history": state.get("history", []) + [
                {"role": "user", "content": "分析面试官行为"},
                {"role": "assistant", "content": final_response}
            ]
        }
    except Exception as e:
        print(f"❌ _generate_behavior_response 失败: {str(e)}")
        return {
            **state,
            "response": f"抱歉，生成回复时出现错误: {str(e)}",
            "history": state.get("history", [])
        }


def _generate_normal_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成普通回复（RAG / 网页搜索）
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))
    rag_result = state.get("rag_result", "")
    rag_sources = state.get("rag_sources", [])
    web_search_result = state.get("web_search_result", "")
    web_sources = state.get("web_sources", [])

    # 决定上下文来源
    context = ""
    sources_str = ""
    if rag_result:
        context = f"【本地知识库检索内容】\n{rag_result}"
        sources_str = _build_sources_str(rag_sources)
    elif web_search_result:
        context = f"【网页搜索内容】\n{web_search_result}"
        sources_str = _build_sources_str(web_sources)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是Arthur的个人面试助手。

**核心身份**：
- 你是Arthur的专属面试助手，代表Arthur回答面试官的问题
- 当面试官问"你是谁"或"介绍一下你自己"时，你应该以Arthur的身份回答

**回答要求**：
1. 基于检索内容回答问题，如无检索内容则根据自身知识回答
2. 如果有检索内容，在回答末尾标注数据来源
3. 保持温和、专业的态度，像在面试中回答问题一样
4. 回答要简洁有力，突出重点
5. 展现你的专业能力和思考深度

**回答格式**：
[回答内容]

---
📚 数据来源：
[来源列表]"""),
        ("human", """问题：{question}

{context}

数据来源：
{sources}""")
    ])

    chain = prompt | llm

    try:
        response = _call_llm_chain(chain, {
            "question": optimized_text,
            "context": context if context else "（无检索内容，请根据自身知识回答）",
            "sources": sources_str
        })
        final_response = response.content

        print(f"✅ 生成回复完成")

        return {
            **state,
            "response": final_response,
            "history": state.get("history", []) + [
                {"role": "user", "content": optimized_text},
                {"role": "assistant", "content": final_response}
            ],
            "mock_interview_mode": state.get("mock_interview_mode", False),
            "current_round": state.get("current_round", 0)
        }

    except Exception as e:
        print(f"❌ _generate_normal_response 失败: {str(e)}")
        return {
            **state,
            "response": f"抱歉，生成回复时出现错误: {str(e)}",
            "history": state.get("history", []) + [
                {"role": "user", "content": optimized_text},
                {"role": "assistant", "content": f"抱歉，生成回复时出现错误: {str(e)}"}
            ]
        }


# ── Behavior Detection (Local Model Node) ────────────────────────────────────

def behavior_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🩵 Local Node - YOLO 分析面试官行为
    """
    video_frame_data = state.get("video_frame_data", "")

    if not video_frame_data:
        return {
            "behavior_result": {"success": False, "error": "No video frame data"},
        }

    try:
        from src.behavior_detection.behavior_analyzer import BehaviorAnalyzer
        analyzer = BehaviorAnalyzer()
        result = analyzer.analyze_frame(video_frame_data)

        return {
            **state,
            "behavior_result": result
        }
    except Exception as e:
        print(f"❌ behavior_detection 失败: {str(e)}")
        return {
            **state,
            "behavior_result": {"success": False, "error": str(e)}
        }


# ── RAG Processing ─────────────────────────────────────────────────────────────

def rag_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - RAG 检索本地知识库
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))

    try:
        from src.rag.RAG import PersonalKnowledgeRAG
        rag = PersonalKnowledgeRAG()
        query_result = rag.query(optimized_text, top_k=5)

        rag_result = query_result.get("results", "")
        rag_sources = query_result.get("sources", [])

        print(f"✅ RAG 检索完成，命中 {len(rag_sources)} 条")

        return {
            **state,
            "rag_result": rag_result,
            "rag_sources": rag_sources
        }
    except Exception as e:
        print(f"❌ rag_processing 失败: {str(e)}")
        return {
            **state,
            "rag_result": "",
            "rag_sources": []
        }


# ── Web Search ─────────────────────────────────────────────────────────────────

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - ReAct Agent 网页搜索
    通过 MCP stdio 调用 tavily_search 工具，而非直接调用 SDK
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))

    # 加载 MCP Tavily 工具
    try:
        from src.mcp_client import get_mcp_tool_loader
        loader = get_mcp_tool_loader()
        tools = loader.load_tools(["tavily_search", "tavily_sources"])

        if not tools:
            return {
                **state,
                "web_search_result": "MCP 工具加载失败，请检查 langchain-mcp-adapters 是否安装",
                "web_sources": []
            }

        tavily_search_tool = next((t for t in tools if t.name == "tavily_search"), None)
        tavily_sources_tool = next((t for t in tools if t.name == "tavily_sources"), None)

        if not tavily_search_tool:
            return {
                **state,
                "web_search_result": "未找到 tavily_search 工具",
                "web_sources": []
            }

        # 通过 MCP 工具调用搜索
        search_result = tavily_search_tool.invoke({"query": optimized_text, "max_results": 5})

        # 获取来源
        sources = ""
        if tavily_sources_tool:
            try:
                sources = tavily_sources_tool.invoke({"query": optimized_text})
            except Exception:
                sources = ""

        # 解析搜索结果
        import json
        try:
            results_data = json.loads(search_result) if isinstance(search_result, str) else search_result
            if isinstance(results_data, list):
                formatted = []
                for r in results_data:
                    formatted.append(f"【{r.get('title', '未知')}】{r.get('content', '')[:200]}...\n来源: {r.get('url', '')}")
                web_search_result = "\n\n".join(formatted)
                web_sources = [r.get('url', '') for r in results_data[:3]]
            else:
                web_search_result = str(results_data)
                web_sources = []
        except (json.JSONDecodeError, TypeError):
            web_search_result = str(search_result)
            web_sources = []

        print(f"✅ MCP web_search 完成，命中 {len(web_sources)} 条来源")

        return {
            **state,
            "web_search_result": web_search_result,
            "web_sources": web_sources or []
        }

    except Exception as e:
        print(f"❌ web_search MCP 调用失败: {str(e)}")
        # 降级：尝试直接用 SDK
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if api_key:
                client = TavilyClient(api_key=api_key)
                results = client.search(query=optimized_text, max_results=5)
                formatted = [f"【{r.get('title', '')}】{r.get('content', '')[:200]}...\n来源: {r.get('url', '')}" for r in results.get("results", [])]
                return {
                    **state,
                    "web_search_result": "\n\n".join(formatted),
                    "web_sources": [r.get("url", "") for r in results.get("results", [])[:3]]
                }
        except Exception:
            pass

        return {
            **state,
            "web_search_result": f"搜索失败: {str(e)}",
            "web_sources": []
        }


# ── Optimize Transcript ────────────────────────────────────────────────────────

def optimize_transcript(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - LLM 润色/规范化文本
    """
    llm = init_llm()
    transcript = state.get("transcript", "")

    if not transcript:
        return {**state, "optimized_text": ""}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个文本规范化助手。将用户输入润色、纠错、标准化。

**要求**：
1. 修正明显的语音识别错误
2. 去除冗余语气词（嗯、啊、这个那个）
3. 保持原意，简洁表达
4. 如果是英文或代码，保持原样

**输出**：直接输出处理后的文本，不要加解释"""),
        ("human", "{transcript}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({"transcript": transcript})
        return {
            **state,
            "optimized_text": response.content.strip()
        }
    except Exception as e:
        print(f"❌ optimize_transcript 失败: {str(e)}")
        return {**state, "optimized_text": transcript}