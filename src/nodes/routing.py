# Intent recognition + agent routing nodes

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import init_llm
from src.core.retry import with_retry


# ── Intent Recognition ─────────────────────────────────────────────────────────

def intent_recognition(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 识别问题类型（技术问题/个人问题/最新知识/开放性问题）
    + 识别职业意图模式（mock_interview / interview_review / career_planning）
    """
    llm = init_llm()

    transcript = state.get("transcript", state.get("input_text", ""))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个意图识别助手。请分析用户问题，判断其类型。

**问题类型（单选）**：
- 技术问题：对编程、框架、算法、系统设计的提问
- 个人问题：关于候选人的经历、项目、技能、学历的提问
- 最新知识：需要实时信息、新闻、数据的问题
- 开放性问题：闲聊、自我介绍、价值观类问题

**职业意图（可多选，主要识别一个）**：
- mock_interview：触发词"模拟面试"、"来面试"、"面试一下"、"开始面试"
- interview_review：触发词"复盘"、"面试表现怎么样"、"我今天面了"、"面试怎么样"
- career_planning：触发词"怎么发展"、"职业方向"、"要不要转"、"该学什么"
- normal_chat：以上都不是

**输出格式（JSON）**：
{
  "question_type": "技术问题|个人问题|最新知识|开放性问题",
  "intent_mode": "mock_interview|interview_review|career_planning|normal_chat",
  "execution_plan": "简短说明接下来的处理策略"
}"""),
        ("human", "{input_text}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({"input_text": transcript})
        import json
        content = response.content.strip()

        # Try to extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)
        print(f"✅ 意图识别完成: {result['question_type']} / {result['intent_mode']}")

        return {
            **state,
            "intent": {
                "question_type": result.get("question_type", "开放性问题"),
                "execution_plan": result.get("execution_plan", ""),
                "intent_mode": result.get("intent_mode", "normal_chat")
            },
            "intent_mode": result.get("intent_mode", "normal_chat")
        }
    except Exception as e:
        print(f"❌ intent_recognition 失败: {str(e)}")
        return {
            **state,
            "intent": {"question_type": "开放性问题", "execution_plan": "", "intent_mode": "normal_chat"},
            "intent_mode": "normal_chat"
        }


# ── Agent Router ──────────────────────────────────────────────────────────────

def agent_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 根据意图决定路由目标
    """
    intent = state.get("intent", {})
    question_type = intent.get("question_type", "开放性问题")
    intent_mode = state.get("intent_mode", "normal_chat")

    # 职业意图优先路由
    if intent_mode in ("mock_interview", "interview_review", "career_planning"):
        return {**state, "route_decision": intent_mode}

    # 普通问答路由
    if question_type == "技术问题":
        route = "rag_processing"
    elif question_type == "个人问题":
        route = "rag_processing"
    elif question_type == "最新知识":
        route = "web_search"
    else:
        route = "generate_response"

    return {**state, "route_decision": route}


# ── RAG Result Checker ─────────────────────────────────────────────────────────

def check_rag_result(state: Dict[str, Any]) -> str:
    """
    🟠 Condition Node - 检查 RAG 是否有结果
    """
    rag_result = state.get("rag_result", "")
    if rag_result and len(rag_result.strip()) > 10:
        return "has_content"
    return "no_content"


# ── Intent Mode Helper ─────────────────────────────────────────────────────────

def _get_intent_mode(state: Dict[str, Any]) -> str:
    """
    从 state 中提取 intent_mode 字段
    """
    return state.get("intent_mode", "normal")