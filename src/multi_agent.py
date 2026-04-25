# LangChain AI Stack - 单例 Agent + SessionManager 架构
# 职责：图编排（只负责 build_graph + get_singleton_agent）
# 所有节点逻辑已拆分到 src/nodes/ 和 src/core/

import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Import nodes from split modules ───────────────────────────────────────────

from src.nodes.preprocessing import (
    pre_router, ocr_processing, document_parsing,
    process_speech_to_text,
)
from src.nodes.generation import (
    optimize_transcript, rag_processing, web_search,
    generate_response, behavior_detection,
)
from src.nodes.routing import (
    intent_recognition, agent_router, check_rag_result, _get_intent_mode,
)
from src.nodes.career_intents import (
    mock_interview, interview_review, career_planning,
)
from src.core.state import AgentState
from src.skill_loader import get_skill_loader, get_skill


# ── Singleton agent management ─────────────────────────────────────────────────

_singleton_agent = None


def get_singleton_agent():
    """获取全局单例 Agent（线程安全，进程级单例）"""
    global _singleton_agent
    if _singleton_agent is None:
        _singleton_agent = create_multi_agent()
    return _singleton_agent


def create_multi_agent():
    """
    创建统一的 LangGraph 智能体，按照流程图编排节点

    节点列表（15个）：
      pre_router / ocr_processing / document_parsing / behavior_detection /
      process_speech_to_text / optimize_transcript / intent_recognition /
      agent_router / rag_processing / web_search / generate_response /
      mock_interview / interview_review / career_planning
    """
    workflow = StateGraph(AgentState)

    # ── Register all nodes ──────────────────────────────────────────────────
    workflow.add_node("pre_router", pre_router)
    workflow.add_node("ocr_processing", ocr_processing)
    workflow.add_node("document_parsing", document_parsing)
    workflow.add_node("behavior_detection", behavior_detection)
    workflow.add_node("process_speech_to_text", process_speech_to_text)
    workflow.add_node("optimize_transcript", optimize_transcript)
    workflow.add_node("intent_recognition", intent_recognition)
    workflow.add_node("agent_router", agent_router)
    workflow.add_node("check_rag_result", check_rag_result)
    workflow.add_node("rag_processing", rag_processing)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("mock_interview", mock_interview)
    workflow.add_node("interview_review", interview_review)
    workflow.add_node("career_planning", career_planning)

    workflow.set_entry_point("pre_router")

    # ── Pre-router: file type routing ──────────────────────────────────────
    def pre_route_decision(state: Dict[str, Any]) -> str:
        pre_route = state.get("pre_route", "text_input")
        mapping = {
            "file_ocr": "ocr_processing",
            "file_document": "document_parsing",
            "video_input": "behavior_detection",
        }
        return mapping.get(pre_route, "process_speech_to_text")

    workflow.add_conditional_edges(
        "pre_router",
        pre_route_decision,
        {
            "ocr_processing": "ocr_processing",
            "document_parsing": "document_parsing",
            "behavior_detection": "behavior_detection",
            "process_speech_to_text": "process_speech_to_text"
        }
    )

    # ── Post-file-processing: always go to optimize ─────────────────────────
    workflow.add_edge("ocr_processing", "optimize_transcript")
    workflow.add_edge("document_parsing", "optimize_transcript")
    workflow.add_edge("behavior_detection", "generate_response")
    workflow.add_edge("process_speech_to_text", "optimize_transcript")

    # ── Intent recognition + routing ─────────────────────────────────────────
    workflow.add_edge("optimize_transcript", "intent_recognition")
    workflow.add_edge("intent_recognition", "agent_router")

    # ── Agent router: route to appropriate node ─────────────────────────────
    def decide_next_node(state: Dict[str, Any]) -> str:
        """
        路由决策：通过 SkillLoader 动态匹配，或降级到默认路由

        SkillLoader 查询逻辑：
        - intent_mode 匹配 → career intent 节点（mock_interview 等）
        - question_type 匹配 → 基础路由节点
        """
        route = state.get("route_decision", "generate_response")
        intent_mode = state.get("intent_mode", "normal")
        intent = state.get("intent", {})
        question_type = intent.get("question_type", "")

        # ── 尝试 SkillLoader 动态匹配 ────────────────────────────────────
        loader = get_skill_loader()

        # 1. 优先：职业意图模式（intent_mode）
        if intent_mode not in ("normal", "", None):
            skill_id = loader.match(intent_mode=intent_mode)
            if skill_id:
                print(f"[SkillLoader] intent_mode={intent_mode} → skill={skill_id}")
                return skill_id

        # 2. 尝试：问题类型匹配（question_type → skill 映射）
        if question_type:
            skill_id = loader.match(question_type=question_type)
            if skill_id:
                print(f"[SkillLoader] question_type={question_type} → skill={skill_id}")
                return skill_id

        # 3. 降级：默认路由（route 来自 agent_router 的判断）
        return route

    workflow.add_conditional_edges(
        "agent_router",
        decide_next_node,
        {
            "rag_processing": "rag_processing",
            "web_search": "web_search",
            "generate_response": "generate_response",
            "mock_interview": "mock_interview",
            "interview_review": "interview_review",
            "career_planning": "career_planning",
        }
    )

    # ── RAG → check → generate or fallback to web search ────────────────────
    workflow.add_conditional_edges(
        "rag_processing",
        check_rag_result,
        {
            "has_content": "generate_response",
            "no_content": "web_search"
        }
    )

    # ── Web search → generate_response ─────────────────────────────────────
    workflow.add_edge("web_search", "generate_response")

    # ── Career intent nodes → output ────────────────────────────────────────
    workflow.add_edge("mock_interview", END)
    workflow.add_edge("interview_review", END)
    workflow.add_edge("career_planning", END)

    # ── generate_response → END ─────────────────────────────────────────────
    workflow.add_edge("generate_response", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    memory_dir = Path(__file__).parent.parent / "data"
    memory_dir.mkdir(exist_ok=True)
    db_path = str((memory_dir / "checkpoints.db").resolve())
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer, debug=False)