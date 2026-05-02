# Core session management - SessionManager + AgentSingleton
# 从 main.py 拆分出来，按职责单一化

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)


class SessionContext:
    """单个会话的上下文"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.tool_count = 0

    def touch(self):
        self.last_active = datetime.now()


class AgentSingleton:
    """单例 Agent — 进程启动时编译一次，所有会话共享"""

    _instance: Optional["AgentSingleton"] = None
    _agent = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """初始化（只在首次调用时执行一次）"""
        if self._agent is None:
            logger.info("[AgentSingleton] 开始编译 LangGraph...")
            from src.multi_agent import get_singleton_agent
            self._agent = get_singleton_agent()
            logger.info("[AgentSingleton] ✅ LangGraph 编译完成")

    @property
    def agent(self):
        if self._agent is None:
            self.initialize()
        return self._agent


class SessionManager:
    """单例 SessionManager — 统一管理所有会话的生命周期"""

    _instance: Optional["SessionManager"] = None
    _sessions: Dict[str, SessionContext] = {}
    _agent_singleton: AgentSingleton

    def __new__(cls, agent_singleton: AgentSingleton = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
            if agent_singleton is None:
                agent_singleton = AgentSingleton()
                agent_singleton.initialize()
            cls._instance._agent_singleton = agent_singleton
        return cls._instance

    def get_session(self, session_id: str) -> SessionContext:
        """获取或创建会话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id)
            logger.info(f"[SessionManager] 创建新会话: {session_id}")
        ctx = self._sessions[session_id]
        ctx.touch()
        return ctx

    def get_history(self, session_id: str) -> list:
        """获取会话历史"""
        return self.get_session(session_id).history

    def update_history(self, session_id: str, history: list):
        """更新会话历史"""
        self.get_session(session_id).history = history

    def cleanup(self, session_id: str):
        """销毁会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"[SessionManager] 销毁会话: {session_id}")

    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """列出所有活跃会话"""
        result = {}
        for sid, ctx in self._sessions.items():
            result[sid] = {
                "session_id": sid,
                "created_at": ctx.created_at.isoformat(),
                "last_active": ctx.last_active.isoformat(),
                "tool_count": ctx.tool_count,
            }
        return result


def build_langgraph_config(session_id: str) -> dict:
    """构建 LangGraph invoke 所需的 config（绑定 thread=session_id）"""
    return {
        "configurable": {
            "thread_id": session_id
        },
        "recursion_limit": 100
    }


# ── Global singletons (lazy init) ───────────────────────────────────────────────

_agent_singleton: Optional[AgentSingleton] = None
_session_manager: Optional[SessionManager] = None


def get_agent_singleton() -> AgentSingleton:
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = AgentSingleton()
        _agent_singleton.initialize()
    return _agent_singleton


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(get_agent_singleton())
    return _session_manager