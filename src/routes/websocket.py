# WebSocket handler - 从 main.py 拆分出来
# 职责：WebSocket 连接管理 + 消息处理

import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket 聊天接口 — 所有会话共用 AgentSingleton
    
    消息格式（前端 → 后端）:
      {"type": "chat", "content": "用户输入"}
      {"type": "init", "content": ""}  # 初始化 session
      {"type": "reset", "content": ""}  # 重置会话
    
    消息格式（后端 → 前端）:
      {"type": "text", "content": "AI 回复片段"}
      {"type": "complete", "content": "...", "intent_mode": "normal", ...}
      {"type": "error", "content": "错误信息"}
      {"type": "reset_complete"}
    """
    from src.core.session_manager import get_session_manager, get_agent_singleton, build_langgraph_config

    await websocket.accept()

    sm = get_session_manager()
    sm.get_session(session_id)  # 确保 session 存在

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "chat":
                query = data["content"]
                logger.info(f"[WS] session={session_id} msg={query[:50]}...")

                history = sm.get_history(session_id)
                agent_input = {"input_text": query, "history": history}
                config = build_langgraph_config(session_id)

                try:
                    agent = get_agent_singleton().agent
                    result = agent.invoke(agent_input, config)

                    new_history = result.get("history", history)
                    sm.update_history(session_id, new_history)

                    await websocket.send_json({
                        "type": "text",
                        "content": result.get("response", "")
                    })
                    await websocket.send_json({
                        "type": "complete",
                        "content": result.get("response", ""),
                        "intent_mode": result.get("intent_mode", "normal"),
                        "mock_interview_mode": result.get("mock_interview_mode", False),
                        "current_round": result.get("current_round", 0)
                    })

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"[WS] 执行失败: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "content": str(e)
                    })

            elif data["type"] == "reset":
                sm.update_history(session_id, [])
                await websocket.send_json({"type": "reset_complete"})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        sm.cleanup(session_id)