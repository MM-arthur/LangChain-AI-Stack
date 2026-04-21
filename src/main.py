import sys
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uuid
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver

from src.speech_recognition.speech_to_text import SpeechToTextService
from src.speech_recognition.sensevoice import WhisperService
from src.multi_agent import get_singleton_agent

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "null", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SessionManager + AgentSingleton
# 参考 Hermes 架构思想：单例 Agent + per-session 会话隔离
# ============================================================

# 全局变量 - 语音转文字服务
stt_service = None
sensevoice_service = None

# 语音识别引擎选择: "paddlespeech" / "sensevoice"
SPEECH_ENGINE = os.getenv("SPEECH_ENGINE", "sensevoice")

# MCP Config
CONFIG_FILE_PATH = "../mcp_config.json"

OUTPUT_TOKEN_INFO = {
    "moonshot-v1-8k": {"max_tokens": 8000},
}


class SessionContext:
    """单个会话的上下文"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.checkpointer = MemorySaver()
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

    def __new__(cls, agent_singleton: AgentSingleton):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
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

    def get_checkpointer(self, session_id: str):
        """获取会话的 checkpointer（用于 LangGraph thread）"""
        return self.get_session(session_id).checkpointer

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


# 初始化单例
_agent_singleton = AgentSingleton()
_agent_singleton.initialize()
_session_manager = SessionManager(_agent_singleton)

logger.info(f"[SessionManager] 初始化完成，当前会话数: 0")


# 辅助函数
def load_config_from_json():
    """加载 MCP 配置文件"""
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["src/mcp_server/mcp_server_time.py"],
            "transport": "stdio"
        }
    }

    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                if not config:
                    return default_config
                return config
        else:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return default_config


def get_speech_service():
    """获取语音识别服务实例"""
    global sensevoice_service, stt_service

    if SPEECH_ENGINE == "sensevoice":
        if sensevoice_service is None:
            sensevoice_service = WhisperService()
        return sensevoice_service
    else:
        if stt_service is None:
            stt_service = SpeechToTextService()
        return stt_service


def build_langgraph_config(session_id: str) -> dict:
    """构建 LangGraph invoke 所需的 config（绑定 thread=session_id）"""
    return {
        "configurable": {
            "thread_id": session_id
        },
        "recursion_limit": 100
    }


# ============================================================
# API 端点
# ============================================================

@app.get("/api/models")
async def get_available_models():
    """获取可用模型列表"""
    return {"models": ["moonshot-v1-8k", "API_custom_model"]}


@app.get("/api/sessions")
async def list_sessions():
    """列出所有活跃会话"""
    return {"sessions": _session_manager.list_sessions()}


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """获取指定会话信息"""
    if session_id not in _session_manager._sessions:
        return {"initialized": False}
    ctx = _session_manager.get_session(session_id)
    return {
        "initialized": True,
        "session_id": session_id,
        "created_at": ctx.created_at.isoformat(),
        "last_active": ctx.last_active.isoformat(),
        "history_length": len(ctx.history),
    }


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket 聊天接口 — 所有会话共用 AgentSingleton"""
    await websocket.accept()
    _session_manager.get_session(session_id)  # 确保 session 存在

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "chat":
                query = data["content"]
                logger.info(f"[WS] session={session_id} msg={query[:50]}...")

                history = _session_manager.get_history(session_id)

                agent_input = {
                    "input_text": query,
                    "history": history
                }

                config = build_langgraph_config(session_id)

                try:
                    result = _agent_singleton.agent.invoke(
                        agent_input,
                        config
                    )

                    new_history = result.get("history", history)
                    _session_manager.update_history(session_id, new_history)

                    await websocket.send_json({
                        "type": "text",
                        "content": result.get("response", "")
                    })
                    await websocket.send_json({
                        "type": "complete",
                        "content": result.get("response", "")
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
                _session_manager.update_history(session_id, [])
                await websocket.send_json({"type": "reset_complete"})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _session_manager.cleanup(session_id)


@app.get("/api/config")
async def get_config():
    """获取当前 MCP 配置"""
    return load_config_from_json()


@app.post("/api/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """处理音频文件，返回语音转文字 + AI 回复"""
    try:
        speech_service = get_speech_service()
        transcript = await speech_service.convert_audio_to_text(audio)

        if "失败" in transcript or "无法识别" in transcript:
            return {"error": transcript}

        history = _session_manager.get_history(conversation_id)

        agent_input = {
            "input_text": transcript,
            "history": history
        }

        config = build_langgraph_config(conversation_id)
        result = _agent_singleton.agent.invoke(agent_input, config)

        _session_manager.update_history(conversation_id, result.get("history", history))

        return {
            "transcript": transcript,
            "optimized_text": result.get("optimized_text", ""),
            "intent": result.get("intent", {}),
            "route_decision": result.get("route_decision", ""),
            "response": result.get("response", ""),
            "conversation_id": conversation_id
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"服务器处理失败: {str(e)}"}


@app.post("/api/speech_to_text")
async def speech_to_text(audio: UploadFile = File(...)):
    """仅处理语音转文字"""
    try:
        speech_service = get_speech_service()
        transcript = await speech_service.convert_audio_to_text(audio)
        if "失败" in transcript or "无法识别" in transcript:
            return {"error": transcript}
        return {"transcript": transcript}
    except Exception as e:
        return {"error": f"语音转文字失败: {str(e)}"}


@app.post("/api/reset_conversation")
async def reset_conversation(conversation_id: str = Form(default="default")):
    """重置对话历史"""
    _session_manager.update_history(conversation_id, [])
    return {"status": "ok", "message": f"对话 {conversation_id} 已重置"}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "LangChain-AI-Stack",
        "agent_singleton": "initialized",
        "active_sessions": len(_session_manager._sessions)
    }


@app.post("/api/analyze_behavior")
async def analyze_behavior(
    frame: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """分析面试官行为（从摄像头视频帧）"""
    import base64

    try:
        file_content = await frame.read()
        if len(file_content) == 0:
            return {"success": False, "error": "上传的文件为空"}

        frame_base64 = base64.b64encode(file_content).decode('utf-8')
        logger.info(f"收到视频帧，大小: {len(file_content)} bytes")

        _session_manager.get_session(conversation_id)
        history = _session_manager.get_history(conversation_id)

        agent_input = {
            "input_text": "分析面试官的行为和状态",
            "video_frame_data": frame_base64,
            "history": history,
            "pre_route": "video_input"
        }

        config = build_langgraph_config(conversation_id)
        result = _agent_singleton.agent.invoke(agent_input, config)

        _session_manager.update_history(conversation_id, result.get("history", []))

        return {
            "success": True,
            "behavior_result": result.get("behavior_result", {}),
            "response": result.get("response", ""),
            "conversation_id": conversation_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"行为分析失败: {str(e)}"}


@app.post("/api/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """上传文件并进行 OCR 或文档解析"""
    import tempfile
    import shutil
    from pathlib import Path

    temp_file_path = None

    try:
        file_suffix = Path(file.filename).suffix.lower()
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'webp',
                             '.pdf', '.xlsx', '.xls', '.docx', '.doc']
        if file_suffix not in allowed_extensions:
            return {
                "success": False,
                "error": f"不支持的文件类型: {file_suffix}"
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        logger.info(f"文件已上传: {file.filename}, 大小: {os.path.getsize(temp_file_path)} bytes")

        _session_manager.get_session(conversation_id)
        history = _session_manager.get_history(conversation_id)

        agent_input = {
            "input_text": f"请处理上传的文件: {file.filename}",
            "file_path": temp_file_path,
            "history": history
        }

        config = build_langgraph_config(conversation_id)
        result = _agent_singleton.agent.invoke(agent_input, config)

        _session_manager.update_history(conversation_id, result.get("history", history))

        return {
            "success": True,
            "filename": file.filename,
            "file_type": result.get("file_type", "unknown"),
            "extracted_text": result.get("document_content", ""),
            "ocr_result": result.get("ocr_result", {}),
            "response": result.get("response", ""),
            "conversation_id": conversation_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"文件处理失败: {str(e)}"}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@app.post("/api/ocr")
async def ocr_process(
    file: UploadFile = File(...),
    enable_structure: bool = Form(default=False)
):
    """仅进行 OCR 处理"""
    import tempfile
    import shutil
    from pathlib import Path

    temp_file_path = None

    try:
        from src.ocr.ocr_service import OCRService

        file_suffix = Path(file.filename).suffix.lower()
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf']
        if file_suffix not in allowed_extensions:
            return {"success": False, "error": f"不支持的文件类型: {file_suffix}"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        ocr_service = OCRService()
        result = ocr_service.process_file(temp_file_path, enable_structure=enable_structure)

        return {
            "success": result.get("success", False),
            "filename": file.filename,
            "text": result.get("text", ""),
            "text_lines": result.get("text_lines", []),
            "structure": result.get("structure", {}),
            "total_pages": result.get("total_pages", 1),
            "error": result.get("error", "")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"OCR处理失败: {str(e)}"}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@app.post("/api/parse_document")
async def parse_document(file: UploadFile = File(...)):
    """仅进行文档解析"""
    import tempfile
    import shutil
    from pathlib import Path

    temp_file_path = None

    try:
        from src.document_parser.document_parser_service import DocumentParserService

        file_suffix = Path(file.filename).suffix.lower()
        allowed_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.doc']
        if file_suffix not in allowed_extensions:
            return {"success": False, "error": f"不支持的文件类型: {file_suffix}"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        parser = DocumentParserService()
        result = parser.parse_document(temp_file_path)

        return {
            "success": result.get("success", False),
            "filename": file.filename,
            "file_type": result.get("file_type", "unknown"),
            "full_text": result.get("full_text", ""),
            "metadata": result.get("metadata", {}),
            "total_pages": result.get("total_pages", 0),
            "total_sheets": result.get("total_sheets", 0),
            "total_paragraphs": result.get("total_paragraphs", 0),
            "error": result.get("error", "")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"文档解析失败: {str(e)}"}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)