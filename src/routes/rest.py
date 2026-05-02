# REST API routes - 从 main.py 拆分出来
# 所有 /api/* 端点

import os
import json
import base64
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, WebSocket
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── MCP Config ─────────────────────────────────────────────────────────────────

CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "mcp_config.json"
SPEECH_ENGINE = os.getenv("SPEECH_ENGINE", "sensevoice")

# ── Lazy imports（避免循环依赖）────────────────────────────────────────────────

_speech_service = None


def get_speech_service():
    global _speech_service
    if _speech_service is None:
        if SPEECH_ENGINE == "sensevoice":
            from src.speech_recognition.sensevoice import WhisperService
            _speech_service = WhisperService()
        else:
            from src.speech_recognition.speech_to_text import SpeechToTextService
            _speech_service = SpeechToTextService()
    return _speech_service


def get_session_manager():
    from src.core.session_manager import get_session_manager
    return get_session_manager()


def get_agent_singleton():
    from src.core.session_manager import get_agent_singleton
    return get_agent_singleton()


def build_langgraph_config(session_id: str) -> dict:
    from src.core.session_manager import build_langgraph_config as _build
    return _build(session_id)


# ── MCP Config Loader ─────────────────────────────────────────────────────────

def load_config_from_json():
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["src/mcp_server/mcp_server_time.py"],
            "transport": "stdio"
        }
    }

    try:
        if CONFIG_FILE_PATH.exists():
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


# ── MCP Tools ────────────────────────────────────────────────────────────────

@router.get("/api/config")
async def get_config():
    return load_config_from_json()


@router.get("/api/mcp/tools")
async def get_mcp_tools():
    try:
        from src.mcp_client import get_mcp_tool_loader
        loader = get_mcp_tool_loader()
        all_tools = loader.get_all_available_tools()
        return {
            "available_tools": all_tools,
            "description": "通过 MultiServerMCPClient 动态加载的 MCP 工具列表",
            "usage": "POST /api/mcp/load with [tools] to load specific tools"
        }
    except Exception as e:
        return {"error": str(e), "available_tools": []}


@router.post("/api/mcp/load")
async def load_mcp_tools_endpoint(body: dict = None):
    if body is None:
        body = {}
    tool_names = body.get("tools", [])
    if not tool_names:
        return {"error": "请指定要加载的工具名列表", "loaded": []}

    try:
        from src.mcp_client import get_mcp_tool_loader
        loader = get_mcp_tool_loader()
        tools = loader.load_tools(tool_names)
        return {"loaded": [t.name for t in tools], "count": len(tools)}
    except Exception as e:
        return {"error": str(e), "loaded": []}


# ── Session APIs ─────────────────────────────────────────────────────────────

@router.get("/api/models")
async def get_available_models():
    return {"models": ["moonshot-v1-8k", "API_custom_model"]}


@router.get("/api/sessions")
async def list_sessions():
    sm = get_session_manager()
    return {"sessions": sm.list_sessions()}


@router.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    sm = get_session_manager()
    if session_id not in sm._sessions:
        return {"initialized": False}
    ctx = sm.get_session(session_id)
    return {
        "initialized": True,
        "session_id": session_id,
        "created_at": ctx.created_at.isoformat(),
        "last_active": ctx.last_active.isoformat(),
        "history_length": len(ctx.history),
    }


@router.post("/api/reset_conversation")
async def reset_conversation(conversation_id: str = Form(default="default")):
    sm = get_session_manager()
    sm.update_history(conversation_id, [])
    return {"status": "ok", "message": f"对话 {conversation_id} 已重置"}


@router.get("/api/health")
async def health_check():
    sm = get_session_manager()
    return {
        "status": "ok",
        "service": "LangChain-AI-Stack",
        "agent_singleton": "initialized",
        "active_sessions": len(sm._sessions)
    }


# ── Speech / Audio ───────────────────────────────────────────────────────────

@router.post("/api/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    try:
        speech_service = get_speech_service()
        transcript = await speech_service.convert_audio_to_text(audio)

        if "失败" in transcript or "无法识别" in transcript:
            return {"error": transcript}

        sm = get_session_manager()
        history = sm.get_history(conversation_id)

        config = build_langgraph_config(conversation_id)
        result = get_agent_singleton().agent.invoke(
            {"input_text": transcript, "history": history},
            config
        )

        sm.update_history(conversation_id, result.get("history", history))

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


@router.post("/api/speech_to_text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        speech_service = get_speech_service()
        transcript = await speech_service.convert_audio_to_text(audio)
        if "失败" in transcript or "无法识别" in transcript:
            return {"error": transcript}
        return {"transcript": transcript}
    except Exception as e:
        return {"error": f"语音转文字失败: {str(e)}"}


# ── Behavior Analysis ─────────────────────────────────────────────────────────

@router.post("/api/analyze_behavior")
async def analyze_behavior(
    frame: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    try:
        file_content = await frame.read()
        if len(file_content) == 0:
            return {"success": False, "error": "上传的文件为空"}

        frame_base64 = base64.b64encode(file_content).decode('utf-8')
        logger.info(f"收到视频帧，大小: {len(file_content)} bytes")

        sm = get_session_manager()
        sm.get_session(conversation_id)
        history = sm.get_history(conversation_id)

        config = build_langgraph_config(conversation_id)
        result = get_agent_singleton().agent.invoke(
            {
                "input_text": "分析面试官的行为和状态",
                "video_frame_data": frame_base64,
                "history": history,
                "pre_route": "video_input"
            },
            config
        )

        sm.update_history(conversation_id, result.get("history", []))

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


# ── File Upload / OCR ────────────────────────────────────────────────────────

@router.post("/api/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    temp_file_path = None

    try:
        file_suffix = Path(file.filename).suffix.lower()
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'webp',
                             '.pdf', '.xlsx', '.xls', '.docx', '.doc']
        if file_suffix not in allowed_extensions:
            return {"success": False, "error": f"不支持的文件类型: {file_suffix}"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        logger.info(f"文件已上传: {file.filename}, 大小: {os.path.getsize(temp_file_path)} bytes")

        sm = get_session_manager()
        sm.get_session(conversation_id)
        history = sm.get_history(conversation_id)

        config = build_langgraph_config(conversation_id)
        result = get_agent_singleton().agent.invoke(
            {
                "input_text": f"请处理上传的文件: {file.filename}",
                "file_path": temp_file_path,
                "history": history
            },
            config
        )

        sm.update_history(conversation_id, result.get("history", history))

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


@router.post("/api/ocr")
async def ocr_process(
    file: UploadFile = File(...),
    enable_structure: bool = Form(default=False)
):
    # OCR 专用端点（不经过完整 agent）
    try:
        from src.ocr.ocr_service import OCRService
        service = OCRService()
        result = await service.process_image(file.file, enable_structure=enable_structure)
        return {"success": True, **result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"OCR 失败: {str(e)}"}