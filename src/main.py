import sys
import os
import logging

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import uuid
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


from src.speech_recognition.speech_to_text import SpeechToTextService
from src.speech_recognition.sensevoice import WhisperService

from src.multi_agent import create_multi_agent

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

# 全局变量存储会话
sessions: Dict[str, Dict[str, Any]] = {}

# 全局变量 - 语音转文字服务
stt_service = None
sensevoice_service = None

# 语音识别引擎选择: "paddlespeech" / "sensevoice"
SPEECH_ENGINE = os.getenv("SPEECH_ENGINE", "sensevoice")

# 全局变量 - 对话历史
conversation_history = {}

# MCP Config
CONFIG_FILE_PATH = "../mcp_config.json"

OUTPUT_TOKEN_INFO = {
    "moonshot-v1-8k": {"max_tokens": 8000},
}

class SessionSettings(BaseModel):
    model: str = "moonshot-v1-8k"
    timeout_seconds: int = 120
    recursion_limit: int = 100
    session_id: str

# 辅助函数
def load_config_from_json():
    """加载配置文件"""
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
                # 如果配置为空或不包含任何服务，则使用默认配置
                if not config:
                    return default_config
                return config
        else:
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return default_config

def save_config_to_json(config):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False


async def cleanup_mcp_client(session_id: str):
    """清理MCP客户端"""
    # 在新版本中，不再需要显式清理客户端
    # 客户端会在不再被引用时自动清理
    if session_id in sessions and sessions[session_id].get("mcp_client"):
        # 清理逻辑（如果需要）
        sessions[session_id]["mcp_client"] = None

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

async def initialize_agent(session_id: str, model_name: str, mcp_config: Dict[str, Any]):
    """初始化AI代理"""
    # 清理旧客户端
    await cleanup_mcp_client(session_id)
    
    # 初始化MCP客户端
    print(f"[DEBUG] 初始化MCP客户端，配置: {mcp_config}")
    client = MultiServerMCPClient(mcp_config)

    try:
        tools = await client.get_tools()
        print(f"[DEBUG] 成功获取 {len(tools)} 个工具")
    except Exception as e:
        print(f"[ERROR] 获取MCP工具时出错: {e}")
        import traceback
        traceback.print_exc()
        # 即使工具获取失败，也继续初始化代理（但不带工具）
        tools = []
    
    # 初始化模型 - 只使用MoonshotChat和CustomAPIModel，避免依赖问题
    if model_name.startswith("moonshot-"):
        model = MoonshotChat(
            model=model_name,
            temperature=0.1,
            max_tokens=OUTPUT_TOKEN_INFO.get(model_name, {}).get("max_tokens", 8000),
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    elif model_name == "API_custom_model":
        from custom_api_llm.model import CustomAPIModel
        model = CustomAPIModel(
            model_name="model",  # 根据需要调整
            username=os.environ.get("API_custom_model_username"),
            password=os.environ.get("API_custom_model_password"),
            api_base=os.environ.get("API_custom_model_url"),
        )
    else:
        # 默认使用Moonshot模型
        model = MoonshotChat(
            model="moonshot-v1-8k",
            temperature=0.1,
            max_tokens=8000,
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    
    
    # 创建StateGraph
    workflow = StateGraph(dict)
    
    # LLM节点
    def llm_node(state):
        messages = state.get("messages", [])
        # 添加系统提示词
        if not messages or not isinstance(messages[0], AIMessage):
            # 为了简单起见，我们不使用系统提示词，直接让模型处理
            pass
        response = model.invoke(messages)
        return {"messages": messages + [response]}
    
    # 工具调用节点
    def tool_node(state):
        messages = state.get("messages", [])
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        
        # 执行工具调用
        tool_results = []
        for tool_call in tool_calls:
            # 查找工具
            tool_name = getattr(tool_call, "name", None) or tool_call.get("name")
            tool_args = getattr(tool_call, "args", None) or tool_call.get("args")
            
            # 执行工具
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        tool_results.append({
                            "tool_call": tool_call,
                            "output": result
                        })
                    except Exception as e:
                        tool_results.append({
                            "tool_call": tool_call,
                            "output": f"Error: {str(e)}"
                        })
                    break
        
        # 构建工具响应消息
        tool_messages = []
        for result in tool_results:
            tool_call = result["tool_call"]
            tool_message = ToolMessage(
                content=str(result["output"]),
                tool_call_id=getattr(tool_call, "id", None) or str(uuid.uuid4())
            )
            tool_messages.append(tool_message)
        
        return {"messages": messages + tool_messages}
    
    # 条件边 - 决定是使用工具还是直接返回结果
    def should_continue(state):
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    # 添加节点
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    
    # 添加边
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", should_continue, {
        "tools": "tools",
        END: END
    })
    workflow.add_edge("tools", "llm")
    
    # 编译Agent
    agent = workflow.compile(checkpointer=MemorySaver())
    
    # 保存到会话
    if session_id not in sessions:
        sessions[session_id] = {}
    
    sessions[session_id].update({
        "agent": agent,
        "mcp_client": client,
        "tool_count": len(tools),
        "model": model_name,
        "thread_id": str(uuid.uuid4()),
        "history": []
    })
    
    return {"tool_count": len(tools), "status": "initialized"}

# API端点
@app.get("/api/models")
async def get_available_models():
    """获取可用模型列表"""
    available_models = ["moonshot-v1-8k", "API_custom_model"]
    return {"models": available_models}

@app.post("/api/initialize")
async def initialize_session(settings: SessionSettings):
    """初始化会话"""
    try:
        config = load_config_from_json()
        result = await initialize_agent(settings.session_id, settings.model, config)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}









@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天接口"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                query = data["content"]
                
                print(f"[DEBUG] 收到消息: {query}")
                
                # 获取或初始化对话历史
                history = conversation_history.get(session_id, [])
                
                # 调用统一的LangGraph Agent处理打字输入
                agent = create_multi_agent()
                
                # 构建输入
                agent_input = {
                    "input_text": query,
                    "history": history
                }
                
                # 调用Agent
                try:
                    result = agent.invoke(agent_input)
                    
                    # 更新对话历史
                    conversation_history[session_id] = result["history"]
                    
                    # 向前端发送回复
                    await websocket.send_json({
                        "type": "text",
                        "content": result["response"]
                    })
                    
                    # 向前端发送完成信号
                    await websocket.send_json({
                        "type": "complete",
                        "content": result["response"]
                    })
                    
                    print(f"[DEBUG] 查询完成: {query}")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ERROR] 查询执行失败: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "content": str(e)
                    })
            
            elif data["type"] == "reset":
                if session_id in conversation_history:
                    del conversation_history[session_id]
                await websocket.send_json({
                    "type": "reset_complete"
                })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket error: {e}")
    finally:
        await cleanup_mcp_client(session_id)




@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    return load_config_from_json()

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    if session_id not in sessions:
        return {"initialized": False}
    
    session = sessions[session_id]
    return {
        "initialized": True,
        "tool_count": session.get("tool_count", 0),
        "model": session.get("model", ""),
        "history": session.get("history", [])
    }

# 语音转文字API端点
@app.post("/api/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """
    处理音频文件，返回语音转文字结果和AI回复
    """
    try:
        # 1. 语音转文字
        speech_service = get_speech_service()

        # 修复：确保audio对象被正确处理
        transcript = await speech_service.convert_audio_to_text(audio)
        
        if "失败" in transcript or "无法识别" in transcript:
            return {"error": transcript}
        
        # 2. 调用统一的LangGraph Agent处理
        agent = create_multi_agent()
        
        # 获取对话历史
        history = conversation_history.get(conversation_id, [])
        
        # 构建输入
        agent_input = {
            "input_text": transcript,
            "history": history
        }
        
        # 调用Agent
        result = agent.invoke(agent_input)
        
        # 更新对话历史
        conversation_history[conversation_id] = result["history"]
        
        # 3. 返回完整结果
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
    """
    仅处理语音转文字，不生成AI回复
    """
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
    """
    重置对话历史
    """
    try:
        if conversation_id in conversation_history:
            del conversation_history[conversation_id]
        
        return {
            "status": "ok",
            "message": f"对话 {conversation_id} 已重置"
        }
    except Exception as e:
        return {"error": f"重置对话失败: {str(e)}"}

@app.get("/api/health")
async def health_check():
    """
    健康检查端点
    """
    return {
        "status": "ok",
        "service": "融合智能体API"
    }

@app.post("/api/analyze_behavior")
async def analyze_behavior(
    frame: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """
    分析面试官行为（从摄像头视频帧）
    接收前端上传的视频帧图片，分析面试官的表情、视线、注意力等
    帮助作为面试者的你更好地了解面试官的状态
    """
    import base64
    import tempfile
    import shutil
    from pathlib import Path

    temp_file_path = None

    try:
        # 读取视频帧文件
        file_content = await frame.read()

        if len(file_content) == 0:
            return {"success": False, "error": "上传的文件为空"}

        # 转换为 base64
        frame_base64 = base64.b64encode(file_content).decode('utf-8')

        logger.info(f"收到视频帧，大小: {len(file_content)} bytes")

        # 调用统一的 LangGraph Agent 处理
        agent = create_multi_agent()

        # 获取对话历史
        history = conversation_history.get(conversation_id, [])

        # 构建输入 - 分析面试官
        agent_input = {
            "input_text": "分析面试官的行为和状态",
            "video_frame_data": frame_base64,
            "history": history,
            "pre_route": "video_input"
        }

        # 调用 Agent
        result = agent.invoke(agent_input)

        # 更新对话历史
        conversation_history[conversation_id] = result.get("history", [])

        # 返回结果
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
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

@app.post("/api/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: str = Form(default="default")
):
    """
    上传文件并进行OCR或文档解析
    支持的文件类型：图片(png/jpg/jpeg/bmp/tiff/webp)、PDF、Excel(xlsx/xls)、Word(docx/doc)
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    temp_file_path = None
    
    try:
        # 保存上传的文件到临时目录
        file_suffix = Path(file.filename).suffix.lower()
        
        # 验证文件类型
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf', '.xlsx', '.xls', '.docx', '.doc']
        if file_suffix not in allowed_extensions:
            return {
                "success": False,
                "error": f"不支持的文件类型: {file_suffix}。支持的类型: {', '.join(allowed_extensions)}"
            }
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"文件已上传: {file.filename}, 大小: {os.path.getsize(temp_file_path)} bytes")
        
        # 调用统一的LangGraph Agent处理
        agent = create_multi_agent()
        
        # 获取对话历史
        history = conversation_history.get(conversation_id, [])
        
        # 构建输入
        agent_input = {
            "input_text": f"请处理上传的文件: {file.filename}",
            "file_path": temp_file_path,
            "history": history
        }
        
        # 调用Agent
        result = agent.invoke(agent_input)
        
        # 更新对话历史
        conversation_history[conversation_id] = result["history"]
        
        # 返回结果
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
        return {
            "success": False,
            "error": f"文件处理失败: {str(e)}"
        }
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"临时文件已清理: {temp_file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {temp_file_path}, 错误: {str(e)}")

@app.post("/api/ocr")
async def ocr_process(
    file: UploadFile = File(...),
    enable_structure: bool = Form(default=False)
):
    """
    仅进行OCR处理，不生成AI回复
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    temp_file_path = None
    
    try:
        from ocr.ocr_service import OCRService
        
        # 保存上传的文件
        file_suffix = Path(file.filename).suffix.lower()
        
        # 验证文件类型
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf']
        if file_suffix not in allowed_extensions:
            return {
                "success": False,
                "error": f"不支持的文件类型: {file_suffix}。OCR支持的类型: {', '.join(allowed_extensions)}"
            }
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"OCR处理文件: {file.filename}")
        
        # 调用OCR服务
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
        return {
            "success": False,
            "error": f"OCR处理失败: {str(e)}"
        }
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")

@app.post("/api/parse_document")
async def parse_document(file: UploadFile = File(...)):
    """
    仅进行文档解析，不生成AI回复
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    temp_file_path = None
    
    try:
        from document_parser.document_parser_service import DocumentParserService
        
        # 保存上传的文件
        file_suffix = Path(file.filename).suffix.lower()
        
        # 验证文件类型
        allowed_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.doc']
        if file_suffix not in allowed_extensions:
            return {
                "success": False,
                "error": f"不支持的文件类型: {file_suffix}。支持的类型: {', '.join(allowed_extensions)}"
            }
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"文档解析: {file.filename}")
        
        # 调用文档解析服务
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
        return {
            "success": False,
            "error": f"文档解析失败: {str(e)}"
        }
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)