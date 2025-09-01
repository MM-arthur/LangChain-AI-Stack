from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # 增加数据类型概念，数据验证, JSON, 序列化
from typing import Optional, Dict, List, Any # 类型注解
import asyncio
import json
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage

from custom_api_model.model import CustomAPIModel

from agent_driver import astream_graph

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储会话
sessions: Dict[str, Dict[str, Any]] = {}

# MCP Config
CONFIG_FILE_PATH = "mcp_config.json"

# 系统提示词 - 示例
SYSTEM_PROMPT = """
<ROLE>
You are a demo agent for tool usage examples. Keep responses simple.
</ROLE>

<INSTRUCTIONS>
1. Analyze the question briefly
2. Use tools
3. Answer concisely
4. Add source if available
</INSTRUCTIONS>

<OUTPUT_FORMAT>
[Answer]

**Source**(optional)
- [URL]
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
}

# Pydantic模型
class ChatMessage(BaseModel):
    content: str
    session_id: str

class ConfigUpdate(BaseModel):
    config: Dict[str, Any]
    session_id: str

class SessionSettings(BaseModel):
    model: str = "API_custom_model"
    timeout_seconds: int = 120
    recursion_limit: int = 100
    session_id: str

# 辅助函数
def load_config_from_json():
    """加载配置文件"""
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_server/mcp_server_time.py"],
            "transport": "stdio"
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
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

async def initialize_agent(session_id: str, model_name: str, mcp_config: Dict[str, Any]):
    """初始化AI代理"""
    # 清理旧客户端
    await cleanup_mcp_client(session_id)
    
    # 初始化MCP客户端
    client = MultiServerMCPClient(mcp_config)

    tools = await client.get_tools()
    
    # 初始化模型
    if model_name in ["claude-3-7-sonnet-latest"]:
        model = ChatAnthropic(
            model=model_name,
            temperature=0.1,
            max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
        )
    elif model_name == "API_custom_model":
        model = CustomAPIModel(
            model_name="model",  # 根据需要调整
            username=os.environ.get("API_custom_model_username"),
            password=os.environ.get("API_custom_model_password"),
            api_base=os.environ.get("API_custom_model_url"),
        )
        model = model.bind_tools(tools)
    else:
        model = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=OUTPUT_TOKEN_INFO[model_name]["max_tokens"],
        )
    
    
    # 创建代理
    agent = create_react_agent(
        model,
        tools,
        checkpointer=MemorySaver(),
        prompt=SYSTEM_PROMPT,
    )
    
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
    available_models = []
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_models.extend([
            "claude-3-7-sonnet-latest",
        ])
    
    if os.environ.get("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o"])
    
    # 总是添加 'API_custom_model' 模型
    available_models.append("API_custom_model")
    
    return {"models": available_models}

@app.post("/api/initialize")
async def initialize_session(settings: SessionSettings):
    """初始化会话"""
    config = load_config_from_json()
    result = await initialize_agent(settings.session_id, settings.model, config)
    return result

@app.post("/api/update-config")
async def update_config(update: ConfigUpdate):
    """更新配置"""
    save_config_to_json(update.config)
    result = await initialize_agent(
        update.session_id, 
        sessions.get(update.session_id, {}).get("model", "API_custom_model"),
        update.config
    )
    return result







@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天接口"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                if session_id not in sessions or not sessions[session_id].get("agent"):
                    await websocket.send_json({
                        "type": "error",
                        "content": "Session not initialized"
                    })
                    continue
                
                query = data["content"]
                session = sessions[session_id]
                agent = session["agent"]
                
                print(f"[DEBUG] 收到消息: {query}")
                
                # 用于后端历史记录的文本累积
                backend_accumulated_text = []
                
                # 定义一个内部回调函数，astream_graph (来自 utils.py) 将调用它
                async def backend_internal_callback(data_to_send_to_frontend):
                    """
                    此回调函数由 astream_graph (来自 utils.py) 调用，
                    接收预格式化的 {"type": ..., "content": ...} 字典。
                    它将数据发送到前端，并为后端历史记录累积文本。
                    """
                    print(f"[DEBUG] 发送WebSocket消息到前端: {data_to_send_to_frontend}")
                    await websocket.send_json(data_to_send_to_frontend)
                    
                    if data_to_send_to_frontend["type"] == "text":
                        backend_accumulated_text.append(data_to_send_to_frontend["content"])
                    # 工具调用/响应不会累积到最终文本中。它们是不同的消息。

                # 执行查询，通过 backend_internal_callback 流式传输事件
                try:
                    print(f"[DEBUG] 开始执行查询: {query}")
                    
                    await astream_graph(
                        agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=backend_internal_callback, # 传递内部回调函数
                        config=RunnableConfig(
                            recursion_limit=session.get("recursion_limit", 100),
                            thread_id=session["thread_id"],
                        ),
                    )
                    
                    # 流式传输完成后，将完整的用户和助手消息保存到历史记录
                    session["history"].append({"role": "user", "content": query})
                    session["history"].append({"role": "assistant", "content": "".join(backend_accumulated_text)})
                    
                    # 向前端发送完成信号
                    await websocket.send_json({
                        "type": "complete",
                        "content": "".join(backend_accumulated_text) # 可选：发送最终累积的文本
                    })
                    print(f"[DEBUG] 查询完成: {query}")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc() # 打印完整的堆栈跟踪以进行调试
                    print(f"[ERROR] 查询执行失败: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "content": str(e)
                    })
            
            elif data["type"] == "reset":
                if session_id in sessions:
                    sessions[session_id]["thread_id"] = str(uuid.uuid4())
                    sessions[session_id]["history"] = []
                    await websocket.send_json({
                        "type": "reset_complete"
                    })
    
    except Exception as e:
        import traceback
        traceback.print_exc() # 打印完整的堆栈跟踪以进行调试
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)