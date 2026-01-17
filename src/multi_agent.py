# 统一LangGraph智能体实现

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

# 加载环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# 定义智能体状态
class AgentState:
    def __init__(self):
        self.input_text: str = ""  # 输入文本
        self.transcript: str = ""  # 语音转文字结果
        self.optimized_text: str = ""  # 优化后的文本
        self.intent: Dict[str, Any] = {}  # 意图识别结果
        self.route_decision: str = ""  # 路由决策
        self.response: str = ""  # AI回复
        self.history: List[Dict[str, str]] = []  # 对话历史
        self.messages: List = []  # 用于ReAct Agent的消息列表

# 输出token配置
OUTPUT_TOKEN_INFO = {
    "moonshot-v1-8k": {"max_tokens": 8000},
    "moonshot-v1-32k": {"max_tokens": 32000},
    "moonshot-v1-128k": {"max_tokens": 128000},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000}
}

# 初始化大模型
def init_llm(model_name: str = None):
    if model_name is None:
        model_name = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    
    return MoonshotChat(
        model=model_name,
        temperature=0.1,
        max_tokens=OUTPUT_TOKEN_INFO.get(model_name, {}).get("max_tokens", 8000),
        api_key=os.getenv("MOONSHOT_API_KEY")
    )

# 初始化ReAct Agent模板
def create_react_agent(llm, tools: List[Tool] = None, system_prompt: str = ""):
    """
    创建ReAct风格的Agent
    """
    if tools is None:
        tools = []
    
    # ReAct Agent核心逻辑
    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        
        # 构建完整的消息列表
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        # 调用模型获取ReAct响应
        response = llm.invoke(full_messages)
        
        return {
            **state,
            "messages": messages + [response]
        }
    
    return agent_node

# RAG处理节点
def rag_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用RAG生成回复
    """
    try:
        # 尝试导入RAG模块
        try:
            from rag.RAG import run_rag
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from rag.RAG import run_rag
        
        # 获取优化后的文本作为问题
        question = state["optimized_text"]
        
        # 使用用户提供的CSDN参考知识网站列表
        urls=[
            "https://blog.csdn.net/qq_40550384/article/details/149981605",
            "https://blog.csdn.net/qq_40550384/article/details/156398145",
            "https://blog.csdn.net/qq_40550384/article/details/146161589",
            "https://blog.csdn.net/qq_40550384/article/details/147500510",
            "https://blog.csdn.net/qq_40550384/article/details/132668097",
            "https://doi.org/10.1016/j.apacoust.2020.107647",
            "https://xueshu.baidu.com/usercenter/paper/show?paperid=1s740c805s0d0p10f53y0j10pq344792&site=xueshu_se",
            "https://xueshu.baidu.com/usercenter/paper/show?paperid=1u3a0ar0r50n0ev0582t0mk008192348&site=xueshu_se",
            "https://blog.csdn.net/qq_40550384/article/details/131190132",
            "https://blog.csdn.net/qq_40550384/article/details/131660477",
            "https://blog.csdn.net/qq_40550384/article/details/131423567",
            "https://blog.csdn.net/qq_40550384/article/details/131287048",
            "https://blog.csdn.net/qq_40550384/article/details/131261254",
            "https://blog.csdn.net/qq_40550384/article/details/114605673",
            "https://blog.csdn.net/qq_40550384/article/details/112346545",
            "https://blog.csdn.net/qq_40550384/article/details/112985413",
            "https://blog.csdn.net/qq_40550384/article/details/111309984",
            "https://blog.csdn.net/qq_40550384/article/details/110039473",
            "https://blog.csdn.net/qq_40550384/article/details/108440628",
            "https://blog.csdn.net/qq_40550384/article/details/109659915",
        ]
        
        # 调用RAG功能
        rag_result = run_rag(question=question, urls=urls)
        
        # 更新状态
        return {
            **state,
            "response": rag_result["answer"],
            "history": state.get("history", []) + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": rag_result["answer"]}
            ]
        }
    except Exception as e:
        print(f"❌ rag_processing 失败: {str(e)}")
        print(f"❌ 失败详情: {type(e).__name__}: {e}")
        # RAG失败时，回退到直接使用大模型
        return generate_response(state)

# 大模型节点 - 生成回复
def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用大模型生成回复
    """
    llm = init_llm()
    
    # 构建对话历史
    messages = [
        SystemMessage(content="你是一个友好的智能助手，用中文回答用户的问题。")
    ]
    
    # 添加历史对话
    for msg in state.get("history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # 添加当前用户输入（使用优化后的文本）
    messages.append(HumanMessage(content=state["optimized_text"]))
    
    # 调用大模型
    response = llm.invoke(messages)
    
    # 更新状态
    return {
        **state,
        "response": response.content,
        "history": state.get("history", []) + [
            {"role": "user", "content": state["optimized_text"]},
            {"role": "assistant", "content": response.content}
        ]
    }

# 语音转文字处理节点（只是传递，实际转换在API端点完成）
def process_speech_to_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理语音转文字结果
    """
    return {
        **state,
        "transcript": state["input_text"]  # 这里的input_text实际上是已经转换好的文字
    }

# 优化语音识别内容节点（使用ReAct Agent）
def optimize_transcript(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用ReAct Agent优化语音识别的内容
    """
    llm = init_llm()
    
    # 创建ReAct风格的优化Agent
    optimize_agent = create_react_agent(
        llm,
        system_prompt="你是一个专业的文本优化助手，请将语音识别的文本优化为通顺、准确的中文。保持原意不变，去除冗余信息，使句子更加规范。"
    )
    
    # 构建初始消息
    initial_state = {
        **state,
        "messages": [
            HumanMessage(content=f"请优化以下文本：{state.get('transcript', state.get('input_text', ''))}")
        ]
    }
    
    # 执行ReAct Agent
    result = optimize_agent(initial_state)
    
    # 提取优化后的文本
    optimized_text = result["messages"][-1].content
    
    return {
        **state,
        "optimized_text": optimized_text
    }

# 意图识别节点（使用ReAct Agent）
def intent_recognition(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用ReAct Agent识别意图
    """
    llm = init_llm()
    
    # 创建ReAct风格的意图识别Agent
    intent_agent = create_react_agent(
        llm,
        system_prompt='''你是一个专业的意图分析助手，请严格按照JSON格式输出以下信息：
        - question_type: 问题类型，如'技术问题'、'行为问题'、'项目经验问题'等
        - position_direction: 岗位方向，如'全栈开发'、'前端开发'、'后端开发'、'AI大模型应用开发'等
        - technical_fields: 涉及的技术领域列表，如['JavaScript', 'React', 'Python', 'Node.js', '大模型']等
        - core_topic: 问题的核心主题，简要描述
        
        注意：
        1. 只输出JSON，不要添加任何其他文字
        2. 岗位方向只从提供的选项中选择
        3. 技术领域尽量具体
        4. 核心主题要简洁明了'''
    )
    
    # 构建初始消息
    initial_state = {
        **state,
        "messages": [
            HumanMessage(content=f"请分析以下问题的意图：{state['optimized_text']}")
        ]
    }
    
    # 执行ReAct Agent
    result = intent_agent(initial_state)
    
    # 解析意图结果
    import json
    try:
        intent = json.loads(result["messages"][-1].content)
    except Exception as e:
        print(f"❌ 意图解析失败: {str(e)}")
        # 如果解析失败，返回默认意图
        intent = {
            "question_type": "其他问题",
            "position_direction": "全栈开发",
            "technical_fields": [],
            "core_topic": "无法识别的问题"
        }
    
    return {
        **state,
        "intent": intent
    }

# Agent Router节点
def agent_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据意图识别结果决定路由策略
    """
    try:
        intent = state["intent"]
        
        # 用户熟悉的岗位方向
        familiar_positions = ["全栈开发", "前端开发", "后端开发", "AI大模型应用开发"]
        
        # 默认使用RAG
        route_decision = "rag"
        
        # 分析意图，决定路由
        question_type = intent.get("question_type", "")
        position_direction = intent.get("position_direction", "")
        
        # 路由决策逻辑
        if question_type in ["技术问题", "项目经验问题", "行为问题"] and position_direction in familiar_positions:
            route_decision = "rag"
        else:
            route_decision = "generate_response"
        
        return {
            **state,
            "route_decision": route_decision
        }
    except Exception as e:
        print(f"❌ agent_router 失败: {str(e)}")
        # 如果路由决策失败，默认使用大模型生成
        return {
            **state,
            "route_decision": "generate_response"
        }

# 创建统一的LangGraph智能体
def create_multi_agent():
    """
    创建统一的LangGraph智能体，同时支持打字输入和语音输入
    """
    # 初始化状态图
    workflow = StateGraph(dict)
    
    # 添加节点
    workflow.add_node("process_speech_to_text", process_speech_to_text)
    workflow.add_node("optimize_transcript", optimize_transcript)
    workflow.add_node("intent_recognition", intent_recognition)
    workflow.add_node("agent_router", agent_router)
    workflow.add_node("rag_processing", rag_processing)
    workflow.add_node("generate_response", generate_response)
    
    # 添加边
    workflow.set_entry_point("process_speech_to_text")
    workflow.add_edge("process_speech_to_text", "optimize_transcript")
    workflow.add_edge("optimize_transcript", "intent_recognition")
    workflow.add_edge("intent_recognition", "agent_router")
    
    # 根据路由决策选择下一个节点
    def decide_next_node(state: Dict[str, Any]) -> str:
        return "rag_processing" if state.get("route_decision") == "rag" else "generate_response"
    
    workflow.add_conditional_edges(
        "agent_router",
        decide_next_node,
        {
            "rag_processing": "rag_processing",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_edge("rag_processing", END)
    workflow.add_edge("generate_response", END)
    
    # 编译图 - 暂时不使用checkpointer，避免测试时需要提供configurable keys
    agent = workflow.compile()
    
    return agent

# 测试智能体
if __name__ == "__main__":
    agent = create_multi_agent()
    
    # 测试打字输入
    test_input = {
        "input_text": "请你解释一下React中的虚拟DOM是什么？它有什么优势？",
        "history": []
    }
    
    print("=== 测试打字输入 ===")
    result = agent.invoke(test_input)
    
    print(f"原始输入：{test_input['input_text']}")
    print(f"优化后文本：{result['optimized_text']}")
    print(f"意图识别结果：{result['intent']}")
    print(f"路由决策：{result['route_decision']}")
    print(f"生成回复：{result['response']}")
