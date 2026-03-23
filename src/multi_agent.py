# 统一LangGraph智能体实现 - 面试助手Agent

import os
import json
from typing import Dict, Any, List, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
import operator

try:
    from ocr.ocr_service import OCRService
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR服务未安装，OCR功能将不可用")

try:
    from document_parser.document_parser_service import DocumentParserService
    DOCUMENT_PARSER_AVAILABLE = True
except ImportError:
    DOCUMENT_PARSER_AVAILABLE = False
    print("⚠️  文档解析服务未安装，文档解析功能将不可用")

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

OUTPUT_TOKEN_INFO = {
    "moonshot-v1-8k": {"max_tokens": 8000},
    "moonshot-v1-32k": {"max_tokens": 32000},
    "moonshot-v1-128k": {"max_tokens": 128000},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000}
}

def init_llm(model_name: str = None):
    if model_name is None:
        model_name = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    
    return MoonshotChat(
        model=model_name,
        temperature=0.1,
        max_tokens=OUTPUT_TOKEN_INFO.get(model_name, {}).get("max_tokens", 8000),
        api_key=os.getenv("MOONSHOT_API_KEY")
    )

class AgentState(TypedDict):
    input_text: str
    transcript: str
    optimized_text: str
    intent: Dict[str, Any]
    route_decision: str
    response: str
    history: Annotated[List[Dict[str, str]], operator.add]
    messages: Annotated[List, operator.add]
    file_path: str
    file_type: str
    ocr_result: Dict[str, Any]
    document_content: str
    pre_route: str
    rag_result: Optional[str]
    rag_sources: Optional[List[str]]
    web_search_result: Optional[str]
    web_sources: Optional[List[str]]

def pre_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 前置路由：判断是否有文件需要上传
    """
    file_path = state.get("file_path", "")
    
    if file_path and os.path.exists(file_path):
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf']:
            print(f"📁 检测到图片/PDF文件: {file_path}")
            return {
                **state,
                "route_decision": "ocr_processing",
                "pre_route": "file_ocr",
                "file_type": file_ext
            }
        elif file_ext in ['.xlsx', '.xls', '.docx', '.doc']:
            print(f"📁 检测到文档文件: {file_path}")
            return {
                **state,
                "route_decision": "document_parsing",
                "pre_route": "file_document",
                "file_type": file_ext
            }
        else:
            print(f"⚠️  不支持的文件类型: {file_ext}")
            return {
                **state,
                "route_decision": "process_speech_to_text",
                "pre_route": "unsupported_file"
            }
    else:
        return {
            **state,
            "pre_route": "text_input"
        }

def ocr_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🩵 Local Model Node - OCR处理图片/PDF文件（PaddleOCR本地模型）
    """
    if not OCR_AVAILABLE:
        print("⚠️  OCR服务不可用")
        return {
            **state,
            "ocr_result": {"success": False, "error": "OCR服务不可用"},
            "transcript": "OCR服务不可用"
        }
    
    try:
        file_path = state.get("file_path", "")
        
        if not file_path:
            print("⚠️  未提供文件路径，跳过OCR处理")
            return {
                **state,
                "ocr_result": {"success": False, "error": "未提供文件路径"},
                "transcript": state.get("input_text", "")
            }
        
        print(f"🔍 开始OCR处理: {file_path}")
        
        ocr_service = OCRService()
        result = ocr_service.process_file(file_path, enable_structure=True)
        
        if result.get("success"):
            extracted_text = result.get("text", "")
            print(f"✅ OCR成功，提取文本长度: {len(extracted_text)}")
            
            return {
                **state,
                "ocr_result": result,
                "transcript": extracted_text,
                "document_content": extracted_text
            }
        else:
            error_msg = result.get("error", "OCR处理失败")
            print(f"❌ OCR失败: {error_msg}")
            return {
                **state,
                "ocr_result": result,
                "transcript": f"OCR处理失败: {error_msg}"
            }
            
    except Exception as e:
        print(f"❌ ocr_processing 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "ocr_result": {"success": False, "error": str(e)},
            "transcript": f"OCR处理异常: {str(e)}"
        }

def document_parsing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 解析Excel/Word文档，提取问题文本
    """
    if not DOCUMENT_PARSER_AVAILABLE:
        print("⚠️  文档解析服务不可用")
        return {
            **state,
            "document_content": "文档解析服务不可用",
            "transcript": state.get("input_text", "")
        }
    
    try:
        file_path = state.get("file_path", "")
        
        if not file_path:
            print("⚠️  未提供文件路径，跳过文档解析")
            return {
                **state,
                "document_content": "",
                "transcript": state.get("input_text", "")
            }
        
        print(f"📄 开始文档解析: {file_path}")
        
        parser = DocumentParserService()
        result = parser.parse_document(file_path)
        
        if result.get("success"):
            extracted_text = result.get("full_text", "")
            file_type = result.get("file_type", "unknown")
            
            print(f"✅ 文档解析成功，文件类型: {file_type}, 文本长度: {len(extracted_text)}")
            
            return {
                **state,
                "document_content": extracted_text,
                "transcript": extracted_text,
                "file_type": file_type
            }
        else:
            error_msg = result.get("error", "文档解析失败")
            print(f"❌ 文档解析失败: {error_msg}")
            return {
                **state,
                "document_content": f"文档解析失败: {error_msg}",
                "transcript": f"文档解析失败: {error_msg}"
            }
            
    except Exception as e:
        print(f"❌ document_parsing 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "document_content": f"文档解析异常: {str(e)}",
            "transcript": f"文档解析异常: {str(e)}"
        }

def process_speech_to_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🩵 Local Model Node - 语音转文字处理（Whisper/PaddleSpeech本地模型）
    注意：实际语音转换在API端点完成，这里只是传递文本
    """
    input_text = state.get("input_text", "")
    transcript = state.get("transcript", "")
    
    if transcript:
        return {
            **state,
            "transcript": transcript
        }
    else:
        return {
            **state,
            "transcript": input_text
        }

def optimize_transcript(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 优化所有输入文本（LLM润色）
    注意：优化的是所有输入文本，不只是语音输入
    """
    llm = init_llm()
    
    transcript = state.get("transcript", state.get("input_text", ""))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个文本规范化助手。请对输入文本进行最小化处理。

处理规则：
1. **保持原意绝对不变** - 这是最高优先级
2. **简单问题直接返回** - 如果输入已经是清晰的问题（如"你是谁"、"现在几点"），直接返回原文
3. **只处理以下情况**：
   - 语音识别错误：修正明显的识别错误（如"你好吗"被识别成"你好嘛"）
   - 文件提取混乱：整理从PDF/图片提取的乱序文本
   - 口语化表达：将过于口语化的表达转为书面语
4. **不要**：
   - 不要扩展或改写问题
   - 不要添加任何解释
   - 不要改变问题的核心意图

只输出处理后的文本，不要添加任何解释。"""),
        ("human", "{text}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"text": transcript})
        optimized_text = response.content
        
        print(f"✅ 文本优化完成")
        print(f"   原文: {transcript[:100]}...")
        print(f"   优化后: {optimized_text[:100]}...")
        
        return {
            **state,
            "optimized_text": optimized_text
        }
    except Exception as e:
        print(f"❌ 文本优化失败: {str(e)}")
        return {
            **state,
            "optimized_text": transcript
        }

def intent_recognition(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 意图识别并生成执行计划（LLM分析）
    """
    llm = init_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''你是一个专业的意图分析助手，请分析用户输入并生成执行计划。

请严格按照JSON格式输出以下信息：
{{
    "question_type": "问题类型（技术问题/个人问题/最新知识/开放性问题）",
    "technical_fields": ["涉及的技术领域列表"],
    "core_topic": "问题的核心主题",
    "execution_plan": {{
        "steps": [
            {{
                "step_number": 1,
                "action": "动作名称",
                "description": "动作描述",
                "reason": "执行原因"
            }}
        ],
        "expected_output": "预期输出类型"
    }}
}}

判断规则：
1. **技术问题**：涉及编程、框架、算法、开发等技术内容 → 走RAG检索本地知识库
2. **个人问题**：询问候选人（Arthur）的项目经历、技能、工作经验等简历相关问题 → 走RAG检索本地知识库
3. **最新知识**：需要实时信息的问题 → 走网页搜索
   - 包含关键词："最新"、"最近"、"当前"、"今天"、"现在"、"几点"
   - 需要实时数据：天气、股价、新闻、时间等
4. **开放性问题**：闲聊、建议类、身份询问类 → 直接生成回复
   - 如："你是谁"、"介绍一下你自己"、"你好"

重要判断优先级：
- 如果问题包含时间相关词（几点、今天、现在、当前），优先判断为"最新知识"
- 如果问题是询问身份或自我介绍，判断为"开放性问题"
- 只有明确的技术问题或简历相关问题才判断为"技术问题"或"个人问题"

注意：
1. 只输出JSON，不要添加任何其他文字
2. execution_plan必须包含具体的执行步骤'''),
        ("human", "请分析以下输入的意图：\n\n{text}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"text": state.get("optimized_text", "")})
        intent_str = response.content
        
        if intent_str.startswith("```json"):
            intent_str = intent_str.split("```json")[1].split("```")[0].strip()
        elif intent_str.startswith("```"):
            intent_str = intent_str.split("```")[1].split("```")[0].strip()
        
        intent = json.loads(intent_str)
        
        print(f"✅ 意图识别成功:")
        print(f"   问题类型: {intent.get('question_type', '未知')}")
        print(f"   核心主题: {intent.get('core_topic', '未知')}")
        
    except Exception as e:
        print(f"❌ 意图解析失败: {str(e)}")
        intent = {
            "question_type": "开放性问题",
            "technical_fields": [],
            "core_topic": "无法识别的问题",
            "execution_plan": {
                "steps": [
                    {
                        "step_number": 1,
                        "action": "generate_response",
                        "description": "直接生成回复",
                        "reason": "意图识别失败，使用默认回复"
                    }
                ],
                "expected_output": "文本回复"
            }
        }
    
    return {
        **state,
        "intent": intent
    }

def agent_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 根据意图决定路由（LLM分类决策）
    """
    try:
        intent = state.get("intent", {})
        question_type = intent.get("question_type", "开放性问题")
        
        if question_type in ["技术问题", "个人问题"]:
            route_decision = "rag_processing"
            print(f"🎯 路由决策: RAG检索（技术问题/个人问题）")
        elif question_type == "最新知识":
            route_decision = "web_search"
            print(f"🎯 路由决策: 网页搜索（最新知识）")
        else:
            route_decision = "generate_response"
            print(f"🎯 路由决策: 直接生成回复（开放性问题）")
        
        return {
            **state,
            "route_decision": route_decision
        }
        
    except Exception as e:
        print(f"❌ agent_router 失败: {str(e)}")
        return {
            **state,
            "route_decision": "generate_response"
        }

def rag_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - RAG检索增强生成（LLM+向量检索本地知识库）
    """
    try:
        try:
            from rag.RAG import run_rag
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from rag.RAG import run_rag
        
        question = state["optimized_text"]
        
        urls = [
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
        
        print(f"🔍 开始RAG检索: {question}")
        rag_result = run_rag(question=question, urls=urls)
        
        answer = rag_result.get("answer", "")
        documents = rag_result.get("documents", [])
        
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source and source not in sources:
                sources.append(source)
        
        if answer and "无法回答" not in answer and len(answer) > 50:
            print(f"✅ RAG检索成功，找到相关内容")
            return {
                **state,
                "rag_result": answer,
                "rag_sources": sources
            }
        else:
            print(f"⚠️  RAG检索无相关内容")
            return {
                **state,
                "rag_result": "",
                "rag_sources": []
            }
            
    except Exception as e:
        print(f"❌ rag_processing 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "rag_result": "",
            "rag_sources": []
        }

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 网页搜索Agent（LangGraph ReAct范式）
    通过ReAct循环：思考→行动→观察 来调用搜索工具
    """
    from langchain_core.tools import tool
    from langchain.agents import create_agent
    import json
    
    @tool
    def tavily_search(query: str) -> str:
        """
        使用Tavily进行网页搜索，获取最新信息。
        
        Args:
            query: 搜索关键词
            
        Returns:
            搜索结果的JSON字符串，包含标题、链接、内容摘要
        """
        try:
            from tavily import TavilyClient
            
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return json.dumps({"error": "TAVILY_API_KEY未配置"}, ensure_ascii=False)
            
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            results = response.get("results", [])
            if not results:
                return json.dumps({"error": "未找到相关结果"}, ensure_ascii=False)
            
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append({
                    "index": i,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500]
                })
            
            return json.dumps(formatted, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"搜索失败: {str(e)}"}, ensure_ascii=False)
    
    @tool
    def get_search_sources(query: str) -> str:
        """
        获取搜索结果的来源链接列表。
        
        Args:
            query: 搜索关键词
            
        Returns:
            来源链接列表
        """
        try:
            from tavily import TavilyClient
            
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return json.dumps({"error": "TAVILY_API_KEY未配置"}, ensure_ascii=False)
            
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            results = response.get("results", [])
            sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]
            
            return json.dumps(sources, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k"),
            temperature=0.1,
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1"
        )
        
        tools = [tavily_search, get_search_sources]
        
        question = state.get("optimized_text", state.get("input_text", ""))
        
        print(f"🔍 启动ReAct搜索Agent...")
        print(f"   问题: {question}")
        
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""你是一个网页搜索助手，需要帮助用户搜索最新信息。

你可以使用以下工具：
1. tavily_search: 进行网页搜索，获取详细内容
2. get_search_sources: 获取搜索结果的来源链接

工作流程：
1. 分析用户问题，确定搜索关键词
2. 使用 tavily_search 进行搜索
3. 整理搜索结果，提取关键信息
4. 标注信息来源

最后请输出整理后的搜索结果，包括：
1. 关键信息摘要
2. 信息来源列表"""
        )
        
        result = agent.invoke({
            "messages": [HumanMessage(content=f"请帮我搜索以下问题的最新信息：{question}")]
        })
        
        final_message = result["messages"][-1] if result.get("messages") else None
        
        if final_message:
            search_result = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            sources = []
            for msg in result.get("messages", []):
                if hasattr(msg, 'content') and 'http' in str(msg.content):
                    import re
                    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', str(msg.content))
                    sources.extend(urls)
            
            sources = list(set(sources))[:5]
            
            print(f"✅ ReAct搜索Agent完成")
            
            return {
                **state,
                "web_search_result": search_result,
                "web_sources": sources,
                "messages": result.get("messages", [])
            }
        else:
            print(f"⚠️  ReAct搜索Agent无结果")
            return {
                **state,
                "web_search_result": "",
                "web_sources": []
            }
            
    except Exception as e:
        print(f"❌ web_search ReAct Agent 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "web_search_result": "",
            "web_sources": []
        }

def check_rag_result(state: Dict[str, Any]) -> str:
    """
    🟠 Condition Node - RAG结果检查
    """
    rag_result = state.get("rag_result", "")
    
    if rag_result and len(rag_result) > 50:
        print(f"✅ RAG结果检查: 有相关内容")
        return "has_content"
    else:
        print(f"⚠️  RAG结果检查: 无相关内容，回退到网页搜索")
        return "no_content"

def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 统一生成最终回复
    整合内容、标注数据源、温和回答
    """
    llm = init_llm()
    
    optimized_text = state.get("optimized_text", state.get("input_text", ""))
    rag_result = state.get("rag_result", "")
    rag_sources = state.get("rag_sources", [])
    web_search_result = state.get("web_search_result", "")
    web_sources = state.get("web_sources", [])
    
    context = ""
    sources_str = ""
    
    if rag_result:
        context = f"【本地知识库检索内容】\n{rag_result}"
        sources_str = "\n".join([f"- {s}" for s in rag_sources[:3]])
    elif web_search_result:
        context = f"【网页搜索内容】\n{web_search_result}"
        sources_str = "\n".join([f"- {s}" for s in web_sources[:3]])
    
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
        response = chain.invoke({
            "question": optimized_text,
            "context": context if context else "（无检索内容，请根据自身知识回答）",
            "sources": sources_str if sources_str else "（无外部数据来源）"
        })
        
        final_response = response.content
        
        print(f"✅ 生成回复完成")
        
        return {
            **state,
            "response": final_response,
            "history": state.get("history", []) + [
                {"role": "user", "content": optimized_text},
                {"role": "assistant", "content": final_response}
            ]
        }
        
    except Exception as e:
        print(f"❌ generate_response 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            **state,
            "response": f"抱歉，生成回复时出现错误: {str(e)}",
            "history": state.get("history", []) + [
                {"role": "user", "content": optimized_text},
                {"role": "assistant", "content": f"抱歉，生成回复时出现错误: {str(e)}"}
            ]
        }

def create_multi_agent():
    """
    创建统一的LangGraph智能体，按照流程图编排节点
    """
    workflow = StateGraph(dict)
    
    workflow.add_node("pre_router", pre_router)
    workflow.add_node("ocr_processing", ocr_processing)
    workflow.add_node("document_parsing", document_parsing)
    workflow.add_node("process_speech_to_text", process_speech_to_text)
    workflow.add_node("optimize_transcript", optimize_transcript)
    workflow.add_node("intent_recognition", intent_recognition)
    workflow.add_node("agent_router", agent_router)
    workflow.add_node("rag_processing", rag_processing)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)
    
    workflow.set_entry_point("pre_router")
    
    def pre_route_decision(state: Dict[str, Any]) -> str:
        pre_route = state.get("pre_route", "text_input")
        
        if pre_route == "file_ocr":
            return "ocr_processing"
        elif pre_route == "file_document":
            return "document_parsing"
        else:
            return "process_speech_to_text"
    
    workflow.add_conditional_edges(
        "pre_router",
        pre_route_decision,
        {
            "ocr_processing": "ocr_processing",
            "document_parsing": "document_parsing",
            "process_speech_to_text": "process_speech_to_text"
        }
    )
    
    workflow.add_edge("ocr_processing", "process_speech_to_text")
    workflow.add_edge("document_parsing", "process_speech_to_text")
    workflow.add_edge("process_speech_to_text", "optimize_transcript")
    workflow.add_edge("optimize_transcript", "intent_recognition")
    workflow.add_edge("intent_recognition", "agent_router")
    
    def decide_next_node(state: Dict[str, Any]) -> str:
        route = state.get("route_decision", "generate_response")
        
        if route == "rag_processing":
            return "rag_processing"
        elif route == "web_search":
            return "web_search"
        else:
            return "generate_response"
    
    workflow.add_conditional_edges(
        "agent_router",
        decide_next_node,
        {
            "rag_processing": "rag_processing",
            "web_search": "web_search",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_conditional_edges(
        "rag_processing",
        check_rag_result,
        {
            "has_content": "generate_response",
            "no_content": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate_response")
    workflow.add_edge("generate_response", END)
    
    agent = workflow.compile()
    
    return agent

if __name__ == "__main__":
    agent = create_multi_agent()
    
    test_input = {
        "input_text": "请你解释一下React中的虚拟DOM是什么？它有什么优势？",
        "history": []
    }
    
    print("=== 测试打字输入 ===")
    result = agent.invoke(test_input)
    
    print(f"\n原始输入：{test_input['input_text']}")
    print(f"优化后文本：{result['optimized_text']}")
    print(f"意图识别结果：{result['intent']}")
    print(f"路由决策：{result['route_decision']}")
    print(f"生成回复：{result['response']}")
