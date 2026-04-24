# 统一LangGraph智能体实现 - 面试助手Agent

import os
import sys
import json
from typing import Dict, Any, List, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.tools import Tool
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import operator

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.ocr.ocr_service import OCRService
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR service not installed, OCR will be unavailable")

try:
    from src.document_parser.document_parser_service import DocumentParserService
    DOCUMENT_PARSER_AVAILABLE = True
except ImportError:
    DOCUMENT_PARSER_AVAILABLE = False
    print("Document parser not installed, document parsing will be unavailable")

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

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
    # 行为分析结果
    behavior_result: Optional[Dict[str, Any]]
    video_frame_data: Optional[str]  # base64 编码的视频帧

def pre_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 前置路由：判断是否有文件需要上传或视频帧数据
    """
    file_path = state.get("file_path", "")
    video_frame_data = state.get("video_frame_data", "")

    # 优先检测视频帧数据（来自摄像头）
    if video_frame_data:
        print(f"📹 检测到视频帧数据，进行行为分析")
        return {
            **state,
            "pre_route": "video_input"
        }

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

def behavior_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🩵 Local Model Node - 面试官行为分析（YOLO目标检测 + 姿态估计）
    分析内容：面试官的人体检测、面部表情、视线方向、坐姿姿态、注意力水平
    帮助作为面试者的你更好地了解面试官的状态和反应
    """
    try:
        from src.behavior_detection.behavior_analyzer import BehaviorAnalyzer

        video_frame_data = state.get("video_frame_data", "")
        file_path = state.get("file_path", "")

        analyzer = BehaviorAnalyzer()

        if video_frame_data:
            print(f"📹 开始分析面试官视频帧...")
            result = analyzer.analyze_video_frame(video_frame_data)
        elif file_path and os.path.exists(file_path):
            print(f"📹 开始分析面试官图像: {file_path}")
            result = analyzer.analyze_image_file(file_path)
        else:
            print(f"⚠️  未提供视频帧数据或文件路径")
            result = {"success": False, "error": "未提供视频数据"}

        if result.get("success", False):
            print(f"✅ 面试官行为分析完成")
            print(f"   姿势: {result.get('posture', {}).get('state', 'unknown')}")
            print(f"   表情: {result.get('expression', {}).get('state', 'unknown')}")
            print(f"   视线: {result.get('gaze', {}).get('direction', 'unknown')}")
            print(f"   注意力: {result.get('attention', {}).get('level', 'unknown')}")
        else:
            print(f"❌ 面试官行为分析失败: {result.get('error', '未知错误')}")

        return {
            **state,
            "behavior_result": result
        }

    except ImportError:
        print("⚠️  行为分析服务未安装")
        return {
            **state,
            "behavior_result": {"success": False, "error": "行为分析模块未安装"}
        }
    except Exception as e:
        print(f"❌ behavior_detection 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "behavior_result": {"success": False, "error": str(e)}
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
5. **模拟面试**：用户想练习面试、希望 AI 扮演面试官提问 → 走 mock_interview
   - 包含关键词："模拟面试"、"来面试"、"面试一下"、"帮我面试"、"扮演面试官"
6. **面试复盘**：用户想复盘过去的面试表现、分析面试结果 → 走 interview_review
   - 包含关键词："复盘"、"面试表现"、"我今天面了"、"我上次面了"、"面试怎么样"
7. **职业规划**：询问职业发展方向、学习路径、是否该转方向 → 走 career_planning
   - 包含关键词："怎么发展"、"职业方向"、"要不要转"、"该学什么"、"往哪走"

重要判断优先级：
- 如果问题包含"模拟面试"、"来面试"等关键词 → 优先判断为"模拟面试"
- 如果问题包含"复盘"、"面试表现"、"我今天面了" → 优先判断为"面试复盘"
- 如果问题包含职业发展方向相关词 → 优先判断为"职业规划"
- 如果问题包含时间相关词（几点、今天、现在、当前），优先判断为"最新知识"
- 如果问题是询问身份或自我介绍，判断为"开放性问题"

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
        elif question_type == "模拟面试":
            route_decision = "mock_interview"
            print(f"🎯 路由决策: 模拟面试")
        elif question_type == "面试复盘":
            route_decision = "interview_review"
            print(f"🎯 路由决策: 面试复盘")
        elif question_type == "职业规划":
            route_decision = "career_planning"
            print(f"🎯 路由决策: 职业规划")
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

def _get_intent_mode(state: Dict[str, Any]) -> str:
    """从 state 推断当前意图模式，用于前端 UI 展示"""
    if state.get("mock_interview_mode"):
        return "mock_interview"
    route = state.get("route_decision", "")
    if route == "mock_interview":
        return "mock_interview"
    elif route == "interview_review":
        return "interview_review"
    elif route == "career_planning":
        return "career_planning"
    return "normal"

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
    behavior_result = state.get("behavior_result", {})

    context = ""
    sources_str = ""

    # 处理行为分析结果
    if behavior_result and behavior_result.get("success", False):
        posture = behavior_result.get("posture", {})
        expression = behavior_result.get("expression", {})
        gaze = behavior_result.get("gaze", {})
        attention = behavior_result.get("attention", {})
        warnings = behavior_result.get("warnings", [])

        behavior_desc = f"""【面试官行为分析】
- 坐姿状态：{posture.get('state', 'unknown')}（置信度: {posture.get('confidence', 0):.0%}）
- 面部表情：{expression.get('state', 'unknown')}（置信度: {expression.get('confidence', 0):.0%}）
- 视线方向：{gaze.get('direction', 'unknown')}（置信度: {gaze.get('confidence', 0):.0%}）
- 注意力水平：{attention.get('level', 'unknown')}（得分: {attention.get('score', 0):.0%}）
"""

        if warnings:
            behavior_desc += f"- 警告信息：{'；'.join(warnings)}"

        context = behavior_desc

    elif rag_result:
        context = f"【本地知识库检索内容】\n{rag_result}"
        sources_str = "\n".join([f"- {s}" for s in rag_sources[:3]])
    elif web_search_result:
        context = f"【网页搜索内容】\n{web_search_result}"
        sources_str = "\n".join([f"- {s}" for s in web_sources[:3]])

    # 如果是行为分析，生成专门的面试官分析反馈
    if behavior_result and behavior_result.get("success", False):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的面试观察助手。

**你的身份**：
- 你在帮助面试者（用户）分析面试官的行为
- 用户不是面试官，而是来参加面试的候选人

**分析维度**：
1. 表情：面试官表情是否严肃/友善/中性？是否在思考？
2. 视线：面试官视线在哪里？是在看你、看屏幕、还是走神？
3. 姿势：面试官坐姿是否放松？身体前倾表示感兴趣？
4. 注意力：面试官是否专注？还是显得不耐烦？

**输出要求**：
1. 先给出一个总体判断（1-2句话）
2. 针对每个维度给出简短分析
3. 根据面试官的状态，给出应对建议
4. 保持积极正面的语气，帮助用户建立信心

**格式**：
【面试官状态】
...（整体评价）

【细节观察】
- 表情：...
- 视线：...
- 姿势：...
- 注意力：...

【应对建议】
...（如何调整自己的回答策略）"""),
            ("human", "{behavior_analysis}")
        ])

        chain = prompt | llm

        try:
            response = chain.invoke({
                "behavior_analysis": context
            })

            final_response = response.content

            print(f"✅ 行为分析回复生成完成")

            return {
                **state,
                "response": final_response,
                "history": state.get("history", []) + [
                    {"role": "user", "content": "分析面试官行为"},
                    {"role": "assistant", "content": final_response}
                ],
                "intent_mode": _get_intent_mode(state)
            }
        except Exception as e:
            print(f"❌ generate_response 失败: {str(e)}")
            return {
                **state,
                "response": f"抱歉，生成回复时出现错误: {str(e)}",
                "history": state.get("history", []),
                "intent_mode": _get_intent_mode(state)
            }

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
            ],
            "intent_mode": _get_intent_mode(state),
            "mock_interview_mode": state.get("mock_interview_mode", False),
            "current_round": state.get("current_round", 0)
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
            ],
            "intent_mode": _get_intent_mode(state)
        }

# ============================================================
# 职业意图节点（Personal AI Coach - 面试复盘/职业规划/模拟面试）
# ============================================================

def mock_interview(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🤝 模拟面试节点
    识别用户要求模拟面试，启动多轮面试对话流程
    """
    llm = init_llm()
    
    user_input = state.get("optimized_text", state.get("input_text", ""))
    interview_history = state.get("interview_history", [])
    mock_mode = state.get("mock_interview_mode", False)
    current_round = state.get("current_round", 0)
    
    # 判断是启动面试还是继续面试
    is_start = (
        "模拟面试" in user_input or
        "来面试" in user_input or
        "面试一下" in user_input or
        "帮我面试" in user_input
    )
    
    if is_start:
        # 启动新面试
        interview_history = []
        current_round = 0
        mock_mode = True
        
        # 从 RAG 召回 Arthur 简历和 JD（如果有配置）
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from rag.RAG import load_personal_knowledge
            personal_docs = load_personal_knowledge()
            resume_context = ""
            for doc in personal_docs:
                if doc.metadata.get("type") == "personal_resume":
                    resume_context += doc.page_content + "\n\n"
        except Exception as e:
            resume_context = ""
            print(f"⚠️  简历加载失败: {e}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的中文面试官，正在为候选人进行模拟面试。

你的职责：
1. 根据候选人的背景（简历）选择合适的面试问题
2. 先问2-3个热身问题（自我介绍、项目介绍）
3. 然后深入问技术问题，每次问1个，追问细节
4. 保持专业、友好的面试氛围

输出要求：
- 第一轮：直接输出第一道面试题，不要废话
- 后续轮次：先对上一轮回答做1-2句简短点评，然后问下一道题
- 问题要具体、循序渐进"""),
            ("human", f"""候选人背景：
{resume_context if resume_context else '暂无简历信息，按通用问题提问'}

面试正式开始。请输出第一道面试题。""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        
        interview_history.append({"role": "interviewer", "content": response})
        current_round = 1
        
        return {
            **state,
            "response": response,
            "mock_interview_mode": True,
            "interview_history": interview_history,
            "current_round": current_round
        }
    
    elif mock_mode:
        # 继续面试：用户回答了问题
        interview_history.append({"role": "candidate", "content": user_input})
        
        # 检查是否结束面试
        end_keywords = ["结束", "面试完了", "好了", "不用了", "结束面试"]
        if any(kw in user_input for kw in end_keywords):
            return {
                **state,
                "mock_interview_mode": False,
                "interview_finished": True,
                "interview_history": interview_history,
                "route_decision": "generate_response"
            }
        
        # 追问或生成下一题
        history_str = "\n".join([f"{'面试官' if h['role']=='interviewer' else '候选人'}：{h['content']}" for h in interview_history[-6:]])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是专业面试官。
- 先对候选人上一轮回答做1-2句简短点评（肯定优点或指出不足）
- 然后继续追问或问下一道相关问题
- 如果已经问了5轮以上，可以输出"面试结束，感谢参与！"
- 问题要具体，结合他的简历背景"""),
            ("human", f"""面试对话记录：
{history_str}

请继续面试（或结束面试）。""")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        
        interview_history.append({"role": "interviewer", "content": response})
        current_round += 1
        
        return {
            **state,
            "response": response,
            "mock_interview_mode": True,
            "interview_history": interview_history,
            "current_round": current_round
        }
    else:
        # 非 mock_interview 模式，不处理
        return {**state}


def interview_review(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    📋 面试复盘节点
    Arthur 输入面试内容，AI 对照 JD 和简历做结构化复盘
    """
    llm = init_llm()
    
    user_input = state.get("optimized_text", state.get("input_text", ""))
    
    # 从 RAG 召回 Arthur 简历和 JD
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from rag.RAG import load_personal_knowledge
        personal_docs = load_personal_knowledge()
        
        resume_context = ""
        jd_context = ""
        for doc in personal_docs:
            if doc.metadata.get("type") == "personal_resume":
                resume_context += doc.page_content + "\n\n"
            elif doc.metadata.get("type") == "personal_note":
                jd_context += doc.page_content + "\n\n"
    except Exception as e:
        resume_context = ""
        jd_context = ""
        print(f"⚠️  知识库加载失败: {e}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的面试复盘助手，帮助候选人（Arthur）分析面试表现。

分析维度：
1. 【技术问题】- 问了什么技术？答得怎么样？有没有遗漏关键点？
2. 【项目经验】- 面试官对哪个项目最感兴趣？有没有被问住的地方？
3. 【表达与逻辑】- 回答是否有条理？STAR法则用了吗？
4. 【整体评价】- 这次面试表现如何？几分？（10分制）
5. 【下次改进】- 具体可操作的2-3条改进建议

输出格式（Markdown）：
## 📋 面试复盘报告

### 整体评分：X/10
### 1. 技术问题分析
...
### 2. 项目经验分析
...
### 3. 表达与逻辑
...
### 4. 下次改进建议
...
"""),
        ("human", f"""请分析以下面试表现：

候选人背景：
{resume_context if resume_context else '暂无简历信息'}

Arthur 描述的面试内容：
{user_input}

请生成结构化复盘报告。""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    
    return {
        **state,
        "response": response,
        "rag_sources": ["Arthur 个人简历", "面试复盘分析"],
        "intent_mode": "interview_review"
    }


def career_planning(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🧭 职业规划节点
    结合 Arthur 简历 + 历史对话，给出个性化职业发展建议
    """
    llm = init_llm()
    
    user_input = state.get("optimized_text", state.get("input_text", ""))
    
    # 从 RAG 召回 Arthur 简历
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from rag.RAG import load_personal_knowledge
        personal_docs = load_personal_knowledge()
        resume_context = ""
        for doc in personal_docs:
            if doc.metadata.get("type") == "personal_resume":
                resume_context += doc.page_content + "\n\n"
    except Exception as e:
        resume_context = ""
        print(f"⚠️  简历加载失败: {e}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的职业规划顾问，为技术人才的职业发展提供个性化建议。

分析维度：
1. 【优势分析】- 结合 Arthur 的学历（天大硕士）、经历（Siemens/Lenovo/渤海银行）、技术栈（Java/Vue/AI Agent）
2. 【市场定位】- 当前 AI 行业趋势 + 他的背景最适合什么岗位方向
3. 【路径建议】- 短期（3-6个月）、中期（1-2年）、长期（3-5年）分别怎么走
4. 【具体行动】- 每个阶段最值得做的一件事

输出格式（Markdown）：
## 🧭 Arthur 职业发展规划

### 1. 个人优势分析
...
### 2. 目标岗位定位
...
### 3. 发展路径建议
   - 短期（3-6个月）：...
   - 中期（1-2年）：...
   - 长期（3-5年）：...
### 4. 当前最值得做的一件事
...
"""),
        ("human", f"""Arthur 的背景：
{resume_context if resume_context else '暂无简历信息'}

他的职业发展问题/困惑：
{user_input}

请给出个性化职业规划建议。""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    
    return {
        **state,
        "response": response,
        "rag_sources": ["Arthur 个人简历", "职业规划分析"],
        "intent_mode": "career_planning"
    }


def create_multi_agent():
    """
    创建统一的LangGraph智能体，按照流程图编排节点
    """
    workflow = StateGraph(dict)
    
    workflow.add_node("pre_router", pre_router)
    workflow.add_node("ocr_processing", ocr_processing)
    workflow.add_node("document_parsing", document_parsing)
    workflow.add_node("behavior_detection", behavior_detection)  # YOLO行为分析节点
    workflow.add_node("process_speech_to_text", process_speech_to_text)
    workflow.add_node("optimize_transcript", optimize_transcript)
    workflow.add_node("intent_recognition", intent_recognition)
    workflow.add_node("agent_router", agent_router)
    workflow.add_node("rag_processing", rag_processing)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_response", generate_response)
    # 职业意图节点（Personal AI Coach）
    workflow.add_node("mock_interview", mock_interview)
    workflow.add_node("interview_review", interview_review)
    workflow.add_node("career_planning", career_planning)

    workflow.set_entry_point("pre_router")

    def pre_route_decision(state: Dict[str, Any]) -> str:
        pre_route = state.get("pre_route", "text_input")

        if pre_route == "file_ocr":
            return "ocr_processing"
        elif pre_route == "file_document":
            return "document_parsing"
        elif pre_route == "video_input":
            return "behavior_detection"
        else:
            return "process_speech_to_text"

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

    workflow.add_edge("ocr_processing", "process_speech_to_text")
    workflow.add_edge("document_parsing", "process_speech_to_text")
    workflow.add_edge("behavior_detection", "generate_response")  # 行为分析直接生成回复
    workflow.add_edge("process_speech_to_text", "optimize_transcript")
    workflow.add_edge("optimize_transcript", "intent_recognition")
    workflow.add_edge("intent_recognition", "agent_router")
    
    def decide_next_node(state: Dict[str, Any]) -> str:
        route = state.get("route_decision", "generate_response")
        
        if route == "rag_processing":
            return "rag_processing"
        elif route == "web_search":
            return "web_search"
        elif route == "mock_interview":
            return "mock_interview"
        elif route == "interview_review":
            return "interview_review"
        elif route == "career_planning":
            return "career_planning"
        else:
            return "generate_response"
    
    workflow.add_conditional_edges(
        "agent_router",
        decide_next_node,
        {
            "rag_processing": "rag_processing",
            "web_search": "web_search",
            "mock_interview": "mock_interview",
            "interview_review": "interview_review",
            "career_planning": "career_planning",
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
    workflow.add_edge("mock_interview", "generate_response")
    workflow.add_edge("interview_review", "generate_response")
    workflow.add_edge("career_planning", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # SqliteSaver 持久化：所有 session 共用一个 checkpointer，靠 thread_id 隔离
    import sqlite3
    db_path = Path(project_root) / "data" / "checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    sqlite_checkpointer = SqliteSaver(conn)
    
    agent = workflow.compile(checkpointer=sqlite_checkpointer)
    
    return agent

# ============================================================
# 单例模式入口（Hermes 架构思想）
# 进程启动时调用一次，所有会话共享编译好的 Agent
# ============================================================

_agent_singleton_instance = None


def get_singleton_agent():
    """
    返回单例 Agent（编译一次，进程级共享）
    main.py 在启动时调用此函数初始化 AgentSingleton
    """
    global _agent_singleton_instance
    if _agent_singleton_instance is None:
        _agent_singleton_instance = create_multi_agent()
        print(f"[get_singleton_agent] ✅ Agent 编译完成")
    return _agent_singleton_instance


if __name__ == "__main__":
    # 测试模式：直接运行 multi_agent.py
    agent = create_multi_agent()

    test_input = {
        "input_text": "请你解释一下React中的虚拟DOM是什么？它有什么优势？",
        "history": []
    }

    print("=== 测试文本输入 ===")
    result = agent.invoke(test_input)

    print(f"\n原始输入：{test_input['input_text']}")
    print(f"优化后文本：{result.get('optimized_text', '')}")
    print(f"意图识别结果：{result.get('intent', {})}")
    print(f"路由决策：{result.get('route_decision', '')}")
    print(f"生成回复：{result.get('response', '')}")
