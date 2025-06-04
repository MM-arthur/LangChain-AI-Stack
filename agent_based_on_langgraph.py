from imports import *
from config import memory, model, tavily, sqlite_lock

# ========== 数据模型定义 ==========
class ClassificationResult(BaseModel):
    is_technical: bool

class StyleTransferResult(BaseModel):
    formatted_answer: str

# ========== 状态类型定义 ==========
class AgentState(TypedDict):
    question: str
    is_technical: Optional[bool]
    blog_content: Optional[str]
    web_content: Optional[str]
    final_answer: Optional[str]
    answer_source: Optional[Literal["blog", "web", "blocked"]]
    is_complete: bool
    current_node: Optional[str]  # 新增当前节点字段

# ========== 默认初始状态 ==========
DEFAULT_STATE = {
    "is_technical": None,
    "blog_content": None,
    "web_content": None,
    "final_answer": None,
    "answer_source": None,
    "is_complete": False,
    "current_node": None
}

# ========== 知识库配置 ==========
BLOG_KNOWLEDGE = {
    "python装饰器": {
        "content": "我在《Python高级技巧》博客中详细讨论过，装饰器本质上是语法糖...",
        "source": "https://myblog.com/python-decorator",
        "style_markers": ["喜欢用'语法糖'比喻", "习惯先讲原理再给示例"]
    },
    "redis持久化": {
        "content": "正如我在2023年架构设计分享中分析的，Redis持久化就像...",
        "source": "https://myblog.com/redis-persistence",
        "style_markers": ["常用物流系统类比", "强调CAP理论应用"]
    }
}

# ========== 提示模板 ==========
TECHNICAL_PROMPT = """请严格按以下JSON格式返回：
{
  "is_technical": true|false
}

判断规则：
1. 包含以下关键词则为技术问题：原理、实现、优化、架构、源码、设计、算法、机制、流程、协议、模型、框架、系统
2. 涉及技术概念解释或实现细节的问题
3. 其他情况视为非技术问题

当前问题：{question}"""

STYLE_TRANSFER_PROMPT = """请严格按以下JSON格式返回：
{{
  "formatted_answer": "改写后的内容"
}}

要求：
1. 必须使用JSON格式
2. 包含我的写作风格：{style_markers}
3. 必须添加博客引用提示（如"如我在...中提到的"）
4. 保持回答在100字以内

示例正确响应：
{{
  "formatted_answer": "Kafka的消息顺序性保证类似我在《分布式系统设计》中提到的队列处理机制，通过分区有序性..."
}}

原始内容：
{web_content}"""

# ========== 核心函数 ==========
def classify_question(state: AgentState):
    """问题分类路由"""
    try:
        messages = [
            SystemMessage(content=TECHNICAL_PROMPT.format(question=state["question"])),
            HumanMessage(content="请严格使用指定JSON格式返回")
        ]
        response = model.invoke(messages)
        
        # 改进的JSON提取逻辑
        cleaned = response.content.strip()
        if cleaned.startswith('{') and cleaned.endswith('}'):
            pass  # 已经是纯JSON
        elif "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
        result = json.loads(cleaned)
        validated = ClassificationResult(**result)
        return {"is_technical": validated.is_technical}
    
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print(f"[警告] 分类解析失败，使用保守策略: {str(e)}")
        # 更全面的技术关键词检测
        tech_keywords = ["原理", "实现", "优化", "架构", "源码", "设计", "算法", 
                        "机制", "流程", "协议", "模型", "框架", "系统"]
        return {"is_technical": any(kw in state["question"] for kw in tech_keywords)}
def search_blog(state: AgentState):
    """博客知识检索"""
    if not state.get("is_technical", False):
        return {
            "answer_source": "blocked",
            "final_answer": "请真人回答该问题",
            "is_complete": True
        }
    
    simplified_q = state["question"].lower().replace(" ", "")[:15]
    blog_data = BLOG_KNOWLEDGE.get(simplified_q)
    
    if blog_data:
        return {
            "blog_content": blog_data["content"],
            "answer_source": "blog",
            "final_answer": f"{blog_data['content']}\n（源自我的博客：{blog_data['source']}）",
            "is_complete": True
        }
    return {"blog_content": ""}

def search_web(state: AgentState):
    """网络检索与风格迁移"""
    if state.get("final_answer"):
        return {}
    
    try:
        # 执行网络搜索（添加重试逻辑）
        search_result = tavily.search(
            query=state["question"],
            max_results=3,
            search_depth="advanced"
        )
        web_content = "\n".join(
            r.get("content", "")[:500] for r in search_result.get("results", [])[:2]  # 限制内容长度
        )[:2000]  # 总长度限制
    except Exception as e:
        print(f"[错误] 网络搜索失败: {str(e)}")
        web_content = ""
    
    if not web_content.strip():
        return {
            "final_answer": "暂时无法获取相关信息，建议直接查阅官方文档",
            "answer_source": "web",
            "is_complete": True
        }
    
    style_markers = BLOG_KNOWLEDGE.get("_global_style", ["常用现实案例类比", "喜欢分点论述"])
    
    try:
        messages = [
            SystemMessage(content=STYLE_TRANSFER_PROMPT.format(
                style_markers=", ".join(style_markers),
                web_content=web_content
            )),
            HumanMessage(content="请严格按示例格式返回JSON")
        ]
        response = model.invoke(messages)
        
        # 提取JSON内容
        raw_content = response.content
        if "```json" in raw_content:
            cleaned = raw_content.split("```json")[1].split("```")[0].strip()
        else:
            cleaned = raw_content
        
        parsed = json.loads(cleaned)
        validated = StyleTransferResult(**parsed)
        formatted = validated.formatted_answer
        
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print(f"[警告] 风格迁移失败: {str(e)}")
        formatted = web_content[:150] + "..." if web_content else "信息处理失败"
    
    return {
        "web_content": formatted,
        "answer_source": "web",
        "final_answer": f"{formatted}\n（根据行业实践整理）",
        "is_complete": True
    }

def block_non_tech(state: AgentState):
    """非技术问题拦截"""
    return {
        "final_answer": "该问题需要人工沟通，请联系我的微信：tech_interview",
        "answer_source": "blocked",
        "is_complete": True
    }
def wrap_node_function(node_name, func):
    """自动添加当前节点名称到状态"""
    def wrapped_func(state):
        result = func(state)
        result["current_node"] = node_name
        return result
    return wrapped_func
# ========== 工作流构建 ==========
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("classify", wrap_node_function("classify", classify_question))
builder.add_node("blog_search", wrap_node_function("blog_search", search_blog))
builder.add_node("web_search", wrap_node_function("web_search", search_web))
builder.add_node("block", wrap_node_function("block", block_non_tech))

# 设置路由
builder.set_entry_point("classify")

# 条件分支
builder.add_conditional_edges(
    "classify",
    lambda state: "tech" if state.get("is_technical") else "non_tech",
    {"tech": "blog_search", "non_tech": "block"}
)

builder.add_conditional_edges(
    "blog_search",
    lambda state: "has_answer" if state.get("blog_content") else "need_web",
    {"has_answer": END, "need_web": "web_search"}
)

builder.add_edge("web_search", END)
builder.add_edge("block", END)

# 编译工作流
graph = builder.compile(checkpointer=memory)

# ========== 测试执行 ==========
# ========== 测试执行 ==========
if __name__ == "__main__":
    test_cases = [
        {"question": "Python装饰器的实现原理是什么？"},
        {"question": "Kafka如何保证消息顺序性？"},
        {"question": "为什么选择我们公司？"}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*30}")
        print(f"测试用例 {i+1}: {case['question']}")
        
        init_state = {**DEFAULT_STATE, **case}
        config = {"configurable": {"thread_id": f"test_{i}"}}
        
        try:
            for step in graph.stream(init_state, config=config):
                print(f"\n[步骤状态]")
                print(f"当前节点: {step.get('__pydantic_initial__', {}).get('node')}")
                print(f"状态变化:")
                for k, v in step.items():
                    if not k.startswith('__') and k not in ['node', 'config']:
                        print(f"  {k}: {v}")
                
                if step.get("final_answer"):
                    print(f"\n最终回答：{step['final_answer']}")
                    print(f"来源：{step.get('answer_source', 'unknown')}")
        except Exception as e:
            print(f"[严重错误] 流程执行失败: {str(e)}")