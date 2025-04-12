import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from threading import Lock
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_community.chat_models.moonshot import MoonshotChat

from dotenv import load_dotenv
_ = load_dotenv()
sqlite_lock = Lock()

conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
model = MoonshotChat(
    model="moonshot-v1-8k",
    temperature=0.8,
    max_tokens=1024,
    api_key=os.getenv("MOONSHOT_API_KEY")
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class AgentState(TypedDict):
    job_title: str
    location: str
    search_queries: List[str]
    jd_list: List[str]
    analysis: str
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]


PLAN_PROMPT = """作为BAT等大厂招聘专家，请为{job_title}({location})职位生成搜索策略，重点关注：
1. 技术栈关键词(如React/Vue/Spring Boot等)
2. AI相关技能要求(如TensorFlow/PyTorch/LLM等)
3. 大厂常用筛选条件"""

JD_ANALYSIS_PROMPT = """请深度分析以下职位描述，提取：
1. 核心技术栈(前端/后端/数据库/AI框架)
2. AI相关技能要求
3. 薪资范围
4. 公司名称
5. 职位详情链接(如有)
按以下格式返回：
公司: [公司名]
技术栈: [技术列表] 
AI技能: [AI相关要求]
薪资: [薪资范围]
链接: [职位链接]"""



RESEARCH_PLAN_PROMPT = """为{job_title}({location})生成3个精准搜索查询，要求:
1. 包含"阿里巴巴/腾讯/字节跳动/美团"等公司名
2. 包含"全栈"和"AI"相关关键词
3. 包含薪资范围
4. 使用"site:zhipin.com"或"site:lagou.com"限定招聘网站
示例:
全栈开发工程师 北京 阿里巴巴 AI site:zhipin.com 薪资20k-40k"""



# 节点函数
def plan_node(state: AgentState):
    print("\n正在执行plan_node...")
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=f"职位: {state['job_title']}\n地点: {state['location']}")
    ]
    print("发送给模型的提示:", messages)
    response = model.invoke(messages)
    print("模型响应:", response.content)
    return {"search_queries": [response.content]}


def research_plan_node(state: AgentState):
    with sqlite_lock:
        print("\n正在执行research_plan_node...")
        messages = [
            SystemMessage(content=RESEARCH_PLAN_PROMPT.format(
                job_title=state['job_title'],
                location=state['location']
            ))
        ]
        print("发送给模型的提示:", messages)
        response = model.invoke(messages)
        print("模型响应:", response.content)
        
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        print("解析出的查询:", queries)
        
        jd_list = []
        for q in queries[:3]:
            try:
                search_result = tavily.search(query=q, max_results=3)
                for r in search_result['results']:
                    jd_list.append({
                        "company": r.get('title', '未知').split('|')[0],
                        "content": r['content'],
                        "url": r.get('url', '')
                    })
            except Exception as e:
                print(f"搜索出错: {str(e)}")
        return {"jd_list": jd_list}

def generation_node(state: AgentState):
    print("\n正在执行generation_node...")
    content = "\n\n".join([
        f"公司: {jd['company']}\n描述: {jd['content']}\n链接: {jd['url']}" 
        for jd in state['jd_list']
    ])
    print("分析的内容:", content[:200] + "...")  # 打印前200个字符避免输出过长
    
    messages = [
        SystemMessage(content=JD_ANALYSIS_PROMPT),
        HumanMessage(content=content)
    ]
    print("发送给模型的提示:", messages)
    response = model.invoke(messages)
    print("模型响应:", response.content)
    return {
        "analysis": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }



def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# 构建工作流
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("research_plan", research_plan_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_conditional_edges("generate", should_continue, {END: END})

graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}
input_state = {
    'job_title': "全栈开发工程师",
    'location': "北京",
    "max_revisions": 1,
    "revision_number": 1,
}

print("开始执行工作流...")
for s in graph.stream(input_state, thread):
    print("\n当前状态:", s.keys())
    if 'analysis' in s:
        print("\n职位分析报告:")
        print(s['analysis'])
    elif 'jd_list' in s:
        print(f"获取到 {len(s['jd_list'])} 条职位信息")
    elif 'search_queries' in s:
        print("生成的搜索查询:", s['search_queries'])

print("\n工作流执行完成")