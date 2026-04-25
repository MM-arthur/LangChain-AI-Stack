# Career intent nodes: mock_interview, interview_review, career_planning

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import init_llm
from src.core.retry import with_retry


# ── Mock Interview ─────────────────────────────────────────────────────────────

def mock_interview(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 模拟面试多轮对话节点
    流程：启动 → 问热身问题 → 技术追问 → 点评 → 循环 → 结束输出评估报告
    """
    llm = init_llm()
    intent = state.get("intent", {})
    transcript = state.get("optimized_text", state.get("input_text", ""))
    interview_history: List[str] = state.get("interview_history", [])
    current_round = state.get("current_round", 0)
    mock_interview_mode = state.get("mock_interview_mode", False)

    # 启动面试
    if not mock_interview_mode:
        mock_interview_mode = True
        current_round = 0
        start_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是 Arthur 的模拟面试官。

**角色**：
- 你是一个专业、技术深入的面试官
- 面试者是 Arthur（你的学生）
- 目标是通过多轮问答评估 Arthur 的能力

**流程**：
1. 先问一个热身问题（1-2分钟）：自我介绍、项目概述
2. 逐步深入技术问题（根据回答调整难度）
3. 每轮结束后简短点评，给出下一题提示
4. 3-5轮后给出整体评估报告

**开场白**：
"你好 Arthur，欢迎来参加今天的模拟面试。我会从自我介绍开始，逐步深入到技术细节。让我们开始吧！请简单介绍一下你自己，以及你最近在做的一个项目。"

**回复格式**：
直接输出一句话面试官发言，不要加"面试官："前缀。"""),
            ("human", "开始模拟面试")
        ])
        chain = start_prompt | llm
        response = chain.invoke({})
        first_question = response.content

        interview_history = [f"面试官：{first_question}"]
        print(f"✅ 模拟面试启动，第一题: {first_question[:50]}...")

        return {
            **state,
            "response": first_question,
            "mock_interview_mode": True,
            "current_round": 1,
            "interview_history": interview_history,
            "history": state.get("history", []) + [
                {"role": "assistant", "content": first_question}
            ]
        }

    # 候选回答 → 分析 + 追问 or 结束
    candidate_answer = transcript

    # 更新历史
    interview_history.append(f"Arthur：{candidate_answer}")

    # 分析回答质量，决定下一步
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是模拟面试官。当前正在对 Arthur 进行多轮面试。

**历史对话**：
{history}

**最新回答来自 Arthur**：
{latest_answer}

**你的任务**：
1. 简短点评 Arthur 的回答（1-2句话）
2. 决定是继续追问还是结束面试

**决策规则**：
- current_round >= 5：结束面试，输出评估报告
- current_round < 5 且回答质量高：追问深入
- current_round < 5 且回答一般：换方向继续问

**输出格式（JSON）**：
{
  "feedback": "简短点评 Arthur 的回答（1-2句话）",
  "next_question": "下一个问题（如果继续）或评估报告（如果结束）",
  "should_end": true/false
}"""),
        ("human", f"Arthur 回答：{candidate_answer}")
    ])

    chain = analysis_prompt | llm

    try:
        response = chain.invoke({})
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        import json
        result = json.loads(content)
        feedback = result.get("feedback", "")
        next_question = result.get("next_question", "")
        should_end = result.get("should_end", False)

        interview_history.append(f"面试官点评：{feedback}")
        interview_history.append(f"面试官：{next_question}")

        if should_end:
            # 生成最终评估报告
            report = _generate_interview_report(interview_history, current_round)
            print(f"✅ 模拟面试结束，共 {current_round} 轮")

            return {
                **state,
                "response": f"{feedback}\n\n{report}",
                "mock_interview_mode": False,
                "current_round": 0,
                "interview_history": [],
                "history": state.get("history", []) + [
                    {"role": "user", "content": candidate_answer},
                    {"role": "assistant", "content": feedback + "\n\n" + report}
                ]
            }

        current_round += 1
        print(f"✅ 面试第 {current_round} 轮: {next_question[:50]}...")

        return {
            **state,
            "response": f"{feedback}\n\n{next_question}",
            "current_round": current_round,
            "interview_history": interview_history,
            "history": state.get("history", []) + [
                {"role": "user", "content": candidate_answer},
                {"role": "assistant", "content": feedback + "\n\n" + next_question}
            ]
        }

    except Exception as e:
        print(f"❌ mock_interview 失败: {str(e)}")
        return {
            **state,
            "response": f"模拟面试出错: {str(e)}",
            "history": state.get("history", [])
        }


def _generate_interview_report(history: List[str], rounds: int) -> str:
    """生成结构化面试评估报告"""
    llm = init_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的面试评估专家。根据面试历史，生成结构化评估报告。

**评估维度**：
1. 技术深度：候选人在自己领域的知识深度
2. 表达能力：逻辑是否清晰、回答是否简洁
3. 项目经验：项目经历的真实性和深度
4. 临场反应：回答问题的速度和灵活性
5. 整体建议：需要提高的方向

**格式要求**：
使用 Markdown，结构清晰，适合直接给候选人看。

**报告标题**：
## 🎯 模拟面试评估报告"""),
        ("human", "面试历史：\n{history}\n\n面试轮次：{rounds}")
    ])

    chain = prompt | llm
    history_text = "\n".join(history[-20:])  # 最近20条
    response = chain.invoke({"history": history_text, "rounds": str(rounds)})
    return response.content


# ── Interview Review ──────────────────────────────────────────────────────────

def interview_review(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 面试复盘分析
    Arthur 输入面试内容，AI 对照 JD 和简历做结构化复盘
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))

    # 从 RAG 召回 Arthur 简历和 JD
    try:
        from src.rag.RAG import PersonalKnowledgeRAG
        rag = PersonalKnowledgeRAG()
        rag_results = rag.query(optimized_text, top_k=5)
        resume_context = rag_results.get("results", "")
    except Exception:
        resume_context = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的面试复盘助手，帮助候选人分析面试表现。

**复盘维度**：
1. 【技术面】面试问了哪些技术问题？回答得如何？
2. 【项目经验】项目经历有没有被深挖？描述是否清晰？
3. 【表达逻辑】回答是否有逻辑？结构是否清晰？
4. 【优缺点】这次面试中你的表现有哪些优缺点？
5. 【改进建议】下次面试可以从哪些方面提高？

**输出要求**：
- 使用 Markdown，结构清晰
- 评分用 ⭐ 表示（1-5星）
- 每维度 1-2 句话分析
- 最后给出 3 条具体可执行的改进建议

**格式**：
## 📋 面试复盘报告

### 技术面
...

### 项目经验
...

### 表达逻辑
...

### 优缺点分析
...

### 🎯 改进建议
1. ...
2. ...
3. ... """),
        ("human", """Arthur 描述的面试内容：
{interview_content}

参考简历/职位描述：
{resume_context}""")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "interview_content": optimized_text,
            "resume_context": resume_context or "（无简历上下文）"
        })
        review_report = response.content

        print(f"✅ 面试复盘完成")

        return {
            **state,
            "review_report": review_report,
            "response": review_report,
            "history": state.get("history", []) + [
                {"role": "user", "content": f"复盘：{optimized_text[:50]}..."},
                {"role": "assistant", "content": review_report}
            ],
            "rag_sources": ["Arthur 个人简历", "面试复盘分析"]
        }
    except Exception as e:
        print(f"❌ interview_review 失败: {str(e)}")
        return {
            **state,
            "review_report": f"复盘失败: {str(e)}",
            "response": f"复盘失败: {str(e)}",
            "history": state.get("history", [])
        }


# ── Career Planning ────────────────────────────────────────────────────────────

def career_planning(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔵 LLM Node - 职业发展规划
    结合 Arthur 简历 + 历史对话，给出个性化职业发展建议
    """
    llm = init_llm()
    optimized_text = state.get("optimized_text", state.get("input_text", ""))

    # 从 RAG 召回 Arthur 简历
    try:
        from src.rag.RAG import PersonalKnowledgeRAG
        rag = PersonalKnowledgeRAG()
        rag_results = rag.query(optimized_text, top_k=5)
        resume_context = rag_results.get("results", "")
    except Exception:
        resume_context = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个职业发展顾问，帮助用户规划职业路径。

**分析框架**：
1. 【优势分析】结合学历、经历、技术栈，找出核心竞争力
2. 【目标定位】当前最合适的岗位方向是什么
3. 【短期路径】（3-6个月）需要重点提升什么技能
4. 【中期路径】（1-2年）如何从初级到资深
5. 【长期路径】（3-5年）职业天花板如何突破

**输出要求**：
- 结合 Arthur 的具体背景（天大硕士、Siemens/Lenovo/渤海银行经历、Java/Vue/AI Agent 技术栈）
- 每个阶段给出具体可执行的建议
- 使用 Markdown，结构清晰

**格式**：
## 🧭 Arthur 职业发展规划

### 1. 优势分析
...

### 2. 目标定位
...

### 3. 短期路径（3-6个月）
...

### 4. 中期路径（1-2年）
...

### 5. 长期路径（3-5年）
... """),
        ("human", """Arthur 的问题/背景：
{question}

Arthur 的简历信息：
{resume_context}""")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": optimized_text,
            "resume_context": resume_context or "（无简历上下文）"
        })
        career_plan = response.content

        print(f"✅ 职业规划完成")

        return {
            **state,
            "career_plan": career_plan,
            "response": career_plan,
            "history": state.get("history", []) + [
                {"role": "user", "content": optimized_text},
                {"role": "assistant", "content": career_plan}
            ],
            "rag_sources": ["Arthur 个人简历", "职业规划分析"]
        }
    except Exception as e:
        print(f"❌ career_planning 失败: {str(e)}")
        return {
            **state,
            "career_plan": f"职业规划失败: {str(e)}",
            "response": f"职业规划失败: {str(e)}",
            "history": state.get("history", [])
        }