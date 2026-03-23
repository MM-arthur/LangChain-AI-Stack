# Agent功能说明文档

## 📖 项目概述

本项目是一个基于 **LangGraph** 的**面试助手智能体**，支持多种输入方式（文本、语音、文件上传），并集成了OCR识别、文档解析、RAG检索增强、网页搜索等高级功能。

### 核心特性

- ✅ **多模态输入** - 支持文本、语音、图片、PDF、Excel、Word等多种输入
- ✅ **智能路由** - 根据问题类型自动选择最佳处理流程
- ✅ **本地知识库** - RAG检索个人博客/技术文档
- ✅ **网页搜索** - 获取最新知识信息
- ✅ **OCR识别** - PaddleOCR本地模型处理图片/PDF
- ✅ **语音识别** - Whisper/PaddleSpeech本地模型
- ✅ **温和回答** - 面试场景专业回复

---

## 🏗️ 整体架构

### 系统架构图

```
用户输入
   ↓
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  pre_router (🟢 Tool Node) - 前置路由                 │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│        ┌────────────┼────────────┐                         │
│        │            │            │                         │
│        ▼            ▼            ▼                         │
│   📷 OCR处理   📄 文档解析   📝 文本输入                    │
│   (🩵本地模型)  (🟢工具节点)   (🩵本地模型)                 │
│        │            │            │                         │
│        └────────────┴────────────┘                         │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  optimize_transcript (🔵 LLM) - 文本优化              │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  intent_recognition (🔵 LLM) - 意图识别               │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  agent_router (🔵 LLM) - 路由决策                     │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│        ┌────────────┼────────────┐                         │
│        │            │            │                         │
│        ▼            ▼            ▼                         │
│   🔍 RAG检索   🌐 网页搜索   💬 直接生成                   │
│   (🔵 LLM)     (🔵 LLM)      (🔵 LLM)                      │
│        │            │            │                         │
│        ▼            │            │                         │
│   ┌─────────┐       │            │                         │
│   │结果检查 │       │            │                         │
│   └────┬────┘       │            │                         │
│    有内容│无内容     │            │                         │
│        │└───────────┘            │                         │
│        └────────────┬────────────┘                         │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  generate_response (🔵 LLM) - 统一生成回复            │   │
│  │  整合内容 + 标注数据源 + 温和回答                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
   ↓
AI回复（带数据来源）
```

---

## 🎯 核心功能模块

### 节点类型说明

| 颜色 | 类型 | 说明 |
|------|------|------|
| 🔵 蓝色 | LLM Node | 云端大语言模型调用节点 |
| 🩵 青色 | Local Model Node | 本地模型节点（PaddleOCR/Whisper等） |
| 🟢 绿色 | Tool Node | 纯功能/工具节点，不涉及模型调用 |
| 🟠 橙色 | Condition Node | 条件判断节点 |

---

### 1. 前置路由模块 (pre_router) 🟢

**节点类型**: Tool Node（纯逻辑判断）

**功能定位**: 系统入口，判断是否有文件上传

**处理逻辑**:
```
检测file_path → 判断文件扩展名 → 返回路由决策
```

**路由规则**:
| 文件类型 | 路由目标 |
|---------|---------|
| 图片/PDF | `ocr_processing` |
| Excel/Word | `document_parsing` |
| 无文件 | `process_speech_to_text` |

---

### 2. OCR处理模块 (ocr_processing) 🩵

**节点类型**: Local Model Node（PaddleOCR本地模型）

**功能定位**: 图片和PDF文字识别

**核心能力**:
- 图片OCR识别
- PDF智能处理（自动检测文字版/扫描版）
- 表格结构识别
- 多语言支持

**技术栈**:
- PaddleOCR - 核心OCR引擎
- PyMuPDF - PDF渲染
- OpenCV - 图像处理

**输出**: 提取的文本内容 → 传递给 `process_speech_to_text`

---

### 3. 文档解析模块 (document_parsing) 🟢

**节点类型**: Tool Node（纯工具解析）

**功能定位**: 结构化文档解析

**支持格式**:
- PDF: 提取文本、表格、图片
- Excel: 多工作表、数据、公式
- Word: 段落、表格、样式

**输出**: 提取的文本内容 → 传递给 `process_speech_to_text`

---

### 4. 语音文本处理模块 (process_speech_to_text) 🩵

**节点类型**: Local Model Node（Whisper/PaddleSpeech）

**功能定位**: 处理语音转文字结果

**注意**: 实际语音转换在API端点完成，此节点负责传递文本

**输入来源**:
- 语音转文字结果
- OCR提取文本
- 文档解析文本
- 直接文本输入

---

### 5. 文本优化模块 (optimize_transcript) 🔵

**节点类型**: LLM Node（云端大模型）

**功能定位**: 优化**所有输入文本**（不只是语音）

**优化内容**:
- 去除冗余信息
- 文本规范化
- 专业术语保持
- 整理文件提取的文本

---

### 6. 意图识别模块 (intent_recognition) 🔵

**节点类型**: LLM Node（云端大模型）

**功能定位**: 识别问题类型，生成执行计划

**输出内容**:
```json
{
    "question_type": "技术问题/个人问题/最新知识/开放性问题",
    "technical_fields": ["涉及的技术领域"],
    "core_topic": "核心主题",
    "execution_plan": {
        "steps": [...]
    }
}
```

**问题类型判断**:
| 类型 | 关键词/特征 | 路由目标 |
|------|-----------|---------|
| 技术问题 | 技术术语、编程相关 | RAG检索 |
| 个人问题 | 项目经验、简历相关 | RAG检索 |
| 最新知识 | "最新"、"最近"、"当前" | 网页搜索 |
| 开放性问题 | 闲聊、建议类 | 直接生成 |

---

### 7. 路由决策模块 (agent_router) 🔵

**节点类型**: LLM Node（云端大模型）

**功能定位**: 根据意图决定路由

**路由规则**:
```
技术问题/个人问题 → rag_processing
最新知识 → web_search
开放性问题 → generate_response
```

---

### 8. RAG检索模块 (rag_processing) 🔵

**节点类型**: LLM Node（LLM + 向量检索）

**功能定位**: 从本地知识库检索相关内容

**知识库来源**:
- 个人CSDN博客
- 技术文档
- 项目经验文档

**技术栈**:
- FAISS - 向量存储
- Sentence Transformers - 文本嵌入
- HuggingFace - all-MiniLM-L6-v2

**输出**:
- `rag_result`: 检索到的答案
- `rag_sources`: 数据来源列表

---

### 9. RAG结果检查 (check_rag_result) 🟠

**节点类型**: Condition Node（条件判断）

**判断逻辑**:
```python
if rag_result and len(rag_result) > 50:
    return "has_content"  # → generate_response
else:
    return "no_content"   # → web_search
```

---

### 10. 网页搜索模块 (web_search) 🔵

**节点类型**: LLM Node（LangGraph ReAct Agent）

**功能定位**: 获取最新知识，通过ReAct范式自主调用搜索工具

**技术栈**:
- LangGraph `create_react_agent` - ReAct框架
- Tavily API - 网页搜索
- `@tool` 装饰器 - 工具定义

**ReAct工作流程**:
```
Thought: 思考下一步该做什么
    ↓
Action: 选择工具并执行
    ↓
Observation: 观察工具返回结果
    ↓
(循环直到获得完整答案)
```

**可用工具**:
| 工具名称 | 功能 | 参数 |
|---------|------|------|
| `tavily_search` | 进行网页搜索，获取详细内容 | query |
| `get_search_sources` | 获取搜索结果的来源链接 | query |

**输出**:
- `web_search_result`: 搜索内容（LLM整理后）
- `web_sources`: 来源链接

---

### 11. 回复生成模块 (generate_response) 🔵

**节点类型**: LLM Node（云端大模型）

**功能定位**: 统一生成最终回复

**核心能力**:
1. **内容整合**: 整合RAG或网页搜索结果
2. **数据源标注**: 标注内容来源
3. **温和回答**: 面试场景专业回复

**输入来源**:
| 来源 | 内容 | 数据源 |
|------|------|--------|
| RAG有结果 | `rag_result` + `rag_sources` | 本地知识库 |
| 网页搜索 | `web_search_result` + `web_sources` | 互联网 |
| 开放性问题 | 无额外内容 | LLM自身知识 |

**回复格式**:
```
[回答内容]

---
📚 数据来源：
- [来源链接1]
- [来源链接2]
```

---

## 🔄 典型使用场景

### 场景1: 技术问题（RAG检索）

```
用户: "请解释React中的虚拟DOM是什么？"
   ↓
pre_router → 文本输入
   ↓
optimize_transcript → 优化文本
   ↓
intent_recognition → 识别为"技术问题"
   ↓
agent_router → 路由到RAG
   ↓
rag_processing → 检索本地知识库
   ↓
check_rag_result → 有内容
   ↓
generate_response → 整合回答 + 标注来源
```

---

### 场景2: 最新知识（网页搜索）

```
用户: "2024年最新的大模型技术有哪些？"
   ↓
pre_router → 文本输入
   ↓
optimize_transcript → 优化文本
   ↓
intent_recognition → 识别为"最新知识"
   ↓
agent_router → 路由到网页搜索
   ↓
web_search (ReAct Agent)
   ├─ Thought: 分析问题，确定搜索策略
   ├─ Action: tavily_search("2024 大模型技术")
   ├─ Observation: 获取搜索结果
   ├─ Thought: 整理关键信息
   └─ 输出整理后的结果
   ↓
generate_response → 整合回答 + 标注来源
```

---

### 场景3: RAG无结果回退

```
用户: "某个非常冷门的技术问题"
   ↓
pre_router → 文本输入
   ↓
intent_recognition → 识别为"技术问题"
   ↓
agent_router → 路由到RAG
   ↓
rag_processing → 检索无结果
   ↓
check_rag_result → 无内容
   ↓
web_search → 网页搜索
   ↓
generate_response → 整合回答
```

---

### 场景4: 文件上传

```
用户: 上传图片/PDF
   ↓
pre_router → 检测到文件
   ↓
ocr_processing → OCR提取文本
   ↓
process_speech_to_text → 传递文本
   ↓
optimize_transcript → 整理文本
   ↓
后续流程同上...
```

---

## 🎨 技术栈总览

### 核心框架
- **LangGraph** - Agent编排框架
- **LangChain** - LLM应用框架
- **FastAPI** - Web服务框架

### AI能力
- **Moonshot AI** - 云端大语言模型
- **PaddleOCR** - 本地OCR识别
- **PaddleSpeech/Whisper** - 本地语音识别

### 搜索能力
- **Tavily** - 网页搜索API
- **MCP** - 工具调用协议

### 向量检索
- **FAISS** - 向量数据库
- **Sentence Transformers** - 文本嵌入

---

## 🔧 MCP工具配置

### mcp_config.json

```json
{
  "get_current_time": {
    "command": "python",
    "args": ["src/mcp_server/mcp_server_time.py"],
    "transport": "stdio"
  },
  "web_search": {
    "command": "python",
    "args": ["src/mcp_server/mcp_server_web_search.py"],
    "transport": "stdio"
  }
}
```

### 可用工具

| 工具名称 | 功能 | 参数 |
|---------|------|------|
| `get_current_time` | 获取当前时间 | timezone |
| `web_search` | 网页搜索 | query, max_results |
| `get_search_sources` | 获取搜索来源 | query, max_results |

---

## 📊 状态定义

```python
class AgentState(TypedDict):
    input_text: str                    # 用户输入文本
    transcript: str                    # 语音转文字结果
    optimized_text: str                # 优化后的文本
    intent: Dict[str, Any]             # 意图识别结果
    route_decision: str                # 路由决策
    response: str                      # 最终回复
    history: List[Dict[str, str]]      # 对话历史
    messages: List                     # 消息列表
    file_path: str                     # 文件路径
    file_type: str                     # 文件类型
    ocr_result: Dict[str, Any]         # OCR结果
    document_content: str              # 文档内容
    pre_route: str                     # 前置路由结果
    rag_result: Optional[str]          # RAG检索结果
    rag_sources: Optional[List[str]]   # RAG数据源列表
    web_search_result: Optional[str]   # 网页搜索结果
    web_sources: Optional[List[str]]   # 网页数据源列表
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# .env文件
MOONSHOT_API_KEY=your_api_key
TAVILY_API_KEY=your_tavily_key
```

### 3. 启动服务

```bash
python src/main.py
```

### 4. 访问服务

- API文档: http://localhost:8000/docs
- 前端界面: http://localhost:8080

---

## 📝 API端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/ws/chat/{session_id}` | WebSocket | 对话接口 |
| `/api/upload_file` | POST | 文件上传 |
| `/api/ocr` | POST | OCR处理 |
| `/api/parse_document` | POST | 文档解析 |
| `/api/process_audio` | POST | 语音处理 |
| `/api/config` | GET | 获取MCP配置 |

---

## 🎯 最佳实践

### 1. 面试场景使用
- 技术问题会从本地知识库检索
- 最新技术动态会走网页搜索
- 回答会标注数据来源，增加可信度

### 2. 文件处理
- 上传面试题图片，自动OCR识别
- 上传简历PDF，自动解析内容

### 3. 多轮对话
- 系统保持对话历史
- 可以追问和深入讨论

---

## 🔮 未来规划

- [ ] 支持更多本地模型
- [ ] 添加语音输出（TTS）
- [ ] 支持视频分析
- [ ] 多语言支持
- [ ] 面试模拟模式

---

## 📞 技术支持

如有问题，请查看：
- 流程图文档: `docs/Agent流程图.md`
- 项目README: `README.md`
