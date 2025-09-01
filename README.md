# LangChain & LangGraph-based-AI Agent-full-stack-application

个人本地面试助手（开发中...）

Thanks & References:
- teddynote LAB: https://github.com/teddynote-lab/langgraph-mcp-agents
- Bytebase dbhub-main: https://github.com/bytebase/dbhub
- PaddleOCR

核心：

1. LangGraph ReAct Agent 
推理（Thought）、行动（Action）、观察（Observation），循环上述过程，直到最终答案（Final Answer）

2. LangChain supported model & Custom API model
将仅支持单次问答的朴素Restful HTTP API接口, 封装为支持langChain & LangGraph Agent 的 LLM 形态 

3. MCP

4. 异步全流程输出跟踪与socket前后端交互

TODO
5.轻量级RAG with Agent
6.轻量级OCR with Agent






