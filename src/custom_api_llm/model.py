from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig
import requests
import json
import base64
import asyncio
import aiohttp
import re
from pydantic import Field

class CustomAPIModel(BaseChatModel):
    """
    将仅支持单次问答的朴素LLM Restful HTTP API接口, 封装为支持langChain & LangGraph Agent 的 LLM 形态 
    CustomAPIModel 为 LangChain Agent 中 LLM 的角色。它不负责执行工具，也不负责管理多轮的 ReAct 循环（思考-行动-观察）
    这些都由 create_react_agent 和其底层的 LangGraph 框架来处理。
    模型只专注于一件事：根据给定的消息历史和工具定义，决定是给出最终答案，还是发出工具调用指令，并以 LangChain 期望的 AIMessage 格式返回。

    Wrap a simple LLM Restful HTTP API that only supports single-turn Q&A into an LLM format compatible with LangChain & LangGraph Agent
    CustomAPIModel serves as the LLM role in LangChain Agent. It is not responsible for executing tools nor managing the multi-turn ReAct loop (Thought-Action-Observation)
    These are handled by create_react_agent and its underlying LangGraph framework.
    The model focuses solely on one thing: based on the given message history and tool definitions, decide whether to provide a final answer or issue tool call instructions, and return it in the AIMessage format expected by LangChain.
    """
    
    # 根据你的Restful API参数自定义
    model_name: str = Field(description="模型")
    username: str = Field(description="用户名")
    password: str = Field(description="密码")
    api_base: str = Field(description="请求地址")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=2048, description="最大token数")
    tools: List[Any] = Field(default=[], description="工具列表") # 如果原生支持，则不需要下文如此处理
    

    def _format_tools_for_prompt(self) -> str:
        """将工具信息格式化为提示文本
        
        Returns:
            格式化后的工具描述文本
        """
        if not self.tools:
            return ""
        
        tools_text = "\n\n=== 可用工具列表 ===\n"
        
        for i, tool in enumerate(self.tools, 1):
            tool_name = getattr(tool, 'name', f'tool_{i}')
            tool_description = getattr(tool, 'description', '无描述')
            
            tools_text += f"{i}. 工具名称: {tool_name}\n"
            tools_text += f"   描述: {tool_description}\n"
            
            # 尝试获取参数信息
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
                    if 'properties' in schema:
                        tools_text += f"   参数: {', '.join(schema['properties'].keys())}\n"
                        # 添加参数详细信息
                        for param_name, param_info in schema['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_desc = param_info.get('description', '无描述')
                            tools_text += f"     - {param_name} ({param_type}): {param_desc}\n"
                except Exception as e:
                    tools_text += f"   参数: 无法获取参数信息\n"
            
            tools_text += "\n"
        
        tools_text += "=== 工具调用格式 ===\n"
        tools_text += "如果需要使用工具，请严格按照以下JSON格式回复：\n"
        tools_text += '```json\n'
        tools_text += '{\n'
        tools_text += '  "action": "use_tool",\n'
        tools_text += '  "tool_name": "工具名称",\n'
        tools_text += '  "parameters": {\n'
        tools_text += '    "参数名": "参数值"\n'
        tools_text += '  }\n'
        tools_text += '}\n'
        tools_text += '```\n'
        tools_text += "如果不需要使用工具，请直接回复内容，不要包含上述JSON格式。\n"
        tools_text += "========================\n\n"
        
        return tools_text
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """将LangChain消息转换为API所需的格式
        
        Args:
            messages: LangChain消息列表
            
        Returns:
            转换后的用户输入文本
        """
        prompt_parts = []
        
        # 处理消息历史
        # LangChain Agent 会在每次迭代中传递完整的消息历史
        # 包括 HumanMessage, AIMessage (可能包含tool_calls), ToolMessage
        # 将这些都转换成大模型能理解的纯文本对话格式
        
        # 先添加工具信息，确保每次模型调用都能看到可用工具
        tools_info = self._format_tools_for_prompt()
        if tools_info:
            prompt_parts.append(tools_info)
        
        prompt_parts.append("=== 对话历史 ===")
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"[系统消息] {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"[用户] {message.content}")
            elif isinstance(message, AIMessage):
                # 如果AIMessage有tool_calls，说明是模型上次决定调用工具
                if message.tool_calls:
                    # 将tool_calls转换为模型能理解的文本格式
                    tool_call_strs = []
                    for tc in message.tool_calls:
                        tool_call_strs.append(json.dumps({
                            "action": "use_tool",
                            "tool_name": tc['name'],
                            "parameters": tc['args']
                        }, ensure_ascii=False))
                    prompt_parts.append(f"[助手] 决定调用工具:\n```json\n{tool_call_strs[0]}\n```\n") # 假设只处理一个tool_call
                else:
                    # 否则是普通文本回复
                    prompt_parts.append(f"[助手] {message.content}")
            elif isinstance(message, ToolMessage):
                prompt_parts.append(f"[工具执行结果] {message.content}")
        prompt_parts.append("==================\n")
        
        # 提示模型在当前轮次进行响应
        if self.tools:
            prompt_parts.append("请分析上述对话历史和工具信息，如果需要使用工具来获取信息或执行操作，请按照上述JSON格式回复。如果可以直接回答，请直接提供答案。")
        else:
            prompt_parts.append("请分析上述对话历史，并提供回答。")

        return "\n".join(prompt_parts)
    
    def _parse_tool_calls(self, response_content: str) -> tuple[str, List[Dict[str, Any]]]:
        """解析模型响应中的工具调用
        
        Args:
            response_content: 模型响应内容
            
        Returns:
            (清理后的响应内容, 工具调用列表)
        """
        tool_calls = []
        cleaned_content = response_content
        
        # 查找JSON格式的工具调用
        json_pattern = r'```json\s*(\{[^`]+\})\s*```'
        json_matches = re.findall(json_pattern, response_content, re.DOTALL)
        
        for match in json_matches:
            try:
                tool_call_data = json.loads(match.strip())
                if (tool_call_data.get('action') == 'use_tool' and 
                    'tool_name' in tool_call_data and 
                    'parameters' in tool_call_data):
                    
                    # LangChain的tool_calls需要'name'和'args'
                    tool_calls.append({
                        'name': tool_call_data['tool_name'],
                        'args': tool_call_data['parameters']
                    })
                    
                    # 从响应中移除JSON代码块
                    cleaned_content = re.sub(
                        r'```json\s*' + re.escape(match.strip()) + r'\s*```',
                        '',
                        cleaned_content,
                        flags=re.DOTALL
                    ).strip()
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON解析失败: {e}, 内容: {match}")
                continue
        
        # 也尝试查找不在代码块中的JSON (如果整个响应就是工具调用JSON)
        if not tool_calls and cleaned_content.strip().startswith('{') and cleaned_content.strip().endswith('}'):
            try:
                tool_call_data = json.loads(cleaned_content.strip())
                if (tool_call_data.get('action') == 'use_tool' and 
                    'tool_name' in tool_call_data and 
                    'parameters' in tool_call_data):
                    
                    tool_calls.append({
                        'name': tool_call_data['tool_name'],
                        'args': tool_call_data['parameters']
                    })
                    cleaned_content = "" # 整个响应都是工具调用
            except json.JSONDecodeError:
                pass
        
        return cleaned_content, tool_calls
    
    def _call_api(self, user_input: str) -> str:
        """调用自定义模型API
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            模型响应
        """
        
        credentials = "TESTfdsfsddf"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json"
        }
        
        # 构造请求体 - 取决于你得API参数
        payload = {
            "question": "",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
        }
        
        
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            
            try:
                response_data = response.json()
                res = response_data.get("ans", str(response_data))
                return res
            except Exception as e:
                print(f"错误: {str(e)}")
                return response.text
                
        except Exception as e:
            print(f"错误: {str(e)}")
            raise
    
    # override. _generate, _agenerate, _llm_type, invoke, ainvoke, bind_tools too
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        """生成聊天响应"""

        # 1. 转换消息格式为API所需的纯文本提示
        user_input = self._convert_messages_to_prompt(messages)
        
        # 2. 调用自定义API获取原始响应
        response_content = self._call_api(user_input)
        
        # 3. 解析响应，判断是工具调用还是最终答案
        cleaned_content, tool_calls = self._parse_tool_calls(response_content)
        
        # 4. 根据解析结果，构造并返回 LangChain 期望的 AIMessage
        if tool_calls:
            # 如果模型决定调用工具，返回带有 tool_calls 的 AIMessage
            # LangChain Agent 会接管并执行这些工具
            message = AIMessage(
                content=cleaned_content, # 这里的content可以是模型在决定调用工具前的一些思考，也可以是空字符串
                tool_calls=[
                    {"name": tc['name'], "args": tc['args'], "id": f"call_{i}"} 
                    for i, tc in enumerate(tool_calls) # 添加一个简单的id
                ]
            )
            
        else:
            # 如果模型没有决定调用工具，返回纯文本 AIMessage (最终答案)
            message = AIMessage(content=cleaned_content)
            
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def invoke(
        self, 
        input: Union[List[BaseMessage], str], 
        config: Optional[RunnableConfig] = None, 
        **kwargs: Any
    ) -> AIMessage:
        """调用模型生成响应"""
        # 对于 LangChain Agent，通常会直接调用 _generate 或 _agenerate 但为了兼容性，保留 invoke
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
            
        result = self.generate([messages], **kwargs)
        return result.generations[0][0].message # 注意这里是 generations[0] 而不是 generations[0][0]
    
    async def _call_api_async(self, user_input: str) -> str:
        credentials = "TESTfdsfsddf"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/json"
        }
        
        # 构造请求体 - 取决于你得API参数
        payload = {
            "question": "",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
            "XXX": "XXX",
        }
        
        
        # 使用aiohttp进行异步请求
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
                ) as response:

                    response.raise_for_status()
                    
                    # 尝试解析JSON响应
                    try:
                        response_text = await response.text()
                        
                        response_data = json.loads(response_text)
                        res = response_data.get("ans", str(response_data))

                        
                        return res
                    except Exception as e:

                        print(f"错误: {str(e)}")
                        text_content = await response.text()
                        return text_content
                        
        except Exception as e:
            print(f"错误: {str(e)}")
            raise
    
    '''
    实现 _generate (或 _agenerate) 方法并返回正确的 ChatResult
    这是最关键的部分。create_react_agent 期望其底层的 LLM 在被调用时，能够根据其“思考”返回两种类型的 AIMessage
    工具调用Action: 如果模型决定使用工具, 它应该返回一个 AIMessage 对象，其 tool_calls 属性被填充（包含工具名称、参数和可选的 ID), 而 content 属性可以为空或包含模型的“思考”过程。
    最终答案Final Answer: 如果模型认为可以直接回答,它应该返回一个 AIMessage 对象，其 content 属性包含最终答案，而 tool_calls 属性为空。
    '''
    # override
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        
        # 1. 转换消息格式为API所需的纯文本提示
        user_input = self._convert_messages_to_prompt(messages)

        
        # 2. 异步调用自定义API获取原始响应
        response_content = await self._call_api_async(user_input)
        
        # 3. 解析响应，判断是工具调用还是最终答案
        cleaned_content, tool_calls = self._parse_tool_calls(response_content)

        
        # 4. 根据解析结果，构造并返回 LangChain 期望的 AIMessage
        if tool_calls:
            # 如果模型决定调用工具，返回带有 tool_calls 的 AIMessage
            message = AIMessage(
                content=cleaned_content,
                tool_calls=[
                    {"name": tc['name'], "args": tc['args'], "id": f"call_{i}"} 
                    for i, tc in enumerate(tool_calls)
                ]
            )
            
        else:
            # 如果模型没有决定调用工具，返回纯文本 AIMessage (最终答案)
            message = AIMessage(content=cleaned_content)
            
            

        
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def ainvoke(
        self, 
        input: Union[List[BaseMessage], str], 
        config: Optional[RunnableConfig] = None, 
        **kwargs: Any
    ) -> AIMessage:
        """异步调用模型生成响应"""
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
            
        result = await self.agenerate([messages], **kwargs)
        return result.generations[0][0].message
    
    @property
    def _llm_type(self) -> str:
        return "custom_api_model_for_react_agent"
    
    def bind_tools(self, tools: List[Any]) -> "CustomAPIModel":
        """绑定工具到模型"""
        new_instance = self.__class__(
            model_name=self.model_name,
            username=self.username,
            password=self.password,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools
        )
        return new_instance