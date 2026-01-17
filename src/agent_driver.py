import json
import uuid
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from typing import Dict, Any, Callable, Optional, List
from langchain_core.messages.ai import AIMessageChunk


# Global flags and sets for tracking state across events within a single query
_final_text_sent_for_query = False
_sent_tool_calls_canonical = set() 


'''
astream_graph 是一个中间层，它将 LangGraph Agent 复杂的、内部的、细粒度的执行事件流，
适配并转换为你的前端界面能够直接理解和渲染的、统一的、高层次的、用户友好的实时消息流。
它不负责 Agent 的决策和核心计算, 但它负责驱动Agent的执行, 然后观察、解释、格式化和广播 Agent 的执行过程
'''

async def astream_graph(
    agent_runnable: CompiledStateGraph,
    input_data: Dict[str, Any],
    callback,
    config: RunnableConfig
):
    """
    Streams events from LangGraph, extracts text and tool calls, and sends them via callback.
    """
    global _final_text_sent_for_query, _sent_tool_calls_canonical
    _final_text_sent_for_query = False # Reset for each new query
    _sent_tool_calls_canonical = set() # Reset for each new query

    async for event in agent_runnable.astream_events(input_data, config=config, version="v1"):
        kind = event["event"]
        print(f"[DEBUG][utils.py] Received event kind: {kind}, data_keys: {event.get('data', {}).keys()}")
        
        # --- Handle on_tool_end events first ---
        # This is where the tool's actual output becomes available
        if kind == "on_tool_end":
            tool_output_obj = event["data"].get("output")
            tool_call_id = getattr(tool_output_obj, 'tool_call_id', None) 

            tool_content_to_send = ""
            if isinstance(tool_output_obj, BaseMessage):
                # If it's a BaseMessage (like ToolMessage), get its actual content
                tool_content_to_send = tool_output_obj.content
            elif isinstance(tool_output_obj, (dict, list)):
                # If it's a dict/list (e.g., tool returned JSON), serialize it
                try:
                    tool_content_to_send = json.dumps(tool_output_obj, ensure_ascii=False, indent=2)
                except TypeError:
                    tool_content_to_send = repr(tool_output_obj) # Fallback
            else:
                # Otherwise, just convert to string
                tool_content_to_send = str(tool_output_obj)

            await callback({
                "type": "tool_response",
                "content": tool_content_to_send,
                "tool_call_id": tool_call_id
            })
            print(f"[DEBUG][utils.py] Sent tool response message from on_tool_end: {tool_content_to_send}")
            continue

        # --- Handle message-containing events (on_chat_model_stream, on_chain_stream, on_chain_end, on_final_output) ---
        
        message_or_chunk = None
        if kind == "on_chat_model_stream":
            message_or_chunk = event["data"].get("chunk")
        elif kind == "on_chain_stream":
            message_or_chunk = event["data"].get("chunk")
            if isinstance(message_or_chunk, dict) and "messages" in message_or_chunk and message_or_chunk["messages"]:
                message_or_chunk = message_or_chunk["messages"][-1]
        elif kind == "on_chain_end":
            message_or_chunk = event["data"].get("output")
            if isinstance(message_or_chunk, dict) and "messages" in message_or_chunk and message_or_chunk["messages"]:
                message_or_chunk = message_or_chunk["messages"][-1]
        elif kind == "on_final_output":
            message_or_chunk = event["data"].get("output")
            if isinstance(message_or_chunk, dict) and "messages" in message_or_chunk and message_or_chunk["messages"]:
                message_or_chunk = message_or_chunk["messages"][-1]
        
        # Only process if we successfully extracted a BaseMessage or AIMessageChunk
        if not isinstance(message_or_chunk, (BaseMessage, AIMessageChunk)):
            if kind == "on_tool_start":
                print(f"[DEBUG][utils.py] Skipping on_tool_start event as it's handled by on_tool_end for response and AIMessage for call.")
            continue 

        # --- Process extracted message_or_chunk ---
        
        # 1. Extract and send tool calls (from AIMessage/AIMessageChunk)
        if hasattr(message_or_chunk, 'tool_calls') and message_or_chunk.tool_calls:
            for tool_call in message_or_chunk.tool_calls:
                tool_name = getattr(tool_call, 'name', None)
                if tool_name is None: 
                    tool_name = tool_call.get('name')

                tool_args = getattr(tool_call, 'args', None)
                if tool_args is None: 
                    tool_args = tool_call.get('args')

                if tool_name is not None and tool_args is not None:
                    tool_args_for_canonical = tool_args
                    if not isinstance(tool_args, dict):
                        try:
                            tool_args_for_canonical = tool_args.model_dump() if hasattr(tool_args, 'model_dump') else dict(tool_args)
                        except Exception:
                            tool_args_for_canonical = tool_args 

                    try:
                        canonical_args = json.dumps(tool_args_for_canonical, sort_keys=True, ensure_ascii=False)
                    except TypeError:
                        canonical_args = repr(tool_args_for_canonical) 
                    
                    canonical_tool_call = f"{tool_name}:{canonical_args}"

                    if canonical_tool_call in _sent_tool_calls_canonical:
                        print(f"[DEBUG][utils.py] Skipping duplicate tool call message for canonical: {canonical_tool_call}. Already sent.")
                        continue

                    tool_call_id_for_frontend = getattr(tool_call, 'id', str(uuid.uuid4()))

                    await callback({
                        "type": "tool",
                        "content": json.dumps({"tool_name": tool_name, "tool_input": tool_args}, ensure_ascii=False),
                        "tool_call_id": tool_call_id_for_frontend 
                    })
                    _sent_tool_calls_canonical.add(canonical_tool_call)
                    print(f"[DEBUG][utils.py] Sent tool (from {kind} tool_calls): {tool_name}, args: {tool_args}, Canonical: {canonical_tool_call}")
                else:
                    print(f"[DEBUG][utils.py] WARNING: Could not extract tool_name or tool_args from tool_call: {tool_call}. Skipping this tool event.")
        
        # Anthropic style content list with "tool_use" dicts (fallback/alternative)
        if isinstance(message_or_chunk.content, list):
            for part in message_or_chunk.content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_use_id = part.get('id', None)
                    if tool_use_id is None:
                        tool_use_id = str(uuid.uuid4())
                        print(f"[DEBUG] WARNING: Anthropic tool_use has no ID. Generating new ID: {tool_use_id} for {part}")

                    canonical_tool_call = f"anthropic_tool_use:{tool_use_id}"

                    if canonical_tool_call in _sent_tool_calls_canonical:
                        print(f"[DEBUG][utils.py] Skipping duplicate anthropic tool_use message for canonical: {canonical_tool_call}. Already sent.")
                        continue

                    await callback({
                        "type": "tool",
                        "content": json.dumps(part, ensure_ascii=False),
                        "tool_call_id": tool_use_id
                    })
                    _sent_tool_calls_canonical.add(canonical_tool_call)
                    print(f"[DEBUG][utils.py] Sent tool (from {kind} content list tool_use): {part}, Canonical: {canonical_tool_call}")

        # 2. Extract and send text content (with is_final flag)
        text_content = ""
        # MODIFIED: Only extract text content if the message is an AIMessage
        # AND it does NOT contain tool_calls (meaning it's a pure text response from LLM)
        if isinstance(message_or_chunk, AIMessage) and not message_or_chunk.tool_calls:
            if isinstance(message_or_chunk.content, str):
                text_content = message_or_chunk.content
            elif isinstance(message_or_chunk.content, list): # For multi-modal content
                for part in message_or_chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        text_content += part["text"]
        
        if text_content:
            is_final_output_event = (kind == "on_final_output") 
            is_chain_end_event_of_final_message = (kind == "on_chain_end" and isinstance(message_or_chunk, AIMessage))

            is_final_flag = False
            if is_final_output_event:
                is_final_flag = True
            elif not is_final_output_event and is_chain_end_event_of_final_message:
                is_final_flag = True

            if is_final_flag:
                if not _final_text_sent_for_query:
                    await callback({"type": "text", "content": text_content, "is_final": True})
                    print(f"[DEBUG][utils.py] Sent FINAL text from {kind}: {text_content}")
                    _final_text_sent_for_query = True
                else:
                    print(f"[DEBUG][utils.py] Skipped sending duplicate final text from {kind}.")
            elif not _final_text_sent_for_query:
                await callback({"type": "text", "content": text_content, "is_final": False})
                print(f"[DEBUG][utils.py] Sent streaming text from {kind}: {text_content}")
            else:
                print(f"[DEBUG][utils.py] Skipped sending streaming text after final answer from {kind}.")



'''
若只要结果
async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
) -> Dict[str, Any]:
    pass
'''


