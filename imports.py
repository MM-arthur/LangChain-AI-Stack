from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import json
from pydantic import BaseModel, ValidationError