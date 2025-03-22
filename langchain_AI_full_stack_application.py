import os
from dotenv import load_dotenv
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("MOONSHOT_API_KEY") # if you have paid API, it will be more powerful

if not api_key:
    raise ValueError("api key not find")

chat = MoonshotChat(
    model="moonshot-v1-8k",
    temperature=0.8,
    max_tokens=20,
    api_key=api_key
)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="什么是AI?")
]

response = chat(messages)
print(response)
