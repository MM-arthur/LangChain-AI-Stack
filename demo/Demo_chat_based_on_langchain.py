import os
from dotenv import load_dotenv
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict

load_dotenv()

# output parser
response_schemas = [
    ResponseSchema(name="answer", description="The main answer to the question"),
    ResponseSchema(name="confidence", description="Confidence score from 0-1")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，请用JSON格式回答，包含answer和confidence字段"),
    ("human", "{input_question}")
])

# LLM
chat = MoonshotChat(
    model="moonshot-v1-8k",
    temperature=0.8,
    max_tokens=1024,
    api_key=os.getenv("MOONSHOT_API_KEY")
)

# chain
chain = (
    {"input_question": RunnablePassthrough()}
    | prompt
    | chat
    | output_parser
)

# Run the chain
def process_query(query: str) -> Dict:
    try:
        result = chain.invoke(query)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    question = "什么是人工智能？"
    result = process_query(question)
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")