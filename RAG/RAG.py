from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Optional, Union
from langchain_core.documents import Document
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

model = MoonshotChat(
    model="moonshot-v1-8k",
    temperature=0.8,
    max_tokens=1024,
    api_key=os.getenv("MOONSHOT_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL_NAME_PATH"),
    model_kwargs={'device': 'cpu'},
    cache_folder=os.getenv("EMBEDDING_MODEL_CACHE_PATH")
)

# 定义RAG状态类型
class RAGState(TypedDict):
    """
    RAG工作流的状态定义
    Attributes:
        question: str - 用户问题
        documents: List[Document] - 处理后的文档列表
        answer: str - 生成的答案
        vector_store: Optional[FAISS] - 向量存储(延迟初始化)
    """
    question: str
    documents: List[Document]
    answer: str
    vector_store: Optional[FAISS] = None

# 文档加载和预处理函数
def load_and_chunk_documents(urls: List[str] = None, file_path: str = None) -> List[Document]:
    """
    从URL或本地文件加载文档并进行分块处理
    Args:
        urls: List[str] - 网页URL列表(可选)
        file_path: str - 本地文件路径(可选)
    Returns:
        List[Document] - 分块后的文档列表
    """
    # 从URL加载文档
    if urls:
        loader = WebBaseLoader(urls)
        raw_docs = loader.load()
    # 从本地文件加载文档
    elif file_path:
        loader = DirectoryLoader(file_path)
        raw_docs = loader.load()
    else:
        raise ValueError("必须提供urls或file_path参数")
    
    # 文档分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(raw_docs)

# 创建向量存储函数
def create_vector_store(documents: List[Document]) -> FAISS:
    """
    从文档创建FAISS向量存储
    Args:
        documents: List[Document] - 文档列表
    Returns:
        FAISS - 向量存储实例
    """
    return FAISS.from_documents(documents, embeddings)

# 检索节点函数
def retrieve_documents(state: RAGState):
    """
    检索相关文档
    Args:
        state: RAGState - 当前RAG状态
    Returns:
        dict - 包含检索到文档的新状态
    """
    # 如果向量存储不存在则创建
    if not state.get("vector_store"):
        state["vector_store"] = create_vector_store(state["documents"])
    
    # 设置检索器并获取相关文档
    retriever = state["vector_store"].as_retriever(
        search_kwargs={"k": 5}  # 控制检索文档数量
    )
    docs = retriever.get_relevant_documents(state["question"])
    return {"documents": docs}

# 生成答案函数
def generate_answer(state: RAGState):
    """
    根据检索到的文档生成答案
    Args:
        state: RAGState - 当前RAG状态
    Returns:
        dict - 包含生成答案的新状态
    """
    # 构建上下文字符串
    context = "\n\n".join(
        f"Source {i+1} (Page {doc.metadata.get('page', '?')}): {doc.page_content}"
        for i, doc in enumerate(state["documents"])
    )
    
    # 设置提示模板
    prompt = ChatPromptTemplate.from_template(
        """你是一个专业的问答助手。请仅使用以下上下文信息回答问题：
        {context}
        
        当前日期: {date}
        问题: {question}
        
        回答要求：
        - 如果上下文不包含答案，请说明"根据已知信息无法回答该问题"
        - 包含内容来源的引用标记 [Source X]
        - 使用简洁的中文回答
        """
    )
    
    # 构建处理链
    chain = (
        {"context": lambda _: context, 
         "question": lambda x: x["question"],
         "date": lambda _: os.getenv("CURRENT_DATE", "2023-10-01")}
        | prompt
        | model
        | StrOutputParser()
    )
    
    # 调用链生成答案
    answer = chain.invoke(state)
    return {"answer": answer}

# 构建状态图
workflow = StateGraph(RAGState)

# 添加节点
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# 设置工作流
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 编译工作流
rag_app = workflow.compile()

# 运行入口函数
def run_rag(question: str, urls: List[str] = None, file_path: str = None):
    """
    RAG运行入口
    Args:
        question: str - 用户问题
        urls: List[str] - 网页URL列表(可选)
        file_path: str - 本地文件路径(可选)
    Returns:
        str - 生成的答案
    """
    # 预处理文档
    chunked_docs = load_and_chunk_documents(urls=urls, file_path=file_path)
    
    # 初始化状态
    initial_state = {
        "question": question,
        "documents": chunked_docs,
        "answer": "",
        "vector_store": None  # 延迟初始化
    }
    
    # 执行工作流
    result = rag_app.invoke(initial_state)
    return result

# 使用示例
if __name__ == "__main__":
    # 从URL获取答案
    url_response = run_rag(
        question="LangChain是什么？",
        urls=["https://python.langchain.com/docs/get_started/introduction"]
    )
    print("从URL获取的答案:\n", url_response["answer"])
    print("\n检索到的内容:\n")
    for i, doc in enumerate(url_response["documents"]):
        print(f"文档 {i+1} (来源: {doc.metadata.get('source', '未知')}):\n{doc.page_content}\n")
    
    # 从本地文件获取答案(示例路径，请根据实际情况修改)
    # file_response = run_rag(
    #     question="文档中提到的关键概念是什么？",
    #     file_path="path/to/your/local/files"
    # )
    # print("从本地文件获取的答案:\n", file_response["answer"])
    # print("\n检索到的内容:\n")
    # for i, doc in enumerate(file_response["documents"]):
    #     print(f"文档 {i+1} (来源: {doc.metadata.get('source', '未知')}):\n{doc.page_content}\n")