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
from tavily import TavilyClient
from langchain_community.document_loaders import PlaywrightLoader
import logging

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
        # 使用Playwright加载动态网页
        loader = PlaywrightLoader(
            urls=urls,
            remove_selectors=["header", "footer", "nav"]  # 移除无关元素
        )
        raw_docs = loader.load()
        # 添加延迟确保内容加载
        raw_docs = loader.load(wait_until="networkidle", timeout=10000)
        logger.info(f"成功加载{len(raw_docs)}个文档")
    elif file_path:
        logger.info(f"开始从本地路径{file_path}加载文档")
        loader = DirectoryLoader(file_path)
        raw_docs = loader.load()
        logger.info(f"成功加载{len(raw_docs)}个文档")
    else:
        raise ValueError("必须提供urls或file_path参数")
    
    # 文档分块处理
    logger.info("开始对文档进行分块处理")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )


    chunks = text_splitter.split_documents(raw_docs)
    # 添加文档质量过滤
    filtered_chunks = []
    for chunk in chunks:
        # 过滤掉太短的文档块
        if len(chunk.page_content) > 100:
            filtered_chunks.append(chunk)
        else:
            logger.debug(f"跳过太短的文档块: {len(chunk.page_content)}字符")
    
    logger.info(f"文档分块完成，共{len(filtered_chunks)}个有效块(过滤掉{len(chunks)-len(filtered_chunks)}个无效块)")



    return filtered_chunks

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
    # 查询扩展
    original_question = state["question"]
    # 可以使用LLM生成相关查询
    expanded_questions = [original_question]
    logger.info(f"原始查询: {original_question}")
    
    # 如果向量存储不存在则创建
    if not state.get("vector_store"):
        logger.info("向量存储不存在，开始创建")
        state["vector_store"] = create_vector_store(state["documents"])
        logger.info("向量存储创建完成")
    
    # 设置检索器并获取相关文档
    logger.info(f"开始检索与问题相关的文档")
    retriever = state["vector_store"].as_retriever(
        search_kwargs={"k": 5, "score_threshold": 0.7}
    )
    
    # 合并多个查询的结果
    all_docs = []
    for q in expanded_questions:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)
    
    # 去重
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    # 按相关性排序
    unique_docs = sorted(unique_docs, key=lambda x: x.metadata.get('score', 0), reverse=True)[:5]
    
    logger.info(f"检索完成，共{len(unique_docs)}个相关文档")
    return {"documents": list(unique_docs)}


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
        - 仔细分析上下文，提取关键信息
        - 如果上下文不包含答案，请说明"根据已知信息无法回答该问题"
        - 包含内容来源的引用标记 [Source X]，其中X是文档编号
        - 对复杂概念进行清晰解释
        - 使用简洁的中文回答
        - 保持客观中立的态度
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

    logger.info(f"生成答案使用的上下文长度: {len(context)}字符")
    logger.debug(f"最终答案: {answer}")

    return {"answer": answer}

# 构建状态图
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

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
        question="什么是MySQL库表设计?",
        urls=[
            "https://blog.csdn.net/qq_40550384/article/details/146161589",
            "https://blog.csdn.net/qq_40550384/article/details/147500510",
            "https://blog.csdn.net/qq_40550384/article/details/132668097",
            "https://doi.org/10.1016/j.apacoust.2020.107647",
            "https://xueshu.baidu.com/usercenter/paper/show?paperid=1s740c805s0d0p10f53y0j10pq344792&site=xueshu_se",
            "https://xueshu.baidu.com/usercenter/paper/show?paperid=1u3a0ar0r50n0ev0582t0mk008192348&site=xueshu_se",
            "https://blog.csdn.net/qq_40550384/article/details/131190132",
            "https://blog.csdn.net/qq_40550384/article/details/131660477",
            "https://blog.csdn.net/qq_40550384/article/details/131423567",
            "https://blog.csdn.net/qq_40550384/article/details/131287048",
            "https://blog.csdn.net/qq_40550384/article/details/131261254",
            "https://blog.csdn.net/qq_40550384/article/details/114605673",
            "https://blog.csdn.net/qq_40550384/article/details/112346545",
            "https://blog.csdn.net/qq_40550384/article/details/112985413",
            "https://blog.csdn.net/qq_40550384/article/details/111309984",
            "https://blog.csdn.net/qq_40550384/article/details/110039473",
            "https://blog.csdn.net/qq_40550384/article/details/108440628",
            "https://blog.csdn.net/qq_40550384/article/details/109659915",
        ]
    )
    print(f"\n=== 答案 ===\n{url_response['answer']}\n")
    print(f"\n=== 检索内容 ===\n")
    for i, doc in enumerate(url_response["documents"]):
        print(f"文档 {i+1} (来源: {doc.metadata.get('source', '未知')}):\n{doc.page_content}\n{'='*50}")
    
    # 从本地文件获取答案(示例路径，请根据实际情况修改)
    # file_response = run_rag(
    #     question="文档中提到的关键概念是什么？",
    #     file_path="path/to/your/local/files"
    # )
    # print("从本地文件获取的答案:\n", file_response["answer"])
    # print("\n检索到的内容:\n")
    # for i, doc in enumerate(file_response["documents"]):
    #     print(f"文档 {i+1} (来源: {doc.metadata.get('source', '未知')}):\n{doc.page_content}\n")