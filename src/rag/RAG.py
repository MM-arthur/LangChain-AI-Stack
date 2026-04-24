from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Optional, Union
from langchain_core.documents import Document
from langchain_community.chat_models.moonshot import MoonshotChat
import os
from dotenv import load_dotenv
from tavily import TavilyClient
import logging

# 加载环境变量
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# 初始化模型
def init_model(model_name: str = None):
    if model_name is None:
        model_name = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
    
    if model_name.startswith("moonshot-"):
        from langchain_community.chat_models.moonshot import MoonshotChat
        return MoonshotChat(
            model=model_name,
            temperature=0.8,
            max_tokens=1024,
            api_key=os.getenv("MOONSHOT_API_KEY")
        )
    elif model_name == "claude-3-7-sonnet-latest":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=0.8,
            max_tokens=1024,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif model_name == "gpt-4o":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0.8,
            max_tokens=1024,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        from custom_api_llm.model import CustomAPIModel
        return CustomAPIModel(
            model_name="model",
            username=os.environ.get("API_custom_model_username"),
            password=os.environ.get("API_custom_model_password"),
            api_base=os.environ.get("API_custom_model_url"),
        )

# 初始化默认模型
model = init_model()

current_dir = Path(__file__).parent
cache_path = current_dir.parent / "rag" / "models"

embeddings = None

def get_embeddings():
    """延迟获取embeddings实例，支持从ModelScope加载Qwen3-Embedding-0.6B"""
    global embeddings
    if embeddings is None:
        import os
        import sys
        
        torch_lib_path = None
        try:
            import torch
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
            print(f"✅ 找到PyTorch lib目录: {torch_lib_path}")
            
            if torch_lib_path not in os.environ['PATH']:
                os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ['PATH']
                print(f"✅ 将PyTorch lib目录添加到PATH: {torch_lib_path}")
        except Exception as e:
            print(f"⚠️  获取PyTorch lib目录失败: {e}")
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
        
        if embedding_model == "qwen3-embedding-0.6b":
            print(f"🔄 加载 Qwen3-Embedding-0.6B 模型...")
            from modelscope import snapshot_download
            
            model_dir = snapshot_download(
                'Qwen/Qwen3-Embedding-0.6B',
                cache_dir=str(cache_path)
            )
            print(f"✅ 模型下载完成: {model_dir}")
            
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=model_dir,
                model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Qwen3-Embedding-0.6B 加载成功")
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_path = os.getenv("EMBEDDING_MODEL_NAME_PATH", str(cache_path / "all-MiniLM-L6-v2" / "all-MiniLM-L6-v2-main"))
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'},
                cache_folder=str(cache_path)
            )
    return embeddings

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
    import requests
    from bs4 import BeautifulSoup
    
    # 从URL加载文档
    if urls:
        raw_docs = []
        print(f"📄 开始加载 {len(urls)} 个URL...")
        for i, url in enumerate(urls):
            try:
                print(f"   [{i+1}/{len(urls)}] 加载: {url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                article = soup.find('article') or soup.find('div', class_='article-content') or soup.find('div', id='article_content') or soup.find('div', class_='markdown_views')
                
                if article:
                    content = article.get_text(separator='\n', strip=True)
                else:
                    content = soup.get_text(separator='\n', strip=True)
                
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                clean_content = '\n'.join(lines)
                
                doc = Document(
                    page_content=clean_content,
                    metadata={'source': url}
                )
                raw_docs.append(doc)
                print(f"      ✅ 成功，内容长度: {len(clean_content)} 字符")
            except Exception as e:
                print(f"      ❌ 失败: {e}")
                continue
        
        print(f"✅ 成功加载 {len(raw_docs)} 个文档")
    elif file_path:
        logger.info(f"开始从本地路径{file_path}加载文档")
        loader = DirectoryLoader(file_path)
        raw_docs = loader.load()
        logger.info(f"成功加载{len(raw_docs)}个文档")
    else:
        raise ValueError("必须提供urls或file_path参数")
    
    print("📝 开始对文档进行分块处理...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )


    chunks = text_splitter.split_documents(raw_docs)
    print(f"   分块完成，共 {len(chunks)} 个块")
    
    filtered_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > 100:
            filtered_chunks.append(chunk)
        else:
            pass
    
    print(f"✅ 文档处理完成，共 {len(filtered_chunks)} 个有效块")



    return filtered_chunks

def create_vector_store(documents: List[Document]) -> FAISS:
    """
    从文档创建FAISS向量存储
    Args:
        documents: List[Document] - 文档列表
    Returns:
        FAISS - 向量存储实例
    """
    print(f"🔨 创建向量存储，共 {len(documents)} 个文档块...")
    print(f"⏳ 正在进行向量化（CPU模式，预计需要 1-3 分钟）...")
    
    embeddings = get_embeddings()
    
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        progress = min(i + batch_size, len(texts))
        print(f"   向量化进度: {progress}/{len(texts)} ({progress*100//len(texts)}%)")
    
    print(f"   构建FAISS索引...")
    text_embedding_pairs = list(zip(texts, all_embeddings))
    vs = FAISS.from_embeddings(text_embedding_pairs, embeddings, metadatas=metadatas)
    
    print(f"✅ 向量存储创建完成")
    return vs

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
        search_kwargs={"k": 5}
    )
    
    # 合并多个查询的结果
    all_docs = []
    for q in expanded_questions:
        docs = retriever.invoke(q)
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
        f"Source {doc.metadata.get('source', '未知')} (Page {doc.metadata.get('page', '?')}): {doc.page_content}"
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
        - 包含内容来源的引用标记，引用格式为[Source URL或文件路径]，其中URL或文件路径是上下文中标注的完整来源地址
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

_vector_store_cache = None
_cache_urls_key = None

# FAISS 持久化路径
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_FILE = FAISS_INDEX_DIR / "meta.json"


def _load_personal_docs_timestamp(knowledge_base_path: str) -> float:
    """获取个人知识库最新文件的修改时间戳（用于判断是否需要重建索引）"""
    import time
    if not os.path.exists(knowledge_base_path):
        return 0.0
    latest_mtime = 0.0
    for root, dirs, files in os.walk(knowledge_base_path):
        for f in files:
            if f.endswith(('.pdf', '.md')):
                fp = os.path.join(root, f)
                mtime = os.path.getmtime(fp)
                if mtime > latest_mtime:
                    latest_mtime = mtime
    return latest_mtime


def _save_vector_store_meta(cache_key: str, personal_mtime: float):
    """保存 FAISS 索引元数据"""
    import json
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "cache_key": cache_key,
        "personal_mtime": personal_mtime,
        "saved_at": os.path.getmtime(str(FAISS_INDEX_FILE)) if FAISS_INDEX_FILE.exists() else None
    }
    with open(FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _load_vector_store_meta() -> dict:
    """加载 FAISS 索引元数据"""
    import json
    if not FAISS_META_FILE.exists():
        return {}
    with open(FAISS_META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_personal_knowledge(knowledge_base_path: str = None) -> List[Document]:
    """
    加载 Arthur 个人知识库：简历、JD、行业知识等
    敏感文件，留在本地，不上传 GitHub
    """
    if knowledge_base_path is None:
        knowledge_base_path = os.getenv("RAG_KNOWLEDGE_BASE", "")
    
    if not knowledge_base_path or not os.path.exists(knowledge_base_path):
        logger.info("未配置个人知识库路径或路径不存在，跳过")
        return []
    
    docs = []
    
    # 加载 PDF 文件（简历、工作证明等）
    pdf_files = [f for f in os.listdir(knowledge_base_path) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(knowledge_base_path, pdf_file)
        try:
            logger.info(f"📄 加载 PDF: {pdf_file}")
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["source"] = f"个人资料/{pdf_file}"
                doc.metadata["type"] = "personal_resume"
            docs.extend(pdf_docs)
            logger.info(f"  ✅ 加载 {len(pdf_docs)} 页")
        except Exception as e:
            logger.warning(f"  ⚠️  PDF 加载失败 {pdf_file}: {e}")
    
    # 加载 Markdown 文件（SOP、笔记等）
    md_files = [f for f in os.listdir(knowledge_base_path) if f.endswith(".md")]
    for md_file in md_files:
        md_path = os.path.join(knowledge_base_path, md_file)
        try:
            logger.info(f"📝 加载 Markdown: {md_file}")
            loader = UnstructuredMarkdownLoader(md_path)
            md_docs = loader.load()
            for doc in md_docs:
                doc.metadata["source"] = f"个人资料/{md_file}"
                doc.metadata["type"] = "personal_note"
            docs.extend(md_docs)
            logger.info(f"  ✅ 加载 {len(md_docs)} 段")
        except Exception as e:
            logger.warning(f"  ⚠️  Markdown 加载失败 {md_file}: {e}")
    
    logger.info(f"个人知识库加载完成，共 {len(docs)} 个文档")
    return docs

def add_documents_to_vector_store(existing_vs: FAISS, new_docs: List[Document]) -> FAISS:
    """
    将新文档追加到已有的 FAISS 向量存储（不重建索引）
    """
    if not new_docs:
        return existing_vs
    
    texts = [doc.page_content for doc in new_docs]
    metadatas = [doc.metadata for doc in new_docs]
    
    embeddings = get_embeddings()
    texts_with_embeddings = list(zip(texts, embeddings.embed_documents(texts)))
    
    existing_vs.add_embeddings(texts_with_embeddings, metadatas=metadatas)
    logger.info(f"已追加 {len(new_docs)} 个文档到向量存储")
    return existing_vs

def get_or_create_vector_store(urls: List[str] = None, file_path: str = None) -> FAISS:
    """
    获取或创建向量存储（带缓存 + FAISS 持久化）
    
    知识库构成：
    1. CSDN 博客（通用技术知识）
    2. Arthur 个人知识库（简历 + JD + 行业知识，本地私有）
    
    持久化策略：
    - 启动时检查 faiss_index/ 是否存在且 cache_key 匹配
    - 如果个人文档有更新（mtime 变化），强制重建
    - 重建后自动保存到 faiss_index/
    """
    global _vector_store_cache, _cache_urls_key
    
    cache_key = str(sorted(urls)) if urls else file_path
    personal_base = os.getenv("RAG_KNOWLEDGE_BASE", "")
    personal_mtime = _load_personal_docs_timestamp(personal_base) if personal_base else 0.0
    
    # Step 1: 尝试从持久化索引加载
    if FAISS_INDEX_FILE.exists():
        meta = _load_vector_store_meta()
        if (
            meta.get("cache_key") == cache_key
            and meta.get("personal_mtime", 0) >= personal_mtime
        ):
            try:
                logger.info("📦 从 FAISS 持久化索引加载...（启动加速）")
                embeddings = get_embeddings()
                _vector_store_cache = FAISS.load_local(
                    str(FAISS_INDEX_DIR),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                _cache_urls_key = cache_key
                logger.info("✅ FAISS 持久化索引加载成功")
                return _vector_store_cache
            except Exception as e:
                logger.warning(f"⚠️  FAISS 索引加载失败，将重建: {e}")
    
    # Step 2: 内存缓存命中
    if _vector_store_cache is not None and _cache_urls_key == cache_key:
        print("📦 使用内存缓存的向量存储")
        return _vector_store_cache
    
    # Step 3: 需要重建索引
    print("🔄 重建向量存储...")
    
    # 3a. 加载 CSDN 博客
    if urls or file_path:
        chunked_docs = load_and_chunk_documents(urls=urls, file_path=file_path)
        _vector_store_cache = create_vector_store(chunked_docs)
        logger.info("✅ CSDN 博客索引创建完成")
    else:
        _vector_store_cache = None
    
    # 3b. 追加 Arthur 个人知识库
    personal_docs = load_personal_knowledge()
    if personal_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            add_start_index=True
        )
        personal_chunks = text_splitter.split_documents(personal_docs)
        logger.info(f"个人文档分块完成，共 {len(personal_chunks)} 个块")
        
        if _vector_store_cache is None:
            _vector_store_cache = create_vector_store(personal_chunks)
        else:
            _vector_store_cache = add_documents_to_vector_store(_vector_store_cache, personal_chunks)
        logger.info("✅ 个人知识库追加完成")
    
    _cache_urls_key = cache_key
    
    # Step 4: 持久化到磁盘
    try:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vector_store_cache.save_local(str(FAISS_INDEX_DIR))
        _save_vector_store_meta(cache_key, personal_mtime)
        logger.info(f"✅ FAISS 索引已保存到 {FAISS_INDEX_DIR}")
    except Exception as e:
        logger.warning(f"⚠️  FAISS 索引保存失败: {e}")
    
    return _vector_store_cache

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()

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
    vector_store = get_or_create_vector_store(urls=urls, file_path=file_path)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    context = "\n\n".join(
        f"Source {doc.metadata.get('source', '未知')}: {doc.page_content}"
        for doc in docs
    )
    
    prompt = ChatPromptTemplate.from_template(
        """你是一个专业的问答助手。请仅使用以下上下文信息回答问题：
        {context}
        
        当前日期: {date}
        问题: {question}
        
        回答要求：
        - 仔细分析上下文，提取关键信息
        - 如果上下文不包含答案，请说明"根据已知信息无法回答该问题"
        - 包含内容来源的引用标记，引用格式为[Source URL或文件路径]
        - 使用简洁的中文回答
        """
    )
    
    chain = (
        {"context": lambda _: context, 
         "question": lambda x: x["question"],
         "date": lambda _: os.getenv("CURRENT_DATE", "2023-10-01")}
        | prompt
        | model
        | StrOutputParser()
    )
    
    answer = chain.invoke({"question": question})
    
    return {"answer": answer, "documents": docs}

# 使用示例
if __name__ == "__main__":
    # 从URL获取答案
    url_response = run_rag(
        question="什么是MySQL库表设计?",
        urls=[
            "https://blog.csdn.net/qq_40550384/article/details/149981605",
            "https://blog.csdn.net/qq_40550384/article/details/156398145",
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