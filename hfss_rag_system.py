import re
import string
import torch
import os
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据清洗函数
def clean_hfss_manual(text):
    # 1. 去除特殊标记
    text = re.sub(r'<.*?>', '', text)

    # 2. 处理公式排版问题
    text = re.sub(r'\n+', '\n', text)  # 合并连续换行
    text = re.sub(r'(\d)\n(\d)', r'\1\2', text)  # 修复公式换行

    # 3. 去除冗余页脚
    text = re.sub(r'^.*?微波仿真论坛.*$', '', text, flags=re.MULTILINE)

    # 4. 标准化术语
    term_mapping = {
        '集总端口': '集总参数端口',
        '波端口': '波导端口',
        '有限元法': '有限元法(FEM)',
        '电压驻波比': '电压驻波比(VSWR)',
        '雷达横截面': '雷达横截面(RCS)',
        '电磁兼容性': '电磁兼容性(EMC)',
        '电磁干扰': '电磁干扰(EMI)',
        'HFSS': 'HFSS(High Frequency Structure Simulator)',
        '仿真': '电磁仿真',
        '天线': '天线设计',
        '滤波器': '滤波器设计',
        '参数扫描': '参数扫描分析',
        '优化': '电磁结构优化',
    }
    for old_term, new_term in term_mapping.items():
        text = text.replace(old_term, new_term)

    # 5. 保留标点符号，但移除非打印字符
    printable = set(string.printable + '，。！？：；''""【】《》、（）…—')
    text = ''.join(c for c in text if c in printable)

    # 6. 合并段落
    text = re.sub(r'\n\s*\n', '\n', text)  # 保留单空行
    return text

# 集成数据清洗的文档加载与分块
def load_and_chunk_documents(file_paths):
    all_chunks = []
    for file_path in file_paths:
        try:
            logging.info(f"开始加载PDF文件: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logging.info(f"PDF加载完成，共 {len(documents)} 页")

            # 合并所有页面内容
            all_text = ""
            for doc in documents:
                all_text += doc.page_content + "\n\n"

            logging.info("开始清洗文本...")
            cleaned_text = clean_hfss_manual(all_text)
            logging.info(f"文本清洗完成，清洗后字符数: {len(cleaned_text)}")

            logging.info("开始文本分块...")
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            chunks = text_splitter.split_text(cleaned_text)
            logging.info(f"文本分块完成，共 {len(chunks)} 个块")

            # 为每个块添加来源信息
            file_name = os.path.basename(file_path)
            chunks = [f"[来源: {file_name}] " + chunk for chunk in chunks]
            all_chunks.extend(chunks)

        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}")
            continue

    logging.info(f"总共处理了 {len(file_paths)} 个文件，生成了 {len(all_chunks)} 个文本块")
    return all_chunks

# 向量化与知识库构建
def build_knowledge_base(chunks):
    try:
        logging.info("开始构建向量数据库...")
        embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        logging.info("向量数据库构建完成")
    except Exception as e:
        logging.error(f"构建向量数据库时出错: {e}")
        return None
    return knowledge_base

# 保存和加载知识库
def save_knowledge_base(knowledge_base, directory="knowledge_base"):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        knowledge_base.save_local(directory)
        logging.info(f"知识库已保存至 {directory} 目录")
    except Exception as e:
        logging.error(f"保存知识库时出错: {e}")

def load_knowledge_base(directory="knowledge_base", embeddings=None):
    try:
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        knowledge_base = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"已从 {directory} 加载知识库")
    except Exception as e:
        logging.error(f"加载知识库时出错: {e}")
        return None
    return knowledge_base

# 大语言模型配置
def setup_llm():
    try:
        logging.info("正在加载语言模型...")
        model_name = "deepseek-ai/DeepSeek-V3-0324"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("语言模型加载完成")
    except Exception as e:
        logging.error(f"加载语言模型时出错: {e}")
        return None
    return llm

# 构建检索增强生成链
def build_rag_chain(knowledge_base, llm):
    try:
        logging.info("构建RAG检索链...")
        retriever = knowledge_base.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        logging.info("RAG检索链构建完成")
    except Exception as e:
        logging.error(f"构建RAG检索链时出错: {e}")
        return None
    return qa_chain

# 问答交互
def ask_question(qa_chain, question):
    try:
        logging.info(f"\n问题: {question}")
        logging.info("正在查询...")
        result = qa_chain({"query": question})

        logging.info("\n回答:")
        logging.info(result["result"])

        logging.info("\n参考文档片段:")
        for i, doc in enumerate(result["source_documents"][:3], 1):
            logging.info(f"\n--- 文档 {i} ---")
            logging.info(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    except Exception as e:
        logging.error(f"问答交互时出错: {e}")
    return result

def interactive_qa(qa_chain):
    print("\n=== 电磁兼容知识库问答系统 ===")
    print("输入问题进行查询，输入'exit'退出")

    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'exit':
            print("谢谢使用，再见!")
            break

        result = ask_question(qa_chain, question)
        print("\n回答:")
        print(result["result"])

        print("\n参考文档片段:")
        for i, doc in enumerate(result["source_documents"][:3], 1):
            print(f"\n--- 文档 {i} ---")
            content = doc.page_content
            source_info = ""
            if "[来源:" in content:
                source_end = content.find("]")
                if source_end > 0:
                    source_info = content[:source_end+1]
                    content = content[source_end+1:].strip()
            print(source_info)
            print(content[:300] + "..." if len(content) > 300 else content)

if __name__ == "__main__":
    knowledge_base_dir = "RAG知识库"
    kb_directory = "hfss_knowledge_base"

    if os.path.exists(kb_directory):
        logging.info(f"发现已有知识库: {kb_directory}")
        embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
        knowledge_base = load_knowledge_base(kb_directory, embeddings)
    else:
        # 获取知识库中所有PDF文件的路径
        pdf_files = []
        if os.path.exists(knowledge_base_dir):
            for file in os.listdir(knowledge_base_dir):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(knowledge_base_dir, file))

            if not pdf_files:
                logging.error(f"在 {knowledge_base_dir} 目录中未找到PDF文件")
                exit(1)

            logging.info(f"找到以下PDF文件：")
            for pdf_file in pdf_files:
                logging.info(f"- {os.path.basename(pdf_file)}")

            # 加载并分块所有文档（含清洗）
            chunks = load_and_chunk_documents(pdf_files)
            # 构建知识库
            knowledge_base = build_knowledge_base(chunks)
            if knowledge_base:
                save_knowledge_base(knowledge_base, kb_directory)
        else:
            logging.error(f"知识库目录 {knowledge_base_dir} 不存在")
            exit(1)

    if knowledge_base:
        llm = setup_llm()
        if llm:
            rag_chain = build_rag_chain(knowledge_base, llm)
            if rag_chain:
                interactive_qa(rag_chain)
    else:
        logging.error("无法加载或构建知识库，程序退出")