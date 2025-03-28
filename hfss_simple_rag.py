import re
import string
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
def load_and_chunk_documents(file_path):
    print(f"开始加载PDF文件: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"PDF加载完成，共 {len(documents)} 页")

    # 合并所有页面内容
    all_text = ""
    for doc in documents:
        all_text += doc.page_content + "\n\n"

    print("开始清洗文本...")
    cleaned_text = clean_hfss_manual(all_text)
    print(f"文本清洗完成，清洗后字符数: {len(cleaned_text)}")

    print("开始文本分块...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"文本分块完成，共 {len(chunks)} 个块")
    return chunks

class SimpleRAG:
    def __init__(self, chunks=None, embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', llm_model_name='Qwen/Qwen7B-Chat'):
        print(f"初始化RAG系统...")
        print(f"加载嵌入模型: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print("嵌入模型加载成功")
        except Exception as e:
            print(f"加载嵌入模型失败: {e}")
            raise

        self.chunks = chunks or []
        self.embeddings = None
        self.llm = None
        self.llm_chain = None

        if chunks:
            print("检测到文本块，开始计算嵌入向量...")
            self.compute_embeddings()
        else:
            print("未提供文本块，跳过嵌入向量计算")

        print(f"加载语言模型: {llm_model_name}")
        self.setup_llm(llm_model_name)

    def compute_embeddings(self):
        print("计算文本块嵌入向量...")
        self.embeddings = self.embedding_model.encode(self.chunks)
        print(f"嵌入向量计算完成，维度: {self.embeddings.shape}")

    def setup_llm(self, model_name='uer/gpt2-chinese-cluecorpussmall'):
        """设置大语言模型和生成链"""
        print(f"正在加载语言模型: {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # 配置生成管道
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )

            # 创建LangChain LLM包装器
            self.llm = HuggingFacePipeline(pipeline=pipe)

            # 创建提示模板
            template = """基于以下信息回答问题:

{context}

问题: {question}
回答:"""

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )

            # 创建LLM链
            self.llm_chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )

            print("语言模型加载完成")
        except Exception as e:
            print(f"加载语言模型时出错: {e}")
            self.llm = None
            self.llm_chain = None

    def save(self, directory="hfss_simple_rag_data"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存文本块
        with open(os.path.join(directory, "chunks.txt"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk + "\n===CHUNK_SEPARATOR===\n")

        # 保存嵌入向量
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)

        print(f"数据已保存至 {directory} 目录")

    def load(self, directory="hfss_simple_rag_data"):
        print(f"\n=== 开始加载数据 ===")
        print(f"数据目录: {directory}")

        # 加载文本块
        chunks_path = os.path.join(directory, "chunks.txt")
        print(f"检查文本块文件: {chunks_path}")

        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    print("正在读取文本块...")
                    content = f.read()
                    self.chunks = content.split("\n===CHUNK_SEPARATOR===\n")
                    # 移除最后一个空元素（如果有）
                    if self.chunks and self.chunks[-1].strip() == "":
                        self.chunks.pop()
                    print(f"成功加载文本块: {len(self.chunks)} 个")
            except Exception as e:
                print(f"读取文本块时出错: {e}")
                raise
        else:
            print(f"警告: 文本块文件不存在 ({chunks_path})")
            self.chunks = []

        # 加载嵌入向量
        embeddings_path = os.path.join(directory, "embeddings.npy")
        print(f"检查嵌入向量文件: {embeddings_path}")

        if os.path.exists(embeddings_path):
            try:
                print("正在加载嵌入向量...")
                self.embeddings = np.load(embeddings_path)
                print(f"成功加载嵌入向量: 形状为 {self.embeddings.shape}")
            except Exception as e:
                print(f"加载嵌入向量时出错: {e}")
                raise
        else:
            print(f"警告: 嵌入向量文件不存在 ({embeddings_path})")
            self.embeddings = None

        # 检查数据一致性
        if self.chunks and self.embeddings is not None:
            if len(self.chunks) != self.embeddings.shape[0]:
                print(f"警告: 文本块数量 ({len(self.chunks)}) 与嵌入向量数量 ({self.embeddings.shape[0]}) 不一致")
            else:
                print("数据一致性检查通过")

        print("=== 数据加载完成 ===\n")

    def search(self, query, top_k=5):
        if self.embeddings is None or len(self.chunks) == 0:
            print("错误：没有可用的嵌入向量或文本块")
            return []

        print(f"开始搜索，文本块总数: {len(self.chunks)}")
        print(f"嵌入向量维度: {self.embeddings.shape}")

        # 计算查询的嵌入向量
        print("计算查询的嵌入向量...")
        query_embedding = self.embedding_model.encode([query])[0]
        print(f"查询向量维度: {query_embedding.shape}")

        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取最相关的文本块索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "similarity": similarities[idx]
            })

        return results

    def generate_answer(self, query, top_k=3):
        """使用检索结果和LLM生成回答"""
        print("\n=== 开始生成回答 ===")
        print(f"问题: {query}")

        if self.llm_chain is None:
            print("错误：LLM链未初始化")
            return "抱歉，语言模型未正确加载，无法生成回答。"

        print(f"\n1. 检索相关文本 (top_k={top_k})")
        # 检索相关文本块
        results = self.search(query, top_k=top_k)

        if not results:
            return "抱歉，我找不到相关的信息来回答您的问题。"

        # 将检索到的文本块合并为上下文
        print("\n2. 准备上下文信息")
        context = "\n\n".join([f"文档片段 {i+1}:\n{result['chunk']}" for i, result in enumerate(results)])
        print(f"合并后的上下文长度: {len(context)} 字符")

        # 截断过长的上下文以适应模型的最大序列长度
        max_length = 1024  # 假设模型的最大序列长度为1024
        if len(context) > max_length:
            context = context[:max_length]
            print(f"上下文过长，已截断至 {max_length} 字符")

        try:
            print("\n3. 使用语言模型生成回答")
            # 使用LLM生成回答
            response = self.llm_chain.invoke({"context": context, "question": query})  # 修改为使用 invoke 方法
            print("回答生成完成")
            return response
        except Exception as e:
            print(f"生成回答时出错: {e}")
            # 如果生成失败，返回最相关的文本块
            return f"生成回答时出现问题，以下是最相关的信息:\n\n{results[0]['chunk']}"

def interactive_qa(rag_system):
    print("\n=== HFSS中文手册增强问答系统 ===")
    print("输入问题进行查询，输入'exit'退出")
    print("输入'raw'切换到仅检索模式，输入'gen'切换到生成回答模式")

    mode = "gen"  # 默认使用生成模式

    while True:
        try:
            question = input("\n请输入您的问题: ")
        except (KeyboardInterrupt, EOFError):
            print("\n检测到中断，退出程序...")
            break
        if question.lower() == 'exit':
            print("谢谢使用，再见!")
            break
        elif question.lower() == 'raw':
            mode = "raw"
            print("已切换到仅检索模式")
            continue
        elif question.lower() == 'gen':
            mode = "gen"
            print("已切换到生成回答模式")
            continue

        if mode == "raw":
            # 仅检索模式
            results = rag_system.search(question, top_k=5)

            if results:
                print("\n找到以下相关内容:")
                for i, result in enumerate(results, 1):
                    print(f"\n--- 结果 {i} (相似度: {result['similarity']:.4f}) ---")
                    # 显示前300个字符
                    chunk_text = result["chunk"]
                    print(chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text)
            else:
                print("未找到相关内容")
        else:
            # 生成回答模式
            print("\n正在生成回答...")
            answer = rag_system.generate_answer(question, top_k=3)
            print("\n回答:")
            print(answer)

            # 显示参考文档
            print("\n参考文档片段:")
            results = rag_system.search(question, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"\n--- 文档 {i} (相似度: {result['similarity']:.4f}) ---")
                chunk_text = result["chunk"]
                print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)

if __name__ == "__main__":
    try:
        file_path = "HFSS中文手册.pdf"
        data_directory = "hfss_simple_rag_data"

        # 检查是否已存在数据
        if os.path.exists(data_directory):
            print(f"发现已有数据: {data_directory}")
            # 加载现有数据
            rag_system = SimpleRAG()
            rag_system.load(data_directory)
        else:
            # 加载并分块文档（含清洗）
            chunks = load_and_chunk_documents(file_path)
            # 构建RAG系统
            rag_system = SimpleRAG(chunks)
            # 保存数据以便下次使用
            rag_system.save(data_directory)

        # 进入交互式问答
        print("\n系统初始化完成，准备开始交互...")
        interactive_qa(rag_system)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
