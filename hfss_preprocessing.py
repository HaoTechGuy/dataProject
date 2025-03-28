import re
import pdfplumber
import pandas as pd
from transformers import AutoTokenizer, TFBertModel
import numpy as np
import tensorflow as tf
import faiss

# 1. 文档清洗与格式转换
def pdf_to_text(pdf_path):
    """
    将PDF文件转换为文本
    :param pdf_path: PDF文件路径
    :return: 提取的文本
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def remove_redundant_info(text):
    """
    去除文本中的冗余信息，如页眉、页脚、版权声明等
    :param text: 原始文本
    :return: 去除冗余信息后的文本
    """
    # 示例：假设版权声明包含 © 字符，可根据实际情况修改正则表达式
    text = re.sub(r'©.*', '', text)
    # 去除页眉页脚，假设页眉页脚包含特定关键词，这里简单示例
    text = re.sub(r'第.*页', '', text)
    return text

# 2. 分块策略
def chunk_text(text, chunk_size=512):
    """
    按指定大小对文本进行分块
    :param text: 输入文本
    :param chunk_size: 分块大小
    :return: 分块后的文本列表
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def extract_tables(text):
    """
    提取文本中的表格数据（这里简单示例，实际情况可能更复杂）
    :param text: 输入文本
    :return: 表格数据列表
    """
    # 假设表格以多行数据且有分隔符形式存在，这里简单用 \n 分割行
    lines = text.split('\n')
    tables = []
    table = []
    for line in lines:
        if line.strip():
            table.append(line.split())
        elif table:
            tables.append(pd.DataFrame(table))
            table = []
    if table:
        tables.append(pd.DataFrame(table))
    return tables

# 3. 结构化数据提取
def extract_key_parameters(text):
    """
    提取关键参数，如边界条件、材料属性等
    :param text: 输入文本
    :return: 关键参数字典
    """
    parameters = {}
    # 示例：提取电导率，假设文本中有 电导率: 58e6 S/m 这样的表述
    conductivity_match = re.search(r'电导率: (\d+e\d+) S/m', text)
    if conductivity_match:
        parameters['电导率'] = conductivity_match.group(1)
    # 可以继续添加其他参数的提取逻辑
    return parameters

# 4. 术语标准化
def standardize_terms(text, term_mapping):
    """
    对文本中的术语进行标准化
    :param text: 输入文本
    :param term_mapping: 术语映射字典
    :return: 标准化后的文本
    """
    for old_term, new_term in term_mapping.items():
        text = text.replace(old_term, new_term)
    return text

# 5. 向量化与存储
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = TFBertModel.from_pretrained('bert-base-chinese')

def get_embedding(text):
    """
    获取文本的嵌入向量
    :param text: 输入文本
    :return: 嵌入向量
    """
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    mean_embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return mean_embedding.numpy()[0]

def create_vector_index(chunks):
    """
    创建向量索引并存储
    :param chunks: 分块后的文本列表
    :return: 向量索引和元数据列表
    """
    dimension = 768  # BERT的输出维度
    index = faiss.IndexFlatL2(dimension)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings = np.array(embeddings)
    index.add(embeddings)
    metadata = [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]
    return index, metadata

# 6. 质量控制
def check_duplicates(chunks):
    """
    检查文本块中的重复内容
    :param chunks: 分块后的文本列表
    :return: 重复内容的索引列表
    """
    duplicate_indices = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if chunks[i] == chunks[j]:
                duplicate_indices.append(j)
    return duplicate_indices

# 主函数
def main():
    pdf_path = 'HFSS中文手册.pdf'

    # 1. 文档清洗与格式转换
    text = pdf_to_text(pdf_path)
    text = remove_redundant_info(text)

    # 2. 分块策略
    chunks = chunk_text(text)
    tables = extract_tables(text)

    # 3. 结构化数据提取
    parameters = extract_key_parameters(text)

    # 4. 术语标准化
    term_mapping = {
        "Perfect E": "理想电边界",
        "RCS": "雷达横截面"
    }
    standardized_text = standardize_terms(text, term_mapping)

    # 5. 向量化与存储
    index, metadata = create_vector_index(chunks)

    # 6. 质量控制
    duplicate_indices = check_duplicates(chunks)

    print("预处理完成！")
    print("关键参数:", parameters)
    print("重复内容索引:", duplicate_indices)

if __name__ == "__main__":
    main()