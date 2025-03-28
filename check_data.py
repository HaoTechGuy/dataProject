import os
import numpy as np

def check_data_directory(directory="hfss_simple_rag_data"):
    print(f"检查数据目录: {directory}")

    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return False

    # 检查chunks.txt文件
    chunks_path = os.path.join(directory, "chunks.txt")
    if not os.path.exists(chunks_path):
        print(f"错误: 文件 {chunks_path} 不存在")
        return False

    # 读取chunks.txt文件
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = content.split("\n===CHUNK_SEPARATOR===\n")
            if chunks and chunks[-1].strip() == "":
                chunks.pop()
        print(f"成功读取文本块: {len(chunks)} 个")
        if chunks:
            print(f"第一个文本块 (前100个字符): {chunks[0][:100]}...")
    except Exception as e:
        print(f"读取文本块时出错: {e}")
        return False

    # 检查embeddings.npy文件
    embeddings_path = os.path.join(directory, "embeddings.npy")
    if not os.path.exists(embeddings_path):
        print(f"错误: 文件 {embeddings_path} 不存在")
        return False

    # 读取embeddings.npy文件
    try:
        embeddings = np.load(embeddings_path)
        print(f"成功读取嵌入向量: 形状为 {embeddings.shape}")
    except Exception as e:
        print(f"读取嵌入向量时出错: {e}")
        return False

    # 检查数据一致性
    if len(chunks) != embeddings.shape[0]:
        print(f"警告: 文本块数量 ({len(chunks)}) 与嵌入向量数量 ({embeddings.shape[0]}) 不一致")
    else:
        print(f"数据一致性检查通过: 文本块数量与嵌入向量数量一致 ({len(chunks)})")

    return True

if __name__ == "__main__":
    check_data_directory()