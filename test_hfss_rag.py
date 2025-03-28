import sys
import traceback
from hfss_simple_rag import SimpleRAG

def test_rag_system():
    print("开始测试RAG系统...")
    # 初始化RAG系统
    try:
        print("正在初始化RAG系统...")
        rag_system = SimpleRAG()
        print("正在加载数据...")
        rag_system.load("hfss_simple_rag_data")
        print("数据加载完成")
    except Exception as e:
        print(f"初始化RAG系统时出错: {e}")
        traceback.print_exc()
        return

    # 测试问题列表
    test_questions = [
        "HFSS软件的主要功能是什么？",
        "如何设置边界条件？",
        "什么是S参数？",
        "如何进行参数扫描？",
        "天线设计中需要注意哪些问题？"
    ]

    print("\n=== RAG系统测试 ===\n")

    for i, question in enumerate(test_questions):
        print(f"\n[{i+1}/{len(test_questions)}] 问题: {question}")
        sys.stdout.flush()  # 确保输出被刷新

        # 测试检索功能
        print("\n1. 检索结果:")
        sys.stdout.flush()
        try:
            results = rag_system.search(question, top_k=2)
        except Exception as e:
            print(f"检索过程出错: {e}")
            traceback.print_exc()
            continue
        for i, result in enumerate(results, 1):
            print(f"\n--- 文档 {i} (相似度: {result['similarity']:.4f}) ---")
            chunk_text = result["chunk"]
            print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)

        # 测试生成功能
        print("\n2. 生成的回答:")
        sys.stdout.flush()
        try:
            answer = rag_system.generate_answer(question, top_k=2)
        except Exception as e:
            print(f"生成回答时出错: {e}")
            traceback.print_exc()
            continue
        print(answer)
        print("\n" + "="*50)

if __name__ == "__main__":
    try:
        test_rag_system()
        print("\n测试完成")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()