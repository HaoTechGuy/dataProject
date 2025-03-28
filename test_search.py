from hfss_simple_rag import SimpleRAG

def test_search():
    print("初始化RAG系统...")
    rag = SimpleRAG()

    print("加载数据...")
    rag.load("hfss_simple_rag_data")

    # 测试搜索功能
    query = "HFSS软件的主要功能是什么？"
    print(f"\n执行搜索: '{query}'")

    try:
        results = rag.search(query, top_k=3)

        print("\n搜索结果:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n--- 结果 {i} (相似度: {result['similarity']:.4f}) ---")
                # 显示前200个字符
                chunk_text = result["chunk"]
                print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
        else:
            print("未找到相关内容")
    except Exception as e:
        print(f"搜索过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search()