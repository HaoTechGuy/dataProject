import sys
import traceback
from sentence_transformers import SentenceTransformer

def test_embedding_model():
    print("\n=== 测试嵌入模型加载 ===")
    model_name = 'shibing624/text2vec-base-chinese'

    try:
        print(f"尝试加载模型: {model_name}")
        model = SentenceTransformer(model_name)
        print("模型加载成功")

        # 测试编码功能
        print("测试编码功能...")
        test_text = "这是一个测试句子"
        embedding = model.encode([test_text])[0]
        print(f"编码成功，向量维度: {embedding.shape}")

        return True
    except Exception as e:
        print(f"模型加载或使用过程中出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_model()
    if success:
        print("\n测试成功完成")
    else:
        print("\n测试失败")
        sys.exit(1)