import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def test_llm_model():
    print("\n=== 测试语言模型加载 ===")
    model_name = 'uer/gpt2-chinese-cluecorpussmall'

    try:
        print(f"尝试加载模型: {model_name}")
        print("1. 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("分词器加载成功")

        print("2. 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("模型加载成功")

        print("3. 创建生成管道...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        print("生成管道创建成功")

        # 测试生成功能
        print("4. 测试文本生成...")
        test_prompt = "人工智能是"
        result = pipe(test_prompt)
        print(f"生成结果: {result[0]['generated_text']}")

        return True
    except Exception as e:
        print(f"模型加载或使用过程中出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_model()
    if success:
        print("\n测试成功完成")
    else:
        print("\n测试失败")
        sys.exit(1)