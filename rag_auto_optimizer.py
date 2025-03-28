import json
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from hfss_simple_rag import SimpleRAG, load_and_chunk_documents

# 配置日志记录
logging.basicConfig(
    filename=f'rag_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RAGOptimizer:
    def __init__(self, pdf_path="HFSS中文手册.pdf", data_dir="hfss_simple_rag_data"):
        self.pdf_path = pdf_path
        self.data_dir = data_dir
        self.chunks = None
        self.test_questions = self.load_test_questions()
        self.results = []
        self.best_params = None
        self.best_score = -1

    def load_test_questions(self):
        """加载测试问题"""
        # 这里可以从文件加载或者手动定义一些测试问题
        return [
            "HFSS软件的主要功能是什么？",
            "如何设置边界条件？",
            "什么是S参数？",
            "如何进行参数扫描？",
            "天线设计中需要注意哪些问题？",
            "电磁兼容性分析如何进行？",
            "波导端口如何设置？",
            "如何优化仿真模型？",
            "有限元法的基本原理是什么？",
            "如何分析电磁场分布？"
        ]

    def load_or_create_chunks(self):
        """加载或创建文本块"""
        if os.path.exists(self.data_dir):
            logging.info(f"从{self.data_dir}加载现有数据")
            temp_rag = SimpleRAG()
            temp_rag.load(self.data_dir)
            self.chunks = temp_rag.chunks
        else:
            logging.info(f"从PDF文件{self.pdf_path}创建新的文本块")
            self.chunks = load_and_chunk_documents(self.pdf_path)

        return self.chunks

    def evaluate_rag_system(self, params):
        """评估RAG系统在给定参数下的性能"""
        try:
            # 创建RAG系统
            rag_system = SimpleRAG(
                chunks=self.chunks,
                embedding_model_name=params.get('embedding_model', 'shibing624/text2vec-base-chinese'),
                llm_model_name=params.get('llm_model', 'uer/gpt2-chinese-cluecorpussmall')
            )

            # 性能指标
            response_times = []
            answer_lengths = []
            errors = 0

            # 对每个测试问题进行评估
            for question in self.test_questions:
                try:
                    # 测量检索时间
                    start_time = time.time()
                    search_results = rag_system.search(
                        question,
                        top_k=params.get('top_k', 3)
                    )
                    retrieval_time = time.time() - start_time

                    # 测量生成时间
                    start_time = time.time()
                    answer = rag_system.generate_answer(
                        question,
                        top_k=params.get('top_k', 3)
                    )
                    generation_time = time.time() - start_time

                    # 记录指标
                    response_times.append(retrieval_time + generation_time)
                    answer_lengths.append(len(answer) if answer else 0)

                except Exception as e:
                    logging.error(f"问题'{question}'处理失败: {str(e)}")
                    errors += 1

            # 计算综合评分
            avg_response_time = np.mean(response_times) if response_times else float('inf')
            avg_answer_length = np.mean(answer_lengths) if answer_lengths else 0
            error_rate = errors / len(self.test_questions) if self.test_questions else 1

            # 综合评分 (响应时间越短越好，答案长度适中，错误率越低越好)
            score = (10 / (1 + avg_response_time)) * (min(avg_answer_length, 200) / 200) * (1 - error_rate)

            result = {
                'params': params,
                'avg_response_time': avg_response_time,
                'avg_answer_length': avg_answer_length,
                'error_rate': error_rate,
                'score': score
            }

            logging.info(f"参数 {params} 的评估结果: 分数={score:.4f}, 响应时间={avg_response_time:.4f}秒")
            return result

        except Exception as e:
            logging.error(f"评估参数 {params} 时出错: {str(e)}")
            return {
                'params': params,
                'error': str(e),
                'score': -1
            }

    def grid_search(self):
        """执行网格搜索以找到最佳参数"""
        # 确保已加载文本块
        if self.chunks is None:
            self.load_or_create_chunks()

        # 定义参数网格
        param_grid = {
            'embedding_model': ['shibing624/text2vec-base-chinese'],
            'llm_model': ['uer/gpt2-chinese-cluecorpussmall'],
            'top_k': [1, 3, 5],
            # 可以添加其他参数
        }

        # 生成所有参数组合
        grid = ParameterGrid(param_grid)
        logging.info(f"开始网格搜索，共{len(grid)}种参数组合")

        # 评估每种参数组合
        for params in grid:
            result = self.evaluate_rag_system(params)
            self.results.append(result)

            # 更新最佳参数
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_params = params
                logging.info(f"找到新的最佳参数: {params}, 分数: {self.best_score:.4f}")

        # 保存结果
        self.save_results()

        return self.best_params, self.best_score

    def save_results(self):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存为JSON
        with open(f'optimization_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'best_params': self.best_params,
                'best_score': self.best_score
            }, f, ensure_ascii=False, indent=2)

        # 生成可视化
        self.visualize_results(timestamp)

        logging.info(f"优化结果已保存，最佳参数: {self.best_params}, 分数: {self.best_score:.4f}")

    def visualize_results(self, timestamp):
        """可视化优化结果"""
        try:
            # 转换为DataFrame便于分析
            df = pd.DataFrame(self.results)

            # 创建图表目录
            charts_dir = "optimization_charts"
            os.makedirs(charts_dir, exist_ok=True)

            # 绘制响应时间与参数关系图
            plt.figure(figsize=(10, 6))
            for top_k in sorted(set(p['top_k'] for p in [r['params'] for r in self.results])):
                subset = df[df['params'].apply(lambda x: x.get('top_k') == top_k)]
                if not subset.empty:
                    plt.plot(subset.index, subset['avg_response_time'],
                             marker='o', label=f'top_k={top_k}')

            plt.title('响应时间与参数关系')
            plt.xlabel('参数组合索引')
            plt.ylabel('平均响应时间 (秒)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{charts_dir}/response_time_{timestamp}.png")

            # 绘制综合评分与参数关系图
            plt.figure(figsize=(10, 6))
            for top_k in sorted(set(p['top_k'] for p in [r['params'] for r in self.results])):
                subset = df[df['params'].apply(lambda x: x.get('top_k') == top_k)]
                if not subset.empty:
                    plt.plot(subset.index, subset['score'],
                             marker='o', label=f'top_k={top_k}')

            plt.title('综合评分与参数关系')
            plt.xlabel('参数组合索引')
            plt.ylabel('评分')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{charts_dir}/score_{timestamp}.png")

            logging.info(f"优化结果可视化已保存到{charts_dir}目录")

        except Exception as e:
            logging.error(f"可视化结果时出错: {str(e)}")

    def optimize_and_apply(self):
        """优化并应用最佳参数"""
        # 执行网格搜索找到最佳参数
        best_params, best_score = self.grid_search()

        if best_params:
            logging.info(f"应用最佳参数: {best_params}")

            # 使用最佳参数创建RAG系统
            optimized_rag = SimpleRAG(
                chunks=self.chunks,
                embedding_model_name=best_params.get('embedding_model', 'shibing624/text2vec-base-chinese'),
                llm_model_name=best_params.get('llm_model', 'uer/gpt2-chinese-cluecorpussmall')
            )

            # 保存优化后的系统
            optimized_data_dir = f"{self.data_dir}_optimized"
            optimized_rag.save(optimized_data_dir)

            logging.info(f"优化后的RAG系统已保存到{optimized_data_dir}")

            # 保存最佳参数配置
            with open(f"{optimized_data_dir}/best_params.json", 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)

            return optimized_rag, best_params
        else:
            logging.warning("未找到有效的最佳参数")
            return None, None

def run_optimization():
    """运行优化过程"""
    try:
        optimizer = RAGOptimizer()
        optimizer.optimize_and_apply()
    except Exception as e:
        logging.error(f"优化过程失败: {str(e)}")

if __name__ == "__main__":
    run_optimization()