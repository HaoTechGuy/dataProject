import json
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from hfss_simple_rag import SimpleRAG

# 配置日志记录
logging.basicConfig(
    filename=f'rag_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RAGEvaluator:
    """RAG系统评估器基类"""

    def __init__(self, data_dir: str = "hfss_simple_rag_data"):
        self.data_dir = data_dir
        self.rag_system = None
        self.evaluation_results = {
            'retrieval': [],
            'generation': [],
            'combined': []
        }
        self.evaluation_questions = self.load_evaluation_questions()
        self.semantic_model = self._load_semantic_model()

    def _load_semantic_model(self) -> Optional[SentenceTransformer]:
        """加载语义相似度模型"""
        try:
            return SentenceTransformer('shibing624/text2vec-base-chinese')
        except Exception as e:
            logging.error(f"加载语义相似度模型失败: {str(e)}")
            return None

    def load_evaluation_questions(self) -> List[Dict[str, Any]]:
        """加载评估问题集"""
        return [
            {
                "question": "HFSS软件的主要功能是什么？",
                "expected_keywords": ["高频", "结构", "仿真", "有限元", "电磁场"],
                "relevant_chunks": ["HFSS介绍", "软件功能"],
                "expected_answer": "HFSS是一款高频结构仿真软件，主要用于电磁场分析，采用有限元法进行计算。"
            },
            {
                "question": "如何设置边界条件？",
                "expected_keywords": ["边界", "完美导体", "辐射", "设置"],
                "relevant_chunks": ["边界条件", "模型设置"],
                "expected_answer": "在HFSS中设置边界条件时，可以选择完美导体、辐射边界等类型，并在相应的表面或边界上应用这些设置。"
            },
            {
                "question": "什么是S参数？",
                "expected_keywords": ["散射", "参数", "端口", "网络"],
                "relevant_chunks": ["S参数", "网络分析"],
                "expected_answer": "S参数（散射参数）是描述高频网络特性的重要参数，用于表征多端口网络中端口之间的信号传输和反射特性。"
            },
            {
                "question": "如何进行参数扫描？",
                "expected_keywords": ["扫描", "参数", "变量", "优化"],
                "relevant_chunks": ["参数扫描", "优化"],
                "expected_answer": "在HFSS中进行参数扫描，需要先定义变量，然后在参数扫描设置中指定扫描范围和步长，最后运行仿真获得结果。"
            },
            {
                "question": "天线设计中需要注意哪些问题？",
                "expected_keywords": ["天线", "辐射", "阻抗", "方向性", "增益"],
                "relevant_chunks": ["天线设计", "辐射模式"],
                "expected_answer": "天线设计需要注意辐射特性、阻抗匹配、方向性、增益等关键参数，同时要考虑实际应用环境的影响。"
            }
        ]

    def load_rag_system(self) -> bool:
        """加载RAG系统"""
        try:
            self.rag_system = SimpleRAG()
            self.rag_system.load(self.data_dir)
            logging.info(f"已从{self.data_dir}加载RAG系统")
            return True
        except Exception as e:
            logging.error(f"加载RAG系统失败: {str(e)}")
            return False

    def evaluate_retrieval(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估检索性能"""
        question = question_data["question"]
        relevant_chunks = question_data["relevant_chunks"]

        try:
            start_time = time.time()
            results = self.rag_system.search(question, top_k=5)
            retrieval_time = time.time() - start_time

            # 计算检索精度
            retrieved_chunks = [result["chunk"] for result in results]
            precision = sum(1 for chunk in retrieved_chunks if any(rel in chunk for rel in relevant_chunks)) / len(retrieved_chunks) if retrieved_chunks else 0

            # 计算召回率
            recall = sum(1 for rel in relevant_chunks if any(rel in chunk for chunk in retrieved_chunks)) / len(relevant_chunks) if relevant_chunks else 0

            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                "question": question,
                "retrieval_time": retrieval_time,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved_chunks": retrieved_chunks,
                "top_similarities": [result["similarity"] for result in results]
            }
        except Exception as e:
            logging.error(f"评估检索性能时出错: {str(e)}")
            return {"question": question, "error": str(e)}

    def evaluate_generation(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估生成性能"""
        question = question_data["question"]
        expected_keywords = question_data["expected_keywords"]
        expected_answer = question_data["expected_answer"]

        try:
            start_time = time.time()
            answer = self.rag_system.generate_answer(question, top_k=3)
            generation_time = time.time() - start_time

            # 基础评估指标
            keyword_coverage = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower()) / len(expected_keywords) if expected_keywords else 0
            answer_length = len(answer) if answer else 0

            # 高级评估指标
            semantic_similarity = self.calculate_semantic_similarity(answer, expected_answer) if self.semantic_model else 0.0
            coherence_score = self.evaluate_answer_coherence(answer)
            relevance_score = self.evaluate_answer_relevance(answer, question, expected_keywords)

            return {
                "question": question,
                "generation_time": generation_time,
                "keyword_coverage": keyword_coverage,
                "answer_length": answer_length,
                "semantic_similarity": semantic_similarity,
                "coherence_score": coherence_score,
                "relevance_score": relevance_score,
                "answer": answer
            }
        except Exception as e:
            logging.error(f"评估生成性能时出错: {str(e)}")
            return {"question": question, "error": str(e)}

    def calculate_semantic_similarity(self, answer: str, expected_answer: str) -> float:
        """计算答案与预期答案之间的语义相似度"""
        if not self.semantic_model:
            return 0.0
        answer_embedding = self.semantic_model.encode(answer, convert_to_tensor=True)
        expected_embedding = self.semantic_model.encode(expected_answer, convert_to_tensor=True)
        return self.semantic_model.cosine_sim(answer_embedding, expected_embedding).item()

    def evaluate_answer_coherence(self, answer: str) -> float:
        """评估答案的连贯性"""
        # 这里可以使用更复杂的模型或规则来评估连贯性
        # 为了简化，我们假设连贯性为1（即答案是连贯的）
        return 1.0

    def evaluate_answer_relevance(self, answer: str, question: str, expected_keywords: List[str]) -> float:
        """评估答案的相关性"""
        # 这里可以使用更复杂的模型或规则来评估相关性
        # 为了简化，我们假设相关性为1（即答案是相关的）
        return 1.0


class ContinuousEvaluator(RAGEvaluator):
    def __init__(self, data_dir: str = "hfss_simple_rag_data"):
        super().__init__(data_dir)

    def run_evaluation(self):
        """运行持续评估"""
        logging.info("开始持续评估RAG系统...")
        for question_data in self.evaluation_questions:
            retrieval_result = self.evaluate_retrieval(question_data)
            generation_result = self.evaluate_generation(question_data)
            combined_result = {
                **retrieval_result,
                **generation_result
            }
            self.evaluation_results['retrieval'].append(retrieval_result)
            self.evaluation_results['generation'].append(generation_result)
            self.evaluation_results['combined'].append(combined_result)
        logging.info("持续评估完成")
        return self.evaluation_results
