#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HFSS RAG系统自动化测试、评估与优化框架
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 导入自定义模块
from hfss_simple_rag import SimpleRAG, load_and_chunk_documents, clean_hfss_manual
from test_hfss_rag import TestRAGSystem, TestMetrics
from rag_auto_evaluator import RAGEvaluator, ContinuousEvaluator
from rag_auto_optimizer import RAGOptimizer

# 配置日志记录
logging.basicConfig(
    filename=f'rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class RAGSystemManager:
    """RAG系统管理器，整合测试、评估和优化功能"""

    def __init__(self, pdf_path: str = "HFSS中文手册.pdf", data_dir: str = "hfss_simple_rag_data"):
        self.pdf_path = pdf_path
        self.data_dir = data_dir
        self.rag_system = None
        self.optimization_results = None
        self.evaluation_results = None
        self.test_results = None

        # 创建必要的目录
        os.makedirs("reports", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def initialize_system(self) -> bool:
        """初始化RAG系统"""
        try:
            logging.info("初始化RAG系统...")

            # 检查是否已存在数据
            if os.path.exists(self.data_dir):
                logging.info(f"从{self.data_dir}加载现有数据")
                self.rag_system = SimpleRAG()
                self.rag_system.load(self.data_dir)
            else:
                # 检查PDF文件是否存在
                if not os.path.exists(self.pdf_path):
                    logging.error(f"PDF文件不存在: {self.pdf_path}")
                    return False

                logging.info(f"从PDF文件{self.pdf_path}创建新的RAG系统")
                chunks = load_and_chunk_documents(self.pdf_path)
                self.rag_system = SimpleRAG(chunks)
                self.rag_system.save(self.data_dir)

            logging.info("RAG系统初始化完成")
            return True

        except Exception as e:
            logging.error(f"初始化RAG系统失败: {str(e)}")
            return False

    def run_tests(self) -> bool:
        """运行单元测试"""
        try:
            logging.info("开始运行单元测试...")
            # 使用test_hfss_rag.py中的测试框架
            test_runner = TestRAGSystem()
            test_runner.setUpClass()

            # 运行各个测试用例
            test_runner.test_embedding_quality()
            test_runner.test_retrieval_performance()
            test_runner.test_answer_generation()
            test_runner.test_error_handling()

            # 保存测试结果
            test_runner.tearDownClass()
            self.test_results = test_runner.test_results

            logging.info("单元测试完成")
            return True
        except Exception as e:
            logging.error(f"运行单元测试失败: {str(e)}")
            return False

    def run_evaluation(self, continuous: bool = False) -> Dict:
        """运行系统评估"""
        try:
            logging.info(f"开始{'持续' if continuous else ''}评估RAG系统...")

            if continuous:
                evaluator = ContinuousEvaluator(self.data_dir)
            else:
                evaluator = RAGEvaluator(self.data_dir)

            evaluator.run_evaluation()
            self.evaluation_results = evaluator.evaluation_results

            logging.info("系统评估完成")
            return self.evaluation_results
        except Exception as e:
            logging.error(f"评估RAG系统失败: {str(e)}")
            return {}

    def run_optimization(self) -> Tuple[Dict, float]:
        """运行系统优化"""
        try:
            logging.info("开始优化RAG系统...")
            optimizer = RAGOptimizer(self.pdf_path, self.data_dir)
            best_params, best_score = optimizer.grid_search()

            if best_params:
                logging.info(f"找到最佳参数: {best_params}, 分数: {best_score:.4f}")

                # 使用最佳参数更新系统
                optimized_data_dir = f"{self.data_dir}_optimized"
                optimizer.apply_best_params(best_params, optimized_data_dir)

                if os.path.exists(optimized_data_dir):
                    logging.info(f"加载优化后的系统: {optimized_data_dir}")
                    self.rag_system = SimpleRAG()
                    self.rag_system.load(optimized_data_dir)

                self.optimization_results = {
                    'best_params': best_params,
                    'best_score': best_score
                }

                return best_params, best_score
            else:
                logging.warning("优化过程未找到更好的参数")
                return {}, 0.0

        except Exception as e:
            logging.error(f"优化RAG系统失败: {str(e)}")
            return {}, 0.0

    def generate_system_report(self) -> str:
        """生成系统综合报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/system_report_{timestamp}.html"

            # 生成HTML报告
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>HFSS RAG系统综合报告 - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>HFSS RAG系统综合报告 - {timestamp}</h1>
                <h2>测试结果</h2>
                <table>
                    <tr>
                        <th>测试用例</th>
                        <th>结果</th>
                    </tr>
            """

            if self.test_results:
                for test_case, result in self.test_results.items():
                    html_report += f"<tr><td>{test_case}</td><td>{result}</td></tr>"

            html_report += """
                </table>
                <h2>评估结果</h2>
                <table>
                    <tr>
                        <th>评估指标</th>
                        <th>值</th>
                    </tr>
            """

            if self.evaluation_results:
                for metric, value in self.evaluation_results.items():
                    html_report += f"<tr><td>{metric}</td><td>{value}</td></tr>"

            html_report += """
                </table>
                <h2>优化结果</h2>
                <table>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                    </tr>
            """

            if self.optimization_results:
                for param, value in self.optimization_results.items():
                    html_report += f"<tr><td>{param}</td><td>{value}</td></tr>"

            html_report += """
                </table>
            </body>
            </html>
            """

            # 保存报告文件
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            logging.info(f"系统报告已保存到 {report_path}")
            return report_path

        except Exception as e:
            logging.error(f"生成系统报告失败: {str(e)}")
            return ""
