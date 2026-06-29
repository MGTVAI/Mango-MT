import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 服务配置
EMBEDDING_SERVICE_URL = "http://10.200.16.72:28004/embed"
# EMBEDDING_SERVICE_URL = "http://218.77.58.37:28004/embed"
COMET_SERVICE_URL = "http://10.200.16.79:8088/comet"

# 评估权重配置
WEIGHTS = {
    "semantic_similarity": 0.25,  # 语义相似度
    "bleu_score": 0.31,  # BLEU-2评分
    "comet_score": 0.44,  # COMET评分
}


def fetch_comet_scores(translation_pairs: List[Dict[str, str]]) -> List[float]:
    """从COMET服务获取翻译质量分数"""
    request_data = {"data": translation_pairs}
    response = requests.post(url=COMET_SERVICE_URL, headers={"Content-Type": "application/json"}, json=request_data)
    return eval(response.text)


def fetch_embeddings(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """从Qwen3-Embedding-4B服务获取文本嵌入向量"""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="获取语义嵌入"):
        batch_texts = texts[i : i + batch_size]
        data = {"texts": batch_texts, "add_instruction": False}

        try:
            response = requests.post(EMBEDDING_SERVICE_URL, headers={"Content-Type": "application/json"}, json=data, timeout=120)
            response.raise_for_status()
            batch_embeddings = response.json()["embeddings"]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"获取第{i//batch_size + 1}批嵌入向量失败: {e}")
            return None

    return all_embeddings


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """计算两个嵌入向量的余弦相似度"""
    if not embedding1 or not embedding2:
        return 0.0

    a = np.array(embedding1)
    b = np.array(embedding2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0


class ComprehensiveTranslationEvaluator:
    """综合翻译质量评估器，支持多维度评估"""

    def __init__(
        self,
        enable_comet: bool = True,
        enable_embedding: bool = True,
        use_local_embedding: bool = False,
        embedding_model_name: str = "../llm_model/models/Qwen/Qwen3-Embedding-4B",
        use_local_comet: bool = False,
        comet_model_name: str = "Unbabel/XCOMET-XL"
    ):
        """
        ComprehensiveTranslationEvaluator 的初始化方法。

        参数:
        ----
        enable_comet: 是否启用COMET分数评估
        enable_embedding: 是否启用语义相似度评估
        use_local_embedding: 是否使用本地嵌入模型
        embedding_model_name: 本地嵌入模型的名称
        use_local_comet: 是否使用本地COMET模型
        comet_model_name: 本地COMET模型的名称
        """
        self.enable_comet = enable_comet
        self.enable_embedding = enable_embedding
        self.use_local_embedding = use_local_embedding
        self.embedding_model = None
        self.use_local_comet = use_local_comet
        self.comet_model = None

        # 如果使用本地嵌入模型，在初始化阶段加载模型
        if self.use_local_embedding and self.enable_embedding:
            print(f"正在加载本地嵌入模型: {embedding_model_name}...")
            try:
                from vllm import LLM
                self.embedding_model = LLM(
                    model=embedding_model_name, 
                    task="embed", 
                    tensor_parallel_size=1, 
                    gpu_memory_utilization=0.8
                )
                print("本地嵌入模型加载成功!")
            except Exception as e:
                print(f"加载本地嵌入模型失败: {e}")
                print("将回退到远程服务")
                self.use_local_embedding = False

        # 如果使用本地COMET模型，在初始化阶段加载模型
        if self.use_local_comet and self.enable_comet:
            print(f"正在加载本地COMET模型: {comet_model_name}...")
            try:
                from comet import download_model, load_from_checkpoint
                model_path = download_model(comet_model_name)
                self.comet_model = load_from_checkpoint(model_path)
                print("本地COMET模型加载成功!")
            except Exception as e:
                print(f"加载本地COMET模型失败: {e}")
                print("将回退到远程服务")
                self.use_local_comet = False

        self.supported_languages = {
            "马来语": "malay",
            "泰语": "thai",
            "英语": "english",
            "印尼语": "id",
            "越南语": "vi",
            "俄语": "russian",
            "法语": "french",
            "日语": "japanese",
            "韩语": "korean",
            "西语": "spanish",
        }

    def _get_languages_in_data(self, data: pd.DataFrame) -> List[str]:
        """Return languages present in data, with known languages ordered first."""
        langs_in_data = data["语言"].dropna().astype(str).unique().tolist()
        known_order = list(self.supported_languages.keys())
        ordered = [lang for lang in known_order if lang in langs_in_data]
        others = sorted(lang for lang in langs_in_data if lang not in known_order)
        return ordered + others

    def _build_stats_row(self, lang: str, lang_data: pd.DataFrame) -> Dict:
        """Build a single statistics row for the given language subset."""
        return {
            "语言": lang,
            "样本数": len(lang_data),
            "语义相似度": float(np.mean(lang_data["semantic_similarity"])),
            "BLEU-2": float(np.mean(lang_data["bleu_score"])),
            "COMET": float(np.mean(lang_data["comet_score"])),
            "综合分数": float(np.mean(lang_data["final_score"])),
        }

    def calculate_single_bleu(self, ref_text: str, hyp_text: str) -> float:
        """计算单个翻译对的BLEU分数

        Args:
            ref_text: 参考译文
            hyp_text: 机器译文

        Returns:
            BLEU分数
        """
        if not ref_text or not hyp_text:
            return 0.0

        ref_tokens = tokenize_text(ref_text)
        hyp_tokens = tokenize_text(hyp_text)

        if not ref_tokens or not hyp_tokens:
            return 0.0

        return sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )

    def compute_bleu_scores(self, dataset: pd.DataFrame) -> List[float]:
        """计算BLEU-2分数"""
        scores = []
        for idx in tqdm(range(len(dataset)), desc="BLEU评分"):
            try:
                ref_text = str(dataset["文本"].iloc[idx]).strip()
                hyp_text = str(dataset["answer"].iloc[idx]).strip()
                score = self.calculate_single_bleu(ref_text, hyp_text)
                scores.append(score)
            except Exception as e:
                print(f"计算第 {idx+1} 条BLEU分数时出错: {e}")
                scores.append(0.0)

        return scores

    def compute_comet_scores(self, dataset: pd.DataFrame) -> List[float]:
        """计算COMET分数，支持本地模型和远程服务"""
        if not self.enable_comet:
            print("COMET评估已禁用")
            return [0.0] * len(dataset)

        print("COMET计算...")
        comet_data, valid_indices = [], []
        for idx in range(len(dataset)):
            try:
                src_text = str(dataset["中文"].iloc[idx]).strip()
                ref_text = str(dataset["文本"].iloc[idx]).strip()
                hyp_text = str(dataset["answer"].iloc[idx]).strip()

                if all([src_text, ref_text, hyp_text]):
                    mt_text = hyp_text[:340]
                    comet_data.append({"src": src_text, "ref": ref_text, "mt": mt_text})
                    valid_indices.append(idx)

            except Exception as e:
                print(f"第{idx+1}条COMET数据准备错误: {e}")
                continue

        if not comet_data:
            print("COMET评估数据不存在")
            return [0.0] * len(dataset)

        try:
            if self.use_local_comet and self.comet_model is not None:
                print("使用本地COMET模型计算...")
                batch_size = 128
                comet_scores = []
                for i in tqdm(range(0, len(comet_data), batch_size), desc="计算COMET分数"):
                    batch = comet_data[i:i + batch_size]
                    model_output = self.comet_model.predict(batch, batch_size=batch_size, gpus=1)
                    comet_scores.extend(model_output.scores)

                print(f"len comet_scores: {len(comet_scores)}")
            else:
                print("使用远程COMET服务...")
                # 使用远程服务
                batch_size = 256
                comet_scores = []
                for i in tqdm(range(0, len(comet_data), batch_size), desc="计算COMET分数"):
                    batch = comet_data[i:i + batch_size]
                    batch_scores = fetch_comet_scores(batch)
                    comet_scores.extend(batch_scores)
                # print(f"len comet_scores: {len(comet_scores)}")

            final_scores = [0.0] * len(dataset)
            for idx, score in zip(valid_indices, comet_scores):
                try:
                    final_scores[idx] = float(score)
                except Exception as e:
                    print(f"第{idx}条COMET分数转换错误: {e}")
                    final_scores[idx] = 0.0

            return final_scores

        except Exception as e:
            print(f"计算COMET错误: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(dataset)

    def compute_semantic_similarity(self, dataset: pd.DataFrame) -> List[float]:
        """计算语义相似度分数"""
        if not self.enable_embedding:
            print("语义相似度评估已禁用")
            return [0.0] * len(dataset)

        print("计算语义相似度...")
        src_texts = []
        hyp_texts = []
        valid_indices = []
        for idx in range(len(dataset)):
            try:
                src_text = str(dataset["文本"].iloc[idx]).strip()
                hyp_text = str(dataset["answer"].iloc[idx]).strip()

                if src_text and hyp_text:
                    src_texts.append(src_text)
                    hyp_texts.append(hyp_text)
                    valid_indices.append(idx)
            except Exception as e:
                print(f"第{idx+1}条语义相似度数据准备错误: {e}")
                continue

        if not src_texts:
            print("语义相似度评估数据不存在")
            return [0.0] * len(dataset)

        try:
            if self.use_local_embedding and self.embedding_model is not None:
                print("使用本地嵌入模型计算...")
                all_texts = src_texts + hyp_texts
                outputs = self.embedding_model.embed(all_texts)
                embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                # 分离源文本和假设文本的嵌入向量
                batch_size_actual = len(src_texts)
                src_embeddings = embeddings[:batch_size_actual].tolist()
                hyp_embeddings = embeddings[batch_size_actual:].tolist()
            else:
                print("使用远程嵌入服务...")
                src_embeddings = fetch_embeddings(src_texts)
                hyp_embeddings = fetch_embeddings(hyp_texts)

            if src_embeddings is None or hyp_embeddings is None:
                print("获取嵌入向量失败")
                return [0.0] * len(dataset)

            # 计算相似度
            similarities = []
            for src_emb, hyp_emb in zip(src_embeddings, hyp_embeddings):
                sim = calculate_cosine_similarity(src_emb, hyp_emb)
                similarities.append(sim)

            # 构建最终分数列表
            final_scores = [0.0] * len(dataset)
            for idx, score in zip(valid_indices, similarities):
                final_scores[idx] = float(score)

            return final_scores
        except Exception as e:
            print(f"计算语义相似度错误: {e}")
            return [0.0] * len(dataset)



    def evaluate_translations(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """执行完整的综合翻译评估流程

        Args:
            dataset: 包含翻译数据的DataFrame，需要包含列：
                   - "中文": 源文本
                   - "文本": 参考译文
                   - "answer": 机器译文
                   - "语言": 目标语言

        Returns:
            评估结果DataFrame和统计信息字典
        """
        valid_data = dataset.copy()

        if len(valid_data) == 0:
            print("没有有效的评估数据")
            return pd.DataFrame(), {}

        print(f"开始评估 {len(valid_data)} 条翻译数据...")
        # 计算各项分数
        valid_data["semantic_similarity"] = self.compute_semantic_similarity(valid_data)
        valid_data["bleu_score"] = self.compute_bleu_scores(valid_data)
        valid_data["comet_score"] = self.compute_comet_scores(valid_data)

        # 计算加权总分
        valid_data["final_score"] = (
            valid_data["semantic_similarity"] * WEIGHTS["semantic_similarity"]
            + valid_data["bleu_score"] * WEIGHTS["bleu_score"] 
            + valid_data["comet_score"] * WEIGHTS["comet_score"]
 
        )
        # 生成统计报告
        stats_df = self._generate_statistics(valid_data)
        self._print_evaluation_report(stats_df, valid_data)

        return valid_data, stats_df

    def _generate_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成评估统计信息"""
        stats_list = []

        stats_list.append(self._build_stats_row("总体", data))

        for lang in self._get_languages_in_data(data):
            lang_data = data[data["语言"] == lang]
            if len(lang_data) > 0:
                stats_list.append(self._build_stats_row(lang, lang_data))

        return pd.DataFrame(stats_list)

    def _print_evaluation_report(self, stats_df: pd.DataFrame, data: pd.DataFrame):
        """打印评估报告"""
        overall = stats_df.iloc[0]

        print(f"\n=== 综合翻译质量评估报告 ===")
        print(f"评估样本数: {overall['样本数']}")
        print(f"\n总体评分:")
        print(f"| 语言 | 样本数 | 语义相似度 | BLEU-2 | COMET | 综合分数 |")
        print("|------|--------|--------|--------|-------|--------|")
        print(
            f"|       | {len(data)} | {overall['语义相似度']:.4f} | {overall['BLEU-2']:.4f} | "
            f"{overall['COMET']:.4f} | {overall['综合分数']:.4f} |"
        )

        for _, row in stats_df.iloc[1:].iterrows():
            print(
                f"| {row['语言']} | {row['样本数']} | "
                f"{row['语义相似度']:.4f} | {row['BLEU-2']:.4f} | "
                f"{row['COMET']:.4f} | {row['综合分数']:.4f} |"
            )

        # self._print_sample_translations(data)

    def _print_sample_translations(self, data: pd.DataFrame):
        """打印最佳和最差翻译样本"""
        print(f"\n{'='*80}")
        print("翻译样本分析")

        for lang in self._get_languages_in_data(data):
            lang_data = data[data["语言"] == lang]
            if len(lang_data) == 0:
                continue

            print(f"\n{lang}翻译样本:")
            print("-" * 100)

            # 最佳翻译样本
            best_idx = lang_data["final_score"].idxmax()
            best_row = lang_data.loc[best_idx]
            print(f"\n最佳翻译 (综合分数: {best_row['final_score']:.4f}):")
            print(f"  中文原文: {best_row['中文']}")
            print(f"  参考译文: {best_row['文本']}")
            print(f"  机器翻译: {best_row['answer']}")
            print(f"  语义相似度: {best_row['semantic_similarity']:.4f}")
            print(f"  BLEU分数: {best_row['bleu_score']:.4f}")
            print(f"  COMET分数: {best_row['comet_score']:.4f}")

            # 最差翻译样本
            worst_idx = lang_data["final_score"].idxmin()
            worst_row = lang_data.loc[worst_idx]
            print(f"\n最差翻译 (综合分数: {worst_row['final_score']:.4f}):")
            print(f"  中文原文: {worst_row['中文']}")
            print(f"  参考译文: {worst_row['文本']}")
            print(f"  机器翻译: {worst_row['answer']}")
            print(f"  语义相似度: {worst_row['semantic_similarity']:.4f}")
            print(f"  BLEU分数: {worst_row['bleu_score']:.4f}")
            print(f"  COMET分数: {worst_row['comet_score']:.4f}")


def tokenize_text(text, language="auto"):
    """
    根据语言类型进行文本分词，标点符号作为单独的token

    Parameters:
    -----------
    text : str
        待分词的文本
    language : str, default='auto'
        语言类型 ('auto', 'thai', 'english', 'chinese', 'japanese', 'korean', 'arabic', etc.)

    Returns:
    --------
    list
        分词后的token列表
    """
    if language == "auto":
        language = detect_language(text)

    if language == "thai":
        from pythainlp.tokenize import word_tokenize
        engine = "newmm"
        tokens = word_tokenize(text, engine=engine)
    elif language == "japanese":
        try:
            from janome.tokenizer import Tokenizer
            tokenizer = Tokenizer()
            tokens = [token.surface for token in tokenizer.tokenize(text)]
        except ImportError:
            print("无法导入janome，使用默认分词!!!")
            tokens = text.split()
    elif language == "korean":
        tokens = text.split()
    elif language == "arabic":
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text, language='arabic')
        except (ImportError, LookupError):
            tokens = text.split()
    else:
        tokens = text.split()

    # 处理末尾的标点符号
    processed_tokens = []
    for token in tokens:
        # 检查token是否以标点符号结尾
        match = re.match(r'^(.*?)([.,!?;:]+)$', token)
        if match:
            word, punct = match.groups()
            if word:  # 如果单词部分不为空
                processed_tokens.append(word)
            processed_tokens.append(punct)  # 添加标点符号
        else:
            processed_tokens.append(token)

    return processed_tokens


def detect_language(text):
    """简单的语言检测"""
    # 泰语Unicode范围：\u0e00-\u0e7f
    thai_chars = len(re.findall(r"[\u0e00-\u0e7f]", text))
    # 中文Unicode范围：\u4e00-\u9fff
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    # 日语：平假名 \u3040-\u309F, 片假名 \u30A0-\u30FF
    japanese_hiragana = len(re.findall(r"[\u3040-\u309F]", text))
    japanese_katakana = len(re.findall(r"[\u30A0-\u30FF]", text))
    # 韩语Unicode范围：\uAC00-\uD7AF
    korean_chars = len(re.findall(r"[\uAC00-\uD7AF]", text))
    # 阿拉伯语Unicode范围：\u0600-\u06FF
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))

    total_chars = len(re.sub(r"\s+", "", text))

    if total_chars == 0:
        return "english"

    thai_ratio = thai_chars / total_chars
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = (japanese_hiragana + japanese_katakana) / total_chars
    korean_ratio = korean_chars / total_chars
    arabic_ratio = arabic_chars / total_chars

    if thai_ratio > 0.5:
        return "thai"
    elif japanese_ratio > 0.3:
        return "japanese"
    elif korean_ratio > 0.3:
        return "korean"
    elif arabic_ratio > 0.3:
        return "arabic"
    elif chinese_ratio > 0.3:
        return "chinese"
    else:
        return "english"



if __name__ == "__main__":
    # 示例用法
    evaluator = ComprehensiveTranslationEvaluator()

    # 创建示例数据
    sample_data = pd.DataFrame(
        {
            "中文": ["你好，世界", "今天天气很好"],
            "文本": ["Hello, world", "The weather is nice today"],
            "answer": ["Hello world", "Today's weather is good"],
            "语言": ["英语", "英语"],
        }
    )

    # 执行评估
    results, stats = evaluator.evaluate_translations(sample_data)
    print("\n评估完成！")
