<<<<<<< Updated upstream
import glob
import os
import re
import sys

import pandas as pd

sys.path.append(".")
from eval.translate_eval import ComprehensiveTranslationEvaluator
from eval.utils import parse_srt2list


def read_srt_file(file_path):
    """读取SRT文件内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().lstrip("\ufeff")


def update_answers_from_srt_files(df, input_files):
    """从SRT文件中更新DataFrame的answer列"""
    df["answer"] = None
    for input_file in input_files:
        data = read_srt_file(input_file)
        df_tmp = parse_srt2list(data, dataframe=True)
        filename = os.path.basename(input_file)
        match = re.match(r"CN_(.*?) (EP\d+)_([^\_]+)_([^\_]+)_字幕翻译\.srt", filename)
        if match:
            drama_name = f"{match.group(1)} {match.group(2)}"
            lang = match.group(3)
            if len(df[(df["片段名"] == drama_name) & (df["语言"] == lang)]) != len(df_tmp):
                print('error:', filename)
            df.loc[(df["片段名"] == drama_name) & (df["语言"] == lang), "answer"] = df_tmp["text"].tolist()
    null_count = len(df[df["answer"].isnull()])
    if null_count != 0:
        print(f"Warning: {null_count} rows have empty answers, they will be dropped.")
        df = df[~df["answer"].isnull()].copy()
    return df


def run_main(
    names,
    save_dir,
    pred_dir="/Users/luoxin/Desktop/mangguo/aitranslator/data/raw/text_data/srt_5lang_use",
    eval_data_file="./data/raw/text_data/final_data_testA_5language_len16128.csv",
):
    """
    主评测流程：
    参数:
        names: 需要评估的模型/方法名称列表
        save_dir: 评测结果保存目录
        pred_dir: SRT预测文件的目录
        eval_data_file: 评测集数据CSV
    """
    df_testa_gpt = pd.read_csv(eval_data_file)
    for name in names:
        print("-" * 50)
        test_paths = os.path.join(pred_dir, f"{name}")
        input_files = glob.glob(os.path.join(test_paths, "**", "*.srt"), recursive=True)
        print(f"{name} SRT files found:", len(input_files))
        df_tmp = df_testa_gpt.copy()
        df_tmp = update_answers_from_srt_files(df_tmp, input_files)
        df_pred = df_tmp[~df_tmp["answer"].isnull()].reset_index()

        save_data = os.path.join(save_dir, f"{name}_pred.parquet")
        df_pred[['剧名', '语言', 'index', '文本', 'answer']].to_parquet(save_data)

        evaluator = ComprehensiveTranslationEvaluator(enable_comet=True, enable_embedding=False)
        valid_data, results = evaluator.evaluate_translations(df_pred)
        valid_data.to_csv(os.path.join(save_dir, f"{name}_valid_data.csv"), index=False)
        results_to_save = results.copy()
        for col in ["综合分数", "BLEU-2", "COMET"]:
            if col in results_to_save.columns:
                results_to_save[col] = results_to_save[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
        results_to_save.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False)
        print(f"评估结果已保存: {name}")


if __name__ == "__main__":
    save_dir = "./data/raw/text_data/eval_result"
    os.makedirs(save_dir, exist_ok=True)
    names = [
        "翻译output_5lang_use_qwen-14B-sft-trans_MangGO_20B_RL_chunk10"
        # "翻译output_5lang_use_MangGO-MT20B",
        # "翻译output_5lang_use_qwen-14B-sft-trans_MangGO_20B_chunk10_Not-compress"
        # "翻译output_5lang_use_MangGO-MT14B",
        # "翻译output_5lang_use_deepseek-v3-0324_SubtitleTranslatorV2",
        # "翻译output_5lang_use_deepseek-v3-0324_oldAPI",
        # "翻译output_5lang_use_MangGO-MT14B-base-202508",
    ]
    run_main(names, save_dir)
=======
import glob
import os
import re
import sys

import pandas as pd

sys.path.append(".")
from eval.translate_eval import ComprehensiveTranslationEvaluator
from eval.utils import parse_srt2list


def read_srt_file(file_path):
    """读取SRT文件内容"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().lstrip("\ufeff")


def update_answers_from_srt_files(df, input_files):
    """从SRT文件中更新DataFrame的answer列"""
    df["answer"] = None
    for input_file in input_files:
        data = read_srt_file(input_file)
        df_tmp = parse_srt2list(data, dataframe=True)
        filename = os.path.basename(input_file)
        match = re.match(r"CN_(.*?) (EP\d+)_([^\_]+)_([^\_]+)_字幕翻译\.srt", filename)
        if match:
            drama_name = f"{match.group(1)} {match.group(2)}"
            lang = match.group(3)
            if len(df[(df["片段名"] == drama_name) & (df["语言"] == lang)]) != len(df_tmp):
                print('error:', filename)
            df.loc[(df["片段名"] == drama_name) & (df["语言"] == lang), "answer"] = df_tmp["text"].tolist()
    null_count = len(df[df["answer"].isnull()])
    if null_count != 0:
        raise ValueError(f"null data: {null_count}")
    return df


def run_main(
    names,
    save_dir,
    pred_dir="./data/raw/text_data/srt_5lang_use",
    eval_data_file="./data/raw/text_data/final_data_testA_5language_len16128.csv",
):
    """
    主评测流程：
    参数:
        names: 需要评估的模型/方法名称列表
        save_dir: 评测结果保存目录
        pred_dir: SRT预测文件的目录
        eval_data_file: 评测集数据CSV
    """
    df_testa_gpt = pd.read_csv(eval_data_file)
    for name in names:
        print("-" * 50)
        test_paths = os.path.join(pred_dir, f"{name}")
        input_files = glob.glob(os.path.join(test_paths, "**", "*.srt"), recursive=True)
        print(f"{name} SRT files found:", len(input_files))
        df_tmp = df_testa_gpt.copy()
        df_tmp = update_answers_from_srt_files(df_tmp, input_files)
        df_pred = df_tmp[~df_tmp["answer"].isnull()].reset_index()

        save_data = os.path.join(save_dir, f"{name}_pred.parquet")
        df_pred[['剧名', '语言', 'index', '文本', 'answer']].to_parquet(save_data)

        evaluator = ComprehensiveTranslationEvaluator(enable_comet=True, enable_embedding=True)
        valid_data, results = evaluator.evaluate_translations(df_pred)
        valid_data.to_csv(os.path.join(save_dir, f"{name}_valid_data.csv"), index=False)
        results_to_save = results.copy()
        for col in ["综合分数", "BLEU-2", "COMET"]:
            if col in results_to_save.columns:
                results_to_save[col] = results_to_save[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
        results_to_save.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False)
        print(f"评估结果已保存: {name}")


if __name__ == "__main__":
    save_dir = "./data/raw/text_data/eval_result"
    os.makedirs(save_dir, exist_ok=True)
    names = [
        "翻译output_5lang_use_MangGO-MT20B",
        # "翻译output_5lang_use_MangGO-MT14B",
        # "翻译output_5lang_use_deepseek-v3-0324_SubtitleTranslatorV2",
        # "翻译output_5lang_use_deepseek-v3-0324_oldAPI",
        # "翻译output_5lang_use_MangGO-MT14B-base-202508",
    ]
    run_main(names, save_dir)
>>>>>>> Stashed changes
