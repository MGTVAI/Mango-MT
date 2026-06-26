import argparse
from pathlib import Path

import pandas as pd

from eval.translate_eval import ComprehensiveTranslationEvaluator

ANSWER_COLUMN = "answer"
OUTPUT_DIR = Path(__file__).parent / "eval_results"


def load_dataset(path: Path, answer_column: str = ANSWER_COLUMN) -> pd.DataFrame:
    print(f"读取数据: {path}")
    df = pd.read_excel(path)

    if answer_column not in df.columns:
        raise ValueError(
            f"找不到机器译文列 '{answer_column}'，当前列名: {df.columns.tolist()}"
        )

    print(f"使用机器译文列: {answer_column}")
    df["answer"] = df[answer_column]

    required_cols = ["中文", "文本", "answer", "语言"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列 {missing_cols}，当前列名: {df.columns.tolist()}")

    df = df.dropna(subset=required_cols).copy()
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()
    df = df[df[required_cols].ne("").all(axis=1)].reset_index(drop=True)

    print(f"有效样本数: {len(df)}")
    if "语言" in df.columns:
        print("语言分布:")
        print(df["语言"].value_counts())

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="机器翻译评估脚本")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="输入 Excel 数据文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_file = args.input.expanduser().resolve()

    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    dataset = load_dataset(data_file)

    evaluator = ComprehensiveTranslationEvaluator(
        enable_comet=True,
        enable_embedding=True,
        use_local_embedding=False,
        use_local_comet=False,
    )

    results, stats = evaluator.evaluate_translations(dataset)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{data_file.stem}_{ANSWER_COLUMN}"
    detail_path = OUTPUT_DIR / f"{stem}_valid_data.csv"
    stats_path = OUTPUT_DIR / f"{stem}_results.csv"

    results.to_csv(detail_path, index=False)
    stats.to_csv(stats_path, index=False)

    print("\n详细评估数据:")
    print(results.head())
    print("\n统计报告:")
    print(stats)
    print(f"\n结果已保存:")
    print(f"  详细数据: {detail_path}")
    print(f"  统计报告: {stats_path}")


if __name__ == "__main__":
    main()
