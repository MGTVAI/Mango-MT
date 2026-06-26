#!/usr/bin/env python3
"""
评估 FLORES-200 多模型、11 语种翻译结果（仅 BLEU 与 chrF++）。

数据目录约定：
  - reference_data/{语种代码}.jsonl   参考译文（如 rus_Cyrl.jsonl）
  - translated_data/{模型}/{语种代码}.jsonl   机器译文
  - src_data/cmn_Hans.jsonl   中文源文本

指标（sacrebleu，语料级）：
  - BLEU（日语 ja-mecab、韩语 ko-mecab，其余 13a）
  - chrF++（CHRF，word_order=2）

输出（默认 evaluation_results/）：
  - {模型}_by_language.csv     各语种 BLEU / chrF++
  - all_models_summary.csv     所有模型×语种汇总
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.metrics import CHRF, BLEU

from lang_config import (
    BLEU_TOKENIZER_BY_LANG,
    DEFAULT_BLEU_TOKENIZER,
    TARGET_LANGUAGES,
    lang_output_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REF_DIR = SCRIPT_DIR / "reference_data"
DEFAULT_PRED_DIR = SCRIPT_DIR / "translated_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_results"
SKIP_MODEL_DIRS = frozenset({"dry-run"})

CHRFPP_METRIC = CHRF(word_order=2)
_BLEU_METRIC_CACHE: dict[str, BLEU] = {}


def bleu_tokenizer_for_lang(lang: str) -> str:
    return BLEU_TOKENIZER_BY_LANG.get(lang, DEFAULT_BLEU_TOKENIZER)


def get_bleu_metric(lang: str) -> BLEU:
    tokenizer = bleu_tokenizer_for_lang(lang)
    if tokenizer not in _BLEU_METRIC_CACHE:
        _BLEU_METRIC_CACHE[tokenizer] = BLEU(tokenize=tokenizer)
    return _BLEU_METRIC_CACHE[tokenizer]


def load_jsonl_by_id(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = int(row["id"])
            if row_id in records:
                raise ValueError(f"{path}: 重复 id={row_id} (行 {line_no})")
            records[row_id] = row
    return records


def align_hyp_ref(
    ref_path: Path,
    pred_path: Path,
) -> tuple[list[str], list[str]]:
    refs = load_jsonl_by_id(ref_path)
    hyps = load_jsonl_by_id(pred_path)
    common_ids = sorted(set(refs) & set(hyps))
    if not common_ids:
        raise ValueError(f"{pred_path}: 与参考无可对齐样本")

    missing = len(set(refs) ^ set(hyps))
    if missing:
        print(
            f"警告 {pred_path.name}: ref={len(refs)} hyp={len(hyps)} "
            f"未对齐 {missing} 条，使用交集 {len(common_ids)} 条"
        )

    references = [str(refs[i]["text"]).strip() for i in common_ids]
    hypotheses = [str(hyps[i]["text"]).strip() for i in common_ids]
    return hypotheses, references


def corpus_scores(
    hypotheses: list[str],
    references: list[str],
    lang: str,
) -> tuple[float, float, str]:
    """返回 (BLEU, chrF++, bleu_tokenizer)，分数均为 0–100。"""
    refs_wrapped = [references]
    tokenizer = bleu_tokenizer_for_lang(lang)
    bleu = get_bleu_metric(lang).corpus_score(hypotheses, refs_wrapped).score
    chrfpp = CHRFPP_METRIC.corpus_score(hypotheses, refs_wrapped).score
    return bleu, chrfpp, tokenizer


def evaluate_model(
    model: str,
    ref_dir: Path,
    pred_dir: Path,
    languages: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for lang in languages:
        ref_path = ref_dir / f"{lang}.jsonl"
        pred_path = pred_dir / model / f"{lang}.jsonl"
        if not ref_path.exists():
            raise FileNotFoundError(f"缺少参考文件: {ref_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少预测文件: {pred_path}")

        hyps, refs = align_hyp_ref(ref_path, pred_path)
        bleu, chrfpp, tokenizer = corpus_scores(hyps, refs, lang)
        label = lang_output_label(lang)
        rows.append(
            {
                "语种": label,
                "样本数": len(hyps),
                "BLEU分词器": tokenizer,
                "BLEU": bleu,
                "chrF++": chrfpp,
            }
        )
        print(
            f"  {label}: n={len(hyps)}  BLEU={bleu:.2f} ({tokenizer})  "
            f"chrF++={chrfpp:.2f}"
        )

    df = pd.DataFrame(rows)
    macro = {
        "语种": "宏平均",
        "样本数": int(df["样本数"].sum()),
        "BLEU": float(df["BLEU"].mean()),
        "chrF++": float(df["chrF++"].mean()),
    }
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def discover_models(pred_dir: Path, models: list[str] | None) -> list[str]:
    if models:
        return models
    found = sorted(
        p.name
        for p in pred_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_MODEL_DIRS
    )
    return found


def run_evaluation(
    models: list[str],
    ref_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    languages: list[str],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for model in models:
        print("=" * 60)
        print(f"评估模型: {model}")
        by_lang = evaluate_model(model, ref_dir, pred_dir, languages)
        by_lang.insert(0, "模型", model)
        by_lang.to_csv(output_dir / f"{model}_by_language.csv", index=False)
        print(f"已保存: {output_dir / f'{model}_by_language.csv'}")

        for _, row in by_lang.iterrows():
            summary_rows.append(row.to_dict())

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "all_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n汇总已保存: {summary_path}")
    _print_comparison_tables(summary_df, models, languages)
    return summary_df


def _print_comparison_tables(
    summary_df: pd.DataFrame,
    models: list[str],
    languages: list[str],
) -> None:
    label_width = 20
    for metric in ("BLEU", "chrF++"):
        print(f"\n=== {metric}（按语种，语料级分数） ===")
        header = f"{'语种':<{label_width}}" + "".join(f"{m:>12}" for m in models)
        print(header)
        print("-" * len(header))

        for lang in languages:
            label = lang_output_label(lang)
            parts = [f"{label:<{label_width}}"]
            for model in models:
                sub = summary_df[
                    (summary_df["模型"] == model) & (summary_df["语种"] == label)
                ]
                parts.append(
                    f"{sub[metric].iloc[0]:>12.2f}" if len(sub) else f"{'N/A':>12}"
                )
            print("".join(parts))

        print(f"{'宏平均':<{label_width}}", end="")
        for model in models:
            sub = summary_df[
                (summary_df["模型"] == model) & (summary_df["语种"] == "宏平均")
            ]
            if len(sub):
                print(f"{sub[metric].iloc[0]:>12.2f}", end="")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 FLORES-200 翻译结果（BLEU + chrF++）"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="模型目录名，默认扫描 translated_data（排除 dry-run）",
    )
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="语种子集（FLORES 语种代码，如 rus_Cyrl），默认 11 个目标语种",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = args.languages or TARGET_LANGUAGES
    models = discover_models(args.pred_dir, args.models)

    if not models:
        raise SystemExit(f"未在 {args.pred_dir} 发现任何模型目录")

    print(f"模型: {models}")
    print(f"语种: {[lang_output_label(lang) for lang in languages]}")

    run_evaluation(
        models=models,
        ref_dir=args.ref_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        languages=languages,
    )


if __name__ == "__main__":
    main()
