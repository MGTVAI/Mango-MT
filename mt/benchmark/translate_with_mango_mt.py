#!/usr/bin/env python3
"""
使用 Mango-MT模型对 CSV / XLSX 字幕文件逐行翻译。

输入列：片段ID, 文本, 中文, 片段名, 剧名, 集数, 语言
输出：在原表基础上新增 answer 列；输出格式与输入一致（.csv 或 .xlsx）。

支持 --batch-size 批量推理（默认 8），需设置 TransformersEngine(max_batch_size=...)。
按行显示翻译进度与预计剩余时间（ETA）。
"""

from __future__ import annotations

import argparse
import ast长度
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm.std import tqdm as TqdmType
except ImportError:
    TqdmType = Any  # type: ignore[misc,assignment]
from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
# 必须在 import torch 相关库之前设置



import pandas as pd
from tqdm import tqdm


os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "128")
os.environ.setdefault("FPS_MAX_FRAMES", "16")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_ADAPTER_DIR = ""
enable_thinking = False
model_name = "checkpoint-6380-merged"

INSTRUCTION_TEMPLATE = (
    "你是一个影视剧语言翻译专家，擅长根据影视剧对白并结合语境把中文翻译成其他语言\n\n"
    "给定影视剧的一段上下文对白，结合当前影视剧的语境,将指定的对白翻译成{target_lang}\n\n"
    "## 格式要求\n"
    "- 输入输出格式必须完全一致：JSON格式，键为字幕编号，值为翻译内容\n"
    "- 确保翻译后的条目数量与原始字幕完全一致\n"
    "- 对专有名词、术语严格按照术语表执行\n"
    "- 给定上下文，翻译当前文本，不要遗漏任何字幕\n\n"
    "## 输出格式\n"
    "严格按照以下JSON格式输出，不得有任何额外内容:\n"
    "```json\n"
    "{{\n"
    '  "1": "翻译内容1",\n'
    '  "2": "翻译内容2",\n'
    "  ...\n"
    "}}\n"
    "```"
)


def parse_segment_ids(segment_id: Any) -> list[str]:
    """解析片段ID列，如 \"['1']\" 或 \"['4', '5']\"。"""
    if pd.isna(segment_id):
        return []
    if isinstance(segment_id, list):
        return [str(x) for x in segment_id]
    text = str(segment_id).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (SyntaxError, ValueError):
        return [text]


def build_instruction(target_lang: str) -> str:
    return INSTRUCTION_TEMPLATE.format(target_lang=target_lang)


def format_current_text_json(text: str) -> str:
    """将当前行中文格式化为训练数据一致的 JSON 片段。"""
    payload = {"1": text}
    return "'''json" + json.dumps(payload, ensure_ascii=False) + "'''"


def build_prompt_input(
    prev_lines: list[str],
    current_text: str,
    next_lines: list[str],
) -> str:
    """构造 user prompt 中的 input 部分（不含 instruction）。

    边界情况：
    - 第一行：无 [上文]，仅 [当前文本] + [下文]（两行）
    - 最后一行：无 [下文]，仅 [上文] + [当前文本]（两行）
    - 中间行：[上文] + [当前文本] + [下文]（三行）
    """
    parts: list[str] = []

    if prev_lines:
        parts.append("[上文]")
        parts.append("\n".join(prev_lines))

    parts.append("[当前文本]")
    parts.append(format_current_text_json(current_text))

    if next_lines:
        parts.append("[下文]")
        parts.append("\n".join(next_lines))

    return "\n\n".join(parts)


def build_user_content(
    target_lang: str,
    prev_lines: list[str],
    current_text: str,
    next_lines: list[str],
) -> str:
    instruction = build_instruction(target_lang)
    prompt_input = build_prompt_input(prev_lines, current_text, next_lines)
    return f"{instruction}\n{prompt_input}"



def strip_thinking_blocks(text: str) -> str:
    """移除 thinking 标签及其内容。"""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    return text.strip()


def repair_json_candidate(candidate: str) -> str:
    """修复模型常见的 JSON 格式错误。"""
    repaired = candidate.strip()
    # {"1": "[译文?]}" -> {"1": "译文"}
    repaired = re.sub(
        r'("(?:\d+)")\s*:\s*"\[([^\]"]*)\]\}?',
        r'\1: "\2"}',
        repaired,
    )
    # {"1": "译文?]} -> {"1": "译文"}
    repaired = re.sub(
        r'("(?:\d+)")\s*:\s*"([^"]*?)\](\s*\})',
        r'\1: "\2"\3',
        repaired,
    )
    # 补全缺失的右花括号
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
    return repaired


def regex_extract_json_dict(text: str) -> dict[str, str] | None:
    """JSON 解析失败时，用正则兜底提取键值。"""
    patterns = [
        r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'"(\d+)"\s*:\s*"\[([^\]"]*)\]',
        r'"(\d+)"\s*:\s*"\[([^\]"]*)\]"',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return {str(k): str(v) for k, v in matches}
    return None


def extract_json_from_response(response: str) -> dict[str, str]:
    """从模型输出中解析 JSON 翻译结果。"""
    text = strip_thinking_blocks(response.strip())

    candidates: list[str] = []

    for match in re.finditer(r"'''json([\s\S]*?)'''", text):
        candidates.append(match.group(1).strip())

    if text.startswith("'''json"):
        tail = text[len("'''json") :].strip().rstrip("'")
        if tail and tail not in candidates:
            candidates.append(tail)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for match in re.finditer(r"\{[\s\S]*?\}", text):
        candidates.append(match.group(0).strip())

    last_error: Exception | None = None
    for candidate in reversed(candidates):
        for attempt in (candidate, repair_json_candidate(candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict) and parsed:
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

    fallback = regex_extract_json_dict(text)
    if fallback:
        return fallback

    # preview = response[:300].replace("\n", "\\n")
    if last_error is not None:
        raise ValueError(f"无法从模型输出中解析 JSON: {response}") from last_error
    raise ValueError(f"无法从模型输出中解析 JSON: {response}")


MAX_PARSE_RETRIES = 2
DEFAULT_BATCH_SIZE = 8
# 固定三行语境：上文 1 行 + 当前行 1 行 + 下文 1 行
CONTEXT_PREV_LINES = 1
CONTEXT_NEXT_LINES = 1

RowContext = tuple[list[str], str, list[str]]
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt}行 "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


class QwenTranslationEngine:
    def __init__(
        self,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        model_name: str = model_name,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.max_batch_size = max(1, max_batch_size)
        print(f"model_name: {model_name}")
        model, processor = get_model_processor(model_name)
        # model = PeftModel.from_pretrained(model, adapter_dir)
        template = get_template(processor, enable_thinking=enable_thinking)
        self.engine = TransformersEngine(
            model,
            template=template,
            max_batch_size=self.max_batch_size,
        )

    def _infer_batch(self, user_contents: list[str]) -> list[str]:
        if not user_contents:
            return []
        infer_requests = [
            InferRequest(messages=[{"role": "user", "content": content}])
            for content in user_contents
        ]
        request_config = RequestConfig(max_tokens=1024, temperature=0)
        resp_list = self.engine.infer(infer_requests, request_config=request_config)
        return [resp.choices[0].message.content for resp in resp_list]

    def _parse_translation_response(self, user_content: str, response: str) -> str:
        result = extract_json_from_response(response)
        if "1" not in result:
            raise KeyError(f'模型返回缺少键 "1"，实际返回: {list(result.keys())}')
        return result["1"]

    def translate_row(
        self,
        target_lang: str,
        prev_lines: list[str],
        current_text: str,
        next_lines: list[str],
    ) -> str:
        answers = self.translate_rows_batch(
            target_lang,
            [(prev_lines, current_text, next_lines)],
            batch_size=1,
        )
        return answers[0]

    def translate_rows_batch(
        self,
        target_lang: str,
        rows: list[RowContext],
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_bar: TqdmType | None = None,
    ) -> list[str]:
        if not rows:
            return []

        results = [""] * len(rows)
        pending_indices = list(range(len(rows)))

        for attempt in range(MAX_PARSE_RETRIES + 1):
            if not pending_indices:
                break

            failed_indices: list[int] = []
            batch_iter = range(0, len(pending_indices), batch_size)
            use_inner_bar = progress_bar is None
            inner_iter = (
                tqdm(
                    batch_iter,
                    desc=(
                        f"{target_lang} batch"
                        if attempt == 0
                        else f"{target_lang} retry#{attempt}"
                    ),
                    leave=False,
                )
                if use_inner_bar
                else batch_iter
            )
            for start in inner_iter:
                chunk_indices = pending_indices[start : start + batch_size]
                user_contents = [
                    build_user_content(target_lang, rows[i][0], rows[i][1], rows[i][2])
                    for i in chunk_indices
                ]
                responses = self._infer_batch(user_contents)

                for idx, user_content, response in zip(
                    chunk_indices, user_contents, responses
                ):
                    try:
                        results[idx] = self._parse_translation_response(
                            user_content, response
                        )
                        if progress_bar is not None:
                            progress_bar.update(1)
                    except (ValueError, KeyError, json.JSONDecodeError):
                        if attempt < MAX_PARSE_RETRIES:
                            failed_indices.append(idx)
                        else:
                            print(
                                f"[警告] 模型输出解析失败（已重试 {MAX_PARSE_RETRIES} 次），"
                                f"该行 answer 置为空字符串\n"
                                f"--- 原始输入 user_content ---\n{user_content}\n"
                                f"--- 模型输出 response ---\n{response}",
                                file=sys.stderr,
                            )
                            results[idx] = ""
                            if progress_bar is not None:
                                progress_bar.update(1)

            pending_indices = failed_indices

        return results


def get_row_context(texts: list[str], idx: int) -> tuple[list[str], str, list[str]]:
    """取当前行的三行语境：上文 1 行、当前行、下文 1 行。"""
    current_text = texts[idx]
    prev_lines = texts[idx - CONTEXT_PREV_LINES : idx] if idx >= CONTEXT_PREV_LINES else []
    next_lines = texts[idx + 1 : idx + 1 + CONTEXT_NEXT_LINES] if idx + 1 < len(texts) else []
    return prev_lines, current_text, next_lines


def collect_group_row_tasks(texts: list[str]) -> list[RowContext]:
    return [get_row_context(texts, idx) for idx in range(len(texts))]


def translate_episode_group(
    group_df: pd.DataFrame,
    engine: QwenTranslationEngine | None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_bar: TqdmType | None = None,
) -> list[str]:
    """翻译同一剧集片段，返回与 group_df 等长的 answer 列表。"""
    texts = [str(x).strip() for x in group_df["中文"].tolist()]
    target_lang = str(group_df["语言"].iloc[0]).strip()
    tasks = collect_group_row_tasks(texts)

    if dry_run:
        answers: list[str] = []
        for idx, (prev_lines, current_text, next_lines) in enumerate(tasks):
            prompt = build_user_content(target_lang, prev_lines, current_text, next_lines)
            print("=" * 80)
            print(f"片段: {group_df['片段名'].iloc[0]} | row {idx + 1}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print("... [truncated]")
            answers.append("[dry-run]")
            if progress_bar is not None:
                progress_bar.update(1)
        return answers

    assert engine is not None
    return engine.translate_rows_batch(
        target_lang,
        tasks,
        batch_size=batch_size,
        progress_bar=progress_bar,
    )


def _shorten_postfix_text(text: str, max_len: int = 24) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def translate_dataframe(
    df: pd.DataFrame,
    engine: QwenTranslationEngine | None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    result_df = df.copy()
    result_df["answer"] = ""

    group_key = ["片段名", "语言"]
    groups = list(result_df.groupby(group_key, sort=False))
    total_rows = len(result_df)
    total_groups = len(groups)

    print(
        f"待翻译: {total_rows} 行, {total_groups} 个片段组, batch_size={batch_size}",
        flush=True,
    )

    with tqdm(
        total=total_rows,
        desc="翻译进度",
        unit="行",
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    ) as progress_bar:
        for group_idx, ((segment_name, lang), group_df) in enumerate(groups, start=1):
            progress_bar.set_postfix(
                组=f"{group_idx}/{total_groups}",
                片段=_shorten_postfix_text(segment_name),
                语言=lang,
                refresh=False,
            )
            indices = group_df.index.tolist()
            answers = translate_episode_group(
                group_df.reset_index(drop=True),
                engine=engine,
                dry_run=dry_run,
                batch_size=batch_size,
                progress_bar=progress_bar,
            )
            for idx, answer in zip(indices, answers):
                result_df.at[idx, "answer"] = answer

    return result_df


SUPPORTED_INPUT_SUFFIXES = {".csv", ".xlsx", ".xls"}
SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".xlsx"}


def load_table(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"不支持的输入格式: {suffix}，请使用 {supported}")

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    required_cols = ["片段ID", "中文", "片段名", "语言"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")
    return df


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        raise ValueError(f"不支持的输出格式: {suffix}，请使用 {supported}")

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用Mango-MT模型翻译 CSV / XLSX 字幕")
    parser.add_argument("--input", "-i", required=True, help="输入 CSV 或 XLSX 路径")
    parser.add_argument(
        "--output",
        "-o",
        help="输出路径（.csv 或 .xlsx），默认在输入文件名后加 _translated，扩展名与输入一致",
    )
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="LoRA adapter 路径")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="GPU batch 大小；同时传给 TransformersEngine(max_batch_size=...)（默认 8）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅翻译前 N 行（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不加载模型")
    parser.add_argument("--preview", action="store_true", help="打印第一条 prompt 样例后退出")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def preview_first_prompt(df: pd.DataFrame) -> None:
    group_df = next(iter(df.groupby(["片段名", "语言"], sort=False)))[1].reset_index(drop=True)
    texts = [str(x).strip() for x in group_df["中文"].tolist()]
    target_lang = str(group_df["语言"].iloc[0]).strip()

    preview_idx = 1 if len(texts) > 1 else 0
    prev_lines, current_text, next_lines = get_row_context(texts, preview_idx)

    prompt = build_user_content(target_lang, prev_lines, current_text, next_lines)
    print(f"=== Prompt 样例（第 {preview_idx + 1} 行，三行语境） ===")
    print(prompt)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        out_suffix = input_path.suffix.lower()
        if out_suffix == ".xls":
            out_suffix = ".xlsx"
        output_path = input_path.with_name(f"{input_path.stem}_qwen3_5_drama{out_suffix}")

    df = load_table(input_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    if args.preview:
        preview_first_prompt(df)
        return 0

    if args.batch_size < 1:
        print("batch-size 必须 >= 1", file=sys.stderr)
        return 1

    engine = None
    if not args.dry_run:
        print("正在加载模型...", flush=True)
        try:
            engine = QwenTranslationEngine(
                adapter_dir=args.adapter_dir,
                max_batch_size=args.batch_size,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "\nCUDA OOM：当前 GPU 显存不足。\n"
                    "请先运行 nvidia-smi 查看是否有其他进程占用显存，"
                    "换一张空闲卡，例如: --gpu 2",
                    file=sys.stderr,
                )
            raise
        print("模型加载完成", flush=True)

    print(
        f"batch_size={args.batch_size} "
        f"(TransformersEngine.max_batch_size={args.batch_size})",
        flush=True,
    )

    result_df = translate_dataframe(
        df,
        engine=engine,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    save_table(result_df, output_path)
    print(f"翻译完成，已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
