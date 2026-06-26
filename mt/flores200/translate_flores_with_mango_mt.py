#!/usr/bin/env python3
"""
使用 Qwen3.5-9B 微调模型翻译 flores200_upload/src_data/cmn_Hans.jsonl。

按 url 字段将连续记录合并为同一段，段内逐行翻译：
- 段首行：当前行 + 下文 1 行
- 段末行：上文 1 行 + 当前行
- 段中间行：上文 1 行 + 当前行 + 下文 1 行

输出写入 translated_data/，每条 JSON 结构与 cmn_Hans.jsonl 一致，仅替换 text 字段。

支持 --batch-size 批量推理（默认 8）。需在 TransformersEngine 上设置
max_batch_size 才会真正走 GPU batch；解析失败的样本会自动单条重试。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "128")
os.environ.setdefault("FPS_MAX_FRAMES", "16")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tqdm import tqdm

from lang_config import DEFAULT_SRC_FILENAME, LANG_DISPLAY_NAMES, TARGET_LANGUAGES

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC_FILE = SCRIPT_DIR / "src_data" / DEFAULT_SRC_FILENAME
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "translated_data/Mango_MT"
DEFAULT_ADAPTER_DIR = (
    "v9-20260608-025318/checkpoint-26000"
)

model_name ="Mango_MT"

enable_thinking = False

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

MAX_PARSE_RETRIES = 2
DEFAULT_BATCH_SIZE = 8

RowContext = tuple[list[str], str, list[str]]


def build_instruction(target_lang: str) -> str:
    return INSTRUCTION_TEMPLATE.format(target_lang=target_lang)


def format_current_text_json(text: str) -> str:
    payload = {"1": text}
    return "'''json" + json.dumps(payload, ensure_ascii=False) + "'''"


def build_prompt_input(
    prev_lines: list[str],
    current_text: str,
    next_lines: list[str],
) -> str:
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
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    return text.strip()


def repair_json_candidate(candidate: str) -> str:
    repaired = candidate.strip()
    repaired = re.sub(
        r'("(?:\d+)")\s*:\s*"\[([^\]"]*)\]\}?',
        r'\1: "\2"}',
        repaired,
    )
    repaired = re.sub(
        r'("(?:\d+)")\s*:\s*"([^"]*?)\](\s*\})',
        r'\1: "\2"\3',
        repaired,
    )
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
    return repaired


def regex_extract_json_dict(text: str) -> dict[str, str] | None:
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

    if last_error is not None:
        raise ValueError(f"无法从模型输出中解析 JSON: {response}") from last_error
    raise ValueError(f"无法从模型输出中解析 JSON: {response}")


class QwenTranslationEngine:
    def __init__(
        self,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        model_name: str = model_name,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        from peft import PeftModel
        from swift import get_model_processor, get_template
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

        self._InferRequest = InferRequest
        self._RequestConfig = RequestConfig
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
            self._InferRequest(messages=[{"role": "user", "content": content}])
            for content in user_contents
        ]
        request_config = self._RequestConfig(max_tokens=1024, temperature=0)
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
            desc = (
                f"{target_lang} batch"
                if attempt == 0
                else f"{target_lang} retry#{attempt}"
            )
            for start in tqdm(batch_iter, desc=desc, leave=False):
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

            pending_indices = failed_indices

        return results


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 解析失败") from exc
    if not records:
        raise ValueError(f"{path} 为空或没有有效记录")
    for rec in records:
        if "text" not in rec:
            raise ValueError(f"{path} 缺少 text 字段")
        if "url" not in rec:
            raise ValueError(f"{path} 缺少 url 字段")
    return records


def group_records_by_url(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按从上到下顺序，将连续且 url 相同的记录合并为一段。"""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for rec in records:
        if current and rec["url"] != current[-1]["url"]:
            groups.append(current)
            current = []
        current.append(rec)

    if current:
        groups.append(current)
    return groups


def get_segment_row_context(
    texts: list[str],
    idx: int,
) -> tuple[list[str], str, list[str]]:
    """段内语境：首行仅下文 1 行，末行仅上文 1 行，中间行三行语境。"""
    current_text = texts[idx]
    n = len(texts)

    if n == 1:
        return [], current_text, []
    if idx == 0:
        return [], current_text, texts[1:2]
    if idx == n - 1:
        return texts[idx - 1 : idx], current_text, []
    return texts[idx - 1 : idx], current_text, texts[idx + 1 : idx + 2]


def collect_row_tasks(groups: list[list[dict[str, Any]]]) -> list[RowContext]:
    tasks: list[RowContext] = []
    for group in groups:
        texts = [str(rec["text"]).strip() for rec in group]
        for idx in range(len(texts)):
            tasks.append(get_segment_row_context(texts, idx))
    return tasks


def translate_records_for_language(
    records: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    target_lang: str,
    engine: QwenTranslationEngine | None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    tasks = collect_row_tasks(groups)

    if len(tasks) != len(records):
        raise RuntimeError(
            f"翻译任务数不一致: 期望 {len(records)}，实际 {len(tasks)}"
        )

    if dry_run:
        answers = ["[dry-run]"] * len(tasks)
    else:
        assert engine is not None
        answers = engine.translate_rows_batch(
            target_lang,
            tasks,
            batch_size=batch_size,
        )

    output_records: list[dict[str, Any]] = []
    for rec, answer in zip(records, answers):
        out = dict(rec)
        out["text"] = answer
        output_records.append(out)
    return output_records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Qwen3.5 微调模型翻译 cmn_Hans.jsonl"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_SRC_FILE),
        help="输入中文源 jsonl 路径（默认 cmn_Hans.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录（每个语种一个 jsonl）",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=TARGET_LANGUAGES,
        help="目标语种列表（FLORES 语种代码，如 rus_Cyrl）",
    )
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="LoRA adapter 路径")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="GPU batch 大小；同时传给 TransformersEngine(max_batch_size=...)（默认 8）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅翻译前 N 条记录（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="不加载模型，输出占位译文")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="打印第一段中间行 prompt 样例后退出",
    )
    return parser.parse_args()


def preview_prompt(groups: list[list[dict[str, Any]]], target_lang: str) -> None:
    group = groups[0]
    texts = [str(rec["text"]).strip() for rec in group]
    idx = 1 if len(texts) > 1 else 0
    prev_lines, current_text, next_lines = get_segment_row_context(texts, idx)
    prompt = build_user_content(target_lang, prev_lines, current_text, next_lines)
    print(f"=== Prompt 样例（url 段内第 {idx + 1} 行，目标语 {target_lang}）===")
    print(f"url: {group[0]['url']}")
    print(f"段内行数: {len(texts)}")
    print(prompt)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]

    groups = group_records_by_url(records)
    print(
        f"已加载 {len(records)} 条记录，按 url 分为 {len(groups)} 段",
        flush=True,
    )

    languages = list(args.languages)
    unknown = [lang for lang in languages if lang not in TARGET_LANGUAGES]
    if unknown:
        print(f"[警告] 以下语种不在默认列表中，仍将尝试翻译: {unknown}", file=sys.stderr)

    if args.preview:
        lang_code = languages[0]
        preview_prompt(groups, LANG_DISPLAY_NAMES.get(lang_code, lang_code))
        return 0

    if args.batch_size < 1:
        print("batch-size 必须 >= 1", file=sys.stderr)
        return 1

    engine: QwenTranslationEngine | None = None
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
                    "请先运行 nvidia-smi 查看是否有其他进程占用显存。",
                    file=sys.stderr,
                )
            raise
        print("模型加载完成", flush=True)

    print(
        f"batch_size={args.batch_size} "
        f"(TransformersEngine.max_batch_size={args.batch_size})",
        flush=True,
    )

    for lang_code in tqdm(languages, desc="语种"):
        display_name = LANG_DISPLAY_NAMES.get(lang_code, lang_code)
        output_path = output_dir / f"{lang_code}.jsonl"
        translated = translate_records_for_language(
            records,
            groups,
            display_name,
            engine=engine,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
        save_jsonl(translated, output_path)
        print(f"已写入: {output_path}", flush=True)

    print(f"全部完成，共 {len(languages)} 个语种 -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
