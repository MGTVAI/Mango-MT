#!/usr/bin/env python3
"""
使用 OpenAI 兼容 API 对 CSV / XLSX 字幕逐行翻译，支持 GPT、DeepSeek、Gemini。

- 三个模型可同时跑（模型级并行）
- 每个模型内部 API 调用默认并发 10（--concurrency 可调）
- 输出列：answer_gpt、answer_deepseek、answer_gemini（按所选模型生成）

API 凭证在脚本顶部 MODEL_API_CONFIG 中按模型分别配置；
仅 --models 中选用的模型需要填写 api_key / base_url，未选用的可留空。
命令行 --gpt-api-key 等参数可临时覆盖对应模型配置。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

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
MAX_API_RETRIES = 3
CONTEXT_PREV_LINES = 1
CONTEXT_NEXT_LINES = 1
DEFAULT_CONCURRENCY = 10
SUPPORTED_MODELS = ("gpt", "deepseek", "gemini")

DEFAULT_MODEL_NAMES: dict[str, str] = {
    "gpt": "gpt-5.4",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3-pro-preview",
}

# ========== API 配置：仅填写 --models 中会跑的模型 ==========
MODEL_API_CONFIG: dict[str, dict[str, str]] = {
    "gpt": {
        "api_key": "",
        "base_url": "",
    },
    "deepseek": {
        "api_key": "",
        "base_url": "",
    },
    "gemini": {
        "api_key": "",
        "base_url": "",
    },
}
# ========================================================

# 命令行参数名映射：模型 -> (api_key 属性, base_url 属性, model 属性)
MODEL_CLI_ATTRS: dict[str, tuple[str, str, str]] = {
    "gpt": ("gpt_api_key", "gpt_base_url", "gpt_model"),
    "deepseek": ("deepseek_api_key", "deepseek_base_url", "deepseek_model"),
    "gemini": ("gemini_api_key", "gemini_base_url", "gemini_model"),
}


@dataclass(frozen=True)
class ApiModelConfig:
    name: str
    api_key: str
    base_url: str | None
    model: str
    concurrency: int = DEFAULT_CONCURRENCY
    temperature: float = 0.0
    max_tokens: int = 2048


def parse_segment_ids(segment_id: Any) -> list[str]:
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


def extract_partial_json_value(text: str, key: str = "1") -> dict[str, str] | None:
    """从完整或截断（max_tokens 截断）的 JSON 文本中提取单个键值。"""
    key_pat = re.escape(key)
    closed = re.search(rf'"{key_pat}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if closed:
        return {key: closed.group(1)}

    truncated = re.search(rf'"{key_pat}"\s*:\s*"(.+)', text, re.DOTALL)
    if not truncated:
        return None

    raw = truncated.group(1)
    for suffix in ('"}', '"\n}', '",', '"', "\n}"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    raw = raw.rstrip().rstrip("\\")
    return {key: raw} if raw else None


def _collect_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for match in re.finditer(r"'''json([\s\S]*?)'''", text):
        candidates.append(match.group(1).strip())

    if text.startswith("'''json"):
        tail = text[len("'''json") :].strip().rstrip("'")
        if tail:
            candidates.append(tail)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # 无闭合 ``` 的截断 markdown 块
    open_fence = re.search(r"```(?:json)?\s*([\s\S]+)", text)
    if open_fence:
        body = open_fence.group(1).strip()
        if body.endswith("```"):
            body = body[:-3].strip()
        if body and body not in candidates:
            candidates.append(body)

    brace = re.search(r"\{[\s\S]*", text)
    if brace:
        candidates.append(brace.group(0).strip())

    for match in re.finditer(r"\{[\s\S]*?\}", text):
        block = match.group(0).strip()
        if block not in candidates:
            candidates.append(block)

    if text not in candidates:
        candidates.append(text)

    return candidates


def extract_json_from_response(response: str) -> dict[str, str]:
    text = strip_thinking_blocks(response.strip())
    candidates = _collect_json_candidates(text)

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

        partial = extract_partial_json_value(candidate, "1")
        if partial:
            return partial

    fallback = regex_extract_json_dict(text)
    if fallback:
        return fallback

    partial = extract_partial_json_value(text, "1")
    if partial:
        return partial

    if last_error is not None:
        raise ValueError(f"无法从模型输出中解析 JSON: {response}") from last_error
    raise ValueError(f"无法从模型输出中解析 JSON: {response}")


def build_api_config(
    model_name: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    concurrency: int,
    temperature: float,
    max_tokens: int = 2048,
) -> ApiModelConfig:
    if model_name not in DEFAULT_MODEL_NAMES:
        raise ValueError(f"未知模型: {model_name}，可选: {', '.join(SUPPORTED_MODELS)}")

    return ApiModelConfig(
        name=model_name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class OpenAITranslationEngine:
    """OpenAI 兼容 Chat Completions API 翻译引擎。"""

    def __init__(self, config: ApiModelConfig):
        self.config = config
        client_kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        if not response.choices:
            return ""
        message = response.choices[0].message
        content = (message.content or "").strip()
        if content:
            return content
        refusal = getattr(message, "refusal", None)
        if refusal:
            return str(refusal).strip()
        return ""

    def _chat_once(self, user_content: str) -> str:
        last_exc: Exception | None = None
        last_finish_reason: str | None = None

        for attempt in range(MAX_API_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = self._extract_message_content(response)
                if content:
                    return content

                choice = response.choices[0]
                last_finish_reason = getattr(choice, "finish_reason", None)
                last_exc = ValueError(
                    f"API 返回空 content（finish_reason={last_finish_reason}）"
                )
            except Exception as exc:
                last_exc = exc

            if attempt < MAX_API_RETRIES - 1:
                time.sleep(2**attempt)
                continue

        print(
            f"[警告][{self.config.name}] API 调用失败（已重试 {MAX_API_RETRIES} 次），"
            f"该行 answer 置为空字符串\n"
            f"原因: {last_exc}\n"
            f"--- 原始输入 user_content ---\n{user_content}",
            file=sys.stderr,
        )
        return ""

    def translate_row(
        self,
        target_lang: str,
        prev_lines: list[str],
        current_text: str,
        next_lines: list[str],
    ) -> str:
        user_content = build_user_content(target_lang, prev_lines, current_text, next_lines)
        last_error: Exception | None = None
        response = ""

        for attempt in range(MAX_PARSE_RETRIES + 1):
            response = self._chat_once(user_content)
            if not response.strip():
                if attempt < MAX_PARSE_RETRIES:
                    continue
                return ""

            try:
                result = extract_json_from_response(response)
                if "1" not in result:
                    raise KeyError(f'模型返回缺少键 "1"，实际返回: {list(result.keys())}')
                return result["1"]
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < MAX_PARSE_RETRIES:
                    continue
                print(
                    f"[警告][{self.config.name}] 模型输出解析失败（已重试 {MAX_PARSE_RETRIES} 次），"
                    f"该行 answer 置为空字符串\n"
                    f"--- 原始输入 user_content ---\n{user_content}\n"
                    f"--- 模型输出 response ---\n{response}",
                    file=sys.stderr,
                )
                return ""

        return ""


def get_row_context(texts: list[str], idx: int) -> tuple[list[str], str, list[str]]:
    current_text = texts[idx]
    prev_lines = texts[idx - CONTEXT_PREV_LINES : idx] if idx >= CONTEXT_PREV_LINES else []
    next_lines = texts[idx + 1 : idx + 1 + CONTEXT_NEXT_LINES] if idx + 1 < len(texts) else []
    return prev_lines, current_text, next_lines


def translate_episode_group(
    group_df: pd.DataFrame,
    engine: OpenAITranslationEngine | None,
    dry_run: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[str]:
    texts = [str(x).strip() for x in group_df["中文"].tolist()]
    target_lang = str(group_df["语言"].iloc[0]).strip()
    n = len(texts)
    answers: list[str] = [""] * n

    if dry_run:
        for idx in range(n):
            prev_lines, current_text, next_lines = get_row_context(texts, idx)
            prompt = build_user_content(target_lang, prev_lines, current_text, next_lines)
            print("=" * 80)
            print(f"片段: {group_df['片段名'].iloc[0]} | row {idx + 1}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print("... [truncated]")
            answers[idx] = "[dry-run]"
        return answers

    assert engine is not None

    def _translate_one(idx: int) -> tuple[int, str]:
        prev_lines, current_text, next_lines = get_row_context(texts, idx)
        try:
            answer = engine.translate_row(target_lang, prev_lines, current_text, next_lines)
        except Exception as exc:
            print(
                f"[警告][{engine.config.name}] 第 {idx + 1} 行翻译异常，answer 置为空字符串: {exc}",
                file=sys.stderr,
            )
            answer = ""
        return idx, answer

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_translate_one, idx) for idx in range(n)]
        for future in as_completed(futures):
            idx, answer = future.result()
            answers[idx] = answer

    return answers


def translate_dataframe_single_model(
    df: pd.DataFrame,
    engine: OpenAITranslationEngine,
    answer_col: str,
    dry_run: bool = False,
) -> pd.Series:
    series = pd.Series(index=df.index, dtype=str, data="")
    group_key = ["片段名", "语言"]
    groups = list(df.groupby(group_key, sort=False))

    for (_, _), group_df in tqdm(groups, desc=f"翻译进度 [{engine.config.name}]"):
        indices = group_df.index.tolist()
        answers = translate_episode_group(
            group_df.reset_index(drop=True),
            engine=engine,
            dry_run=dry_run,
            concurrency=engine.config.concurrency,
        )
        for idx, answer in zip(indices, answers):
            series.at[idx] = answer

    series.name = answer_col
    return series


def translate_dataframe_all_models(
    df: pd.DataFrame,
    engines: list[OpenAITranslationEngine],
    dry_run: bool = False,
) -> pd.DataFrame:
    result_df = df.copy()

    if dry_run:
        first = engines[0]
        series = translate_dataframe_single_model(df, first, "answer_dry_run", dry_run=True)
        result_df["answer_dry_run"] = series
        return result_df

    def _run_model(engine: OpenAITranslationEngine) -> tuple[str, pd.Series]:
        col = f"answer_{engine.config.name}"
        series = translate_dataframe_single_model(df, engine, col, dry_run=False)
        return col, series

    with ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = {executor.submit(_run_model, eng): eng.config.name for eng in engines}
        for future in as_completed(futures):
            col, series = future.result()
            result_df[col] = series

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
    parser = argparse.ArgumentParser(
        description="使用 GPT / DeepSeek / Gemini（OpenAI 兼容 API）并发翻译 CSV / XLSX"
    )
    parser.add_argument("--input", "-i", required=True, help="输入 CSV 或 XLSX 路径")
    parser.add_argument(
        "--output",
        "-o",
        help="输出路径；默认在输入文件名后加 _api_translated",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="要调用的模型，默认三个都跑",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="每个模型内部的 API 并发数（默认 10）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅翻译前 N 行（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不调用 API")
    parser.add_argument("--preview", action="store_true", help="打印第一条 prompt 样例后退出")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="API 最大输出 token 数（默认 2048，避免长译文 JSON 被截断）",
    )

    parser.add_argument(
        "--gpt-api-key",
        default=None,
        help="覆盖 gpt 的 api_key（未传则用 MODEL_API_CONFIG['gpt']）",
    )
    parser.add_argument(
        "--gpt-base-url",
        default=None,
        help="覆盖 gpt 的 base_url（未传则用 MODEL_API_CONFIG['gpt']）",
    )
    parser.add_argument("--gpt-model", default=DEFAULT_MODEL_NAMES["gpt"], help="GPT 模型名")

    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help="覆盖 deepseek 的 api_key（未传则用 MODEL_API_CONFIG['deepseek']）",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=None,
        help="覆盖 deepseek 的 base_url（未传则用 MODEL_API_CONFIG['deepseek']）",
    )
    parser.add_argument(
        "--deepseek-model",
        default=DEFAULT_MODEL_NAMES["deepseek"],
        help="DeepSeek 模型名",
    )

    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="覆盖 gemini 的 api_key（未传则用 MODEL_API_CONFIG['gemini']）",
    )
    parser.add_argument(
        "--gemini-base-url",
        default=None,
        help="覆盖 gemini 的 base_url（未传则用 MODEL_API_CONFIG['gemini']）",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_MODEL_NAMES["gemini"],
        help="Gemini 模型名",
    )
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


def _resolve_cli_or_config(cli_value: str | None, config_value: str) -> str:
    if cli_value is not None:
        return cli_value.strip()
    return (config_value or "").strip()


def _get_model_credentials(
    args: argparse.Namespace,
    model_name: str,
    *,
    require_credentials: bool,
) -> tuple[str, str, str]:
    key_attr, url_attr, model_attr = MODEL_CLI_ATTRS[model_name]
    cfg = MODEL_API_CONFIG.get(model_name, {})
    api_key = _resolve_cli_or_config(getattr(args, key_attr), cfg.get("api_key", ""))
    base_url = _resolve_cli_or_config(getattr(args, url_attr), cfg.get("base_url", ""))
    model = (getattr(args, model_attr) or DEFAULT_MODEL_NAMES[model_name]).strip()

    if require_credentials:
        if not api_key:
            raise ValueError(
                f"使用 {model_name} 时 api-key 不能为空，"
                f"请在 MODEL_API_CONFIG['{model_name}']['api_key'] 中填写，"
                f"或通过 --{model_name}-api-key 传入"
            )
        if not base_url:
            raise ValueError(
                f"使用 {model_name} 时 base-url 不能为空，"
                f"请在 MODEL_API_CONFIG['{model_name}']['base_url'] 中填写，"
                f"或通过 --{model_name}-base-url 传入"
            )
    else:
        api_key = api_key or "dry-run"
        base_url = base_url or "http://localhost"

    return api_key, base_url, model


def build_engines(
    args: argparse.Namespace,
    *,
    require_credentials: bool = True,
) -> list[OpenAITranslationEngine]:
    engines: list[OpenAITranslationEngine] = []
    for name in args.models:
        api_key, base_url, model = _get_model_credentials(
            args, name, require_credentials=require_credentials
        )
        config = build_api_config(
            name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        engines.append(OpenAITranslationEngine(config))
        print(
            f"已配置模型 [{name}]: model={config.model}, "
            f"base_url={config.base_url}, concurrency={config.concurrency}, "
            f"max_tokens={config.max_tokens}",
            flush=True,
        )
    return engines


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        out_suffix = input_path.suffix.lower()
        if out_suffix == ".xls":
            out_suffix = ".xlsx"
        output_path = input_path.with_name(f"{input_path.stem}_api_translated{out_suffix}")

    df = load_table(input_path)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    if args.preview:
        preview_first_prompt(df)
        return 0

    engines = build_engines(args, require_credentials=not args.dry_run)
    if not args.dry_run:
        print(f"开始翻译，模型: {', '.join(e.config.name for e in engines)}", flush=True)

    result_df = translate_dataframe_all_models(df, engines, dry_run=args.dry_run)

    save_table(result_df, output_path)
    print(f"翻译完成，已写入: {output_path}")
    answer_cols = [c for c in result_df.columns if c.startswith("answer_")]
    if answer_cols:
        print(f"输出列: {', '.join(answer_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
