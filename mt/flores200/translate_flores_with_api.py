#!/usr/bin/env python3
"""
使用 OpenAI 兼容 API（GPT / DeepSeek / Gemini）翻译 flores200_upload/src_data/cmn_Hans.jsonl。

- 按 url 将连续记录合并为同一段，段内首/中/末行语境与 translate_flores_with_Mango_MT.py 一致
- 每条样本一次 API 调用，同时翻译 11 个目标语种
- 输出：translated_data/{gpt|deepseek|gemini}/{语种代码}.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from lang_config import (
    DEFAULT_SRC_FILENAME,
    DISPLAY_NAME_TO_CODE,
    LANG_DISPLAY_NAMES,
    TARGET_LANGUAGES,
    TARGET_LANG_DISPLAY_NAMES,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC_FILE = SCRIPT_DIR / "src_data" / DEFAULT_SRC_FILENAME
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "translated_data"

MAX_PARSE_RETRIES = 2
MAX_API_RETRIES = 3
DEFAULT_CONCURRENCY = 20
SUPPORTED_MODELS = ("gpt", "deepseek", "gemini")

DEFAULT_MODEL_NAMES: dict[str, str] = {
    "gpt": "gpt-5.4",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3-pro-preview",
}

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

MODEL_CLI_ATTRS: dict[str, tuple[str, str, str]] = {
    "gpt": ("gpt_api_key", "gpt_base_url", "gpt_model"),
    "deepseek": ("deepseek_api_key", "deepseek_base_url", "deepseek_model"),
    "gemini": ("gemini_api_key", "gemini_base_url", "gemini_model"),
}

MULTI_LANG_INSTRUCTION = (
    "你是一个影视剧语言翻译专家，擅长根据影视剧对白并结合语境把中文翻译成其他语言。\n\n"
    "给定影视剧的一段上下文对白，结合当前影视剧的语境，将【当前文本】同时翻译成以下 "
    f"{len(TARGET_LANG_DISPLAY_NAMES)} 种语言：{', '.join(TARGET_LANG_DISPLAY_NAMES)}。\n\n"
    "## 格式要求\n"
    "- 输出为 JSON 对象，键为语种名称（必须与上述语种名称完全一致），值为对应译文\n"
    "- 必须包含全部目标语种，每种语种各一条译文\n"
    "- 只翻译【当前文本】中的内容，不要翻译上下文\n"
    "- 对专有名词、术语保持一致\n\n"
    "## 输出格式\n"
    "严格按照以下 JSON 格式输出，不得有任何额外内容:\n"
    "```json\n"
    "{{\n"
    '  "俄语": "译文",\n'
    '  "英语": "译文",\n'
    "  ...\n"
    "}}\n"
    "```"
)


@dataclass(frozen=True)
class ApiModelConfig:
    name: str
    api_key: str
    base_url: str | None
    model: str
    concurrency: int = DEFAULT_CONCURRENCY
    temperature: float = 0.0
    max_tokens: int = 8192


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


def build_multi_lang_user_content(
    prev_lines: list[str],
    current_text: str,
    next_lines: list[str],
) -> str:
    prompt_input = build_prompt_input(prev_lines, current_text, next_lines)
    return f"{MULTI_LANG_INSTRUCTION}\n{prompt_input}"


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    return text.strip()


def repair_json_candidate(candidate: str) -> str:
    repaired = candidate.strip()
    repaired = re.sub(
        r'("(?:[^"\\]|\\.)+")\s*:\s*"\[([^\]"]*)\]\}?',
        r'\1: "\2"}',
        repaired,
    )
    if repaired.startswith("{") and not repaired.endswith("}"):
        repaired += "}"
    return repaired


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


def extract_multi_lang_json(response: str, expected_langs: list[str]) -> dict[str, str]:
    text = strip_thinking_blocks(response.strip())
    candidates = _collect_json_candidates(text)

    last_error: Exception | None = None
    for candidate in reversed(candidates):
        for attempt in (candidate, repair_json_candidate(candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict) and parsed:
                    return {
                        lang: str(parsed[lang])
                        for lang in expected_langs
                        if lang in parsed
                    }
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

    # 兜底：按语种名逐个正则提取
    result: dict[str, str] = {}
    for lang in expected_langs:
        pat = re.escape(lang)
        match = re.search(rf'"{pat}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if match:
            result[lang] = match.group(1)
    if result:
        return result

    if last_error is not None:
        raise ValueError(f"无法从模型输出中解析 JSON: {response}") from last_error
    raise ValueError(f"无法从模型输出中解析 JSON: {response}")


class OpenAITranslationEngine:
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
                last_exc = ValueError("API 返回空 content")
            except Exception as exc:
                last_exc = exc

            if attempt < MAX_API_RETRIES - 1:
                time.sleep(2**attempt)

        print(
            f"[警告][{self.config.name}] API 调用失败（已重试 {MAX_API_RETRIES} 次）\n"
            f"原因: {last_exc}\n"
            f"--- user_content 前 500 字 ---\n{user_content[:500]}",
            file=sys.stderr,
        )
        return ""

    def translate_row_all_langs(
        self,
        prev_lines: list[str],
        current_text: str,
        next_lines: list[str],
        target_langs: list[str],
    ) -> dict[str, str]:
        user_content = build_multi_lang_user_content(prev_lines, current_text, next_lines)
        empty = {lang: "" for lang in target_langs}
        response = ""

        for attempt in range(MAX_PARSE_RETRIES + 1):
            response = self._chat_once(user_content)
            if not response.strip():
                if attempt < MAX_PARSE_RETRIES:
                    continue
                return empty

            try:
                result = extract_multi_lang_json(response, target_langs)
                missing = [lang for lang in target_langs if lang not in result]
                if missing:
                    raise KeyError(f"缺少语种: {missing}")
                return result
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt < MAX_PARSE_RETRIES:
                    continue
                print(
                    f"[警告][{self.config.name}] 多语种 JSON 解析失败（已重试 {MAX_PARSE_RETRIES} 次）\n"
                    f"原因: {exc}\n"
                    f"--- 模型输出 ---\n{response[:2000]}",
                    file=sys.stderr,
                )
                return empty

        return empty


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
    current_text = texts[idx]
    n = len(texts)

    if n == 1:
        return [], current_text, []
    if idx == 0:
        return [], current_text, texts[1:2]
    if idx == n - 1:
        return texts[idx - 1 : idx], current_text, []
    return texts[idx - 1 : idx], current_text, texts[idx + 1 : idx + 2]


@dataclass(frozen=True)
class TranslationTask:
    global_idx: int
    prev_lines: list[str]
    current_text: str
    next_lines: list[str]


def build_translation_tasks(
    records: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
) -> list[TranslationTask]:
    tasks: list[TranslationTask] = []
    global_idx = 0

    for group in groups:
        texts = [str(rec["text"]).strip() for rec in group]
        for idx in range(len(texts)):
            prev_lines, current_text, next_lines = get_segment_row_context(texts, idx)
            tasks.append(
                TranslationTask(
                    global_idx=global_idx,
                    prev_lines=prev_lines,
                    current_text=current_text,
                    next_lines=next_lines,
                )
            )
            global_idx += 1

    if global_idx != len(records):
        raise RuntimeError(f"任务数 {global_idx} 与记录数 {len(records)} 不一致")
    return tasks


def translate_all_tasks(
    tasks: list[TranslationTask],
    engine: OpenAITranslationEngine | None,
    target_langs: list[str],
    dry_run: bool = False,
) -> dict[str, list[str]]:
    n = len(tasks)
    results: dict[str, list[str]] = {lang: [""] * n for lang in target_langs}

    if dry_run:
        for task in tasks:
            for lang in target_langs:
                results[lang][task.global_idx] = f"[dry-run:{lang}]"
        return results

    assert engine is not None

    def _translate_one(task: TranslationTask) -> tuple[int, dict[str, str]]:
        try:
            translations = engine.translate_row_all_langs(
                task.prev_lines,
                task.current_text,
                task.next_lines,
                target_langs,
            )
        except Exception as exc:
            print(
                f"[警告][{engine.config.name}] 第 {task.global_idx + 1} 条异常: {exc}",
                file=sys.stderr,
            )
            translations = {lang: "" for lang in target_langs}
        return task.global_idx, translations

    with ThreadPoolExecutor(max_workers=engine.config.concurrency) as executor:
        futures = [executor.submit(_translate_one, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"翻译 [{engine.config.name}]",
        ):
            global_idx, translations = future.result()
            for lang in target_langs:
                results[lang][global_idx] = translations.get(lang, "")

    return results


def build_output_records(
    records: list[dict[str, Any]],
    translations: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rec, text in zip(records, translations):
        out = dict(rec)
        out["text"] = text
        output.append(out)
    return output


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_model_outputs(
    records: list[dict[str, Any]],
    lang_results: dict[str, list[str]],
    output_dir: Path,
) -> None:
    for display_name, texts in lang_results.items():
        lang_code = DISPLAY_NAME_TO_CODE.get(display_name, display_name)
        out_path = output_dir / f"{lang_code}.jsonl"
        save_jsonl(build_output_records(records, texts), out_path)
        print(f"  已写入: {out_path}", flush=True)


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
                f"使用 {model_name} 时需配置 api_key（MODEL_API_CONFIG 或 --{model_name}-api-key）"
            )
        if not base_url:
            raise ValueError(
                f"使用 {model_name} 时需配置 base_url（MODEL_API_CONFIG 或 --{model_name}-base-url）"
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
        config = ApiModelConfig(
            name=name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            concurrency=args.concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        engines.append(OpenAITranslationEngine(config))
        print(
            f"已配置 [{name}]: model={config.model}, base_url={config.base_url}, "
            f"concurrency={config.concurrency}, max_tokens={config.max_tokens}",
            flush=True,
        )
    return engines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 GPT / DeepSeek / Gemini API 翻译 flores200_upload cmn_Hans.jsonl（每条 11 语种）"
    )
    parser.add_argument("--input", "-i", default=str(DEFAULT_SRC_FILE), help="输入中文源 jsonl（默认 cmn_Hans.jsonl）")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出根目录，各模型写入其子目录 gpt/deepseek/gemini",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=list(SUPPORTED_MODELS),
        help="要调用的 API 模型（默认三个都跑）",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=TARGET_LANGUAGES,
        help="目标语种列表（FLORES 语种代码，如 rus_Cyrl）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="每个模型内部 API 并发数",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅翻译前 N 条记录")
    parser.add_argument("--dry-run", action="store_true", help="不调用 API，写入占位译文")
    parser.add_argument("--preview", action="store_true", help="打印多语种 prompt 样例后退出")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192, help="API 最大输出 token")

    parser.add_argument("--gpt-api-key", default=None)
    parser.add_argument("--gpt-base-url", default=None)
    parser.add_argument("--gpt-model", default=DEFAULT_MODEL_NAMES["gpt"])

    parser.add_argument("--deepseek-api-key", default=None)
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--deepseek-model", default=DEFAULT_MODEL_NAMES["deepseek"])

    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-base-url", default=None)
    parser.add_argument("--gemini-model", default=DEFAULT_MODEL_NAMES["gemini"])
    return parser.parse_args()


def preview_prompt(tasks: list[TranslationTask]) -> None:
    task = tasks[1] if len(tasks) > 1 else tasks[0]
    prompt = build_multi_lang_user_content(
        task.prev_lines, task.current_text, task.next_lines
    )
    print("=== 多语种 Prompt 样例（段内中间行或首行）===")
    print(prompt[:3000])
    if len(prompt) > 3000:
        print("... [truncated]")


def run_single_model(
    records: list[dict[str, Any]],
    tasks: list[TranslationTask],
    engine: OpenAITranslationEngine | None,
    output_dir: Path,
    target_langs: list[str],
    dry_run: bool,
) -> None:
    model_dir = output_dir / (engine.config.name if engine else "dry-run")
    print(f"\n>>> 模型 [{engine.config.name if engine else 'dry-run'}] -> {model_dir}", flush=True)

    lang_results = translate_all_tasks(tasks, engine, target_langs, dry_run=dry_run)
    save_model_outputs(records, lang_results, model_dir)


def resolve_target_display_names(lang_codes: list[str]) -> list[str]:
    unknown = [code for code in lang_codes if code not in LANG_DISPLAY_NAMES]
    if unknown:
        raise ValueError(f"未知语种代码: {unknown}")
    return [LANG_DISPLAY_NAMES[code] for code in lang_codes]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    target_codes = list(args.languages)
    target_langs = resolve_target_display_names(target_codes)

    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]

    groups = group_records_by_url(records)
    tasks = build_translation_tasks(records, groups)
    print(
        f"已加载 {len(records)} 条，{len(groups)} 个 url 段，"
        f"每条 API 翻译 {len(target_langs)} 个语种（{', '.join(target_codes)}）",
        flush=True,
    )

    if args.preview:
        preview_prompt(tasks)
        return 0

    if args.dry_run:
        for model_name in args.models:
            model_dir = output_dir / model_name
            print(f"\n>>> [dry-run] 模型 [{model_name}] -> {model_dir}", flush=True)
            lang_results = translate_all_tasks(tasks, None, target_langs, dry_run=True)
            save_model_outputs(records, lang_results, model_dir)
        print(f"\n全部完成（dry-run）-> {output_dir}")
        return 0

    engines = build_engines(args, require_credentials=True)

    def _run(engine: OpenAITranslationEngine) -> str:
        run_single_model(records, tasks, engine, output_dir, target_langs, dry_run=False)
        return engine.config.name

    if len(engines) == 1:
        _run(engines[0])
    else:
        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            futures = {executor.submit(_run, eng): eng.config.name for eng in engines}
            for future in as_completed(futures):
                print(f"模型 [{future.result()}] 已完成", flush=True)

    print(f"\n全部完成 -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
