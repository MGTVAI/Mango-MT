"""FLORES-200 语种代码与中文显示名映射（jsonl 文件名使用语种代码）。"""

from __future__ import annotations

DEFAULT_SRC_FILENAME = "cmn_Hans.jsonl"

# FLORES 语种代码，用于 reference_data / translated_data 下的 jsonl 文件名
TARGET_LANGUAGES: list[str] = [
    "rus_Cyrl",
    "ind_Latn",
    "jpn_Jpan",
    "fra_Latn",
    "tha_Thai",
    "eng_Latn",
    "spa_Latn",
    "vie_Latn",
    "arb_Arab",
    "kor_Hang",
    "zsm_Latn",
]

# 中文语种名，用于翻译 prompt（微调模型与 API 均按中文名识别目标语）
LANG_DISPLAY_NAMES: dict[str, str] = {
    "rus_Cyrl": "俄语",
    "ind_Latn": "印尼语",
    "jpn_Jpan": "日语",
    "fra_Latn": "法语",
    "tha_Thai": "泰语",
    "eng_Latn": "英语",
    "spa_Latn": "西语",
    "vie_Latn": "越南语",
    "arb_Arab": "阿语",
    "kor_Hang": "韩语",
    "zsm_Latn": "马来语",
}

# 评估结果输出用语种名：English（英语）
LANG_OUTPUT_LABELS: dict[str, str] = {
    "rus_Cyrl": "Russian（俄语）",
    "ind_Latn": "Indonesian（印尼语）",
    "jpn_Jpan": "Japanese（日语）",
    "fra_Latn": "French（法语）",
    "tha_Thai": "Thai（泰语）",
    "eng_Latn": "English（英语）",
    "spa_Latn": "Spanish（西语）",
    "vie_Latn": "Vietnamese（越南语）",
    "arb_Arab": "Arabic（阿语）",
    "kor_Hang": "Korean（韩语）",
    "zsm_Latn": "Malay（马来语）",
}


def lang_output_label(lang_code: str) -> str:
    return LANG_OUTPUT_LABELS.get(lang_code, lang_code)

DISPLAY_NAME_TO_CODE: dict[str, str] = {
    display: code for code, display in LANG_DISPLAY_NAMES.items()
}

TARGET_LANG_DISPLAY_NAMES: list[str] = [
    LANG_DISPLAY_NAMES[code] for code in TARGET_LANGUAGES
]

DEFAULT_BLEU_TOKENIZER = "13a"
BLEU_TOKENIZER_BY_LANG: dict[str, str] = {
    "jpn_Jpan": "ja-mecab",
    "kor_Hang": "ko-mecab",
}
