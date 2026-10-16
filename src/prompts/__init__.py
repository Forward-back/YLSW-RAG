"""
动态提示词存储：src/prompts/*.json 每标签一个文件，启动时加载进内存。

标签固定列表（规划器让 LLM 单选）；"兜底"无文件，解析为 None（只拼基础提示词）。
JSON 字段：name / describle / version / content，拼接时只用 content。
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定标签列表，与 prompts/ 下文件名一一对应（"兜底"除外）
PROMPT_TAGS = ["产品咨询", "操作指南", "对比分析", "政策法规", "数据查询", "概念解读", "兜底"]

_store: dict[str, str | None] | None = None


def normalize_tag(raw: str | None) -> str | None:
    """LLM 输出的标签归一化：正则匹配固定标签名，匹配不上返回 None（走兜底）。

    容忍 LLM 输出变形："产品咨询类"、"【产品咨询】"、带引号等均命中"产品咨询"。
    """
    if not raw:
        return None
    text = str(raw).strip().strip('"' "'").strip("【】[]")
    for tag in PROMPT_TAGS:
        if re.search(re.escape(tag), text):
            return tag
    return None


def load_prompt_store() -> dict[str, str | None]:
    """加载全部标签的 content 进内存。兜底无文件 → None。"""
    global _store
    if _store is not None:
        return _store

    store: dict[str, str | None] = {}
    for tag in PROMPT_TAGS:
        path = os.path.join(PROMPT_DIR, f"{tag}.json")
        if not os.path.exists(path):
            logger.info(f"[PROMPT] 标签「{tag}」无提示词文件，按兜底处理（仅基础提示词）")
            store[tag] = None
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            content = str(data.get("content") or "").strip()
            if not content:
                logger.warning(f"[PROMPT] {tag}.json 的 content 为空，按兜底处理")
                store[tag] = None
            else:
                store[tag] = content
                logger.info(
                    f"[PROMPT] 加载 {tag}.json v{data.get('version', '?')} "
                    f"({len(content)} 字符)"
                )
        except Exception as e:
            logger.error(f"[PROMPT] 加载 {path} 失败: {e}，按兜底处理")
            store[tag] = None
    _store = store
    return store


def get_tag_content(tag: str | None) -> str | None:
    """取标签对应 content；未知/兜底 → None。"""
    if not tag:
        return None
    return load_prompt_store().get(tag)
