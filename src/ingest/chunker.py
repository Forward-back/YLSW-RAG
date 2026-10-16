"""
分块策略：md 文件统一按 Markdown 结构切分（标题树 + 标题链前缀），
非 md 文件按文档类型差异化。Parent-Child 架构。
"""

import os
import sys
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

# 确保 src/ 在 path 中（支持直接执行和包导入两种方式）
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import (
    CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_CONTEXT_NEIGHBORS,
)
from ingest.doc_classifier import classify_document_type

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    level: str  # "child" | "parent"
    child_chunks: List["Chunk"] = field(default_factory=list)
    keywords: str = ""
    position_start: int = 0
    position_end: int = 0


# BGE-M3 API 拒绝过短文本（如 OCR 产生的 "14."、":"），<10 字符的 child chunk 直接丢弃
MIN_CHILD_CHUNK_CHARS = 10


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 1 字 ≈ 1 token，英文 1 词 ≈ 1 token。"""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    other_tokens = len(text) - chinese_chars
    return chinese_chars + max(other_tokens // 4, 1)


def _filter_short_children(chunks: List[Chunk]) -> List[Chunk]:
    """丢弃 <MIN_CHILD_CHUNK_CHARS 字符的 child chunk，并同步清理 parent 引用。"""
    dropped_ids = {
        id(c) for c in chunks
        if c.level == "child" and len(c.text.strip()) < MIN_CHILD_CHUNK_CHARS
    }
    if not dropped_ids:
        return chunks

    kept = [c for c in chunks if id(c) not in dropped_ids]
    for c in kept:
        if c.level == "parent":
            c.child_chunks = [cc for cc in c.child_chunks if id(cc) not in dropped_ids]
    logger.info(f"丢弃 {len(dropped_ids)} 个过短 child chunk (<{MIN_CHILD_CHUNK_CHARS} 字符)")
    return kept


def chunk_file(
    full_text: str, file_name: str, tables_markdown: List[str] = None
) -> List[Chunk]:
    """md/xlsx（full_text 为 markdown）走统一 md 结构切分，其余按文档类型选择策略。"""
    doc_type = classify_document_type(file_name)
    tables_markdown = tables_markdown or []

    if file_name.lower().endswith((".md", ".xlsx")):
        chunks = _chunk_md_structure(full_text, doc_type)
    elif doc_type == "regulation":
        chunks = _chunk_by_articles(full_text)
    elif doc_type == "manual":
        chunks = _chunk_by_sections(full_text)
    elif doc_type == "news":
        chunks = _chunk_whole_document(full_text)
    else:
        chunks = _chunk_fixed_window(full_text)

    return _filter_short_children(chunks)


# ───────────────────────────────────────────────────────
# Markdown 结构切分（md 知识库统一策略）
# 标题树切分 + 标题链前缀 + 表格块过滤 + 自顶向下粒度控制
# ───────────────────────────────────────────────────────

MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
FENCE_PREFIX = "```"
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$")
MAX_SMALL_TABLE_ROWS = 5           # ≤5 行的小表保留在 chunk 中
NO_HEADING_WHOLE_DOC_TOKENS = CHILD_CHUNK_SIZE * 2  # 无标题短文档整篇阈值
HARD_CUT_CHARS = CHILD_CHUNK_SIZE * 4               # ~1200 字符硬切步长

BLOCK_HEADING = "heading"
BLOCK_TABLE = "table"
BLOCK_CODE = "code"
BLOCK_PARA = "para"


def _split_md_blocks(full_text: str) -> List[Tuple[str, object, str]]:
    """把 md 切成块：标题 / 表格 / 代码围栏 / 段落。返回 [(kind, meta, text)]。"""
    lines = [l.rstrip("\r") for l in full_text.split("\n")]
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(FENCE_PREFIX):
            buf = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(FENCE_PREFIX):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            blocks.append((BLOCK_CODE, "", "\n".join(buf)))
            continue
        m = MD_HEADING_RE.match(line)
        if m:
            blocks.append((BLOCK_HEADING, len(m.group(1)), line.strip()))
            i += 1
            continue
        if stripped.startswith("|"):
            buf = [line]
            i += 1
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            blocks.append((BLOCK_TABLE, 0, "\n".join(buf)))
            continue
        buf = [line]
        i += 1
        while (i < n and lines[i].strip()
               and not lines[i].strip().startswith(FENCE_PREFIX)
               and not MD_HEADING_RE.match(lines[i])
               and not lines[i].strip().startswith("|")):
            buf.append(lines[i])
            i += 1
        while i < n and not lines[i].strip():
            i += 1
        blocks.append((BLOCK_PARA, 0, "\n".join(buf)))
    return blocks


def _build_md_tree(blocks):
    """标题块构建层级树，其余块归属最近标题节点。"""
    root = {"level": 0, "title": "", "children": [], "blocks": []}
    stack = [root]
    for kind, meta, text in blocks:
        if kind == BLOCK_HEADING:
            node = {"level": meta, "title": text, "children": [], "blocks": []}
            while len(stack) > 1 and stack[-1]["level"] >= meta:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            stack[-1]["blocks"].append((kind, text))
    return root


def _blocks_to_text(blocks) -> str:
    """块列表拼成文本：代码块跳过；>5 行大表跳过（要点由相邻转述段覆盖）；小表保留。"""
    parts = []
    for kind, text in blocks:
        if kind == BLOCK_CODE:
            continue
        if kind == BLOCK_TABLE:
            data_rows = [l for l in text.split("\n")
                         if l.strip().startswith("|")
                         and not TABLE_SEP_RE.match(l.strip())]
            if len(data_rows) <= MAX_SMALL_TABLE_ROWS:
                parts.append(text)
            continue
        parts.append(text)
    return "\n\n".join(parts)


_SENT_END_CJK = "。；！？"
_SENT_END_QUOTES = "\"'」』”’）】"


def _split_sentences(text: str) -> List[str]:
    """按句末标点切句（中文句号/分号/感叹/问号 + 尾随引号；英文标点需前有字母后有空格）。"""
    sents = []
    start, i, n = 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _SENT_END_CJK:
            j = i + 1
            while j < n and text[j] in _SENT_END_QUOTES:
                j += 1
            sents.append(text[start:j].strip())
            start, i = j, j
        elif (ch in ".;!?" and i + 1 < n and text[i + 1] in " \t"
              and i >= 2 and text[i - 1].isalpha() and text[i - 2].isalpha()):
            sents.append(text[start:i + 1].strip())
            start, i = i + 1, i + 1
        else:
            i += 1
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return [s for s in sents if s]


def _try_split_articles(body: str):
    """法规：正文按 第X条 细分，返回 [(条款标题, 条款文本)]；不适用（0 条或 >100 条）返回 None。"""
    parts = ARTICLE_PATTERN.split(body)
    parts = [p.strip() for p in parts if p.strip()]
    units = []
    cur_title, cur_text = "", ""
    for part in parts:
        if ARTICLE_PATTERN.match(part):
            if cur_title:
                units.append((cur_title, cur_text.strip()))
            part = part.replace("**", "")  # 去条款标题的加粗残留
            cur_title = part[:60]
            cur_text = part
        else:
            cur_text += part
    if cur_title:
        units.append((cur_title, cur_text.strip()))
    if not units or len(units) > 100:
        return None
    return units


def _prefixed(crumb_str: str, text: str) -> str:
    return f"{crumb_str}\n\n{text}" if crumb_str else text


def _split_long_body(crumb_str: str, body: str, units: List[Tuple[str, str]]):
    """段落 → 句子 → 硬切 三级兜底，粒度控制在 ~CHILD_CHUNK_SIZE token。"""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    cur = ""
    for p in paras:
        if _estimate_tokens(p) > CHILD_CHUNK_SIZE:
            if cur.strip():
                units.append((crumb_str, _prefixed(crumb_str, cur.strip())))
                cur = ""
            _split_long_para(crumb_str, p, units)
            continue
        if cur and _estimate_tokens(cur + "\n\n" + p) > CHILD_CHUNK_SIZE:
            units.append((crumb_str, _prefixed(crumb_str, cur.strip())))
            cur = p
        else:
            cur = cur + "\n\n" + p if cur else p
    if cur.strip():
        units.append((crumb_str, _prefixed(crumb_str, cur.strip())))


def _split_long_para(crumb_str: str, para: str, units: List[Tuple[str, str]]):
    sents = _split_sentences(para)
    cur = ""
    for s in sents:
        if _estimate_tokens(s) > CHILD_CHUNK_SIZE:
            if cur.strip():
                units.append((crumb_str, _prefixed(crumb_str, cur.strip())))
                cur = ""
            for k in range(0, len(s), HARD_CUT_CHARS):
                piece = s[k:k + HARD_CUT_CHARS].strip()
                if piece:
                    units.append((crumb_str, _prefixed(crumb_str, piece)))
            continue
        if cur and _estimate_tokens(cur + s) > CHILD_CHUNK_SIZE:
            units.append((crumb_str, _prefixed(crumb_str, cur.strip())))
            cur = s
        else:
            cur = cur + s if cur else s
    if cur.strip():
        units.append((crumb_str, _prefixed(crumb_str, cur.strip())))


def _emit_units(crumb_str: str, body: str, units: List[Tuple[str, str]], doc_type: str):
    """把一段正文变成 1..n 个切分单元（keywords=标题链，text=标题链前缀+正文）。"""
    body = body.strip()
    if not body:
        return

    if doc_type == "regulation":
        arts = _try_split_articles(body)
        if arts is not None:
            for art_title, art_text in arts:
                kw = f"{crumb_str} {art_title}" if crumb_str else art_title
                units.append((kw, _prefixed(crumb_str, art_text)))
            return

    if _estimate_tokens(body) <= CHILD_CHUNK_SIZE:
        units.append((crumb_str, _prefixed(crumb_str, body)))
        return

    if not crumb_str and _estimate_tokens(body) <= NO_HEADING_WHOLE_DOC_TOKENS:
        units.append(("", body))
        return

    _split_long_body(crumb_str, body, units)


def _walk_md_tree(node, crumbs: List[str], units: List[Tuple[str, str]], doc_type: str):
    if node["title"]:
        crumbs = crumbs + [node["title"][:80]]  # 单级标题链截断，防止长标题重复膨胀
    crumb_str = " > ".join(crumbs) if crumbs else ""
    body = _blocks_to_text(node["blocks"])
    if body.strip():
        _emit_units(crumb_str, body, units, doc_type)
    for child in node["children"]:
        _walk_md_tree(child, crumbs, units, doc_type)


def _merge_small_units(units: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """过短单元（<20 字符）与相邻单元合并，避免信息随过滤丢失。"""
    out, buf = [], []
    for kw, txt in units:
        buf.append((kw, txt))
        if sum(len(t) for _, t in buf) >= MIN_CHILD_CHUNK_CHARS:
            out.extend(buf)
            buf = []
    if buf:
        if out:
            kw, txt = out[-1]
            extra = "\n\n".join(t for _, t in buf)
            out[-1] = (kw, txt + "\n\n" + extra)
        elif len(buf) == 1:
            out.append(buf[0])
        else:
            out.append((buf[0][0], "\n\n".join(t for _, t in buf)))
    return out


def _chunk_md_structure(full_text: str, doc_type: str) -> List[Chunk]:
    blocks = _split_md_blocks(full_text)
    root = _build_md_tree(blocks)
    units = []
    _walk_md_tree(root, [], units, doc_type)
    units = _merge_small_units(units)
    return _build_parent_child(units, chunk_type="md_section")


# ───────────────────────────────────────────────────────
# 法规：按"第X条"/"第X款"切分（非 md 文件兜底策略）
# ───────────────────────────────────────────────────────

ARTICLE_PATTERN = re.compile(r"((?:\*\*)?第[一二三四五六七八九十百千\d]+条\*{0,2}\s*[^\n]*)")
SECTION_PATTERN = re.compile(r"(第[一二三四五六七八九十百千\d]+款\s*[^\n]*)")


def _chunk_by_articles(full_text: str) -> List[Chunk]:
    articles = ARTICLE_PATTERN.split(full_text)
    articles = [a.strip() for a in articles if a.strip()]

    # 按"条"分组
    article_chunks: List[Tuple[str, str]] = []
    current_article = ""
    current_content = ""
    for part in articles:
        if ARTICLE_PATTERN.match(part):
            if current_article:
                article_chunks.append((current_article, current_content.strip()))
            part = part.replace("**", "")  # 去条款标题的加粗残留
            current_article = part[:60]
            current_content = part
        else:
            current_content += part
    if current_article:
        article_chunks.append((current_article, current_content.strip()))

    # 如果按条切分失败（units 太多或太少），降级为固定窗口
    if len(article_chunks) > 100 or len(article_chunks) == 0:
        logger.info("按条切分不适用，降级为固定窗口")
        return _chunk_fixed_window(full_text)

    return _build_parent_child(_merge_small_units(article_chunks), chunk_type="article")


def _build_parent_child(
    units: List[Tuple[str, str]], chunk_type: str
) -> List[Chunk]:
    """将切分单元组装为 Parent-Child 结构。

    Child = 单个单元文本
    Parent = 前后各 N 个邻居的拼接
    """
    chunks: List[Chunk] = []
    n = len(units)
    N = CHUNK_CONTEXT_NEIGHBORS

    # 先建 child chunks
    child_chunks: List[Chunk] = []
    for i, (title, text) in enumerate(units):
        child = Chunk(
            text=text,
            level="child",
            keywords=title,
            position_start=i,
            position_end=i + 1,
        )
        child_chunks.append(child)
        chunks.append(child)

    # 再建 parent chunks
    for i, (title, texts) in enumerate(units):
        start = max(0, i - N)
        end = min(n, i + N + 1)
        parent_text = "\n\n".join(units[j][1] for j in range(start, end))
        parent = Chunk(
            text=parent_text,
            level="parent",
            keywords=title,
            child_chunks=child_chunks[start:end],
            position_start=start,
            position_end=end,
        )
        chunks.append(parent)

    return chunks


# ───────────────────────────────────────────────────────
# 手册：按章节标题切分（非 md 文件兜底策略）
# ───────────────────────────────────────────────────────

HEADING_PATTERN = re.compile(
    r"^((?:第[一二三四五六七八九十百千\d]+[章节]|"
    r"[一二三四五六七八九十\d]+[、．.]?\s*"
    r"|(?:概述|前言|目录|附录|总结|说明|引言)))",
    re.MULTILINE,
)


def _chunk_by_sections(full_text: str) -> List[Chunk]:
    sections = HEADING_PATTERN.split(full_text)
    sections = [s.strip() for s in sections if s.strip()]

    section_units: List[Tuple[str, str]] = []
    current_title = ""
    current_text = ""
    for part in sections:
        if HEADING_PATTERN.match(part):
            if current_text.strip():
                section_units.append((current_title, current_text.strip()))
            current_title = part[:60]
            current_text = part
        else:
            current_text += "\n" + part
    if current_text.strip():
        section_units.append((current_title, current_text.strip()))

    # 如果章节太大，内部再按段落二次切割
    refined_units: List[Tuple[str, str]] = []
    for title, text in section_units:
        if _estimate_tokens(text) > CHILD_CHUNK_SIZE * 1.5:
            paras = text.split("\n\n")
            current = ""
            sub_idx = 1
            for para in paras:
                if _estimate_tokens(current + para) > CHILD_CHUNK_SIZE and current:
                    refined_units.append((f"{title} (段{sub_idx})", current.strip()))
                    sub_idx += 1
                    current = para
                else:
                    current += "\n\n" + para if current else para
            if current.strip():
                refined_units.append((f"{title} (段{sub_idx})", current.strip()))
        else:
            refined_units.append((title, text))

    return _build_parent_child(_merge_small_units(refined_units), chunk_type="section")


# ───────────────────────────────────────────────────────
# 研究报告：固定窗口 + 重叠（非 md 文件兜底策略）
# ───────────────────────────────────────────────────────

def _chunk_fixed_window(full_text: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    text_len = len(full_text)

    # 按段落边界切分，兜底按单行，再兜底按字符
    paragraphs = full_text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = full_text.split("\n")
    # 如果仍然超大段，强制按字符数切分
    if len(paragraphs) <= 1 and len(full_text) > CHILD_CHUNK_SIZE * 6:
        step = CHILD_CHUNK_SIZE * 4  # ~1200 chars
        paragraphs = [full_text[i:i+step] for i in range(0, len(full_text), step)]

    children: List[Tuple[str, int, int]] = []
    pos = 0
    current_chunk = ""
    chunk_start_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_in_text_start = full_text.find(para, pos) if pos < text_len else pos
        if para_in_text_start == -1:
            para_in_text_start = pos

        if current_chunk and _estimate_tokens(current_chunk + para) > CHILD_CHUNK_SIZE:
            children.append((current_chunk.strip(), chunk_start_pos, para_in_text_start))
            current_chunk = para
            chunk_start_pos = para_in_text_start
        else:
            if not current_chunk:
                chunk_start_pos = para_in_text_start
            current_chunk += "\n\n" + para if current_chunk else para
        pos = para_in_text_start + len(para)

    if current_chunk.strip():
        children.append((current_chunk.strip(), chunk_start_pos, text_len))

    # 过短 child（<20 字符）并入前一个邻居，避免被 _filter_short_children 整块丢弃丢信息
    merged: List[Tuple[str, int, int]] = []
    for text, start, end in children:
        if merged and len(merged[-1][0]) < MIN_CHILD_CHUNK_CHARS:
            prev_text, prev_start, _ = merged[-1]
            merged[-1] = (prev_text + "\n\n" + text, prev_start, end)
        else:
            merged.append((text, start, end))
    if len(merged) > 1 and len(merged[-1][0]) < MIN_CHILD_CHUNK_CHARS:
        prev_text, prev_start, prev_end = merged[-2]
        merged[-2] = (prev_text + "\n\n" + merged[-1][0], prev_start, merged[-1][2])
        merged.pop()
    children = merged

    # 组装 Child + Parent
    n = len(children)
    child_chunks: List[Chunk] = []
    for i, (text, start, end) in enumerate(children):
        child = Chunk(text=text, level="child", position_start=start, position_end=end)
        child_chunks.append(child)
        chunks.append(child)

    for i, (text, start, end) in enumerate(children):
        parent_start = start
        parent_end = end
        parent_text = text

        # 向前扩展
        if i > 0:
            prev_text = child_chunks[i - 1].text[-CHUNK_OVERLAP * 4:]
            parent_text = prev_text + "\n\n" + parent_text
            parent_start = child_chunks[i - 1].position_start

        # 向后扩展
        if i < n - 1:
            next_text = child_chunks[i + 1].text[:CHUNK_OVERLAP * 4]
            parent_text = parent_text + "\n\n" + next_text
            parent_end = child_chunks[i + 1].position_end

        parent = Chunk(
            text=parent_text,
            level="parent",
            child_chunks=[child],
            position_start=parent_start,
            position_end=parent_end,
        )
        chunks.append(parent)

    return chunks


# ───────────────────────────────────────────────────────
# 新闻/短文本：整篇一个 chunk（非 md 文件兜底策略）
# ───────────────────────────────────────────────────────

def _chunk_whole_document(full_text: str) -> List[Chunk]:
    child = Chunk(text=full_text, level="child", position_start=0, position_end=len(full_text))
    parent = Chunk(
        text=full_text,
        level="parent",
        child_chunks=[child],
        position_start=0,
        position_end=len(full_text),
    )
    return [child, parent]
