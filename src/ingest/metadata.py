"""
从文件名提取元数据字段。
新格式: {TYPE}_{DATE}_{ORG}_{PRODUCT}_{TITLE}_{VERSION}_{ZH}.{ext}
旧格式: 兼容兜底，用正则从标题中提取。
"""

import os
import sys
import re
import logging
from dataclasses import dataclass

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from utils import _CN_DIGIT
from ingest.doc_classifier import _classify_legacy, _is_new_format, TYPE_PREFIX_MAP

logger = logging.getLogger(__name__)


# ── 文号正则（从标题中提取，新旧格式通用）─────────────────
DOC_NUMBER_PATTERNS = [
    re.compile(r"银发(?:\{|（)\s*(\d+)\s*(?:）|\})号"),
    re.compile(r"银发\s*(\d+)\s*号"),
    re.compile(r"银监办发(?:\{|（)\s*(\d+)\s*(?:）|\})号"),
    re.compile(r"银监办发\s*(\d+)\s*号"),
    re.compile(r"银监发(?:\{|（)\s*(\d+)\s*(?:）|\})号"),
    re.compile(r"银监发\s*(\d+)\s*号"),
    re.compile(r"银办发(?:\{|（)\s*(\d+)\s*(?:）|\})号"),
    re.compile(r"银办发\s*(\d+)\s*号"),
    re.compile(r"(?:中国人民银行|海关总署)?公告(?:\{|（)\s*(\d+)\s*(?:）|\})\s*第\s*(\d+)\s*号"),
    re.compile(r"中国人民银行令(?:\{|（)\s*(\d+)\s*(?:）|\})\s*第\s*(\d+)\s*号"),
    re.compile(r"汇发\s*(\d+)\s*号"),
]

# ── 旧格式：机构名称正则 ──
ORG_NAME_PATTERNS = [
    re.compile(r"中国人民银行"),
    re.compile(r"中国银监会"),
    re.compile(r"国家外汇管理局"),
    re.compile(r"国务院办公厅"),
    re.compile(r"国家发展改革委"),
    re.compile(r"中国银监会办公厅"),
    re.compile(r"中国人民银行办公厅"),
    re.compile(r"银联商务"),
    re.compile(r"中国银联"),
    re.compile(r"拉卡拉"),
    re.compile(r"移卡"),
    re.compile(r"抖音支付"),
    re.compile(r"支付宝"),
    re.compile(r"浦发银行"),
    re.compile(r"交通银行"),
    re.compile(r"光大银行"),
    re.compile(r"桂林银行"),
    re.compile(r"快手"),
    re.compile(r"洞见科技"),
    re.compile(r"御风算法"),
    re.compile(r"大华捷通"),
    re.compile(r"一川"),
    re.compile(r"中金点睛"),
]

# ── 日期正则（合并新旧格式）─────────────────────────────
DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*年\s*第?\s*([\d一二三四五六七八九十]{1,2})\s*季度"),
    re.compile(r"(\d{4})\s*年\s*([\d一二三四五六七八九十]{1,2})\s*月"),
    re.compile(r"(\d{8})"),
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
]


@dataclass
class DocumentMetadata:
    doc_number: str = ""
    org_name: str = ""
    doc_date_str: str = ""
    doc_title: str = ""
    doc_type: str = ""       # regulation / manual / product / report / news / table / general
    product_name: str = ""   # 产品中文名，"通用" 表示无特定产品


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def extract_metadata(file_name: str) -> DocumentMetadata:
    """从文件名提取元数据，优先走新格式，兜底旧正则。"""
    if _is_new_format(file_name):
        return _extract_new_format(file_name)
    else:
        return _extract_legacy_format(file_name)


# ═══════════════════════════════════════════════════════
# 新格式解析
# ═══════════════════════════════════════════════════════

def _extract_new_format(file_name: str) -> DocumentMetadata:
    """解析新命名: TYPE_DATE_ORG_PRODUCT_TITLE_VERSION_ZH.ext"""
    name_no_ext = re.sub(r"\.(txt|pdf|docx?|xlsx|doc)$", "", file_name, flags=re.IGNORECASE)
    parts = name_no_ext.split("_")

    doc_type = TYPE_PREFIX_MAP.get(parts[0], "general")
    doc_date_str = parts[1] if len(parts) > 1 else ""
    org_name = parts[2] if len(parts) > 2 else ""
    product_name = parts[3] if len(parts) > 3 else ""

    lang_idx = len(parts) - 1
    version_idx = lang_idx - 1
    version = parts[version_idx] if version_idx > 3 else "V1.0"

    title_start = 4
    title_end = version_idx
    if title_end > title_start:
        title = "_".join(parts[title_start:title_end])
    else:
        title = parts[title_start] if title_start < len(parts) else ""

    doc_number = _extract_doc_number(title)

    title_date = _extract_date_from_title(title)
    if title_date:
        doc_date_str = title_date

    return DocumentMetadata(
        doc_number=doc_number,
        org_name=org_name,
        doc_date_str=doc_date_str,
        doc_title=title,
        doc_type=doc_type,
        product_name=product_name,
    )


# ═══════════════════════════════════════════════════════
# 旧格式解析（向后兼容）
# ═══════════════════════════════════════════════════════

def _extract_legacy_format(file_name: str) -> DocumentMetadata:
    """旧格式正则提取，兜底用。"""
    clean = re.sub(r"^\d{3,4}-", "", file_name)
    clean = re.sub(r"\.(txt|pdf|docx?|xlsx|doc)$", "", clean, flags=re.IGNORECASE)

    meta = DocumentMetadata(doc_title=clean.strip())

    meta.doc_number = _extract_doc_number(clean)

    for pattern in ORG_NAME_PATTERNS:
        m = pattern.search(clean)
        if m:
            meta.org_name = m.group(0)
            break

    for pattern in DATE_PATTERNS:
        m = pattern.search(clean)
        if m:
            meta.doc_date_str = m.group(0)
            break

    meta.doc_type = _classify_legacy(file_name)
    meta.product_name = ""

    return meta


# ═══════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════

def _extract_doc_number(text: str) -> str:
    """从文本中提取文号。"""
    for pattern in DOC_NUMBER_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = m.groups()
            if len(groups) >= 2 and groups[0] and groups[1]:
                matched = m.group(0)
                for p in ["银发", "银监办发", "银监发", "银办发", "汇发"]:
                    if p in matched:
                        return f"{p}{groups[0]}{groups[1]}号"
                return re.sub(r"\s+", "", m.group(0))
            elif groups[0]:
                return re.sub(r"\s+", "", m.group(0))
            else:
                return re.sub(r"\s+", "", m.group(0))
    return ""


def _extract_date_from_title(title: str) -> str:
    """从标题文本中提取日期字符串。"""
    for pattern in DATE_PATTERNS:
        m = pattern.search(title)
        if m:
            return m.group(0)
    m = re.search(r"(?:银发|银监发|银监办发|银办发|汇发|银发)\s*\{?\s*(\d{4})", title)
    if m:
        return f"{m.group(1)}0000"
    m = re.search(r"〔(\d{4})〕", title)
    if m:
        return f"{m.group(1)}0000"
    return ""
