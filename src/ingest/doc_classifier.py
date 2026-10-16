"""文档类型分类器 —— 从文件名判断文档类型，供 chunker 使用。"""

import re

# ── 类型前缀映射 ──
TYPE_PREFIX_MAP = {
    "REG": "regulation", "MAN": "manual", "PDT": "product", "RPT": "report",
    "NWS": "news", "TBL": "table", "GEN": "general",
}

# ── 旧格式关键词分类 ──
REGULATION_KEYWORDS = [
    "通知", "办法", "条例", "公告", "规定", "法", "指引",
    "指导意见", "实施方案", "暂行办法", "管理办法", "令",
    "意见", "批复", "监督管理办法", "风险提示",
]
MANUAL_KEYWORDS = [
    "操作手册", "产品手册", "入网手册", "配置手册", "使用指南",
    "产品介绍", "手册", "介绍", "部署手册", "技术指南", "服务指引",
    "改造指引", "授权登录指引", "实施指引",
]
REPORT_KEYWORDS = ["报告", "财报", "研究", "分析", "季报", "年报",
                   "三季度报告", "三季度财报", "运行总体情况"]
NEWS_KEYWORDS = ["来了", "成功举办", "获刑", "退场", "热潮", "值得关注",
                 "飙涨", "增资", "缩水", "盘点", "嵌入", "迈出"]


def _is_new_format(filename: str) -> bool:
    """检测是否新命名规范: TYPE_XXXXXXXX_ORG_PRODUCT_..."""
    name_no_ext = re.sub(r"\.(txt|pdf|docx?|xlsx|doc)$", "", filename, flags=re.IGNORECASE)
    parts = name_no_ext.split("_")
    if len(parts) < 6:
        return False
    return parts[0] in TYPE_PREFIX_MAP


def classify_document_type(file_name: str) -> str:
    """根据文件名判断文档类型。新格式直接取前缀，旧格式用关键词。"""
    if _is_new_format(file_name):
        prefix = file_name.split("_")[0]
        return TYPE_PREFIX_MAP.get(prefix, "general")
    return _classify_legacy(file_name)


def _classify_legacy(file_name: str) -> str:
    """旧格式：关键词匹配判断文档类型。"""
    clean = file_name.lower()
    if clean.endswith(".xlsx"):
        return "table"
    if any(kw in clean for kw in NEWS_KEYWORDS):
        return "news"
    clean_no_academy = clean.replace("研究院", "")
    if any(kw in clean_no_academy for kw in REPORT_KEYWORDS):
        return "report"
    if any(kw in clean for kw in MANUAL_KEYWORDS):
        return "manual"
    if any(kw in clean for kw in REGULATION_KEYWORDS):
        return "regulation"
    return "general"
