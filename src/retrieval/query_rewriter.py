"""
查询模板重写器：产品 × 问法模板 → 预计算检索参数。
覆盖 60% 单文档标准问法，0ms 返回，不调 LLM。
"""

import logging

logger = logging.getLogger(__name__)

# ── 问法模板 ──
# {product} 在匹配时替换为实际产品名
# fragments: 所有片段必须在 query 中出现才算命中
TEMPLATES = [
    {
        "name": "features",
        "fragments": ["{product}",  "功能"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["产品介绍", "产品功能"],
    },
    {
        "name": "how_to_use",
        "fragments": ["{product}", "怎么用"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["使用指南", "操作步骤", "如何使用"],
    },
    {
        "name": "how_to_config",
        "fragments": ["{product}", "怎么配置"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["配置指南", "配置方法", "如何配置"],
    },
    {
        "name": "operation_flow",
        "fragments": ["{product}", "操作流程"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["操作步骤", "流程说明"],
    },
    {
        "name": "what_is",
        "fragments": ["{product}", "是什么"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["产品介绍", "产品概述"],
    },
    {
        "name": "product_intro",
        "fragments": ["{product}", "产品介绍"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["产品概述", "产品说明"],
    },
    {
        "name": "faq",
        "fragments": ["{product}", "常见问题"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["FAQ", "常见问题"],
    },
    {
        "name": "payment_flow",
        "fragments": ["{product}", "支付流程"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["支付步骤", "交易流程"],
    },
    {
        "name": "onboarding",
        "fragments": ["{product}", "商户", "接入"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["商户接入", "接入指南", "入驻"],
    },
    {
        "name": "pricing",
        "fragments": ["{product}", "收费标准"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["费率", "定价", "收费"],
    },
    {
        "name": "account_mgmt",
        "fragments": ["{product}", "账户"],
        "type_hint": "manual",
        "query_type": "single_doc",
        "synonyms": ["账户管理", "账户设置"],
    },
    {
        "name": "report_analysis",
        "fragments": ["{product}", "报表"],
        "type_hint": "report",
        "query_type": "single_doc",
        "synonyms": ["统计报表", "数据报表"],
    },
    # ── cross_doc 模板：比较/跨文档查询 ──
    {
        "name": "compare",
        "fragments": ["{product}", "区别"],
        "type_hint": "manual",
        "query_type": "cross_doc",
        "synonyms": ["对比", "差异", "不同"],
    },
    {
        "name": "compare2",
        "fragments": ["{product}", "对比"],
        "type_hint": "manual",
        "query_type": "cross_doc",
        "synonyms": ["比较", "差异", "区别"],
    },
    {
        "name": "cross_year",
        "fragments": ["{product}", "2023", "2024"],
        "type_hint": "regulation",
        "query_type": "cross_doc",
        "synonyms": ["年度对比", "跨年度"],
    },
    {
        "name": "multi_product",
        "fragments": ["{product}", "分别"],
        "type_hint": "manual",
        "query_type": "cross_doc",
        "synonyms": ["各自", "分别有"],
    },
]


class QueryRewriter:
    """模板驱动的查询重写器。"""

    def __init__(self, products: list[str]):
        self.products = products
        # 预展开：product × template → 匹配结果
        self._build_cache()

    def _build_cache(self):
        """启动时预计算所有 product × template 组合的匹配 key。"""
        self.entries: list[tuple[str, dict]] = []  # [(match_key, template_config), ...]
        for tpl in TEMPLATES:
            for product in self.products:
                # 生成匹配 key：产品名替换模板中的 {product}
                match_key = "|".join(
                    f.replace("{product}", product) for f in tpl["fragments"]
                )
                self.entries.append((match_key, {
                    "product": product,
                    "type_hint": tpl["type_hint"],
                    "query_type": tpl["query_type"],
                    "synonyms": tpl["synonyms"],
                    "template_name": tpl["name"],
                }))
        # 按 match_key 长度降序（长优先，避免 "小U" 误匹配 "小U掌柜" 的模板）
        self.entries.sort(key=lambda x: -len(x[0]))
        logger.info(f"查询模板预加载: {len(TEMPLATES)} 模板 × {len(self.products)} 产品 = {len(self.entries)} 条规则")

    def rewrite(self, query: str) -> dict | None:
        """尝试模板匹配。命中返回改写结果，未命中返回 None。

        Returns: {"product": ..., "type_hint": ..., "query_type": ..., "synonyms": [...]}
        """
        for match_key, config in self.entries:
            # match_key 格式: "小U|有哪些功能|功能"
            fragments = match_key.split("|")
            if all(f in query for f in fragments):
                logger.debug(f"  模板命中: {config['template_name']} → product={config['product']}")
                return config
        logger.debug(f"  模板未命中: {query[:60]}")
        return None
