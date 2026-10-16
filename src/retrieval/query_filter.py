"""
Query 实体分析 + 元数据检索。
从用户问题中提取产品/机构/类型意图，然后在 chunks 表按元数据字段精确查找。
作为 Tier 1 参与 RRF 融合，软加权而非硬过滤。

第一层元数据过滤（机构前置过滤）：
- OrgName 枚举定义全部规范机构名（= chunks.org_name 值）
- ORG_ALIASES 提供简称→规范名映射
- find_mentioned_orgs 从问题中提取全部提及机构（支持多机构）
- build_org_scope 把机构集合转成 chunk_id/vector_id 范围集合
- 检索主流程（app.py）：命中机构 → 语义/词法路只在范围内检索；未命中 → 全范围
"""

import re
import time
import logging
import pymysql
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── 机构枚举（第一层过滤依据，value = chunks.org_name 规范名）──
class OrgName(str, Enum):
    PBOC = "中国人民银行"
    CBRC = "中国银监会办公厅"
    SAFE = "国家外汇管理局"
    STATE_COUNCIL_OFFICE = "国务院办公厅"
    NDRC = "国家发展改革委"
    UNIONPAY = "中国银联"
    UMS = "银联商务"
    LAKALA = "拉卡拉"
    YIKA = "移卡"
    SPDB = "浦发银行"
    BCOMM = "交通银行"
    CEB = "光大银行"
    GLB = "桂林银行"
    INSIGHTONE = "洞见科技"
    YUFENG = "御风算法"
    DAHUA = "大华捷通"
    CICC = "中金点睛"
    WALLSTREET = "华尔街见闻"
    CF40 = "中国金融四十人论坛"
    PROSECUTOR = "检察日报"
    JIEMIAN = "界面新闻"
    IFNEWS = "国际金融报"
    MPAYNEWS = "移动支付网"
    CCTV_FINANCE = "央视财经"
    BOSTON= "波士顿咨询"
    EUDaily="Official Journal of the European Communities"
    ChinaEconomicNet="中国经济网"
    DXYPay="中国电信翼支付"
    JianchaDaily="检察日报"
    Jinduan="锦锻"
# 简称/别名 → 规范机构枚举（谨慎添加：有歧义的短名宁可缺失也不误映射）
ORG_ALIASES = {
    "人民银行": OrgName.PBOC,
    "银监会": OrgName.CBRC,
    "外管局": OrgName.SAFE,
    "发改委": OrgName.NDRC,
    "银监": OrgName.CBRC,
    "银联": {OrgName.UNIONPAY, OrgName.UMS},
    "界面": OrgName.JIEMIAN,
    "四十人": OrgName.CF40,
    "波士顿": OrgName.BOSTON,
    "欧盟日报": OrgName.EUDaily,
    "欧盟时报": OrgName.EUDaily,
    "交行": OrgName.BCOMM,
    "光大": OrgName.CEB,
    "桂林": OrgName.GLB,
    "电信翼": OrgName.DXYPay,
    "银商": OrgName.UMS,
    }

# 动态枚举：启动时 sync_entities_from_db 从 DB 同步（硬编码为兜底）
_DYNAMIC_ORGS: list[str] = []
_DYNAMIC_PRODUCTS: list[str] = []


# ── 已知产品列表（从计划中的产品映射表）───────────────────
KNOWN_PRODUCTS = [
    "insightone","智慧停车场","客服对话分析助手","光大银行B2B网上支付","桂林银行B2B网上支付","交通银行B2B网上支付","浦发银行B2B网上支付","nginx","UPNGX","UPTrace","大华捷通",
    "电票打印","全民付","全民付蓝鲸","商圈魔盒","小U产品","小U点餐","小U集团","小U掌柜","移动远程支付","银商大脑","悦打印",
    "E外卖","E开票","云闪付","MCP智能支付","POS终端支付服务","银联卡特约商户类别码","智能合约","D+1结算产品","御风算法"
]

# ── 类型意图关键词 ──────────────────────────────────────
TYPE_HINTS = {
    "regulation": ["规定", "通知", "办法", "条例", "法规", "公告", "指导意见",
                   "暂行办法", "管理办法", "文号", "发文字号"],
    "manual": ["怎么用", "如何操作", "操作步骤", "怎么配置", "操作手册","步骤",
               "使用指南", "产品介绍", "功能", "如何使用"],
    "report": ["报告", "财报", "运行情况", "统计分析"],
    "news": ["新闻", "最新", "动态", "热点"],
}


@dataclass
class QueryEntities:
    product: str = ""
    org: str = ""  # 首个机构规范名（MetadataSearcher 兼容）
    orgs: list = field(default_factory=list)  # 全部提及机构（OrgName 枚举）
    type_hint: str = ""  # regulation / manual / report / news / ""
    has_entities: bool = False
    products: list = field(default_factory=list)  # 问题中出现的全部产品名（归一化去重）


def _org_names(target) -> list[str]:
    """别名目标 → 规范名列表（OrgName / set[OrgName] / 字符串）。"""
    if isinstance(target, str):
        return [target]
    return [o.value for o in target]


def find_mentioned_orgs(query: str) -> list[str]:
    """找出问题中提及的全部机构（规范名字符串，长优先 + 遮蔽已匹配片段 + 去重）。

    支持多机构："光大银行和交通银行的B2B网上支付" → [光大银行, 交通银行]
    简称映射："人民银行有哪些规定" → [中国人民银行]
    """
    # 候选：别名（短）与规范名（长：硬编码枚举 + DB 动态同步）合并，长优先避免 "银联商务" 被 "银联" 类短词截胡
    candidates = [(k, _org_names(v)) for k, v in ORG_ALIASES.items()]
    for org in OrgName:
        candidates.append((org.value, [org.value]))
    for name in _DYNAMIC_ORGS:
        candidates.append((name, [name]))
    candidates.sort(key=lambda x: -len(x[0]))

    found = []
    remaining = query
    for name, names in candidates:
        if name in remaining:
            found.extend(names)
            remaining = remaining.replace(name, " ", 1)

    seen = set()
    out = []
    for org in found:
        if org not in seen:
            seen.add(org)
            out.append(org)
    return out


def find_mentioned_products(query: str) -> list[str]:
    """找出问题中提及的全部产品名（长优先 + 遮蔽已匹配片段 + 归一化去重）。

    "全民付蓝鲸和银商大脑的关系" → [全民付蓝鲸, 银商大脑]
    """
    found = []
    remaining = query
    for p in sorted(KNOWN_PRODUCTS + _DYNAMIC_PRODUCTS, key=lambda x: -len(x)):
        if p in remaining:
            found.append(_normalize_product_name(p))
            remaining = remaining.replace(p, " ", 1)
    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def analyze_query(query: str) -> QueryEntities:
    """从用户问题中提取实体意图。"""
    result = QueryEntities()

    # 产品匹配 —— 收集全部提及产品（长优先，见 find_mentioned_products）
    result.products = find_mentioned_products(query)
    result.product = result.products[0] if result.products else ""

    # 机构匹配 —— 全部提及机构（长优先 + 别名归一，支持多机构）
    result.orgs = find_mentioned_orgs(query)
    result.org = result.orgs[0] if result.orgs else ""

    # 类型意图
    for hint_type, keywords in TYPE_HINTS.items():
        if any(kw in query for kw in keywords):
            result.type_hint = hint_type
            break

    result.has_entities = bool(result.product or result.org or result.type_hint)
    return result


def _normalize_product_name(name: str) -> str:
    """产品名归一化，与 analyze_query 保持一致。"""
    if name == "B2B":
        return "B2B网上支付"
    if name in ("D+1结算", "D+1结算产品","D+1"):
        return "D+1结算产品"
    return name

def build_product_mentions(db_conn: pymysql.Connection):
    """扫描全部 child chunks 正文，建立产品提及倒排索引。

    遍历 KNOWN_PRODUCTS，用子串匹配找出正文中提到产品名的 chunk，
    写入 product_mentions 表。构建只需 5-10 秒，查询时 <1ms。
    """
    # 归一化后的唯一产品名集合（硬编码 + DB 动态同步）
    unique_products = sorted(set(
        _normalize_product_name(p) for p in KNOWN_PRODUCTS + _DYNAMIC_PRODUCTS
    ))
    logger.info(f"开始构建产品提及索引，{len(unique_products)} 个产品名")

    with db_conn.cursor() as cur:
        # 清空旧数据
        cur.execute("TRUNCATE TABLE product_mentions")

        # 加载全部 child chunk 文本
        cur.execute("SELECT id, chunk_text, document_id FROM chunks WHERE chunk_level = 'child'")
        chunks = cur.fetchall()
        logger.info(f"  已加载 {len(chunks)} 个 chunks")

        # 内存扫描：产品名 → chunk 列表
        mentions = []
        for chunk_id, chunk_text, doc_id in chunks:
            if not chunk_text:
                continue
            for p in unique_products:
                if p in chunk_text:
                    mentions.append((p, chunk_id, doc_id))

        if not mentions:
            logger.warning("  未找到任何产品提及")
            return

        # 批量写入
        cur.executemany(
            "INSERT IGNORE INTO product_mentions (product_name, chunk_id, document_id) "
            "VALUES (%s, %s, %s)",
            mentions,
        )
        try:
            db_conn.commit()
        except Exception as e:
            logger.warning(f"product_mentions commit 失败: {e}")

        # 统计
        cur.execute("SELECT product_name, COUNT(*) FROM product_mentions GROUP BY product_name ORDER BY COUNT(*) DESC")
        for row in cur.fetchall():
            logger.info(f"  {row[0]}: {row[1]} chunks")

    logger.info(f"产品提及索引构建完成，共 {len(mentions)} 条记录")


def sync_entities_from_db(pool) -> tuple[int, int]:
    """启动时从 DB 同步机构/产品动态枚举，返回 (新增机构数, 新增产品数)。

    - 机构：chunks.org_name（child 级）distinct，排除空值与"通用"
    - 产品：chunks.product_name + product_mentions.product_name distinct
    硬编码枚举为兜底；DB 出现新机构/新产品并入动态集后参与实体检测。
    新增产品会触发 product_mentions 全量重建（用扩充后的产品集重扫正文）。
    """
    global _DYNAMIC_ORGS, _DYNAMIC_PRODUCTS

    conn = pool.connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT org_name FROM chunks "
                "WHERE chunk_level='child' AND org_name IS NOT NULL AND org_name != ''"
            )
            db_orgs = [r[0] for r in cur.fetchall()]
            cur.execute(
                "SELECT DISTINCT product_name FROM chunks "
                "WHERE chunk_level='child' AND product_name IS NOT NULL AND product_name != ''"
            )
            db_products = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT product_name FROM product_mentions")
            db_products += [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    hardcoded_orgs = {o.value for o in OrgName}
    hardcoded_products = {_normalize_product_name(p) for p in KNOWN_PRODUCTS}

    new_orgs = sorted({
        n for n in db_orgs
        if n and n != "通用" and len(n) >= 2
        and n not in hardcoded_orgs and n not in _DYNAMIC_ORGS
    })
    new_products = sorted({
        _normalize_product_name(p) for p in db_products
        if p and p != "通用" and len(p) >= 2
        and _normalize_product_name(p) not in hardcoded_products
        and _normalize_product_name(p) not in _DYNAMIC_PRODUCTS
    })

    if new_orgs:
        _DYNAMIC_ORGS.extend(new_orgs)
        logger.info(f"[实体同步] 新增动态机构 {len(new_orgs)} 个: {new_orgs}")
    if new_products:
        _DYNAMIC_PRODUCTS.extend(new_products)
        logger.info(f"[实体同步] 新增动态产品 {len(new_products)} 个: {new_products}")
        logger.info("[实体同步] 检测到新动态产品，重建 product_mentions 倒排...")
        build_product_mentions(pool.connection())

    return len(new_orgs), len(new_products)


def build_org_scope(db_conn: pymysql.Connection, orgs: list) -> dict | None:
    """第一层过滤：机构集合 → 检索范围集合。

    多机构取并集。返回 {"chunk_ids": set, "vector_ids": set}；
    范围为空返回 None（调用方回退全范围）。
    """
    if not orgs:
        return None

    names = list(orgs)
    with db_conn.cursor() as cur:
        ph = ",".join(["%s"] * len(names))
        cur.execute(
            f"SELECT id, vector_id FROM chunks "
            f"WHERE org_name IN ({ph}) AND chunk_level='child'",
            names,
        )
        rows = cur.fetchall()

    chunk_ids = {r[0] for r in rows}
    vector_ids = {r[1] for r in rows if r[1] is not None}
    if not chunk_ids:
        return None
    return {"chunk_ids": chunk_ids, "vector_ids": vector_ids}


class MetadataSearcher:
    """按元数据字段（产品/机构/类型）搜索 chunks，作为 Tier 1 参与融合。

    两路匹配：
    - 元数据标签: chunks.product_name / org_name / doc_type 精确匹配
    - 产品提及倒排: product_mentions 表，捕获正文中提到产品名但标签未标的 chunks
    """

    def __init__(self, pool):
        # 连接池（DBUtils PooledDB）：每次查询借还连接，线程安全无锁竞争
        self.pool = pool

    def _cursor(self):
        conn = self.pool.connection()
        return conn, conn.cursor(pymysql.cursors.DictCursor)

    def search(self, entities: QueryEntities, top_k: int = 20, query_type: str = "single_doc") -> list[dict]:
        """返回匹配的 child chunks 列表。"""
        if not entities.has_entities:
            logger.debug("[META] 无实体意图，跳过元数据检索")
            return []

        t0 = time.perf_counter()
        logger.info(
            f"[META] 实体: products={entities.products} org={entities.org} "
            f"type={entities.type_hint} query_type={query_type}"
        )

        # 全部提及产品（归一化、排除"通用"），支持多产品同时检索
        products = entities.products or ([entities.product] if entities.product else [])
        products = [p for p in products if p and p != "通用"]

        seen: dict[int, dict] = {}  # chunk_id → result, 用于去重合并

        # ── 路 1: 元数据标签精确匹配 ──
        conditions = []
        params = []

        if products:
            ph = ",".join(["%s"] * len(products))
            conditions.append(f"c.product_name IN ({ph})")
            params.extend(products)

        if entities.org and entities.org != "通用":
            conditions.append("c.org_name = %s")
            params.append(entities.org)

        if entities.type_hint:
            conditions.append("c.doc_type = %s")
            params.append(entities.type_hint)

        if conditions:
            where = " AND ".join(f"({c})" for c in conditions)
            sql_and = f"""
                SELECT c.id, c.chunk_text, c.document_id, c.org_name,
                       c.doc_type, c.product_name, d.file_name
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.chunk_level = 'child'
                  AND ({where})
                ORDER BY c.id DESC
                LIMIT %s
            """
            params.append(top_k * 2)

            try:
                conn = self.pool.connection()
                try:
                    with conn.cursor(pymysql.cursors.DictCursor) as cur:
                        cur.execute(sql_and, params)
                        rows = cur.fetchall()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"元数据检索失败: {e}")
                rows = []

            # AND 太严格降级 OR（分别构建 SQL，不做字符串替换）
            if not rows and len(conditions) > 1:
                where_or = " OR ".join(f"({c})" for c in conditions)
                sql_or = f"""
                    SELECT c.id, c.chunk_text, c.document_id, c.org_name,
                           c.doc_type, c.product_name, d.file_name
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.chunk_level = 'child'
                      AND ({where_or})
                    ORDER BY c.id DESC
                    LIMIT %s
                """
                try:
                    conn = self.pool.connection()
                    try:
                        with conn.cursor(pymysql.cursors.DictCursor) as cur:
                            cur.execute(sql_or, params)
                            rows = cur.fetchall()
                    finally:
                        conn.close()
                except Exception:
                    pass

            for row in rows:
                seen[row["id"]] = {
                    "id": row["id"],
                    "chunk_id": row["id"],
                    "text": row["chunk_text"],
                    "score": 0.9,
                    "source": "metadata",
                    "document_id": row["document_id"],
                    "file_name": row.get("file_name", ""),
                    "org_name": row.get("org_name", ""),
                    "doc_type": row.get("doc_type", ""),
                    "product_name": row.get("product_name", ""),
                }

        # ── 路 2: 产品提及倒排索引（cross_doc 时启用，捕获正文中提及产品名的跨文档 chunks）──
        if products and query_type == "cross_doc":
            try:
                conn = self.pool.connection()
                try:
                    with conn.cursor(pymysql.cursors.DictCursor) as cur:
                        ph = ",".join(["%s"] * len(products))
                        cur.execute(f"""
                            SELECT c.id, c.chunk_text, c.document_id, c.org_name,
                                   c.doc_type, c.product_name, d.file_name
                            FROM product_mentions pm
                            JOIN chunks c ON pm.chunk_id = c.id
                            JOIN documents d ON c.document_id = d.id
                            WHERE pm.product_name IN ({ph})
                              AND c.product_name NOT IN ({ph})
                              AND c.chunk_level = 'child'
                            ORDER BY c.document_id, c.id DESC
                            LIMIT %s
                        """, products + products + [top_k * 10])
                        rows = cur.fetchall()
                finally:
                    conn.close()
                doc_quota: dict[int, int] = {}  # 每文档最多取 2 条，保证跨文档多样性
                for row in rows:
                    cid = row["id"]
                    did = row["document_id"]
                    if cid not in seen and doc_quota.get(did, 0) < 2:
                        doc_quota[did] = doc_quota.get(did, 0) + 1
                        # 跨文档提及，与元数据标签同等权重
                        seen[cid] = {
                            "id": cid,
                            "chunk_id": cid,
                            "text": row["chunk_text"],
                            "score": 0.9,
                            "source": "metadata",
                            "document_id": row["document_id"],
                            "file_name": row.get("file_name", ""),
                            "org_name": row.get("org_name", ""),
                            "doc_type": row.get("doc_type", ""),
                            "product_name": row.get("product_name", ""),
                        }
            except Exception as e:
                logger.warning(f"产品提及检索失败: {e}")

        # ── 合并：按跨文档程度动态分配配额 ──
        label_matches = []
        cross_matches = []
        cross_docs: set[int] = set()
        target_products = set(products)

        for item in seen.values():
            if item.get("product_name") in target_products:
                label_matches.append(item)
            else:
                cross_matches.append(item)
                did = item.get("document_id")
                if did:
                    cross_docs.add(did)

        # 各来源内部按 score 排序
        label_matches.sort(key=lambda x: x["score"], reverse=True)
        cross_matches.sort(key=lambda x: x["score"], reverse=True)

        # 动态配额：跨文档覆盖的文档数越多，跨文档占比越高
        num_cross_docs = len(cross_docs)
        if num_cross_docs >= 5:
            cross_ratio = 0.5
        elif num_cross_docs >= 3:
            cross_ratio = 0.4
        elif num_cross_docs >= 2:
            cross_ratio = 0.3
        elif num_cross_docs >= 1:
            cross_ratio = 0.2
        else:
            cross_ratio = 0.0

        n_cross = min(len(cross_matches), max(1, int(top_k * cross_ratio)) if cross_ratio > 0 else 0)
        n_label = min(len(label_matches), top_k - n_cross)

        # 多产品：label 配额按产品轮询分配，避免单一产品挤占全部名额
        by_product: dict[str, list] = {}
        for item in label_matches:
            by_product.setdefault(item.get("product_name"), []).append(item)
        if len(by_product) >= 2:
            balanced = []
            queues = list(by_product.values())
            cursors = [0] * len(queues)
            while len(balanced) < n_label:
                progressed = False
                for i, q in enumerate(queues):
                    if cursors[i] < len(q):
                        balanced.append(q[cursors[i]])
                        cursors[i] += 1
                        progressed = True
                        if len(balanced) >= n_label:
                            break
                if not progressed:
                    break
            label_matches = balanced[:n_label]
            logger.info(
                f"  多产品配额均衡: "
                + ", ".join(f"{k}={len(v)}" for k, v in by_product.items())
            )

        logger.info(
            f"  元数据分配: label={n_label}/{len(label_matches)}, "
            f"cross={n_cross}/{len(cross_matches)} "
            f"(跨文档{num_cross_docs}个, ratio={cross_ratio:.0%})"
        )

        out = label_matches[:n_label] + cross_matches[:n_cross]
        logger.info(f"[META] 返回 {len(out)} 条 ({(time.perf_counter() - t0) * 1000:.0f}ms)")
        return out
