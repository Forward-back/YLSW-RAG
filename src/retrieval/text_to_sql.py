"""
Text-to-SQL 引擎：将自然语言数值查询转为 SQL 执行。

流程：
  1. 用户问"清分交易总金额" → query embedding → 检索 schema_registry
  2. 匹配到 table_name + column_name
  3. LLM 生成 SQL → 安全校验 → 执行 → 返回结果
"""

import os, sys, re, json, time, logging, threading
import pymysql
import requests

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import (
    MYSQL_CONFIG, DEEPSEEK_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    API_TIMEOUT, API_MAX_RETRIES,
)
from http_client import post_json
from retrieval.faiss_search import get_bge_embedding

logger = logging.getLogger(__name__)

# ── SQL 安全：允许的表名白名单 ──
ALLOWED_TABLES = {"tables_structured", "schema_registry"}

# ── SQL 安全：禁止的危险模式（第二道防线，配合 sqlparse 的 SELECT-only 校验）──
FORBIDDEN_SQL_PATTERNS = re.compile(
    r"(?i)(INTO\s+(OUTFILE|DUMPFILE)|LOAD_FILE\s*\(|SLEEP\s*\(|BENCHMARK\s*\(|"
    r"INFORMATION_SCHEMA\b|PERFORMANCE_SCHEMA\b|MYSQL\.|"
    r"\bEXEC\b|\bPREPARE\b|\bGRANT\b|\bREVOKE\b|\bSHUTDOWN\b)"
)


def _strip_code_fences(sql: str) -> str:
    """摘除 LLM 输出中的 ```sql ... ``` 围栏（按内容提取，而非逐字符 strip）。"""
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", sql.strip(), re.DOTALL)
    if m:
        return m.group(1).strip()
    return sql.strip()

# ── SQL 预筛：按实际表内容定义触发词，避免每次都调 DeepSeek ──
SQL_TABLE_TRIGGERS = {
    "清分交易": ["清分", "交易汇总", "机构代码", "批量机构号", "INST_ID", "BAT_INST"],
    "存款准备金率": ["准备金率", "存款准备金", "调整前", "调整后"],
    "信用卡": ["信用卡", "发卡量", "信贷余额", "计息积数", "透支利率", "日利率",
               "预借现金", "透支消费", "期末应偿"],
    "基础信息统计": ["收费站", "运营公司", "路段", "站所", "经纬度", "代投点",
                     "哪个公司", "哪些公司", "公司管", "归谁",
                     "收费部", "财务部", "财司", "高速", "公路", "联网", "站码"],
}

# ── 表级 SQL 模板：列举类高频查询确定性路由（不调 LLM）──
# 值：[(触发关键词, 目标列)]，按序匹配——"站所路段的运营公司"同时含站所/路段/运营公司，
# 运营公司排最前保证命中正确列。
SQL_DISTINCT_TEMPLATES = {
    "基础信息统计": [
        (("运营公司",), "隶属运营公司"),
        (("收费站", "站所"), "收费站名称"),
        (("代投点",), "代投点"),
        (("路段", "高速", "公路", "联网"), "隶属路段联网名称"),
        (("收费部",), "隶属路段集团收费部名称"),
        (("财务部",), "隶属路段融资财务部名称"),
        (("地址",), "地址"),
        (("经度",), "经度"),
        (("纬度",), "纬度"),
    ],
}

# ── 表别名 / 列别名：用户口语词 → 规范表键 / 规范列名 ──
# ingest 时写入 schema_registry.alias，SQL 生成 prompt 自动带出；此处为单一事实源。
TABLE_ALIASES = {
    "基础信息统计": ["收费站表", "站所路段表", "运营公司表"],
}
COLUMN_ALIASES = {
    "基础信息统计": {
        "收费站名称": ["站所", "站点", "收费站"],
        "隶属运营公司": ["运营方", "公司"],
        "隶属路段联网名称": ["路段", "联网名称"],
        "隶属路段集团收费部名称": ["收费部"],
        "隶属路段融资财务部名称": ["财务部"],
        "隶属路段财司户名": ["财司户名"],
        "隶属路段财司账号": ["财司账号"],
        "收费站代码": ["站码"],
        "代投点": ["代投"],
        "地址": ["位置"],
    },
}
SQL_GENERIC_TRIGGERS = [
    "多少", "总计", "合计", "统计", "汇总", "数量", "总共",
    "排名", "最高", "最低", "最大", "最小",
    "占比", "比例", "百分比",
    "分析", "计算", "平均", "同比", "环比",
    "金额", "收入", "利润", "增长率",
]
# ── 表级业务描述：代码表名 DeepSeek 猜不出含义，按子串匹配拼进 schema prompt ──
TABLE_DESC = {
    "BASE_QF_T_TRANS_DAY_HIS": "清分交易汇总信息-基础层交易日历史表字段字典（仅字段定义，无真实交易数据）",
    "MID_QF_M_TRANS_DAY_HIS": "清分交易汇总信息-月交易历史表字段字典（仅字段定义，无真实交易数据）",
    "MID_QF_T_TRANS_DAY_HIS": "清分交易汇总信息-交易日历史表字段字典（仅字段定义，无真实交易数据）",
    "清分交易": "清分交易汇总信息字段字典（3 张表：基础层日/月/交易日历史字段定义，仅字段定义，无真实交易数据）",
    "基础信息统计表": "收费站基础信息统计表（收费站名称/地址/经纬度/隶属运营公司/收费站代码/代投点，含真实数据）",
}
SCHEMA_SEARCH_PROMPT = """你是一个数据库查询助手。根据用户问题，判断是否需要进行数值查询。

用户问题: {question}

可用的数据表:
{schema_info}

判断：
1. 如果问题涉及具体的数值、金额、比例、统计，或询问某字段的全部取值（如"有哪些""列出""分别是哪些""哪些公司"）等需要查表获取的 → 生成 SQL
2. 只有纯概念解释、政策说明、与表中具体数据无关的问题 → 返回 "NO_TABLE_NEEDED"

要求：
- SQL 使用 JSON_EXTRACT(row_data, '$."列名"') 提取字段，中文列名两侧的双引号不能省略；字符串字段用 JSON_UNQUOTE(JSON_EXTRACT(...)) 或 ->> 运算符去除外层引号
- 列名必须从上述给定字段列表中选择，禁止自造字段
- 询问某字段的全部取值（"有哪些""列出""哪些公司"）时，只 SELECT DISTINCT 该单一字段，不要附带其他列，不要写 LIMIT
- 用户提到的名称可能不精确（如"仙桃站"对应表中"仙桃东""仙桃北"），名称类条件用 LIKE '%关键词%' 模糊匹配，不要用等值比较
- 表名用 tables_structured，每个 SQL 必须包含 FROM tables_structured 和 WHERE table_name = '实际表名' 条件
- 数值字段用 CAST 转换类型
- 询问字段取值列表时用 SELECT DISTINCT 或 GROUP BY 去重
- 只输出 SQL 或 "NO_TABLE_NEEDED"，不要其他内容
"""


def _validate_sql(sql: str) -> bool:
    """校验 SQL 安全性：仅允许单条 SELECT 查询白名单中的表，且不含危险函数。"""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        logger.warning(f"SQL 安全拦截: 多语句或空语句 → {sql[:100]}")
        return False

    try:
        import sqlparse
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False
        for stmt in parsed:
            if stmt.get_type() != "SELECT":
                logger.warning(f"SQL 安全拦截: 非 SELECT 语句 → {str(stmt)[:100]}")
                return False
    except ImportError:
        pass

    # 危险模式检查（先剥注释，防止 /* */ 与 -- 绕过关键字匹配）
    no_comments = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    no_comments = re.sub(r"--[^\n]*", "", no_comments)
    if FORBIDDEN_SQL_PATTERNS.search(no_comments):
        logger.warning(f"SQL 安全拦截: 命中危险模式 → {sql[:100]}")
        return False

    tables = set(re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE))
    tables.update(re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE))
    if not tables.issubset(ALLOWED_TABLES):
        logger.warning(f"SQL 安全拦截: 表名不在白名单 → {tables}")
        return False
    return True


class TextToSQLEngine:
    """结构化表格的 Text-to-SQL 查询引擎。"""

    def __init__(self):
        self.conn = pymysql.connect(**MYSQL_CONFIG, autocommit=True)
        self._lock = threading.Lock()  # Flask 多线程共享连接保护
        self._real_name_cache: dict[str, str] = {}
        self._entity_values: set[str] | None = None  # 具体站名/代投点值，模板实体防护用
        self._company_values: set[str] | None = None  # 隶属运营公司全部取值，模板公司过滤用
        self._road_values: list[str] | None = None  # 隶属路段联网名称全部取值（长优先排序）
        self._dept_values: list[str] | None = None  # 隶属路段集团收费部名称全部取值（长优先排序）
        self._field_name_values: set[str] | None = None  # 清分交易字段字典取值（字段名+中文注释）
        self._column_index: list[dict] = []  # 列语义索引（字段模糊问法检索，启动构建）
        self._table_index: dict[str, list[float]] = {}  # 表描述 embedding（schema linking，启动构建）
        self._build_column_index()
        self._build_table_index()

    def _build_column_index(self):
        """列语义索引：表名 + 列名 + 别名 + 含义（row_data 字段字典）jieba 分词。

        服务于模糊字段问法（"与金额相关的字段有哪些"）——词重叠打分直接答，
        不经 SQL 模板也不经 LLM。
        """
        import jieba
        try:
            entries = []
            with self._lock:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_name, column_name, sample_values, alias "
                        "FROM schema_registry"
                    )
                    for tname, cname, samples, alias in cur.fetchall():
                        entries.append({
                            "table": tname, "column": cname,
                            "text": f"{tname} {cname} {alias or ''} {samples or ''}",
                        })
                    # 清分表字段字典的 COMMENTS 含义并入（含中文语义）
                    cur.execute(
                        "SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COLUMN_NAME\"')) AS c, "
                        "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COMMENTS\"')) AS m "
                        "FROM tables_structured WHERE table_name LIKE '%清分交易%'"
                    )
                    for cname, comments in cur.fetchall():
                        if not cname:
                            continue
                        for e in entries:
                            if e["column"] == cname and "COMMENTS" not in e["text"]:
                                e["text"] += f" {comments or ''}"
                                break
            for e in entries:
                e["tokens"] = set(jieba.lcut(e["text"]))
            self._column_index = entries
            logger.info(f"[SQL] 列语义索引构建完成: {len(entries)} 列")
        except Exception as ex:
            logger.warning(f"[SQL] 列语义索引构建失败（字段模糊检索降级关闭）: {ex}")
            self._column_index = []

    def _build_table_index(self):
        """表描述 embedding 索引：多表场景的 schema linking 储备。

        每表条目 = 表名 + 全部列名 + 样例值；BGE 编码存内存。
        关键词剪枝（_hit_table_keys）未命中时按语义相似度取 top-2 表喂 LLM。
        """
        try:
            from retrieval.faiss_search import get_bge_embeddings
            tables: dict[str, list[str]] = {}
            with self._lock:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT table_name, column_name, sample_values "
                        "FROM schema_registry ORDER BY table_name, column_name"
                    )
                    for tname, cname, samples in cur.fetchall():
                        tables.setdefault(tname, []).append(
                            f"{cname}:{samples or ''}"
                        )
            texts = {
                t: f"{t} 列：{'；'.join(cols)}" for t, cols in tables.items()
            }
            embs = get_bge_embeddings(list(texts.values()))
            if embs:
                self._table_index = {
                    t: e for t, e in zip(texts.keys(), embs)
                }
                logger.info(f"[SQL] 表描述索引构建完成: {len(texts)} 表")
        except Exception as ex:
            logger.warning(f"[SQL] 表描述索引构建失败（语义选表降级关闭）: {ex}")
            self._table_index = {}

    def _search_columns(self, question: str) -> list[dict]:
        """模糊字段问法检索：词重叠打分取 top 列条目（零 API 成本）。"""
        if not self._column_index:
            return []
        import jieba
        tokens = set(jieba.lcut(question))
        scored = []
        for e in self._column_index:
            hit = len(tokens & e["tokens"])
            if hit:
                scored.append((hit, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:5]]

    def _load_entities(self) -> set[str]:
        """载入收费站名称/代投点的全部取值（模板实体防护：问题带具体实体时走 LLM 带 WHERE）。"""
        entities: set[str] = set()
        real = self._table_real_name("基础信息统计")
        if real:
            with self._lock:
                with self.conn.cursor() as cur:
                    for col in ("收费站名称", "代投点"):
                        cur.execute(
                            "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, %s)) v "
                            "FROM tables_structured WHERE table_name = %s",
                            (f'$."{col}"', real),
                        )
                        entities |= {str(r[0]) for r in cur.fetchall() if r[0]}
        return {e for e in entities if len(e) >= 2}

    def _question_has_entity(self, question: str) -> bool:
        """问题中含具体站名/代投点值 → 需要 WHERE 条件，模板不适用。

        双路匹配：实体完整出现在问题中；或问题中连续 2-3 字片段是某实体的前缀
        且长度差 ≤1（"仙桃站" → 表内"仙桃东"），拦截变体写法。
        """
        if self._entity_values is None:
            self._entity_values = self._load_entities()
        if any(e in question for e in self._entity_values):
            return True
        for i in range(len(question)):
            for width in (2, 3):
                frag = question[i:i + width]
                if len(frag) < 2:
                    continue
                if any(
                    e.startswith(frag) and len(e) - len(frag) <= 1
                    for e in self._entity_values
                ):
                    return True
        return False

    # 问题关键词 → 查询列（带实体条件的定点查询模板，如"X站的经纬度"）
    ENTITY_QUERY_COLUMNS = [
        (("经纬度",), ["经度", "纬度"]),
        (("地址",), ["地址"]),
        (("公司",), ["隶属运营公司"]),
        (("代投点",), ["代投点"]),
        (("代码", "站码"), ["收费站代码"]),
        (("收费部",), ["隶属路段集团收费部名称"]),
        (("财务部",), ["隶属路段融资财务部名称"]),
        (("财司",), ["隶属路段财司户名", "隶属路段财司账号"]),
        (("路段", "高速", "公路", "联网"), ["隶属路段联网名称"]),
    ]

    def _match_entities(self, question: str) -> list[str]:
        """问题中的站名/代投点实体匹配：完整包含优先，否则 2-4 字片段做前缀匹配。

        "仙桃站" → 片段"仙桃"前缀命中 [仙桃东, 仙桃北]；"上海站" → [上海北站, 上海南站]。
        上限 20 个防爆炸（泛词片段如"设备"命中过多时放弃，交 LLM 路径）。
        """
        if self._entity_values is None:
            self._entity_values = self._load_entities()
        hits = [e for e in self._entity_values if e in question]
        if hits:
            return sorted(hits, key=len)[:20]
        for width in (4, 3, 2):
            frags = {
                question[i:i + width]
                for i in range(len(question) - width + 1)
            }
            for frag in sorted(frags, key=len, reverse=True):
                prefix_hits = [e for e in self._entity_values if e.startswith(frag)]
                if prefix_hits:
                    return sorted(prefix_hits, key=len)[:20]
        return []

    def _try_entity_template(self, question: str) -> tuple[str, str] | None:
        """带实体条件的定点查询模板：'X站的经纬度' → WHERE 收费站名称 IN (...)。

        确定性出 SQL（零 LLM），且多候选实体全部带出（'上海站' → 上海北站+上海南站），
        名称不精确由 IN 列表兜住，不依赖 LLM 写 LIKE。
        """
        cols = next(
            (cs for kws, cs in self.ENTITY_QUERY_COLUMNS if any(k in question for k in kws)),
            None,
        )
        if not cols:
            return None
        entities = self._match_entities(question)
        if not entities:
            return None
        real_name = self._table_real_name("基础信息统计")
        if not real_name:
            return None
        select_cols = ["收费站名称"] + cols
        extracts = ", ".join(
            f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{c}\"')) AS `{c}`"
            for c in select_cols
        )
        in_list = ", ".join(f"'{e}'" for e in entities)
        sql = (
            f"SELECT {extracts} FROM tables_structured "
            f"WHERE table_name = '{real_name}' AND "
            f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"收费站名称\"')) IN ({in_list})"
        )
        logger.info(
            f"[SQL实体模板] {len(entities)} 候选: {entities[:5]} | {question[:60]}"
        )
        return sql, "基础信息统计"

    def _table_real_name(self, table_key: str) -> str | None:
        """表触发键 → tables_structured 里的实际 table_name（缓存）。"""
        cached = self._real_name_cache.get(table_key)
        if cached is not None:
            return cached or None
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT table_name FROM tables_structured "
                    "WHERE table_name LIKE %s",
                    (f"%{table_key}%",),
                )
                row = cur.fetchone()
        name = row[0] if row else None
        self._real_name_cache[table_key] = name or ""
        return name

    def _table_columns(self, table_key: str) -> list[str]:
        """表触发键 → schema_registry 字段名列表。"""
        real = self._table_real_name(table_key)
        if not real:
            return []
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM schema_registry WHERE table_name = %s",
                    (real,),
                )
                return [r[0] for r in cur.fetchall()]

    def _load_companies(self) -> set[str]:
        """载入隶属运营公司的全部取值（问题含具体公司名时，列举模板改 WHERE 过滤）。"""
        if self._company_values is not None:
            return self._company_values
        companies: set[str] = set()
        real = self._table_real_name("基础信息统计")
        if real:
            with self._lock:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, %s)) v "
                        "FROM tables_structured WHERE table_name = %s",
                        (f'$."隶属运营公司"', real),
                    )
                    companies = {str(r[0]) for r in cur.fetchall() if r[0]}
        self._company_values = {c for c in companies if len(c) >= 2}
        return self._company_values

    def _companies_in(self, question: str) -> list[str]:
        return [c for c in self._load_companies() if c in question]

    def _load_field_names(self) -> set[str]:
        """清分交易 3 张字段字典表的 COLUMN_NAME + COMMENTS 全部取值（缓存）。"""
        if self._field_name_values is not None:
            return self._field_name_values
        names: set[str] = set()
        with self._lock:
            with self.conn.cursor() as cur:
                for col in ("COLUMN_NAME", "COMMENTS"):
                    cur.execute(
                        "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, %s)) v "
                        "FROM tables_structured WHERE table_name LIKE %s",
                        (f'$."{col}"', "%清分交易%"),
                    )
                    names |= {str(r[0]) for r in cur.fetchall() if r[0]}
        self._field_name_values = {n for n in names if len(n) >= 2}
        return self._field_name_values

    def _try_field_dict_template(self, question: str) -> tuple[str, str] | None:
        """清分交易字段字典确定性模板（3 张表同族，用 LIKE 前缀一次覆盖）。

        三种问法：
          1. 字段含义：问题含具体字段名/中文注释 + 含义词 → WHERE 字段名或注释 IN
          2. 字段数量：数量词 + 字段 → COUNT(DISTINCT 字段名)（跨 3 表）
          3. 字段列表：问"有哪些字段" → DISTINCT 字段名（跨 3 表）
        """
        if "清分交易" not in self._hit_table_keys(question):
            return None
        family = "TBL_00000000_通用_通用_清分交易汇总信息%"
        if any(k in question for k in ("含义", "是什么", "什么意思", "定义", "代表什么")):
            hits = [v for v in self._load_field_names() if v in question]
            # 只保留最长匹配（"主键ID"命中时丢弃其子串"ID"），避免子串误命中
            hits = [h for h in hits if not any(h != o and h in o for o in hits)]
            if hits:
                in_list = ", ".join(f"'{v}'" for v in hits)
                sql = (
                    "SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COLUMN_NAME\"')) AS 字段名, "
                    "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COMMENTS\"')) AS 含义, "
                    "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"数据类型\"')) AS 数据类型 "
                    f"FROM tables_structured WHERE table_name LIKE '{family}' AND ("
                    f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COLUMN_NAME\"')) IN ({in_list}) OR "
                    f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COMMENTS\"')) IN ({in_list}))"
                )
                logger.info(f"[SQL字段模板] 含义 {hits}: {question[:60]}")
                return sql, "清分交易"
        if any(k in question for k in ("多少个", "几个", "数量")) and "字段" in question:
            sql = (
                "SELECT COUNT(DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COLUMN_NAME\"'))) AS 字段数 "
                f"FROM tables_structured WHERE table_name LIKE '{family}'"
            )
            logger.info(f"[SQL字段模板] 字段数量: {question[:60]}")
            return sql, "清分交易"
        if "字段" in question:
            sql = (
                "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"COLUMN_NAME\"')) AS 取值 "
                f"FROM tables_structured WHERE table_name LIKE '{family}'"
            )
            logger.info(f"[SQL字段模板] 字段列表: {question[:60]}")
            return sql, "清分交易"
        return None

    def _try_template_sql(self, question: str) -> tuple[str, str] | None:
        """列举类查询确定性模板：命中返回 (SQL, 表触发键)，不调 LLM。

        问题含具体实体（站名等）时不走模板——模板只做无 WHERE 的全量列举，
        带实体条件的查询交 LLM 生成 WHERE SQL。
        """
        if self._question_has_entity(question):
            return None
        companies = self._companies_in(question)
        for table_key, tmpls in SQL_DISTINCT_TEMPLATES.items():
            if not any(kw in question for kw in SQL_TABLE_TRIGGERS.get(table_key, ())):
                continue
            if companies:
                # 公司名已被消费为过滤条件："运营公司"关键词只是主语的一部分（如
                # "京珠运营公司 管理 收费站"），优先匹配其他列，仅无其他命中才回退运营公司列
                tmpls = [t for t in tmpls if t[1] != "隶属运营公司"] + [
                    t for t in tmpls if t[1] == "隶属运营公司"
                ]
            if any(kw in question for kw in ("路段", "联网", "分布")):
                # 问"分布在哪些路段"时"收费站"词先命中的是站名列，须让路段列优先
                tmpls = [t for t in tmpls if t[1] == "隶属路段联网名称"] + [
                    t for t in tmpls if t[1] != "隶属路段联网名称"
                ]
            for keywords, column in tmpls:
                if any(kw in question for kw in keywords):
                    real_name = self._table_real_name(table_key)
                    if not real_name:
                        return None
                    # 计数词问法：COUNT(DISTINCT col) 直答，避免列值列举被截断数错
                    if any(k in question for k in ("多少个", "几个", "共有", "一共", "数量")):
                        sql = (
                            f"SELECT COUNT(DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{column}\"'))) AS 数量 "
                            f"FROM tables_structured WHERE table_name = '{real_name}'"
                        )
                        logger.info(f"[SQL模板] 计数 {table_key}.{column}: {question[:60]}")
                        return sql, table_key
                    if companies:
                        in_list = ", ".join(f"'{c}'" for c in companies)
                        sql = (
                            f"SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{column}\"')) AS 取值 "
                            f"FROM tables_structured WHERE table_name = '{real_name}' AND "
                            f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"隶属运营公司\"')) IN ({in_list})"
                        )
                        logger.info(
                            f"[SQL模板] 命中 {table_key}.{column} 过滤公司{companies}: {question[:60]}"
                        )
                    else:
                        sql = (
                            f"SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{column}\"')) AS 取值 "
                            f"FROM tables_structured WHERE table_name = '{real_name}'"
                        )
                        logger.info(
                            f"[SQL模板] 命中 {table_key}.{column}: {question[:60]}"
                        )
                    return sql, table_key
        return None

    def _try_count_template(self, question: str) -> tuple[str, str] | None:
        """计数类确定性模板（三层，防误拦）。

        1. 排序计数："最多/最少/排名前三" → GROUP BY ORDER BY LIMIT
        2. 条件计数：含具体公司名（+可选路段名双条件）→ WHERE 过滤 COUNT
        3. 全表计数：仅当含"总共/一共/共有/总计"信号词（否则放行 LLM SQL，
           防聚合/多条件类问题被 COUNT(*) 短路）
        """
        if not any(k in question for k in ("数量", "多少个", "几个", "几家",
                                           "多少家", "有几", "共计", "共几")):
            return None
        if "收费站" not in question and "站所" not in question:
            return None
        real_name = self._table_real_name("基础信息统计")
        if not real_name:
            return None

        # 1. 排序计数：分组聚合 + 排序 + 限量
        if any(w in question for w in ("最多", "最少", "最大", "最小", "排名")):
            limit = 3 if any(w in question for w in ("前三", "前3")) else 1
            direction = "ASC" if any(w in question for w in ("最少", "最小")) else "DESC"
            group_col = ("隶属路段集团收费部名称" if "收费部" in question
                         else "隶属运营公司")
            sql = (
                f"SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{group_col}\"')) AS 分组, "
                f"COUNT(*) AS 收费站数量 FROM tables_structured "
                f"WHERE table_name = '{real_name}' "
                f"GROUP BY 分组 ORDER BY 收费站数量 {direction} LIMIT {max(limit, 2)}"
            )
            logger.info(f"[SQL计数模板] 排序计数 by {group_col} {direction} limit {limit}: {question[:60]}")
            return sql, "基础信息统计"

        hit_comps = self._companies_in(question)
        cond_col = "隶属运营公司"
        if not hit_comps:
            # 收费部名实体（如"京港澳高速"是收费部值不是公司名）
            hit_comps = self._dept_in(question)
            if hit_comps:
                cond_col = "隶属路段集团收费部名称"

        # 2. 条件计数（公司/收费部 + 可选路段双条件）
        if hit_comps:
            roads = self._road_in(question)
            in_list = ", ".join(f"'{c}'" for c in hit_comps)
            where = (f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{cond_col}\"')) IN ({in_list})")
            if roads:
                r_list = ", ".join(f"'{r}'" for r in roads)
                where += (f" AND JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"隶属路段联网名称\"')) "
                          f"IN ({r_list})")
                sel = (
                    f"SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{cond_col}\"')) AS 分组, "
                    f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"隶属路段联网名称\"')) AS 路段, "
                )
                grp = "分组, 路段"
            else:
                sel = f"SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{cond_col}\"')) AS 分组, "
                grp = "分组"
            sql = (
                sel + f"COUNT(*) AS 收费站数量 FROM tables_structured "
                f"WHERE table_name = '{real_name}' AND {where} "
                f"GROUP BY {grp}"
            )
            logger.info(f"[SQL计数模板] 条件计数 by {cond_col} {hit_comps}{'+路段' if roads else ''}: {question[:60]}")
            return sql, "基础信息统计"

        # 3. 全表计数（收紧触发词，防误拦）
        if any(k in question for k in ("总共", "一共", "共有", "总计")):
            sql = (
                f"SELECT COUNT(*) AS 收费站数量 FROM tables_structured "
                f"WHERE table_name = '{real_name}'"
            )
            logger.info(f"[SQL计数模板] 全表计数: {question[:60]}")
            return sql, "基础信息统计"
        return None

    def _road_in(self, question: str) -> list[str]:
        """问题中提及的联网路段名（表内精确匹配，长优先）。"""
        if self._road_values is None:
            real = self._table_real_name("基础信息统计")
            vals: set[str] = set()
            if real:
                with self._lock:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, %s)) v "
                            "FROM tables_structured WHERE table_name = %s",
                            ('$."隶属路段联网名称"', real),
                        )
                        vals = {str(r[0]) for r in cur.fetchall() if r[0]}
            self._road_values = sorted(vals, key=len, reverse=True)
        return [r for r in self._road_values if r in question]

    def _dept_in(self, question: str) -> list[str]:
        """问题中提及的收费部名（表内精确匹配，长优先）。"""
        if self._dept_values is None:
            real = self._table_real_name("基础信息统计")
            vals: set[str] = set()
            if real:
                with self._lock:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "SELECT DISTINCT JSON_UNQUOTE(JSON_EXTRACT(row_data, %s)) v "
                            "FROM tables_structured WHERE table_name = %s",
                            ('$."隶属路段集团收费部名称"', real),
                        )
                        vals = {str(r[0]) for r in cur.fetchall() if r[0]}
            self._dept_values = sorted(vals, key=len, reverse=True)
        return [d for d in self._dept_values if d in question]

    def _try_extreme_template(self, question: str) -> tuple[str, str] | None:
        """极值模板：'经度最大的收费站' → ORDER BY CAST(数值列) LIMIT 1 取名。"""
        if not any(w in question for w in ("最大", "最小")):
            return None
        col = None
        for kw, c in (("经度", "经度"), ("纬度", "纬度")):
            if kw in question:
                col = c
                break
        if not col:
            return None
        real_name = self._table_real_name("基础信息统计")
        if not real_name:
            return None
        direction = "ASC" if "最小" in question else "DESC"
        sql = (
            f"SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"收费站名称\"')) AS 收费站名称, "
            f"CAST(JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"{col}\"')) AS DECIMAL(20,8)) AS {col} "
            f"FROM tables_structured WHERE table_name = '{real_name}' "
            f"ORDER BY {col} {direction} LIMIT 1"
        )
        logger.info(f"[SQL极值模板] {col} {direction}: {question[:60]}")
        return sql, "基础信息统计"

    def _try_code_lookup(self, question: str) -> tuple[str, str] | None:
        """收费站代码反查：'代码为20213的收费站' → WHERE 收费站代码 = 数字。"""
        if "代码" not in question:
            return None
        m = re.search(r"(\d{4,6})", question)
        if not m:
            return None
        real_name = self._table_real_name("基础信息统计")
        if not real_name:
            return None
        sql = (
            "SELECT JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"收费站代码\"')) AS 收费站代码, "
            "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"收费站名称\"')) AS 收费站名称, "
            "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"地址\"')) AS 地址, "
            "JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"隶属运营公司\"')) AS 隶属运营公司 "
            f"FROM tables_structured WHERE table_name = '{real_name}' AND "
            f"JSON_UNQUOTE(JSON_EXTRACT(row_data, '$.\"收费站代码\"')) = '{m.group(1)}'"
        )
        logger.info(f"[SQL代码反查] {m.group(1)}: {question[:60]}")
        return sql, "基础信息统计"

    @staticmethod
    def _format_hint(table_key: str, rows: list[dict], field_list: list[str]) -> str:
        """SQL 结果文本：表说明 + 字段清单 + 紧凑 JSON，供生成阶段理解字段语义。"""
        body = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(body) > 4000:
            body = body[:4000] + "…(结果已截断)"
        if table_key:
            desc = next((d for k, d in TABLE_DESC.items()
                         if table_key in k or k in table_key), "")
            fields = "、".join(field_list)
            return f"表「{table_key}」说明：{desc}\n字段：{fields}\n结果：\n{body}"
        return f"结果：\n{body}"

    def get_schema_info(self, table_keys: list[str] | None = None) -> str:
        """获取已注册表 schema 信息；table_keys 非空时只返回命中表的 schema（剪枝）。"""
        with self._lock:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute("""
                    SELECT table_name, column_name, column_type, sample_values, alias
                    FROM schema_registry
                    ORDER BY table_name, column_name
                """)
                rows = cur.fetchall()

        if not rows:
            return "（暂无可用表格）"

        if table_keys:
            rows = [
                r for r in rows
                if any(k in r["table_name"] for k in table_keys)
            ]
            if not rows:
                return "（暂无可用表格）"

        tables = {}
        for r in rows:
            tn = r["table_name"]
            if tn not in tables:
                tables[tn] = []
            tables[tn].append(
                f"  {r['column_name']} ({r['column_type']})"
                + (f" - 示例: {r['sample_values']}" if r['sample_values'] else "")
                + (f" / 别名: {r['alias']}" if r['alias'] else "")
            )

        parts = []
        for tn, cols in tables.items():
            desc = next((d for k, d in TABLE_DESC.items() if k in tn), "")
            aliases = next(
                (a for k, a in TABLE_ALIASES.items() if k in tn), []
            )
            head = f"表: {tn}" + (f"（{desc}）" if desc else "")
            if aliases:
                head += f"（别名: {'、'.join(aliases)}）"
            parts.append(head + "\n" + "\n".join(cols))
        return "\n\n".join(parts)

    def _hit_table_keys(self, question: str) -> list[str]:
        """问题命中的表触发键列表（schema 剪枝依据）。"""
        return [
            table_name for table_name, keywords in SQL_TABLE_TRIGGERS.items()
            if any(kw in question for kw in keywords)
        ]

    def _should_query(self, question: str) -> bool:
        """规则预筛：仅当问题命中实际表内容或计算意图时才走 SQL。"""
        if self._hit_table_keys(question):
            for key in self._hit_table_keys(question):
                logger.info(f"  SQL 预筛命中表: {key}")
            return True
        if any(kw in question for kw in SQL_GENERIC_TRIGGERS):
            logger.info("  SQL 预筛命中计算意图")
            return True
        return False

    def query(self, question: str) -> dict | None:
        """执行 Text-to-SQL 查询：模板优先，LLM 生成兜底。

        Returns: {"sql": "...", "result": [...], "answer_hint": "..."} 或 None
        """
        t0 = time.perf_counter()

        def _ms():
            return f"{(time.perf_counter() - t0) * 1000:.0f}ms"

        if not self._should_query(question):
            logger.debug(f"[SQL] 预筛未命中，跳过: {question[:60]}")
            return None

        # 模板优先：极值 > 计数 > 代码反查 > 列举/定点查询，确定性出 SQL，零 LLM 成本
        table_key = None
        use_fields = False
        extreme_hit = self._try_extreme_template(question)
        if extreme_hit:
            sql, table_key = extreme_hit
        else:
            count_hit = self._try_count_template(question)
            if count_hit:
                sql, table_key = count_hit
            else:
                code_hit = self._try_code_lookup(question)
                if code_hit:
                    sql, table_key = code_hit
                else:
                    template_hit = self._try_template_sql(question)
                    if template_hit:
                        sql, table_key = template_hit
                    else:
                        entity_hit = self._try_entity_template(question)
                        if entity_hit:
                            sql, table_key = entity_hit
                        else:
                            field_hit = self._try_field_dict_template(question)
                            if field_hit:
                                sql, table_key = field_hit
                            else:
                                # 模糊字段问法兜底：列语义索引词重叠直接答（零 SQL 零 LLM）
                                if any(k in question for k in ("字段", "列", "含义")):
                                    cols = self._search_columns(question)
                                    if cols:
                                        body = "\n".join(
                                            f"{c['column']}（{c['table'][:40]}）: {c['text'][:150]}"
                                            for c in cols
                                        )
                                        logger.info(
                                            f"[SQL列索引] 模糊字段命中 {len(cols)} 列 ({_ms()})"
                                        )
                                        return {
                                            "sql": "",
                                            "result": [],
                                            "answer_hint": f"[字段语义匹配]\n{body}",
                                        }
                                # schema 剪枝：只喂预筛命中的表。无表 key 命中时不调 LLM 判定
                                # （泛计算词如"多少/统计"的新闻数值题本就无表可查，直接回语料）
                                hit_keys = self._hit_table_keys(question)
                                if not hit_keys:
                                    logger.info(f"[SQL] 无表 key 命中，跳过 LLM 判定 ({_ms()})")
                                    return None
                                schema_info = self.get_schema_info(hit_keys)
                                if not schema_info or "暂无" in schema_info:
                                    logger.info(f"[SQL] 无可用表格 schema，跳过 ({_ms()})")
                                    return None

                                prompt = SCHEMA_SEARCH_PROMPT.format(
                                    question=question, schema_info=schema_info
                                )

                                sql = self._call_llm_for_sql(prompt)
                                if sql is None:
                                    logger.warning(f"[SQL] LLM 生成失败 ({_ms()})")
                                    return None
                                if sql.strip() == "NO_TABLE_NEEDED":
                                    logger.info(f"[SQL] LLM 判定无需查表 ({_ms()})")
                                    return None
                                # LLM 路径：从 SQL 反推涉及的表，拼表说明
                                use_fields = True
                                m = re.search(r"table_name\s*=\s*'([^']+)'", sql)
                                if m:
                                    real = m.group(1)
                                    table_key = next(
                                        (k for k in SQL_TABLE_TRIGGERS if k in real), None
                                    )

        sql = _strip_code_fences(sql)
        logger.info(f"[SQL] 生成: {sql[:200]} ({_ms()})")

        if not _validate_sql(sql):
            logger.warning("[SQL] 安全校验未通过，拒绝执行")
            return None

        try:
            with self._lock:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            logger.info(f"[SQL] 执行成功 {len(rows)} 行 ({_ms()})")
            rows = rows[:100]
            field_list = self._table_columns(table_key) if use_fields else []
            return {
                "sql": sql,
                "result": rows,
                "answer_hint": self._format_hint(table_key, rows, field_list),
            }
        except Exception as e:
            logger.warning(f"[SQL] 执行失败: {e} ({_ms()})")
            return None

    def _call_llm_for_sql(self, prompt: str) -> str | None:
        """调 DeepSeek 生成 SQL。重试策略见 http_client。"""
        resp, err = post_json(
            DEEPSEEK_URL,
            {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0,
                "thinking": {"type": "disabled"},
            },
            {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            API_TIMEOUT,
        )
        if resp is None:
            logger.warning(f"SQL LLM 请求失败: {err}")
            return None
        if resp.status_code != 200:
            logger.warning(f"SQL LLM HTTP {resp.status_code}: {resp.text[:150]}")
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()

    def close(self):
        self.conn.close()
