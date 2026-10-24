"""
Flask HTTP API 主服务 —— 暴露 POST /query 接口。
"""

import os, sys, re, atexit, logging, threading, time
from concurrent.futures import ThreadPoolExecutor

_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from core.utils import setup_windows_console
setup_windows_console()

from core.config import (
    MYSQL_CONFIG, FINAL_TOP_K, FAISS_TOP_K, BM25_TOP_K, LOG_LEVEL,
    SQL_CONTEXT_CAP, RERANK_SKIP_SINGLE, EMBEDDING_DIM,
    BGE_EMBED_BACKEND, BGE_RERANK_BACKEND,
    validate as config_validate,
)
import pymysql

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [req:%(trace)s] %(message)s",
)
logger = logging.getLogger("api")

from core.trace_logging import install_trace_filter, start_trace, clear_trace, submit_with_trace
install_trace_filter()

from flask import Flask, request, jsonify

from retrieval.faiss_search import FAISSSearcher, get_bge_embeddings
from retrieval.bm25_search import BM25Searcher
from retrieval.chunk_store import ChunkStore
from retrieval.text_to_sql import TextToSQLEngine
from retrieval.fusion import FusionEngine
from retrieval.planner import plan_query, execute_plan, generate_query_variants
from retrieval.query_filter import (
    MetadataSearcher, analyze_query, QueryEntities,
    build_product_mentions, sync_entities_from_db,
)
from prompts import get_tag_content, load_prompt_store
from llm.generator import generate_answer, generate_synthesis_answer
from core.prompt_guard import REFUSAL_ANSWER

app = Flask(__name__)
# 竞赛评测按 UTF-8 文本读响应体，禁用 jsonify 默认的 ASCII 转义（\uXXXX）
app.json.ensure_ascii = False
app.json.mimetype = "application/json; charset=utf-8"

# 允许测试页以 file:// 方式直接打开时跨域调用 API
@app.after_request
def _add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

# ── 延迟初始化 ──
faiss_searcher: FAISSSearcher | None = None
bm25_searcher: BM25Searcher | None = None
sql_engine: TextToSQLEngine | None = None
fusion: FusionEngine | None = None
meta_searcher: MetadataSearcher | None = None
# 连接池（DBUtils）：替代原版单连接 + 全局锁的串行访问
db_pool = None
chunk_store: ChunkStore | None = None


def init_services():
    global faiss_searcher, bm25_searcher, sql_engine, fusion, meta_searcher, \
        db_pool, chunk_store
    config_validate()
    logger.info("正在初始化检索服务...")
    from dbutils.pooled_db import PooledDB
    db_pool = PooledDB(
        creator=pymysql, maxconnections=8, mincached=2, maxcached=4,
        blocking=True, autocommit=True, **MYSQL_CONFIG,
    )
    faiss_searcher = FAISSSearcher()
    sql_engine = TextToSQLEngine()
    fusion = FusionEngine()
    chunk_store = ChunkStore(db_pool)
    # metadata 检索与 chunk 数据访问共用连接池（线程安全，无锁竞争）
    meta_searcher = MetadataSearcher(db_pool)
    load_prompt_store()
    atexit.register(_cleanup)

    # BM25 词法路：启动时从池借连接构建内存索引，失败仅降级关闭该路（不阻断启动）
    try:
        bm25_searcher = BM25Searcher(db_pool.connection())
    except Exception as e:
        logger.error(f"BM25 索引初始化异常，词法路降级关闭: {e}")
        bm25_searcher = None

    # 存量数据：product_mentions 空表时自动构建
    _ensure_product_mentions()

    # 本地模型预热（必须在 app 进程内加载：entrypoint 的预热是独立进程，退出即失效；
    # 不预热则首问触发懒加载 ~4.5s，实测首问检索三段 4493ms）
    if BGE_EMBED_BACKEND == "local" or BGE_RERANK_BACKEND == "local":
        try:
            from core.models_singleton import encode_texts, rerank_scores
            if BGE_EMBED_BACKEND == "local":
                encode_texts(["预热"], EMBEDDING_DIM)
                logger.info("本地 embedding 模型已预热")
            if BGE_RERANK_BACKEND == "local":
                rerank_scores("预热", ["预热"])
                logger.info("本地 reranker 模型已预热")
        except Exception as e:
            logger.warning(f"本地模型预热失败（首问会慢）: {e}")

    logger.info("所有服务初始化完成 ✓")


def _cleanup():
    if sql_engine:
        sql_engine.close()
    if db_pool:
        db_pool.close()


def _ensure_product_mentions():
    """实体枚举动态同步 + 倒排索引自愈。

    先 sync_entities_from_db 把 DB 机构/产品并入内存枚举（新增产品会触发重建），
    再检测 product_mentions 空表自动构建。
    """
    try:
        sync_entities_from_db(db_pool)
        conn = db_pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM product_mentions")
                count = cur.fetchone()[0]
        finally:
            conn.close()
        if count == 0:
            logger.info("product_mentions 表为空，自动构建倒排索引...")
            build_product_mentions(db_pool.connection())
    except Exception as e:
        logger.warning(f"product_mentions 检测/构建跳过: {e}")


def _prepend_doc_summaries(contexts: list[str], doc_ids: list[int]) -> list[str]:
    """多步题涉及的文档要点摘要前置到上下文最前（LLM 首因效应）。

    仅多步题（planner_multi）追加；单步题 parent chunk 已覆盖全貌。
    doc_ids 为最终进入上下文的文档（按组装顺序去重，上限 3 篇），
    避免低分候选文档的摘要混入噪声。
    """
    summaries = []
    for d in doc_ids[:3]:
        s = chunk_store.summary_of(d) if chunk_store else None
        if s:
            summaries.append(f"【文档摘要】{s}")
    if summaries:
        logger.info(f"[摘要前置] {len(summaries)} 篇文档摘要加入上下文头部")
    return summaries + contexts


def _run_query(question: str, collect_timing: bool = False, collect_trace: bool = False):
    """主查询管线：实体检测 → 规划器（tag + 计划）→ 并行检索 → RRF → 精排 → Parent → LLM，返回答案文本。

    collect_timing=True 时返回 (answer, timings, trace)，timings 为各阶段耗时字典（毫秒）。
    collect_trace=True 时 trace 为检索链中间数据（评测用）。
    """
    trace_id = start_trace()
    timing: dict = {}
    trace: dict = {}

    def _ms(a, b):
        return f"{(b - a) * 1000:.1f}ms"

    def _rec(key, a, b):
        if collect_timing:
            timing[key] = round((b - a) * 1000, 1)

    def _tr_cand(c):
        if not isinstance(c, dict):
            return None
        rs = c.get("rerank_score")
        return {
            "chunk_id": c.get("chunk_id"),
            "document_id": c.get("document_id"),
            "file_name": c.get("file_name", ""),
            "score": round(float(c.get("score") or 0), 4),
            "rerank_score": round(float(rs), 4) if rs is not None else None,
            "sources": c.get("sources") or ([c.get("source")] if c.get("source") else []),
        }

    def _tr_list(key, items):
        if collect_trace:
            trace[key] = [x for x in (_tr_cand(c) for c in items) if x]

    try:
        logger.info(f">>> 查询开始: {question[:80]}")
        t0 = time.perf_counter()

        # ── 0. 实体检测（非 LLM：产品/机构/类型提示）──
        entities = analyze_query(question)
        expanded_query = question
        logger.info(
            f"[实体] products={entities.products} org={entities.org} "
            f"type_hint={entities.type_hint}"
        )

        # query_type：多产品强制跨文档 > 自动判断
        if len(entities.products) >= 2:
            # 问题涉及多个产品（如 "A和B的关系"），必须跨文档检索
            query_type = "cross_doc"
        else:
            query_type = _detect_query_type(question, entities)
        logger.info(f"  query_type: {query_type}")

        # ── 0.6 问题规划器 + 变体生成（并行：变体只基于原问题，不依赖改写结果）──
        # 标签供生成阶段动态提示词；reject 按 reason 分流：injection/unknown 拒答，
        # chitchat 放行走正常管线（拒答唯一触发源 = LLM 注入判定，无代码层绕过）
        with ThreadPoolExecutor(max_workers=2) as _pex:
            fut_plan = submit_with_trace(
                _pex, plan_query, question,
                products=[p for p in entities.products] if entities.products else None,
                orgs=entities.orgs if entities.orgs else None,
            )
            fut_variants = submit_with_trace(
                _pex, generate_query_variants, question, question
            )
            plan, tag, reject, reject_reason = fut_plan.result()
            _rec("规划改写", t0, time.perf_counter())
            pre_variants = fut_variants.result() or []
            _rec("变体生成", t0, time.perf_counter())
        if reject and reject_reason == "chitchat":
            # 闲聊判定放行走正常管线：业务题可能被误判闲聊（如 ChatGPT 支付题），
            # 检索空时由生成层空上下文文案兜底；注入判定不受影响
            logger.info(f"[PLANNER] chitchat 判定放行: {question[:60]}")
            reject = False
        elif reject and reject_reason != "injection" and (entities.products or entities.orgs):
            # 问题命中已知产品/机构枚举 = 确定性业务问题，覆盖 LLM 的误判拒答
            logger.info(
                f"[PLANNER] 实体命中覆盖拒答 "
                f"(产品={list(entities.products)[:3]} "
                f"机构={entities.orgs[:3]})"
            )
            reject = False
        if reject:
            logger.info(
                f"[PLANNER] 异常问题拒答 (tag={tag or '兜底'}) | "
                f"规划耗时 {_ms(t0, time.perf_counter())}"
            )
            logger.info(f">>> 查询完成（拒答）")
            _rec("规划改写", t0, time.perf_counter())
            _rec("总计", t0, time.perf_counter())
            if collect_trace:
                trace["query_type"] = query_type
                trace["planner_multi"] = False
                trace["faiss_hits"] = []
                trace["bm25_hits"] = []
                trace["meta_results"] = []
                trace["rrf_merged"] = []
                trace["reranked"] = []
                trace["n_context"] = 0
                trace["context_chars"] = 0
            return (REFUSAL_ANSWER, timing, trace) if (collect_timing or collect_trace) else REFUSAL_ANSWER
        tag_content = get_tag_content(tag)
        logger.info(f"  [PLANNER] tag={tag or '兜底'} 动态提示词={'有' if tag_content else '无'}")
        planner_multi = bool(plan and plan["type"] in ("parallel_subq", "multi_hop"))
        if planner_multi:
            query_type = "cross_doc"
        elif plan and plan["type"] == "single":
            planner_q = plan["steps"][0]["query"] or ""
            if planner_q and planner_q != question:
                logger.info(f"[PLANNER] 单步改写: {planner_q[:80]}")
                expanded_query = planner_q
        t_rewrite = time.perf_counter()
        _rec("规划改写", t0, t_rewrite)

        # ── 1-3. 检索 ──
        if planner_multi:
            # 规划路径：step 轻检索（faiss+bm25+meta 的 RRF top-8）由 planner 执行；
            # SQL 路不参与（表格类单问走模板/类1）
            meta_results = meta_searcher.search(entities, 20, query_type)
            t_faiss = t_bm25 = t_meta = time.perf_counter()
            if collect_timing:
                timing["检索_faiss"] = 0.0
                timing["检索_bm25"] = 0.0
                timing["检索_meta"] = 0.0
                timing["检索_sql"] = 0.0
            planner_exec = execute_plan(plan, {
                "faiss": faiss_searcher,
                "bm25": bm25_searcher,
                "resolve": _resolve_faiss_results,
                "fusion": fusion,
                "meta_results": meta_results,
                "sql": sql_engine,
            })
            merged = planner_exec.candidates
            merged = _expand_siblings(merged)
            sql_result = None
            t_sql = t_rrf = time.perf_counter()
            _rec("RRF融合", t_rewrite, t_rrf)
            if collect_trace:
                trace["query_type"] = query_type
                trace["planner_multi"] = True
                _tr_list("faiss_hits", [])
                _tr_list("bm25_hits", [])
                _tr_list("meta_results", meta_results)
        else:
            # 类1完整并行检索（faiss+bm25+meta+sql）；单步题 multi-query 扩展
            # 变体已在规划阶段并行生成（pre_variants），此处零额外 LLM 调用
            variants = pre_variants
            if variants:
                logger.info(f"[MULTI-QUERY] 变体 {len(variants)} 条: {variants}")
            queries = [expanded_query] + variants
            # 查询侧批量 embedding：一次 API 调用嵌入全部查询（替代逐条串行往返）
            embs = get_bge_embeddings(queries) if queries else None
            with ThreadPoolExecutor(max_workers=2 + 2 * len(queries)) as executor:
                if embs:
                    faiss_futures = [
                        submit_with_trace(executor, faiss_searcher.search_vec, e, FAISS_TOP_K)
                        for e in embs
                    ]
                else:
                    # 批量嵌入失败回退逐条（search 内部自带嵌入与缓存）
                    faiss_futures = [
                        submit_with_trace(executor, faiss_searcher.search, q, FAISS_TOP_K)
                        for q in queries
                    ]
                bm25_futures = [
                    submit_with_trace(executor, bm25_searcher.search, q, BM25_TOP_K)
                    for q in queries
                ] if (bm25_searcher and bm25_searcher.ready) else []
                future_meta = submit_with_trace(
                    executor, meta_searcher.search, entities, 20, query_type
                )
                future_sql = submit_with_trace(executor, sql_engine.query, expanded_query)

                faiss_hits = _merge_faiss_multi([f.result() for f in faiss_futures])
                t_faiss = time.perf_counter()
                bm25_results = (
                    _merge_bm25_multi([f.result() for f in bm25_futures])
                    if bm25_futures else []
                )
                t_bm25 = time.perf_counter()
                meta_results = future_meta.result()
                t_meta = time.perf_counter()
                sql_result = future_sql.result()
                t_sql = time.perf_counter()
            _rec("检索_faiss", t_rewrite, t_faiss)
            _rec("检索_bm25", t_rewrite, t_bm25)
            _rec("检索_meta", t_rewrite, t_meta)
            _rec("检索_sql", t_rewrite, t_sql)

            logger.info(
                f"[并行检索] faiss={len(faiss_hits)}条 meta={len(meta_results)}条 "
                f"sql={'命中' if sql_result else '未命中'} | 并行总耗时 {_ms(t_rewrite, t_sql)}"
            )

            faiss_results = _resolve_faiss_results(faiss_hits)
            if collect_trace:
                trace["query_type"] = query_type
                trace["planner_multi"] = False
                _tr_list("faiss_hits", faiss_results)
                _tr_list("bm25_hits", bm25_results)
                _tr_list("meta_results", meta_results)
            merged = fusion.rrf_fusion(faiss_results, sql_result, meta_results, bm25_results)
            merged = _expand_siblings(merged)
            t_rrf = time.perf_counter()
            _rec("RRF融合", t_sql, t_rrf)

        # ── 5. Reranker 精排 + 源配额 ──
        # 空池短路：四路全空（闲聊/完全无关问题）跳过精排与后续管线
        if not merged:
            logger.info("[检索] 融合池为空，跳过精排与生成，直接回未找到")
            _rec("精排rerank", t_rrf, time.perf_counter())
            _rec("parent上下文", t_rrf, time.perf_counter())
            _rec("LLM生成", t_rrf, time.perf_counter())
            _rec("总计", t0, time.perf_counter())
            if collect_trace:
                trace["query_type"] = query_type
                trace["planner_multi"] = planner_multi
                trace["faiss_hits"] = []
                trace["bm25_hits"] = []
                trace["meta_results"] = []
                trace["rrf_merged"] = []
                trace["reranked"] = []
                trace["n_context"] = 0
                trace["context_chars"] = 0
            msg = "抱歉，未找到与您问题相关的信息。请尝试换个方式提问"
            return (msg, timing, trace) if (collect_timing or collect_trace) else msg
        # top_k 留补位余量：上下文组装会跳过与已选 parent 重叠的 child，需要候选补位
        # 规划路径：对原问题 + 各 step 检索文本多路打分取最大值（防第二问语义稀释）
        # A/B 开关：RERANK_SKIP_SINGLE=1 时单步问题跳过精排 API（RRF 排序直出）
        reranked = fusion.rerank(
            question, merged, top_k=FINAL_TOP_K + 4, query_type=query_type,
            sub_queries=planner_exec.queries if planner_multi else None,
            skip_api=(
                RERANK_SKIP_SINGLE and not planner_multi
                and bool(plan) and plan["type"] == "single"
                and query_type == "single_doc"
            ),
        )
        # 邻域补全：入选 chunk 的同文档相邻块（±N）若被精排裁剪，按融合序补入尾部。
        # 法规条款/补贴标准类细节题的正确 chunk 常与已选 chunk 相邻但 rerank 分数低。
        reranked = _append_neighbors(reranked, merged)
        t_rerank = time.perf_counter()
        _rec("精排rerank", t_rrf, t_rerank)
        _tr_list("rrf_merged", merged)
        _tr_list("reranked", reranked)
        logger.info(f"  精排: {len(reranked)} 条")
        for i, c in enumerate(reranked, 1):
            logger.info(
                f"    [{i}] source={c.get('source')} chunk_id={c.get('chunk_id')} "
                f"doc_id={c.get('document_id')} 文号={c.get('doc_number', '')} "
                f"rerank={c.get('rerank_score')} srcs={','.join(c.get('sources') or [])} "
                f"文件={str(c.get('file_name', ''))[:60]}"
            )

        # ── 6. 取 Parent Chunk 上下文 ──
        # 数据查询类问题 SQL 命中时收紧上下文；foreach 展开除外——逐项证据
        # （如"各公司的邮箱"）只存在于 contexts，收紧会丢失答案素材
        has_foreach = bool(plan and any(s.get("foreach") for s in plan["steps"]))
        contexts = _expand_to_parent_contexts(
            reranked,
            sql_cap=(
                SQL_CONTEXT_CAP
                if (tag == "数据查询" and not has_foreach) else None
            ),
            planner_multi=planner_multi,
        )
        if collect_trace:
            trace["n_context"] = len(contexts)
            trace["context_chars"] = sum(len(c) for c in contexts)
        t_ctx = time.perf_counter()
        _rec("parent上下文", t_rerank, t_ctx)
        for i, ctx in enumerate(contexts, 1):
            preview = ctx[:200].replace("\n", " / ")
            logger.info(f"  召回上下文[{i}] ({len(ctx)}字符): {preview}...")

        # ── 7. LLM 生成（基础提示词 + 标签动态内容 + 语言规则）──
        style = compute_answer_style(reranked, question)
        logger.info(
            f"[PROMPT] 标签={tag or '兜底'} 文档语言={style['doc_lang']} "
            f"提问语言={style['answer_lang']} 跨语言={'是' if style['cross_lang'] else '否'}"
        )
        if planner_multi:
            answer = generate_synthesis_answer(
                question, planner_exec.intermediates, contexts, style, tag_content,
                timing=timing if collect_timing else None,
            )
        else:
            answer = generate_answer(question, contexts, style, tag_content,
                                     timing=timing if collect_timing else None)
        t_llm = time.perf_counter()
        _rec("LLM生成", t_ctx, t_llm)
        _rec("总计", t0, t_llm)

        # ── 各阶段耗时汇总 ──
        logger.info(
            f"  耗时: 重写/规划 {_ms(t0, t_rewrite)} | "
            f"并行检索 faiss {_ms(t0, t_faiss)} / bm25 {_ms(t0, t_bm25)} "
            f"/ meta {_ms(t0, t_meta)} / sql {_ms(t0, t_sql)} "
            f"(总 {_ms(t_rewrite, t_sql)}) | "
            f"RRF {_ms(t_sql, t_rrf)} | 精排 {_ms(t_rrf, t_rerank)} | "
            f"Parent {_ms(t_rerank, t_ctx)} | LLM {_ms(t_ctx, t_llm)} | "
            f"总计 {_ms(t0, t_llm)}"
        )

        logger.info(f">>> 查询完成: 答案 {len(answer)} 字")
        return (answer, timing, trace) if (collect_timing or collect_trace) else answer

    except Exception as e:
        logger.error(f"Query 处理异常: {e}", exc_info=True)
        msg = "系统处理出错，请重试"
        return (msg, timing, trace) if (collect_timing or collect_trace) else msg
    finally:
        clear_trace()


@app.route("/chat", methods=["POST"])
def handle_chat():
    """竞赛标准接口：POST /chat，参数 question（JSON 或表单均可），返回 {"answer": ...}。

    JSON 请求带 "timing": true 时响应附 "timings"（各阶段耗时，毫秒）；
    带 "trace": true 时响应附 "trace"（检索链中间数据，评测用）。
    """
    timing_requested = False
    trace_requested = False
    if request.is_json:
        body = request.get_json(silent=True) or {}
        question = body.get("question")
        timing_requested = bool(body.get("timing"))
        trace_requested = bool(body.get("trace"))
    else:
        question = request.form.get("question") or request.args.get("question")
    question = (question or "").strip()
    if not question:
        return jsonify({"answer": "请输入您的问题。"})
    if timing_requested or trace_requested:
        answer, timings, trace = _run_query(
            question, collect_timing=timing_requested, collect_trace=trace_requested
        )
        resp = {"answer": answer}
        if timing_requested:
            resp["timings"] = timings
        if trace_requested:
            resp["trace"] = trace
        return jsonify(resp)
    return jsonify({"answer": _run_query(question)})


_CROSS_DOC_SIGNALS = (
    "对比", "比较", "区别", "差异", "哪些", "所有", "全部",
    "汇总", "全景", "分别", "各自", "综述", "梳理",
    "关系", "联系", "关联", "异同", "相同", "不同",
)


def _detect_query_type(question: str, entities: QueryEntities) -> str:
    """auto 模式：根据问法特征判断单文档还是跨文档检索。"""
    # 问题提及 ≥2 个产品 → 需要跨文档检索（如 "全民付蓝鲸和银商大脑的关系"）
    if len(entities.products) >= 2:
        return "cross_doc"
    if any(s in question for s in _CROSS_DOC_SIGNALS):
        return "cross_doc"
    if "2023" in question and "2024" in question:
        return "cross_doc"
    # 产品+机构同时出现且无明确类型意图 → 可能涉及多份文档
    if entities.has_entities and entities.product and entities.org and not entities.type_hint:
        return "cross_doc"
    return "single_doc"


def _resolve_faiss_results(faiss_hits: list) -> list[dict]:
    """将 FAISS 返回的 (vector_id, score) 转为含文本+元数据的结果（内存直查）。"""
    if not faiss_hits or not chunk_store:
        return []
    return chunk_store.resolve(faiss_hits)


def _merge_faiss_multi(hits_lists: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """多查询 FAISS 结果合并：按 vector_id 去重取最高 score，降序重排。"""
    best: dict[int, tuple[int, float]] = {}
    for hits in hits_lists:
        for vid, score in hits:
            if vid not in best or score > best[vid][1]:
                best[vid] = (vid, score)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


def _merge_bm25_multi(hits_lists: list[list[dict]]) -> list[dict]:
    """多查询 BM25 结果合并：按 chunk id 去重取最高 score，降序重排。"""
    best: dict = {}
    for hits in hits_lists:
        for item in hits:
            key = item.get("id") or item.get("chunk_id")
            if key is None:
                continue
            if key not in best or item.get("score", 0) > best[key].get("score", 0):
                best[key] = item
    return sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)


_VERSION_RE = re.compile(r"V(\d+)[._](\d+)(?:[._]\d+)*")  # 剥整段版本号 V4.2.7/V1_0
_VERSION_PARSE_RE = re.compile(r"V(\d+)[._](\d+)")        # 取前两位比较


def _parse_version(file_name: str) -> tuple[int, int]:
    """文件名版本号 V数字_数字 → (大版本, 小版本)；无版本号 (0, 0)。"""
    m = _VERSION_PARSE_RE.search(file_name or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _doc_group_key(file_name: str) -> str:
    """同文档不同版本归组键：去版本号、语言后缀、类型日期前缀。"""
    name = _VERSION_RE.sub("", file_name or "")
    name = re.sub(r"_(ZH|EN)\.md$", "", name)
    name = re.sub(r"^(REG|MAN|NWS|PDT|RPT|TBL|GEN)_\d{8}_", "", name)
    return name


def _dedup_by_latest_version(candidates: list[dict]) -> list[dict]:
    """元数据去重：同文档多版本同时命中召回时，仅保留版本最新文档的候选。

    版本比较：先比第一个数字，相同再比第二个数字（V4.2.7 → (4,2)）。
    输出保持输入原始顺序（多文档题各组交错，聚合重排会打乱重排序位）。
    """
    groups: dict[str, list[tuple]] = {}
    for c in candidates:
        fn = c.get("file_name") or ""
        groups.setdefault(_doc_group_key(fn), []).append((_parse_version(fn), c))

    kept_ids: set[int] = set()
    for key, items in groups.items():
        if len(items) == 1:
            kept_ids.add(id(items[0][1]))
            continue
        versions = {v for v, _ in items}
        if len(versions) == 1:
            kept_ids.update(id(c) for _, c in items)
            continue
        max_ver = max(versions)
        kept = [c for v, c in items if v == max_ver]
        kept_ids.update(id(c) for c in kept)
        logger.info(
            f"[去重] 文档 {key}: {len(items)} 条候选跨 {len(versions)} 个版本，"
            f"仅保留 V{max_ver[0]}_{max_ver[1]}（剔除 {len(items) - len(kept)} 条旧版）"
        )
    return [c for c in candidates if id(c) in kept_ids]


_CJK_RE = re.compile(r"[一-鿿]")


def detect_language(text: str) -> str:
    """CJK 字符占比 >0.5 判中文，否则英文（混合中英问题时优先英文）。"""
    if not text:
        return "zh"
    letters = sum(1 for ch in text if ch.isalnum())
    cjk = len(_CJK_RE.findall(text))
    return "zh" if letters == 0 or cjk / letters > 0.5 else "en"


def compute_answer_style(reranked: list[dict], question: str) -> dict:
    """统计召回候选元数据 → 语言适配参数。

    - 跳过 sql 候选（无 doc_type/file_name）
    - 文档语言按 file_name _ZH/_EN 取多数（并列取先出现者）
    - answer_lang = 提问语言；cross_lang = 文档语言 ≠ 提问语言
    """
    style = {
        "answer_lang": detect_language(question),
        "doc_lang": None,
        "cross_lang": False,
    }

    doc_cands = [
        c for c in reranked
        if c.get("source") != "sql" and (c.get("doc_type") or c.get("file_name"))
    ]
    if not doc_cands:
        return style

    # 文档语言：_ZH/_EN 多数
    lang_votes = {"zh": 0, "en": 0}
    first_lang = None
    for c in doc_cands:
        fn = c.get("file_name") or ""
        lang = "en" if fn.endswith("_EN.md") else "zh"
        lang_votes[lang] += 1
        if first_lang is None:
            first_lang = lang
    style["doc_lang"] = (
        "zh" if lang_votes["zh"] >= lang_votes["en"] else "en"
    ) if lang_votes["zh"] != lang_votes["en"] else first_lang
    style["cross_lang"] = style["doc_lang"] != style["answer_lang"]
    return style


def _expand_siblings(merged: list[dict], radius: int = 3,
                     max_add: int = 8) -> list[dict]:
    """检索邻域扩展：融合池中每个语料候选的同文档相邻 child 补入池尾部，
    补偿语义检索对相邻条款块（如补贴标准的分段条款）的漏召回。"""
    if not chunk_store:
        return merged
    out = list(merged)
    seen = {str(c.get("chunk_id")) for c in merged}
    added = 0
    for c in merged:
        if added >= max_add:
            break
        doc = c.get("document_id")
        ps = c.get("position_start")
        if not (isinstance(doc, int) and isinstance(ps, int)):
            continue
        for r in chunk_store.neighbors(doc, ps, radius):
            if str(r["id"]) in seen:
                continue
            seen.add(str(r["id"]))
            out.append({
                "id": r["id"],
                "chunk_id": r["id"],
                "text": r["chunk_text"],
                "source": "faiss",
                "document_id": r["document_id"],
                "file_name": r.get("file_name", ""),
                "org_name": r.get("org_name", ""),
                "doc_type": r.get("doc_type", ""),
                "product_name": r.get("product_name", ""),
                "doc_number": r.get("doc_number", ""),
                "doc_date": str(r.get("doc_date", "") or ""),
                "doc_title": r.get("doc_title", ""),
                "position_start": r.get("position_start"),
                "position_end": r.get("position_end"),
            })
            added += 1
            if added >= max_add:
                break
    if added:
        logger.info(f"[邻域扩展] 融合池补入 {added} 条相邻块")
    return out


def _append_neighbors(reranked: list[dict], merged: list[dict],
                      radius: int = 8, max_extra: int = 8) -> list[dict]:
    """邻域补全：把融合链中与已入选 chunk 同文档相邻（chunk 序号差 ≤ radius）
    但被精排裁剪的候选补入重排列表尾部，防条款类细节丢失。"""
    selected = []
    for c in reranked:
        doc = c.get("document_id")
        ps = c.get("position_start")
        if not (isinstance(doc, int) and isinstance(ps, int)):
            continue
        selected.append((doc, ps))
    if not selected:
        return reranked

    out = list(reranked)
    seen_ids = {str(c.get("chunk_id")) for c in reranked}
    extra = 0
    for c in merged:
        if extra >= max_extra:
            break
        cid = str(c.get("chunk_id"))
        if cid in seen_ids:
            continue
        doc = c.get("document_id")
        ps = c.get("position_start")
        if not (isinstance(doc, int) and isinstance(ps, int)):
            continue
        if any(doc == d and abs(ps - p) <= radius for d, p in selected):
            c = dict(c)
            c["rerank_score"] = None  # 补位候选无精排分，置尾部
            out.append(c)
            seen_ids.add(cid)
            extra += 1
    if extra:
        logger.info(f"[邻域补全] 补入 {extra} 条相邻候选")
    return out


def _expand_to_parent_contexts(reranked: list[dict], max_contexts: int = FINAL_TOP_K,
                               sql_cap: int | None = None,
                               planner_multi: bool = False) -> list[str]:
    """分层上下文组装：每源第 1 条取 parent chunk（完整上下文），其余取 child chunk（节省 token）。

    去重：child 完全落在已收录 span 内（同文档）时跳过——Top-1 的 parent 已覆盖其内容，
    跳过后由后续候选补位，最多组装 max_contexts 条不重复上下文。

    合并：同文档上下文 span 重叠或相邻（相接）时合并为一条——取并集区间的 child 单元
    文本重建（无重叠、无遗漏），沿用首条标签。跨章节（span 不相接）不合并，
    保证列表式问题各章节各自成条。

    sql_cap：数据查询类问题 SQL 命中时收紧上下文上限（见 config.SQL_CONTEXT_CAP）。
    """
    # 元数据去重：同文档多版本召回只留最新版
    reranked = _dedup_by_latest_version(reranked)

    # 数据查询类问题 SQL 命中：答案主体在结构化结果里，语料块多为噪声，收紧上限
    if sql_cap and any(c.get("source") == "sql" for c in reranked):
        max_contexts = min(max_contexts, sql_cap)

    if not chunk_store:
        return [c["text"] for c in reranked[:max_contexts]]

    TYPE_LABELS = {
        "regulation": "法规", "manual": "手册", "product": "产品", "report": "报告",
        "news": "新闻", "table": "表格", "general": "通用",
    }

    def _build_label(item: dict) -> str:
        meta_parts = []
        if item.get("doc_number"):
            meta_parts.append(f"文号:{item['doc_number']}")
        if item.get("org_name"):
            meta_parts.append(f"机构:{item['org_name']}")
        if item.get("doc_type"):
            label = TYPE_LABELS.get(item["doc_type"], item["doc_type"])
            meta_parts.append(f"类型:{label}")
        if item.get("product_name"):
            meta_parts.append(f"产品:{item['product_name']}")
        if item.get("file_name"):
            meta_parts.append(f"来源:{item['file_name']}")
        return f"[{' | '.join(meta_parts)}]" if meta_parts else ""

    def _fetch_union_text(doc_id: int, start: int, end: int) -> str:
        """取 [start, end) 区间内全部 child 单元文本按序拼接（内存直查）。"""
        return chunk_store.union_text(doc_id, start, end)

    entries: list[dict] = []  # kind: "sql" | "chunk"；chunk 含 doc_id/start/end/label
    seen = set()
    source_count: dict[str, int] = {}

    for item in reranked:
        src = item.get("source", "")

        if src == "sql":
            if item.get("text"):
                entries.append({"kind": "sql", "text": f"[结构化查询结果]\n{item['text']}"})
                if len(entries) >= max_contexts:
                    break
            continue

        chunk_id = item.get("chunk_id")
        if not chunk_id or chunk_id == -1 or chunk_id in seen:
            continue

        # parent 反查走内存预计算映射（启动时已算好，无 DB 查询）
        row = chunk_store.parent_of(chunk_id)

        if not row:
            continue

        doc_id = row["document_id"]
        c_start, c_end = row["position_start"], row["position_end"]

        # 去重：child 完全落在已收录 span 内 → 与已选 parent/child 内容重复，跳过。
        # 仅当已收录条目是完整文本（top1 parent 或合并后 union 全文）时才可据此跳过；
        # 若已收录条目只是局部 child 文本，其 span 不构成覆盖证据，不得跳过。
        if any(
            e["kind"] == "chunk" and e["doc_id"] == doc_id
            and e["start"] <= c_start and c_end <= e["end"]
            and e.get("full")
            for e in entries
        ):
            continue

        seen.add(chunk_id)
        source_count[src] = source_count.get(src, 0) + 1
        is_top1 = (source_count[src] == 1)
        label = _build_label(item)

        if is_top1 and row.get("parent_text") is not None:
            # parent 瘦身：2776 字符实测过长，截 1500 保留章节主体（尾部多为冗余操作细节）
            text = f"{label}\n{row['parent_text'][:1500]}" if label else row["parent_text"][:1500]
            span = (row["p_start"], row["p_end"])
            is_full = True
        else:
            text = item.get("text", "")
            text = f"{label}\n{text}" if label else text
            span = (c_start, c_end)
            is_full = False

        # 合并：与同文档已有条目 span 重叠或相接 → 并集区间重建正文
        merged = False
        for e in entries:
            if e["kind"] != "chunk" or e["doc_id"] != doc_id:
                continue
            if span[0] <= e["end"] and e["start"] <= span[1]:
                us, ue = min(e["start"], span[0]), max(e["end"], span[1])
                body = _fetch_union_text(doc_id, us, ue)
                if body:
                    keep_label = e["label"] if e["start"] <= span[0] else label
                    e["start"], e["end"] = us, ue
                    e["text"] = f"{keep_label}\n{body}" if keep_label else body
                    e["full"] = True  # union 重建为区间完整文本
                    logger.info(
                        f"  合并上下文: doc_id={doc_id} 并集 [{us},{ue}) "
                        f"(吸收 span [{span[0]},{span[1]})，省 1 条)"
                    )
                merged = True
                break
        if not merged:
            entries.append({
                "kind": "chunk", "doc_id": doc_id,
                "start": span[0], "end": span[1],
                "text": text, "label": label, "full": is_full,
            })

        if len(entries) >= max_contexts:
            break

    total_chars = sum(len(e["text"]) for e in entries)
    logger.info(f"[上下文] 组装 {len(entries)} 条 共 {total_chars} 字符")
    texts = [e["text"] for e in entries]
    if planner_multi:
        doc_ids = []
        for e in entries:
            if e["kind"] == "chunk" and e["doc_id"] not in doc_ids:
                doc_ids.append(e["doc_id"])
        texts = _prepend_doc_summaries(texts, doc_ids)
    return texts


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def _start_keepalive():
    """防网关 serverless 冷池再冷却（冷启动 40-60s，主办方 30s 作答硬约束下首问撞冷=必超时）。

    每 KEEPALIVE_INTERVAL 秒 ping 一次仍在走网关的端点（chat 恒网关；
    embedding/rerank 仅对应开关为 gateway 时），best-effort 失败静默。
    """
    from core.config import (
        DEEPSEEK_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
        BGE_EMBEDDING_URL, BGE_RERANKER_URL, BGE_API_KEY,
        BGE_EMBED_BACKEND, BGE_RERANK_BACKEND, KEEPALIVE_INTERVAL,
    )
    from core.http_client import post_json

    if KEEPALIVE_INTERVAL <= 0:
        return

    h = {"Authorization": f"Bearer {BGE_API_KEY}", "Content-Type": "application/json"}
    targets = [
        ("chat", DEEPSEEK_URL,
         {"model": DEEPSEEK_MODEL,
          "messages": [{"role": "user", "content": "只回复OK"}],
          "max_tokens": 4, "thinking": {"type": "disabled"}}),
    ]
    if BGE_EMBED_BACKEND == "gateway":
        targets.append(("embedding", BGE_EMBEDDING_URL,
                        {"model": "BAAI/bge-m3", "input": ["预热"]}))
    if BGE_RERANK_BACKEND == "gateway":
        targets.append(("rerank", BGE_RERANKER_URL,
                        {"model": "BAAI/bge-reranker-v2-m3",
                         "query": "预热", "documents": ["预热"]}))

    def loop():
        while True:
            time.sleep(KEEPALIVE_INTERVAL)
            for name, url, payload in targets:
                try:
                    post_json(url, payload, h, 10, retries=1)
                except Exception:
                    pass

    threading.Thread(target=loop, daemon=True, name="gateway-keepalive").start()
    logger.info(
        f"[keepalive] 已启动（每 {KEEPALIVE_INTERVAL}s ping "
        f"{'/'.join(t[0] for t in targets)}）"
    )


if __name__ == "__main__":
    init_services()
    _start_keepalive()
    # 主办方固定问答端口经 PORT 环境变量传入（.env），默认 5000
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
