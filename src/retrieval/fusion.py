"""
RRF 融合 + BGE-Reranker 重排序。
"""

import os, sys, time, logging, re
from concurrent.futures import ThreadPoolExecutor
import requests

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from core.config import (
    RRF_K, FINAL_TOP_K, BGE_RERANKER_URL, BGE_API_KEY,
    API_TIMEOUT, API_MAX_RETRIES, RERANK_META_MIN_SCORE,
    RERANK_MAX_CANDIDATES_PER_SOURCE, CHUNK_CONTEXT_NEIGHBORS,
    BGE_RERANK_BACKEND, RERANK_TIMEOUT,
)
from core.http_client import post_json

logger = logging.getLogger(__name__)


_H2_RE = re.compile(r"##\s*([^\n>]+)")
_H3_RE = re.compile(r"###\s*([^\n>]+)")

# 无信息量的工具章节：不进第一轮占位（仅声明/来源类，内容章节不受影响）
_JUNK_H2S = {"声明", "免责声明", "版权声明", "来源", "关于我们", "联系我们", "附录"}

# 未覆盖章节补位的最低 rerank 分数：低于此值视为弱相关，不给章节覆盖名额
COVERAGE_MIN_SCORE = 0.5


def _sec_key(cand: dict):
    """候选所属章节（(document_id, H2标题, H3标题)），缺失层级用 None 区分。

    H2 相同但 H3 众多的长文档按 H3 细分（如法规"全文"下的"一、二、三"各部分），
    否则概述类问题会被单章高分碎片占满、整章缺料。
    """
    head = (cand.get("text") or "")[:400]
    m2 = _H2_RE.search(head)
    m3 = _H3_RE.search(head)
    return (
        cand.get("document_id"),
        m2.group(1).strip() if m2 else None,
        m3.group(1).strip() if m3 else None,
    )


def _neighbor_dup(cand: dict, selected: list[dict]) -> bool:
    """候选与已选 chunk 是否同文档且位置相邻（±CHUNK_CONTEXT_NEIGHBORS）。"""
    ps = cand.get("position_start")
    pe = cand.get("position_end")
    if not (isinstance(ps, int) and isinstance(pe, int) and pe > ps):
        return False
    return any(
        cand.get("document_id") == s.get("document_id")
        and abs(ps - s.get("position_start")) <= CHUNK_CONTEXT_NEIGHBORS
        for s in selected
        if isinstance(s.get("position_start"), int)
        and isinstance(s.get("position_end"), int)
        and s.get("position_end") > s.get("position_start")
    )


def _select_diverse(ranked: list[dict], n: int) -> list[dict]:
    """三轮贪心选择：章节覆盖 + 聚焦补充。

    第一轮：每个 (文档,H2,H3) 章节只取 1 条，且分数不低于最高分的 85%（不低于 0.5）——
    "2024年经济热词"这类各章节分数接近的问题可覆盖全部章节；
    无 H2 的头部引言排最后（名额紧张时优先让位给内容章节），声明/来源类工具章节不参与。
    第二轮：未覆盖章节按分数各补 1 条（分数门槛放宽到 COVERAGE_MIN_SCORE），
    保证概述类问题不会整章缺料（同一 H2 长文档按 H3 细分章节）。
    第三轮：剩余名额按分数补位，跳过已选 chunk 同文档邻域内的候选——
    Parent 已覆盖其邻居，选取邻域 child 属纯冗余。
    """
    if not ranked:
        return []
    scores = [c.get("rerank_score") for c in ranked]
    gate = max(0.5, 0.85 * max(s for s in scores if s is not None)) if any(
        s is not None for s in scores
    ) else None

    selected = []
    seen_sec = set()

    if gate is not None:
        # 内容章节优先
        for cand in ranked:
            s = cand.get("rerank_score") or 0
            if s < gate:
                continue
            doc_id, h2, h3 = _sec_key(cand)
            if h2 is None or h2 in _JUNK_H2S:
                continue
            key = (doc_id, h2, h3)
            if key in seen_sec:
                continue
            seen_sec.add(key)
            selected.append(cand)
            if len(selected) >= n:
                return selected
        # 无 H2 的头部引言后置
        for cand in ranked:
            s = cand.get("rerank_score") or 0
            if s < gate:
                continue
            key = _sec_key(cand)
            if key[1] is not None or key in seen_sec:
                continue
            seen_sec.add(key)
            selected.append(cand)
            if len(selected) >= n:
                return selected

    # 第二轮：未覆盖章节补位（每节最高分 1 条，弱相关不给名额）
    for cand in ranked:
        if cand in selected:
            continue
        s = cand.get("rerank_score") or 0
        if s < COVERAGE_MIN_SCORE:
            continue
        doc_id, h2, h3 = _sec_key(cand)
        if h2 in _JUNK_H2S:
            continue
        key = (doc_id, h2, h3)
        if key in seen_sec or _neighbor_dup(cand, selected):
            continue
        seen_sec.add(key)
        selected.append(cand)
        if len(selected) >= n:
            break

    # 第三轮：剩余名额按分数补位（邻域去重）
    for cand in ranked:
        if cand in selected:
            continue
        if _neighbor_dup(cand, selected):
            continue
        selected.append(cand)
        if len(selected) >= n:
            break
    return selected


class FusionEngine:
    """多路检索融合 + 重排序。"""

    def __init__(self, k: int = None):
        self.rrf_k = k or RRF_K
        self.final_top_k = FINAL_TOP_K

    @staticmethod
    def _make_doc(item, source, chunk_id):
        return {
            "source": source,
            "sources": [source],
            "chunk_id": chunk_id,
            "text": item.get("text", ""),
            "score": item.get("score", 0),
            "document_id": item.get("document_id"),
            "file_name": item.get("file_name", ""),
            "org_name": item.get("org_name", ""),
            "doc_type": item.get("doc_type", ""),
            "product_name": item.get("product_name", ""),
            "doc_number": item.get("doc_number", ""),
            "doc_date": item.get("doc_date", ""),
            "doc_title": item.get("doc_title", ""),
            "position_start": item.get("position_start"),
            "position_end": item.get("position_end"),
        }

    @staticmethod
    def _bucket(cand: dict) -> str:
        """候选归属桶：metadata / corpus（faiss+bm25 语料路）/ sql。

        兼容旧格式候选（无 sources 字段时回退 source）。
        """
        srcs = cand.get("sources") or [cand.get("source", "")]
        if "metadata" in srcs:
            return "metadata"
        if "faiss" in srcs or "bm25" in srcs:
            return "corpus"
        return "sql"

    def rrf_fusion(
        self,
        faiss_results: list[dict],
        sql_results: dict | None,
        meta_results: list[dict] | None = None,
        bm25_results: list[dict] | None = None,
    ) -> list[dict]:
        """RRF 融合四路检索结果（Metadata + FAISS + BM25 + SQL）。

        RRF_score(d) = Σ 1/(k + rank_i(d))

        meta/faiss/bm25 统一按 chunk_id 做 key：同一 chunk 多路命中时
        分数相加（多路共识排名更高），且只占一个候选位。
        元数据精确匹配（Tier 1）使用固定小分母获得中等 boost。
        sql 保持独立 key。
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        logger.info(
            f"[RRF] 输入 meta={len(meta_results or [])} faiss={len(faiss_results)} "
            f"bm25={len(bm25_results or [])} sql={'yes' if sql_results else 'no'}"
        )

        def _add(item, source, boost):
            key = str(item["id"])
            rrf_scores[key] = rrf_scores.get(key, 0) + boost
            if key in doc_map:
                doc = doc_map[key]
                if source not in doc["sources"]:
                    doc["sources"].append(source)
            else:
                doc_map[key] = self._make_doc(item, source, item["id"])

        if meta_results:
            for item in meta_results:
                _add(item, "metadata", 1.0 / (self.rrf_k + 2))

        for rank, item in enumerate(faiss_results):
            _add(item, "faiss", 1.0 / (self.rrf_k + rank + 1))

        if bm25_results:
            for rank, item in enumerate(bm25_results):
                _add(item, "bm25", 1.0 / (self.rrf_k + rank + 1))

        if sql_results:
            sql_key = "sql_result"
            rrf_scores[sql_key] = 1.0 / (self.rrf_k + 1)
            doc_map[sql_key] = {
                "source": "sql",
                "sources": ["sql"],
                "chunk_id": -1,
                "text": sql_results.get("answer_hint", ""),
                "sql": sql_results.get("sql", ""),
                "rows": sql_results.get("result", []),
            }

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, rrf_score in ranked:
            doc = doc_map[key].copy()
            doc["rrf_score"] = rrf_score
            # 归一化 source：metadata 优先，其次语料路（faiss/bm25），最后 sql。
            # 归一化后 _expand_to_parent_contexts 的 source_count 逻辑无需改动。
            doc["source"] = self._bucket(doc)
            results.append(doc)

        bucket_counts: dict[str, int] = {}
        for c in results:
            b = self._bucket(c)
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        logger.info(
            f"[RRF] 输出 {len(results)} 候选 "
            f"(metadata={bucket_counts.get('metadata', 0)}, "
            f"corpus={bucket_counts.get('corpus', 0)}, "
            f"sql={bucket_counts.get('sql', 0)})"
        )
        if results:
            preview = ", ".join(
                f"#{c['chunk_id']}({','.join(c.get('sources') or [])})"
                f"={c['rrf_score']:.4f}"
                for c in results[:5]
            )
            logger.debug(f"[RRF] top5: {preview}")

        return results

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = None,
        query_type: str = "single_doc", sub_queries: list[str] = None,
        skip_api: bool = False,
    ) -> list[dict]:
        """重排序 + 源配额。

        分桶：sql / metadata / corpus（faiss+bm25 合并语料路，互斥）。
        single_doc: meta≤3, corpus 取剩余名额
        cross_doc: meta≤5, corpus 取剩余名额

        sub_queries: 规划器各 step 的实际检索文本——对每条分别 rerank，
        候选取各查询得分的最大值（原问题对第二问语义有稀释，
        "面向什么场景+该场景现状"里现状类文档只对 step2 的查询得分高）。

        skip_api: True 时跳过 rerank API，直接走 RRF 排序 + 配额降级分支
        （A/B 实验用，单步问题省 ~1.3s API 耗时）。
        """
        if top_k is None:
            top_k = self.final_top_k

        if not candidates:
            return []

        sql_hits = [c for c in candidates if self._bucket(c) == "sql"]
        meta_hits = [c for c in candidates if self._bucket(c) == "metadata"]
        corpus_hits = [c for c in candidates if self._bucket(c) == "corpus"]

        rerank_results = None
        if not skip_api:
            # 每源截断前 N 条再送 Reranker：RRF 后低排名候选对最终配额无贡献，
            # 裁剪可减少 rerank API 输入（40+ 条 → ≤16 条），降低该阶段耗时
            rerank_candidates = (
                meta_hits[:RERANK_MAX_CANDIDATES_PER_SOURCE]
                + corpus_hits[:RERANK_MAX_CANDIDATES_PER_SOURCE]
            )

            if rerank_candidates:
                documents = [c["text"][:5000] for c in rerank_candidates]
                if sub_queries:
                    scores: dict[int, float] = {}
                    qs = [query] + list(sub_queries)
                    # 多路打分并行：同一批文档 × 不同查询串，调用互相独立；
                    # 多跳依赖只影响检索顺序，精排阶段候选已固定，并行安全。
                    # 本地 reranker 后端保守串行（CPU 模型并发推理无保障）
                    if BGE_RERANK_BACKEND == "local" or len(qs) == 1:
                        rerank_batches = [self._call_reranker(q, documents) for q in qs]
                    else:
                        with ThreadPoolExecutor(max_workers=len(qs)) as _rex:
                            rerank_batches = list(
                                _rex.map(lambda q: self._call_reranker(q, documents), qs)
                            )
                    for res in rerank_batches:
                        if not res:
                            continue
                        for item in res:
                            idx = item["index"]
                            if idx >= len(documents):
                                continue
                            if idx not in scores or item["relevance_score"] > scores[idx]:
                                scores[idx] = item["relevance_score"]
                    rerank_results = [
                        {"index": i, "relevance_score": s} for i, s in scores.items()
                    ]
                else:
                    rerank_results = self._call_reranker(query, documents)
                logger.info(
                    f"[RERANK] 送入 {len(rerank_candidates)} 条 "
                    f"(meta={len(meta_hits[:RERANK_MAX_CANDIDATES_PER_SOURCE])} "
                    f"corpus={len(corpus_hits[:RERANK_MAX_CANDIDATES_PER_SOURCE])}) "
                    f"{f'{len(sub_queries) + 1} 路打分' if sub_queries else ''} "
                    f"→ API 返回 {len(rerank_results) if rerank_results else 0} 分"
                )

                if rerank_results:
                    for item in rerank_results:
                        idx = item["index"]
                        if idx < len(rerank_candidates):
                            rerank_candidates[idx]["rerank_score"] = item["relevance_score"]

        if rerank_results:
            reranked_meta = [c for c in rerank_candidates if self._bucket(c) == "metadata"]
            reranked_corpus = [c for c in rerank_candidates if self._bucket(c) == "corpus"]
            reranked_meta.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            reranked_corpus.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            # 低分 metadata 不强制占配额：rerank 分数低于阈值的精确匹配视为弱相关，直接丢弃
            n_meta_before = len(reranked_meta)
            reranked_meta = [
                c for c in reranked_meta
                if c.get("rerank_score", 0) >= RERANK_META_MIN_SCORE
            ]
            if n_meta_before > len(reranked_meta):
                logger.info(
                    f"  meta 过滤: 丢弃 {n_meta_before - len(reranked_meta)}/{n_meta_before} 条 "
                    f"低分精确匹配 (阈值 {RERANK_META_MIN_SCORE})"
                )

            meta_quota = 5 if query_type == "cross_doc" else 3
            n_meta = min(len(reranked_meta), meta_quota)
            n_corpus = min(len(reranked_corpus), max(1, top_k - n_meta))
            diverse_corpus = _select_diverse(reranked_corpus, n_corpus)

            n_sql = len(sql_hits)
            final = sql_hits
            final.extend(reranked_meta[:n_meta])
            final.extend(diverse_corpus)
            logger.info(
                f"[RERANK] 配额 meta={n_meta}/{len(reranked_meta)} "
                f"corpus={len(diverse_corpus)}/{len(reranked_corpus)} "
                f"sql={n_sql} → 输出 {min(len(final), top_k)} 条"
            )
            return final[:top_k]
        else:
            logger.info(
                ("[RERANK跳过] 单步问题直走 RRF 排序" if skip_api
                 else "[RERANK降级] Reranker 失败，降级为 RRF 排序")
            )
            meta_quota = 5 if query_type == "cross_doc" else 3
            n_meta = min(len(meta_hits), meta_quota)
            n_corpus = min(len(corpus_hits), max(1, top_k - n_meta))
            diverse_corpus = _select_diverse(corpus_hits, n_corpus)

            n_sql = len(sql_hits)
            final = sql_hits
            final.extend(meta_hits[:n_meta])
            final.extend(diverse_corpus)
            logger.info(
                f"[RRF排序] meta={n_meta}/{len(meta_hits)} "
                f"corpus={len(diverse_corpus)}/{len(corpus_hits)} "
                f"sql={n_sql} → 输出 {min(len(final), top_k)} 条"
            )
            return final[:top_k]

    def _call_reranker(self, query: str, documents: list[str]) -> list[dict] | None:
        """BGE-Reranker 打分（网关 API 或本地模型，BGE_RERANK_BACKEND 控制）。"""
        if BGE_RERANK_BACKEND == "local":
            from core.models_singleton import rerank_scores
            scores = rerank_scores(query, documents)
            if scores is None:
                logger.warning("本地 Reranker 打分失败")
                return None
            return [
                {"index": i, "relevance_score": s}
                for i, s in enumerate(scores)
            ]
        resp, err = post_json(
            BGE_RERANKER_URL,
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
            },
            {
                "Authorization": f"Bearer {BGE_API_KEY}",
                "Content-Type": "application/json",
            },
            RERANK_TIMEOUT,
            retries=1,
        )
        if resp is None:
            logger.warning(f"Reranker 请求失败: {err}")
            return None
        if resp.status_code != 200:
            logger.warning(f"Reranker HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("results", [])
