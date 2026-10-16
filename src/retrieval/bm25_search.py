"""
BM25 词法检索（第 4 路）：jieba 分词 + rank_bm25 内存索引。

启动时从 DB 加载全部 embedded child chunks 构建索引（无状态），
查询时返回与 _resolve_faiss_results 同构的 dict 列表，参与 RRF 融合。
"""

import os, sys, logging, time

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import BM25_TOP_K, BM25_K1, BM25_B

logger = logging.getLogger(__name__)


class BM25Searcher:
    """内存 BM25Okapi 索引，构建失败时 ready=False 且 search 返回空列表。"""

    def __init__(self, db_conn):
        self.ready = False
        self._docs: list[dict] = []
        self._bm25 = None
        try:
            t0 = time.perf_counter()
            import jieba
            jieba.initialize()  # 词典加载从首个请求路径移到启动

            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT c.id, c.chunk_text, c.document_id, "
                    "c.position_start, c.position_end, c.doc_type, c.product_name, "
                    "d.file_name, d.org_name, d.doc_number, d.doc_date, d.doc_title "
                    "FROM chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE c.chunk_level = 'child' AND c.status = 'embedded'"
                )
                rows = cur.fetchall()

            if not rows:
                logger.warning("BM25 索引构建跳过：无 embedded child chunks")
                return

            corpus = []
            for row in rows:
                self._docs.append({
                    "id": row[0],
                    "chunk_id": row[0],
                    "text": row[1] or "",
                    "document_id": row[2],
                    "position_start": row[3],
                    "position_end": row[4],
                    "doc_type": row[5],
                    "product_name": row[6],
                    "file_name": row[7] or "",
                    "org_name": row[8] or "",
                    "doc_number": row[9] or "",
                    "doc_date": str(row[10] or "") if row[10] is not None else "",
                    "doc_title": row[11] or "",
                })
                corpus.append(jieba.lcut(row[1] or ""))

            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
            self.ready = True
            logger.info(
                f"BM25 索引构建完成: {len(corpus)} docs, "
                f"耗时 {(time.perf_counter() - t0):.1f}s"
            )
        except Exception as e:
            logger.error(f"BM25 索引构建失败，词法路降级关闭: {e}")

    def search(self, query: str, top_k: int = None,
               allowed_chunk_ids: set | None = None) -> list[dict]:
        """词法检索 Top-K，返回与 faiss 解析结果同构的 dict 列表。

        allowed_chunk_ids: 机构过滤范围（第一层元数据过滤），
        非 None 时只在该范围内检索（范围外分数置 -1）。
        """
        if not self.ready or not query:
            return []
        if top_k is None:
            top_k = BM25_TOP_K

        t0 = time.perf_counter()
        import jieba
        import numpy as np

        tokens = jieba.lcut(query)
        scores = self._bm25.get_scores(tokens)
        if allowed_chunk_ids is not None:
            allowed = np.array(
                [d["chunk_id"] in allowed_chunk_ids for d in self._docs],
                dtype=bool,
            )
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -1.0)
        order = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:  # 无词重叠的候选是噪声
                continue
            doc = self._docs[idx].copy()
            doc["score"] = score
            doc["source"] = "bm25"
            results.append(doc)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[BM25] query={query[:60]} tokens={len(tokens)} → {len(results)} 条 ({elapsed:.0f}ms)"
        )
        if results:
            preview = ", ".join(
                f"#{r['chunk_id']}:{r['score']:.1f}" for r in results[:3]
            )
            logger.debug(f"[BM25] top3: {preview}")
        return results

    @property
    def size(self) -> int:
        return len(self._docs) if self.ready else 0
