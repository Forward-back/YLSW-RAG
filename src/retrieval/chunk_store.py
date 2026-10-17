"""
内存 ChunkStore —— 启动时一次性加载全部 chunk/document 数据，替代查询期 DB 访问。

优化点（相对原版）：
  1. FAISS 命中解析：vector_id → 内存 dict 直查（原版每查询一次 MySQL IN 查询）
  2. 机构范围过滤：内存扫描（原版每查询一次 MySQL 查询）
  3. Parent 反查与合并重建：启动时预计算 child→parent 映射（原版每候选一次 LEFT JOIN 位置区间反查）
"""

import logging

logger = logging.getLogger(__name__)


class ChunkStore:
    """chunks/documents 的内存镜像（查询期只读）。"""

    def __init__(self, pool):
        self._by_chunk_id: dict = {}       # chunk_id -> chunk 行（含文档元数据）
        self._by_vector_id: dict = {}      # vector_id -> chunk_id
        self._parent_of: dict = {}         # chunk_id -> parent 行（启动时反查好）
        self._children_by_doc: dict = {}   # document_id -> [(start, end, text), ...] 按 start 排序
        self._child_pos: dict = {}         # (document_id, position_start) -> chunk_id
        self._summaries: dict = {}         # document_id -> doc_summary
        self._load(pool)

    def summary_of(self, doc_id: int) -> str | None:
        """文档要点摘要（综述类题上下文前置用），无摘要返回 None。"""
        return self._summaries.get(doc_id)

    # ── 加载 ──────────────────────────────────────────
    def _load(self, pool):
        import pymysql
        conn = pool.connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT c.id, c.chunk_text, c.vector_id, c.document_id, "
                    "c.position_start, c.position_end, c.doc_type, c.product_name, "
                    "d.file_name, d.org_name, d.doc_number, d.doc_date, d.doc_title "
                    "FROM chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE c.chunk_level='child'"
                )
                children = cur.fetchall()

                cur.execute(
                    "SELECT id, document_id, position_start, position_end, chunk_text "
                    "FROM chunks WHERE chunk_level='parent'"
                )
                parents = cur.fetchall()

                cur.execute(
                    "SELECT id, doc_summary FROM documents WHERE doc_summary IS NOT NULL"
                )
                self._summaries = {r["id"]: r["doc_summary"] for r in cur.fetchall()}
        finally:
            conn.close()

        # child 索引
        for row in children:
            self._by_chunk_id[row["id"]] = row
            if row["vector_id"] is not None:
                self._by_vector_id[row["vector_id"]] = row["id"]
            self._children_by_doc.setdefault(row["document_id"], []).append(
                (row["position_start"], row["position_end"], row["chunk_text"])
            )
            self._child_pos[(row["document_id"], row["position_start"])] = row["id"]

        for lst in self._children_by_doc.values():
            lst.sort(key=lambda t: t[0])

        # 启动时反查 child → parent（覆盖该 child 的最小 parent，语义与原 SQL 一致：
        # ORDER BY (p.id IS NULL) ASC, p.position_start DESC LIMIT 1 → 取覆盖该 child 的
        # 多个 parent 中 position_start 最大的那个）
        parents_by_doc: dict = {}
        for p in parents:
            parents_by_doc.setdefault(p["document_id"], []).append(p)
        for doc_id, plist in parents_by_doc.items():
            plist.sort(key=lambda p: p["position_start"])

        for child in children:
            doc_id = child["document_id"]
            cs, ce = child["position_start"], child["position_end"]
            best = None
            for p in parents_by_doc.get(doc_id, []):
                if p["position_start"] <= cs and p["position_end"] >= ce:
                    # 覆盖的 parent 中 position_start 最大者
                    if best is None or p["position_start"] > best["position_start"]:
                        best = p
            if best is not None:
                self._parent_of[child["id"]] = best

        logger.info(
            f"ChunkStore 加载完成: {len(children)} child / {len(parents)} parent / "
            f"{len(self._parent_of)} 条 parent 映射"
        )

    # ── 查询 ──────────────────────────────────────────
    def resolve(self, faiss_hits: list) -> list[dict]:
        """FAISS (vector_id, score) → 含文本+元数据的结果（与原 _resolve_faiss_results 同构）。"""
        results = []
        for vec_id, score in faiss_hits:
            chunk_id = self._by_vector_id.get(vec_id)
            if chunk_id is None:
                continue
            r = self._by_chunk_id[chunk_id]
            results.append({
                "id": r["id"],
                "chunk_id": r["id"],
                "text": r["chunk_text"],
                "score": score,
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
        if len(results) != len(faiss_hits):
            logger.warning(
                f"[FAISS解析] {len(faiss_hits) - len(results)}/{len(faiss_hits)} 个 "
                f"vector_id 在内存索引无对应 chunk"
            )
        return results

    def org_scope(self, orgs: list) -> dict | None:
        """机构集合 → 检索范围集合（内存扫描，语义与原 build_org_scope 一致）。

        返回 {"chunk_ids": set, "vector_ids": set}；范围为空返回 None。
        """
        if not orgs:
            return None
        names = {o.value if hasattr(o, "value") else o for o in orgs}
        chunk_ids = set()
        vector_ids = set()
        for chunk_id, r in self._by_chunk_id.items():
            if r.get("org_name") in names:
                chunk_ids.add(chunk_id)
                if r["vector_id"] is not None:
                    vector_ids.add(r["vector_id"])
        if not chunk_ids:
            return None
        return {"chunk_ids": chunk_ids, "vector_ids": vector_ids}

    def parent_of(self, chunk_id) -> dict | None:
        """child → {document_id, position_start, position_end, parent_text, p_start, p_end}。

        返回键名与原 SQL 查询结果一致；无 parent 时 parent_text/p_start/p_end 为 None。
        """
        child = self._by_chunk_id.get(chunk_id)
        if child is None:
            return None
        parent = self._parent_of.get(chunk_id)
        return {
            "document_id": child["document_id"],
            "position_start": child["position_start"],
            "position_end": child["position_end"],
            "parent_text": parent["chunk_text"] if parent else None,
            "p_start": parent["position_start"] if parent else None,
            "p_end": parent["position_end"] if parent else None,
        }

    def union_text(self, doc_id: int, start: int, end: int) -> str:
        """取 [start, end) 区间内全部 child 文本按序拼接（与原 _fetch_union_text 一致）。"""
        parts = [
            text for s, e, text in self._children_by_doc.get(doc_id, [])
            if s >= start and e <= end
        ]
        return "\n\n".join(parts)

    def neighbors(self, doc_id: int, start: int, radius: int) -> list[dict]:
        """同文档 position 差 ≤ radius 的 child chunk 行（按 position 排序）。"""
        out = []
        for pos in range(start - radius, start + radius + 1):
            cid = self._child_pos.get((doc_id, pos))
            if cid:
                out.append(self._by_chunk_id[cid])
        return out
