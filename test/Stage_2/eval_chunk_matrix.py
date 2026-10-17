# -*- coding: utf-8 -*-
"""Stage_2 视角4：Chunk Size × Overlap 矩阵评测。

4 组参数（基线 300/128 用现有 FAISS 结果 + 重建 200/64、500/128、500/256）：
对 118 个 md 重新切块 → BGE embedding → 内存 IndexFlatIP →
47 题纯向量路 top-50 → 文档级 Recall@5/@10。
输出 chunk_matrix.json。
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import faiss
import numpy as np

from ingest import chunker
from config import DATA_DIR, EMBEDDING_DIM
from retrieval.faiss_search import get_bge_embedding

MATRIX_PARAMS = [(500, 128), (300, 256), (500, 256)]
BASELINE = (300, 128)
TOPK = 50
EMBED_CONCURRENCY = 8


def load_eval_questions():
    qs = json.loads((BASE.parent / "produce_question" / "qa_testset_120.json").read_text(encoding="utf-8"))
    stage1_ids = [r["id"] for r in
                  json.loads((BASE.parent / "Stage_1" / "serial_latency.json").read_text(encoding="utf-8"))]
    qmap = {q["id"]: q for q in qs}
    out = [qmap[i] for i in stage1_ids
           if qmap[i]["类型"] not in ("提示词注入", "SQL注入", "纯闲聊")]
    return [q for q in out if q["id"].endswith(".md")]


def build_chunks(size, overlap):
    """按参数切 118 md，返回 [(file_name, chunk_text), ...]（child 级）。"""
    chunker.CHILD_CHUNK_SIZE = size
    chunker.CHUNK_OVERLAP = overlap
    out = []
    for p in sorted(Path(DATA_DIR).glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunker.chunk_file(text, p.name)
        for c in chunks:
            if c.level == "child":
                out.append((p.name, c.text))
    return out


def embed_all(chunks):
    def work(item):
        i, (fn, text) = item
        return i, fn, get_bge_embedding(text)
    results = {}
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        for i, fn, emb in ex.map(work, enumerate(chunks)):
            if emb is not None:
                results[i] = (fn, emb)
    return results


def build_index(embedded):
    ids = sorted(embedded)
    vecs = np.array([embedded[i][1] for i in ids], dtype=np.float32)
    fns = [embedded[i][0] for i in ids]
    idx = faiss.IndexFlatIP(EMBEDDING_DIM)
    idx.add(vecs)
    return idx, fns


def recall_for(idx, fns, questions, k):
    hits = 0
    total = 0
    for q in questions:
        docs = [r["父文档id"] for r in (q.get("引用文档id") or [])]
        if not docs:
            continue
        emb = get_bge_embedding(q["问题"])
        if emb is None:
            continue
        qv = np.array([emb], dtype=np.float32)
        _, inds = idx.search(qv, TOPK)
        top_fns = {fns[i] for i in inds[0][:k] if 0 <= i < len(fns)}
        total += 1
        if all(d in top_fns for d in docs):
            hits += 1
    return hits / total if total else 0.0


def baseline_recall(questions, k):
    """基线 300/128：用现有 FAISS 索引（根目录 faiss_index.bin），
    vector_id → chunk → document 映射查 MySQL(3307)。"""
    sys.path.insert(0, str(ROOT))
    import pymysql
    from config import MYSQL_CONFIG
    conn = pymysql.connect(host="127.0.0.1", port=3307, user="root",
                           password="ABC123", database="fuck_rag")
    cur = conn.cursor()
    cur.execute("""SELECT c.vector_id, d.file_name FROM chunks c
                   JOIN documents d ON d.id = c.document_id
                   WHERE c.chunk_level='child'""")
    vec_fn = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    idx = faiss.read_index(str(ROOT / "faiss_index.bin"))
    hits = total = 0
    for q in questions:
        docs = [r["父文档id"] for r in (q.get("引用文档id") or [])]
        if not docs:
            continue
        emb = get_bge_embedding(q["问题"])
        if emb is None:
            continue
        qv = np.array([emb], dtype=np.float32)
        _, inds = idx.search(qv, TOPK)
        top_fns = {vec_fn.get(int(i)) for i in inds[0][:k] if int(i) >= 0}
        total += 1
        if all(d in top_fns for d in docs):
            hits += 1
    return hits / total if total else 0.0


def main():
    questions = load_eval_questions()
    print(f"有效题 {len(questions)}（仅 md 主文档）")
    result = {"matrix": [], "n_queries": len(questions)}

    # 基线：现有索引
    t0 = time.time()
    r5 = baseline_recall(questions, 5)
    r10 = baseline_recall(questions, 10)
    result["matrix"].append({"size": BASELINE[0], "overlap": BASELINE[1],
                             "recall_at_5": round(r5, 4), "recall_at_10": round(r10, 4),
                             "baseline": True})
    print(f"基线 {BASELINE}: R@5={r5:.4f} R@10={r10:.4f} ({time.time()-t0:.0f}s)")

    for size, overlap in MATRIX_PARAMS:
        t0 = time.time()
        chunks = build_chunks(size, overlap)
        print(f"参数 {size}/{overlap}: {len(chunks)} child chunks，开始 embedding...")
        embedded = embed_all(chunks)
        print(f"  embedding 完成 {len(embedded)} 条 ({time.time()-t0:.0f}s)，建索引...")
        idx, fns = build_index(embedded)
        r5 = recall_for(idx, fns, questions, 5)
        r10 = recall_for(idx, fns, questions, 10)
        result["matrix"].append({"size": size, "overlap": overlap,
                                 "recall_at_5": round(r5, 4),
                                 "recall_at_10": round(r10, 4),
                                 "n_chunks": len(embedded)})
        print(f"  {size}/{overlap}: R@5={r5:.4f} R@10={r10:.4f} "
              f"chunks={len(embedded)} ({time.time()-t0:.0f}s)")

    (BASE / "chunk_matrix.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成: chunk_matrix.json")


if __name__ == "__main__":
    main()
