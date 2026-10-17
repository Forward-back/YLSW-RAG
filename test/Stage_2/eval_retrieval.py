# -*- coding: utf-8 -*-
"""Stage_2 检索质量评测：47 题 trace 采集 + 视角 1/2/3/5 指标。

视角1 分层召回率：Recall@K（K=1..20），rrf 融合后与 rerank 后两条曲线
视角2 排序质量：每题正确文档在 rerank 后的排名分布
视角3 区分度：rerank 分数相关/不相关分布 → PR 曲线
视角5 Reranker 贡献：正确文档 rrf 排名 → rerank 排名迁移
判定：文档级，多文档题要求全部引用文档命中
"""
import json
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:6151/chat"
TESTSET = BASE.parent / "produce_question" / "qa_testset_120.json"
SERIAL = BASE.parent / "Stage_1" / "serial_latency.json"
MAX_K = 20

SKIP_TYPES = ("提示词注入", "SQL注入", "纯闲聊")


def pick_questions():
    qs = json.loads(TESTSET.read_text(encoding="utf-8"))
    qmap = {q["id"]: q for q in qs}
    stage1_ids = [r["id"] for r in json.loads(SERIAL.read_text(encoding="utf-8"))]
    return [qmap[i] for i in stage1_ids if qmap[i]["类型"] not in SKIP_TYPES]


def ground_truth_docs(q):
    refs = q.get("引用文档id") or []
    return [r["父文档id"] for r in refs]


def doc_rank(cands, file_name):
    """文档在候选列表中的最小排名（1 起），未出现返回 None。"""
    for i, c in enumerate(cands, 1):
        if c.get("file_name") == file_name:
            return i
    return None


def hit_at_k(cands, docs, k):
    """Top-K 内多文档全命中。"""
    return all(doc_rank(cands[:k], d) is not None for d in docs)


def fetch_trace(q, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(URL, json={"question": q["问题"], "trace": True}, timeout=120)
            if r.status_code == 200:
                return r.json().get("trace")
        except Exception:
            pass
        time.sleep(3)
    return None


def main():
    qs = pick_questions()
    print(f"评测 {len(qs)} 题")
    traces = []
    for i, q in enumerate(qs, 1):
        tr = fetch_trace(q)
        if tr is None:
            print(f"[{i}/{len(qs)}] FAIL {q['id'][:40]}")
            continue
        traces.append((q, tr))
        print(f"[{i}/{len(qs)}] OK {q['类型']} reranked={len(tr.get('reranked', []))}")
    out_path = BASE / "traces_47.json"
    out_path.write_text(json.dumps(
        [{"id": q["id"], "问题": q["问题"], "类型": q["类型"], "trace": tr}
         for q, tr in traces], ensure_ascii=False), encoding="utf-8")
    print(f"traces 落盘 {out_path}")

    # ── 视角1 Recall@K ──
    recall_rrf, recall_rerank = [], []
    for q, tr in traces:
        docs = ground_truth_docs(q)
        if not docs:
            continue
        row_rrf, row_rer = {}, {}
        for k in range(1, MAX_K + 1):
            row_rrf[str(k)] = int(hit_at_k(tr.get("rrf_merged", []), docs, k))
            row_rer[str(k)] = int(hit_at_k(tr.get("reranked", []), docs, k))
        recall_rrf.append(row_rrf)
        recall_rerank.append(row_rer)
    n = len(recall_rrf)
    recall_at_k = {
        "n_queries": n,
        "rrf": {str(k): round(sum(r[str(k)] for r in recall_rrf) / n, 4)
                for k in range(1, MAX_K + 1)},
        "rerank": {str(k): round(sum(r[str(k)] for r in recall_rerank) / n, 4)
                   for k in range(1, MAX_K + 1)},
    }
    (BASE / "recall_at_k.json").write_text(
        json.dumps(recall_at_k, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"视角1 Recall@K 完成，n={n}，K=10 rrf={recall_at_k['rrf']['10']} "
          f"rerank={recall_at_k['rerank']['10']}")

    # ── 视角2 排序分布 ──
    rank_dist = []
    for q, tr in traces:
        docs = ground_truth_docs(q)
        for d in docs:
            r = doc_rank(tr.get("reranked", []), d)
            rank_dist.append({"id": q["id"], "doc": d, "rank": r})
    (BASE / "rank_distribution.json").write_text(
        json.dumps(rank_dist, ensure_ascii=False, indent=2), encoding="utf-8")
    hit = [x for x in rank_dist if x["rank"] is not None]
    print(f"视角2 排序分布完成：正确文档 {len(rank_dist)} 个，命中 {len(hit)}，"
          f"中位排名 {sorted(x['rank'] for x in hit)[len(hit)//2] if hit else '-'}")

    # ── 视角3 PR 曲线（rerank 分数阈值）──
    thr_points = []
    thresholds = [i / 20 for i in range(1, 20)]
    pos, neg = [], []
    for q, tr in traces:
        docs = set(ground_truth_docs(q))
        for c in tr.get("reranked", []):
            s = c.get("rerank_score")
            if s is None:
                continue
            (pos if c.get("file_name") in docs else neg).append(s)
    for t in thresholds:
        tp = sum(1 for s in pos if s >= t)
        fp = sum(1 for s in neg if s >= t)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / len(pos) if pos else 0.0
        thr_points.append({"threshold": t, "precision": round(precision, 4),
                           "recall": round(recall, 4)})
    # AUC-PR（梯形法）
    auc = 0.0
    pts = sorted(thr_points, key=lambda p: p["recall"])
    for i in range(1, len(pts)):
        auc += (pts[i]["recall"] - pts[i - 1]["recall"]) * \
               (pts[i]["precision"] + pts[i - 1]["precision"]) / 2
    pr_result = {"points": thr_points, "auc_pr": round(auc, 4),
                 "n_pos": len(pos), "n_neg": len(neg)}
    (BASE / "pr_curve.json").write_text(
        json.dumps(pr_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"视角3 PR 完成：AUC-PR={pr_result['auc_pr']} pos={len(pos)} neg={len(neg)}")

    # ── 视角5 Rank Migration ──
    migration = []
    for q, tr in traces:
        docs = ground_truth_docs(q)
        for d in docs:
            r_before = doc_rank(tr.get("rrf_merged", []), d)
            r_after = doc_rank(tr.get("reranked", []), d)
            migration.append({"id": q["id"], "doc": d,
                              "rank_before": r_before, "rank_after": r_after})
    (BASE / "rank_migration.json").write_text(
        json.dumps(migration, ensure_ascii=False, indent=2), encoding="utf-8")
    both = [m for m in migration if m["rank_before"] is not None and m["rank_after"] is not None]
    improved = sum(1 for m in both if m["rank_after"] < m["rank_before"])
    worsened = sum(1 for m in both if m["rank_after"] > m["rank_before"])
    print(f"视角5 迁移完成：可比较 {len(both)} 对，提升 {improved}，下降 {worsened}，"
          f"不变 {len(both) - improved - worsened}")
    print("全部完成")


if __name__ == "__main__":
    main()
