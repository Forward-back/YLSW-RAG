# -*- coding: utf-8 -*-
"""全量 120 题端到端评测：串行逐题 POST /chat（timing+trace），落 full_eval_result.json。

每记录：
  id / 类型 / 问题 / answer（供后续 LLM-as-judge 复用）
  timings（阶段耗时）/ 总计（客户端端到端）
  trace：reranked 候选 chunk_id 链、n_context、context_chars
  判定：is_refusal（拒答）、hit@K（正确答案 chunk 是否进重排/融合列表 K=1/3/5/10）
"""
import json
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
TEST = BASE.parent
URL = "http://127.0.0.1:6151/chat"
REFUSAL = "抱歉，我暂时无法回答"

STAGES = [
    "规划改写", "检索_faiss", "检索_bm25", "检索_meta", "检索_sql",
    "RRF融合", "精排rerank", "parent上下文", "LLM生成", "llm首token", "总计",
]


def correct_chunk_ids(q):
    ids = set()
    for ref in q.get("引用文档id") or []:
        ids.update(ref.get("子块id") or [])
    return ids


def hit_at_k(chunks, correct, k):
    return any(c in correct for c in chunks[:k])


def main():
    qs = json.loads((TEST / "qa_testset_120.json").read_text(encoding="utf-8"))
    records = []
    for i, q in enumerate(qs, 1):
        correct = correct_chunk_ids(q)
        t0 = time.perf_counter()
        rec = {"id": q["id"], "类型": q["类型"], "问题": q["问题"],
               "正确chunk数": len(correct)}
        try:
            r = requests.post(URL, json={"question": q["问题"],
                                         "timing": True, "trace": True},
                              timeout=120)
            d = r.json()
            rec["answer"] = d.get("answer", "")
            rec.update({s: round(float(d.get("timings", {}).get(s, 0.0)), 1)
                        for s in STAGES[:-1]})
            rec["总计"] = round((time.perf_counter() - t0) * 1000, 1)
            tr = d.get("trace") or {}
            rec["n_context"] = tr.get("n_context", 0)
            rec["context_chars"] = tr.get("context_chars", 0)
            reranked = [(c.get("chunk_id"), c.get("rerank_score"),
                         c.get("sources") or []) for c in tr.get("reranked") or []]
            merged = [(c.get("chunk_id"), c.get("score")) for c in tr.get("rrf_merged") or []]
            rec["is_refusal"] = REFUSAL in (rec["answer"] or "")
            rec["reranked_chunks"] = [c[0] for c in reranked]
            rec["merged_chunks"] = [c[0] for c in merged]
            rec["hit_rerank"] = {f"k{k}": hit_at_k(rec["reranked_chunks"], correct, k)
                                 for k in (1, 3, 5, 10)}
            rec["hit_rrf"] = {f"k{k}": hit_at_k(rec["merged_chunks"], correct, k)
                              for k in (1, 3, 5, 10)}
            status = ("拒答" if rec["is_refusal"]
                      else f"hit10={rec['hit_rerank']['k10']}")
            print(f"[{i}/{len(qs)}] {q['类型']} 总计={rec['总计']}ms {status}")
        except Exception as e:
            print(f"[{i}/{len(qs)}] FAIL {e}")
            rec.update({s: 0.0 for s in STAGES})
            rec["answer"] = ""
            rec["is_refusal"] = False
            rec["reranked_chunks"] = []
            rec["merged_chunks"] = []
            rec["hit_rerank"] = {}
            rec["hit_rrf"] = {}
        records.append(rec)

    (BASE / "full_eval_result.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: full_eval_result.json ({len(records)} 条)")


if __name__ == "__main__":
    main()
