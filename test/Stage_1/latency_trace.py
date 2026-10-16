# -*- coding: utf-8 -*-
"""串行 trace 重采：50 题带 timing+trace，输出 serial_trace.json 并刷新 serial_latency.json。

与 latency_breakdown.py 同 seed 抽样，保证题目一致。
serial_trace.json 每记录：id / 类型 / timings / trace（rrf_merged、reranked 候选链，评测用）。
"""
import json
import time
from pathlib import Path

import requests

from latency_breakdown import STAGES, URL, normalize, sample_questions

BASE = Path(__file__).resolve().parent

TRACE_KEYS = ("rrf_merged", "reranked")


def slim_trace(trace: dict) -> dict:
    out = {"n_context": trace.get("n_context", 0),
           "context_chars": trace.get("context_chars", 0)}
    for key in TRACE_KEYS:
        out[key] = [
            {
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "score": c.get("score"),
                "rerank_score": c.get("rerank_score"),
                "sources": c.get("sources") or [],
            }
            for c in trace.get(key) or []
        ]
    return out


def main():
    qs = sample_questions()
    print(f"抽样 {len(qs)} 题")
    records = []
    for i, q in enumerate(qs, 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                URL,
                json={"question": q["问题"], "timing": True, "trace": True},
                timeout=120,
            )
            d = r.json()
            rec = {"id": q["id"], "类型": q["类型"]}
            rec.update(normalize(d.get("timings", {})))
            rec["总计"] = round((time.perf_counter() - t0) * 1000, 1)
            rec["trace"] = slim_trace(d.get("trace") or {})
            records.append(rec)
            n_merged = len(rec["trace"]["rrf_merged"])
            n_reranked = len(rec["trace"]["reranked"])
            print(f"[{i}/{len(qs)}] {q['类型']} 总计={rec['总计']}ms "
                  f"merged={n_merged} reranked={n_reranked}")
        except Exception as e:
            print(f"[{i}/{len(qs)}] FAIL {e}")
            rec = {"id": q["id"], "类型": q["类型"]}
            rec.update({s: 0.0 for s in STAGES})
            rec["总计"] = 0.0
            rec["trace"] = {k: [] for k in TRACE_KEYS}
            records.append(rec)

    (BASE / "serial_trace.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    latency = [{k: v for k, v in r.items() if k != "trace" and k != "类型"}
               for r in records]
    (BASE / "serial_latency.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成: serial_trace.json + serial_latency.json 已刷新")


if __name__ == "__main__":
    main()
