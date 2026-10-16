# -*- coding: utf-8 -*-
"""主办方服务器本地模型压测（BGE_EMBED_BACKEND/BGE_RERANK_BACKEND=local 部署后运行）。

模式 1（容器内，模型基准）：
  docker exec rag-forward-rookie-app python test/server_bench.py models
模式 2（宿主机，端到端延迟拆解，需 timing=true 接口）：
  python test/server_bench.py e2e

结果直接打印，人工记录对照目标值：
  编码 <300ms；精排 20-60 条 x500字 <3s；端到端总计与网关基线对比。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

QUERIES = ["什么是机器学习", "2024年经济热词有哪些", "扫码支付的费率是多少"]
E2E_QUESTIONS = [
    "2024年中国经济热词有哪些？",
    "云闪付和全民付的费率分别是多少",
    "站所路段的运营公司有哪些，把这些公司的邮箱给我",
]


def bench_models():
    from config import EMBEDDING_DIM
    from models_singleton import encode_texts, rerank_scores

    print("=== ① bge-m3 编码 ===")
    encode_texts(["预热"], EMBEDDING_DIM)
    for q in QUERIES:
        t0 = time.perf_counter()
        v = encode_texts([q], EMBEDDING_DIM)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  {q[:16]!r}: {dt:.0f}ms dim={len(v[0]) if v else 'FAIL'}")

    print("=== ② bge-reranker 精排（真实 chunk 尺寸） ===")
    rerank_scores("预热", ["预热"])
    chunk500 = ("支付清算业务操作指引。" * 50)[:500]
    for n in (20, 60, 100):
        docs = [chunk500] * n
        t0 = time.perf_counter()
        s = rerank_scores(QUERIES[0], docs)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  {n} 条 x500字: {dt:.0f}ms ({dt/n:.0f}ms/条) "
              f"{'OK' if s and len(s) == n else 'FAIL'}")

    print("=== ③ 长文档上限（5000 字截断场景） ===")
    for n in (20, 50):
        docs = [chunk500 * 10] * n
        t0 = time.perf_counter()
        s = rerank_scores(QUERIES[1], docs)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  {n} 条 x5000字: {dt:.0f}ms ({dt/n:.0f}ms/条) "
              f"{'OK' if s and len(s) == n else 'FAIL'}")


def bench_e2e():
    import json
    import requests

    URL = "http://127.0.0.1:6151/chat"
    for q in E2E_QUESTIONS:
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, json={"question": q, "timing": True}, timeout=180)
            d = r.json()
            total = (time.perf_counter() - t0) * 1000
            timings = d.get("timings", {})
            stages = " | ".join(
                f"{k}={v}ms" for k, v in sorted(timings.items(), key=lambda x: -x[1])[:8]
            )
            print(f"\n问: {q}\n  总计 {total:.0f}ms | {stages}")
            ans = (d.get("answer") or "")[:120].replace("\n", " ")
            print(f"  答: {ans}")
        except Exception as e:
            print(f"\n问: {q}\n  FAIL: {e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "models"
    if mode == "models":
        bench_models()
    else:
        bench_e2e()
