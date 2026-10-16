"""本地 BGE 模型冒烟测试：编码/精排延迟 + 与网关向量一致性。

用法（.venv-model，需 BGE_EMBED_BACKEND/BGE_RERANK_BACKEND=local 且 models/ 完整）：
  BGE_EMBED_BACKEND=local BGE_RERANK_BACKEND=local ./.venv-model/Scripts/python test/smoke_local.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import EMBEDDING_DIM
from models_singleton import encode_texts, rerank_scores

QUERIES = ["什么是机器学习", "2024年经济热词有哪些", "扫码支付的费率是多少"]


def bench_embedding():
    # 预热
    encode_texts(["预热"], EMBEDDING_DIM)
    for q in QUERIES:
        t0 = time.perf_counter()
        v = encode_texts([q], EMBEDDING_DIM)
        dt = (time.perf_counter() - t0) * 1000
        ok = v is not None and len(v[0]) == EMBEDDING_DIM
        print(f"  encode {q[:20]!r}: {dt:.0f}ms dim={len(v[0]) if v else 'FAIL'} {'OK' if ok else 'FAIL'}")


def bench_reranker():
    docs = ["测试文档内容，用于精排延迟基准。"] * 20
    rerank_scores("预热", ["预热"])
    t0 = time.perf_counter()
    s = rerank_scores(QUERIES[0], docs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  rerank 20 docs: {dt:.0f}ms scores={[round(x, 3) for x in (s or [])][:3]} "
          f"{'OK' if s and len(s) == 20 else 'FAIL'}")


def bench_long_docs():
    # 真实场景：chunk 文本较长（5000 字截断）
    docs = [("支付清算业务指南。" * 200)[:5000]] * 30
    t0 = time.perf_counter()
    s = rerank_scores(QUERIES[1], docs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  rerank 30 docs x5000字: {dt:.0f}ms {'OK' if s and len(s) == 30 else 'FAIL'}")


if __name__ == "__main__":
    print("=== bge-m3 编码 ===")
    bench_embedding()
    print("=== bge-reranker-v2-m3 精排 ===")
    bench_reranker()
    bench_long_docs()
