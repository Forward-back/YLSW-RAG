"""本地 vs 网关 BGE-M3 向量一致性对比。

结论用途：本地 query 向量与网关索引向量若高度一致（cos>0.99），
可不重建索引直接切换；否则必须重建（reset pending + 删索引）。

用法（.venv-model）：
  BGE_EMBED_BACKEND=local ./.venv-model/Scripts/python test/vector_compat.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import requests

from core.config import BGE_EMBEDDING_URL, BGE_API_KEY, EMBEDDING_DIM, API_TIMEOUT
from core.models_singleton import encode_texts

TEXTS = [
    "2024年经济热词有哪些",
    "扫码支付的费率是多少",
    "银行卡清算是什么意思",
    "站所路段的运营公司有哪些",
]


def gateway_embeddings(texts: list[str]) -> list[list[float]] | None:
    resp = requests.post(
        BGE_EMBEDDING_URL,
        json={"input": texts, "model": "BAAI/bge-m3"},
        headers={"Authorization": f"Bearer {BGE_API_KEY}"},
        timeout=API_TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"gateway HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()["data"]
    data.sort(key=lambda d: d["index"])
    return [item["embedding"][:EMBEDDING_DIM] for item in data]


def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    print("本地编码...")
    local = encode_texts(TEXTS, EMBEDDING_DIM)
    print("网关编码...")
    gw = gateway_embeddings(TEXTS)
    if not local or not gw:
        print("FAIL: 某侧编码失败")
        sys.exit(1)
    for t, l, g in zip(TEXTS, local, gw):
        print(f"  {t[:20]!r}: cos={cos(l, g):.6f}")
