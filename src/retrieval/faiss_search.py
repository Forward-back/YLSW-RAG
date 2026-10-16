import os, sys, time, logging
import numpy as np
import requests

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import (
    FAISS_TOP_K, HNSW_EF_SEARCH, EMBEDDING_DIM,
    BGE_EMBEDDING_URL, BGE_API_KEY, API_TIMEOUT, API_MAX_RETRIES,
    DATA_DIR, BGE_EMBED_BACKEND,
)
from http_client import post_json

logger = logging.getLogger(__name__)


_EMB_CACHE: dict[str, list] = {}
_EMB_CACHE_MAX = 512


def _cache_put(text: str, emb: list):
    if len(_EMB_CACHE) >= _EMB_CACHE_MAX:
        _EMB_CACHE.pop(next(iter(_EMB_CACHE)))
    _EMB_CACHE[text] = emb


def get_bge_embedding(text: str, model: str = "BAAI/bge-m3") -> list | None:
    """单条文本获取 embedding（LRU 缓存，重复 query 零调用）。"""
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    embs = _get_bge_embeddings([text], model)
    if not embs:
        return None
    return embs[0]


def _get_bge_embeddings(texts: list[str], model: str = "BAAI/bge-m3") -> list[list] | None:
    """批量文本调 BGE（网关 API 或本地模型，BGE_EMBED_BACKEND 控制）。"""
    cacheable = [t for t in texts if t not in _EMB_CACHE]
    if not cacheable:
        return [_EMB_CACHE[t] for t in texts]
    if BGE_EMBED_BACKEND == "local":
        from models_singleton import encode_texts
        embs = encode_texts(cacheable, EMBEDDING_DIM)
        if embs is None:
            logger.warning("本地 BGE embedding 失败")
            return None
        for t, emb in zip(cacheable, embs):
            _cache_put(t, emb)
        return [_EMB_CACHE[t] for t in texts]
    resp, err = post_json(
        BGE_EMBEDDING_URL,
        {"input": cacheable, "model": model},
        {"Authorization": f"Bearer {BGE_API_KEY}"},
        API_TIMEOUT,
    )
    if resp is None:
        logger.warning(f"BGE embedding 失败: {err}")
        return None
    if resp.status_code != 200:
        logger.warning(f"BGE embedding HTTP {resp.status_code}: {resp.text[:150]}")
        return None
    data = resp.json()["data"]
    data.sort(key=lambda d: d["index"])
    for t, item in zip(cacheable, data):
        _cache_put(t, item["embedding"][:EMBEDDING_DIM])
    return [_EMB_CACHE[t] for t in texts]


def get_bge_embeddings(texts: list[str], model: str = "BAAI/bge-m3") -> list[list] | None:
    """批量获取 embedding（公开接口，供规划器多 step 一次调用）。"""
    return _get_bge_embeddings(texts, model)


class FAISSSearcher:
    """FAISS HNSW 检索器。"""

    def __init__(self, index_path: str = None):
        import faiss, tempfile
        if index_path is None:
            index_path = os.path.join(os.path.dirname(DATA_DIR), "faiss_index.bin")
        self.index_path = index_path

        if not os.path.exists(index_path):
            logger.error(f"索引文件不存在: {index_path}")
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # FAISS C++ 库不支持中文路径，先读入 bytes 再从临时文件加载
        with open(index_path, "rb") as f:
            index_bytes = f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        try:
            tmp.write(index_bytes)
            tmp.close()
            self.index = faiss.read_index(tmp.name)
        finally:
            os.unlink(tmp.name)

        logger.info(f"FAISS 索引加载: {self.index.ntotal} vectors")
        self.index.hnsw.efSearch = HNSW_EF_SEARCH

    def search(self, query: str, top_k: int = None) -> list[tuple[int, float]]:
        """检索 Top-K 相似向量（query → embedding → HNSW，带缓存）。

        Returns: [(vector_id, score), ...]
        """
        embedding = get_bge_embedding(query)
        if embedding is None:
            logger.error(f"[FAISS] query embedding 获取失败: {query[:60]}")
            return []
        return self.search_vec(embedding, top_k)

    def search_vec(self, embedding: list, top_k: int = None) -> list[tuple[int, float]]:
        """已嵌向量直接检索（批量嵌入场景：规划器多 step 共用一次 API 调用）。"""
        t0 = time.perf_counter()
        if top_k is None:
            top_k = FAISS_TOP_K

        q_vec = np.array([embedding], dtype=np.float32)
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS 无效结果
                continue
            results.append((int(idx), float(dist)))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"[FAISS] vec检索 top_k={top_k} → {len(results)} 条 ({elapsed:.0f}ms)")
        return results

    def search_with_embeddings(
        self, embeddings: list[list[float]], top_k: int = None
    ) -> list[list[tuple[int, float]]]:
        """批量检索（用于多 query 场景）。"""
        if top_k is None:
            top_k = FAISS_TOP_K

        q_vecs = np.array(embeddings, dtype=np.float32)
        distances, indices = self.index.search(q_vecs, top_k)

        batch_results = []
        for i in range(len(embeddings)):
            results = []
            for j in range(top_k):
                idx = indices[i][j]
                dist = distances[i][j]
                if idx < 0:
                    continue
                results.append((int(idx), float(dist)))
            batch_results.append(results)

        return batch_results

    @property
    def size(self) -> int:
        return self.index.ntotal
