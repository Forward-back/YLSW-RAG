"""本地 BGE 模型单例（bge-m3 嵌入 + bge-reranker-v2-m3 精排）。

仅 BGE_EMBED_BACKEND / BGE_RERANK_BACKEND =local 时由调用方惰性加载；进程内全局单例，
多次加载只付一次内存/启动代价。纯 CPU 推理（use_fp16=False）。
"""

import os
import sys
import logging
import threading

_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

# 离线加载本地模型目录，禁止任何 HF hub 网络访问
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from core.config import LOCAL_MODELS_DIR, LOCAL_MODEL_THREADS

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedding_model = None
_reranker_model = None


def _ensure_threads():
    import torch
    torch.set_num_threads(LOCAL_MODEL_THREADS)
    # 线程池并发调 encode 时 OMP 线程数翻倍争抢 CPU，关掉嵌套并行
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def get_embedding_model():
    """bge-m3 嵌入模型单例（BGEM3FlagModel，CPU fp32）。"""
    global _embedding_model
    with _lock:
        if _embedding_model is None:
            _ensure_threads()
            from FlagEmbedding import BGEM3FlagModel
            path = os.path.join(LOCAL_MODELS_DIR, "bge-m3")
            _embedding_model = BGEM3FlagModel(path, use_fp16=False)
            logger.info(f"bge-m3 本地模型加载完成: {path}")
    return _embedding_model


def get_reranker_model():
    """bge-reranker-v2-m3 单例（FlagReranker，CPU fp32）。"""
    global _reranker_model
    with _lock:
        if _reranker_model is None:
            _ensure_threads()
            from FlagEmbedding import FlagReranker
            path = os.path.join(LOCAL_MODELS_DIR, "bge-reranker-v2-m3")
            _reranker_model = FlagReranker(path, use_fp16=False)
            logger.info(f"bge-reranker-v2-m3 本地模型加载完成: {path}")
    return _reranker_model


def encode_texts(texts: list[str], dim: int) -> list[list[float]] | None:
    """批量文本 → 截断前 dim 维向量。与网关 BGE-M3 输出对齐（1024 维截 768）。"""
    try:
        model = get_embedding_model()
        out = model.encode(
            texts,
            batch_size=32,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out.get("dense_vecs")
        if dense is None:
            logger.warning("bge-m3 本地编码未返回 dense_vecs")
            return None
        return [list(map(float, v[:dim])) for v in dense]
    except Exception as e:
        logger.error(f"bge-m3 本地编码失败: {e}")
        return None


def rerank_scores(query: str, documents: list[str]) -> list[float] | None:
    """query 对 documents 逐条打分（normalize 后 0-1，与网关 relevance_score 同语义）。"""
    try:
        model = get_reranker_model()
        pairs = [[query, d] for d in documents]
        scores = model.compute_score(pairs, batch_size=8, normalize=True)
        return [float(s) for s in scores]
    except Exception as e:
        logger.error(f"bge-reranker 本地打分失败: {e}")
        return None
