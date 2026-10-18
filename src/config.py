"""项目全局配置。"""

import os
from pathlib import Path

# 自动加载项目根目录的 .env 文件
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if _dotenv_path.exists():
    with open(_dotenv_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# ── MySQL 连接 ─────────────────────────────────────────
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE", "fuck_rag"),
    "charset": "utf8mb4",
}

# ── 主办方模型网关 API 端点（OpenAI 兼容） ────────────────────
# BGE-M3 Embedding
BGE_EMBEDDING_URL = os.getenv(
    "BGE_EMBEDDING_URL", "http://180.76.46.222:8289/v1/embeddings"
)
BGE_API_KEY = os.getenv("BGE_API_KEY")
BGE_RERANKER_URL = os.getenv(
    "BGE_RERANKER_URL", "http://180.76.46.222:8289/v1/rerank"
)

# ── BGE 后端开关（embedding/reranker 可独立）：gateway=主办方网关 API；local=本地模型推理 ─────
# local 模式需 models/bge-m3、models/bge-reranker-v2-m3 存在（挂载或本地目录），
# 且索引必须用同源向量重建（见 OPS.md §3.2）。
BGE_EMBED_BACKEND = os.getenv("BGE_EMBED_BACKEND", "gateway")
BGE_RERANK_BACKEND = os.getenv("BGE_RERANK_BACKEND", "gateway")
LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", os.path.join(BASE_DIR, "models"))
LOCAL_MODEL_THREADS = int(os.getenv("LOCAL_MODEL_THREADS", "24"))

# 通用问答（主办方模型网关）
DEEPSEEK_URL = os.getenv(
    "DEEPSEEK_URL", "http://180.76.46.222:8289/v1/chat/completions"
)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 多模态图片理解（主办方模型网关）
QWEN_URL = os.getenv(
    "QWEN_URL", "http://180.76.46.222:8289/v1/chat/completions"
)
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

# API 超时 (秒)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))

# ── 网关调用超时（主办方要求 30s 内作答完毕，逐段预算） ─────
# 规划器两段式：短超时快试（挂起即止损）→ 长超时救援（网关慢 5-10s 场景）
PLANNER_TIMEOUT = int(os.getenv("PLANNER_TIMEOUT", "5"))
PLANNER_LONG_TIMEOUT = int(os.getenv("PLANNER_LONG_TIMEOUT", "5"))
# 精排挂起回退 RRF 顺序；生成失败=该题失败（无兜底，健康 ~2s）
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "5"))
GENERATE_TIMEOUT = int(os.getenv("GENERATE_TIMEOUT", "20"))
# keepalive 防冷池再冷却（serverless 冷启动 40-60s，30s 预算下必失败）：0=关闭
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "480"))

# ── Chunking 参数 ─────────────────────────────────────
CHILD_CHUNK_SIZE = 300       # 子 chunk token 数
PARENT_CHUNK_SIZE = 1200     # 父 chunk token 数
CHUNK_OVERLAP = 128          # 固定窗口重叠 token 数
CHUNK_CONTEXT_NEIGHBORS = 3  # 法规上下文邻居条数（前后各N条）

# ── FAISS 参数 ────────────────────────────────────────
# BGE-M3 原生 1024 维，截断前 768 维（官方建议，精度几乎不掉，索引内存/检索 -25%）
EMBEDDING_DIM = 768
FAISS_INDEX_TYPE = "HNSW"
HNSW_M = 32                  # HNSW 图连接数
HNSW_EF_SEARCH = 192    # HNSW 搜索时探索的节点数（实测 30 问：ef=192 时 recall@50 达 100%，延迟仅 +0.2ms/次）

# ── 检索参数 ──────────────────────────────────────────
FAISS_TOP_K = 50
BM25_TOP_K = 50             # BM25 词法路召回数（与 FAISS 对称，不超 reranker 每源截断）
BM25_K1 = 1.5               # BM25Okapi 词频饱和参数（默认值）
BM25_B = 0.75               # BM25Okapi 文档长度归一化参数（默认值）
RRF_K = 60                   # RRF 融合常数
FINAL_TOP_K = 10         # Reranker 后最终送入 LLM 的 chunk 数（meta≤3 + faiss≥1）
# 数据查询类问题 SQL 命中时的上下文上限（1 条 SQL 结果 + 3 条语料）：
# 答案主体在结构化查询结果里，语料块多为噪声，收紧可减 prompt 体积和无关信息
SQL_CONTEXT_CAP = int(os.getenv("SQL_CONTEXT_CAP", "4"))
# 送入 Reranker 的每路候选上限：RRF 后每源截断前 N 条，减少 rerank API 输入量
# 与 FAISS_TOP_K 同值（50）：长文档（55+ 切分单元）也要覆盖全部章节，
# 否则概述类问题会整章缺料（如"南沙金融30条"的"四"章曾完全进不了候选池）
RERANK_MAX_CANDIDATES_PER_SOURCE = int(os.getenv("RERANK_MAX_CANDIDATES_PER_SOURCE", "100"))
# metadata 精确匹配的最低 rerank 分数：低于此值视为弱相关，不强制占用源配额
RERANK_META_MIN_SCORE = float(os.getenv("RERANK_META_MIN_SCORE", "0.1"))
# A/B 实验开关：单步问题跳过精排 API（走 RRF 排序 + 配额降级分支）
RERANK_SKIP_SINGLE = os.getenv("RERANK_SKIP_SINGLE", "0") == "1"

# ── 千问多模态（图片理解）───────────────────────────────
QWEN_MODEL = os.getenv("QWEN_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5")
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "1024"))
QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "20"))
QWEN_CONCURRENCY = int(os.getenv("QWEN_CONCURRENCY", "3"))

# 图片描述总开关（DOCX 内嵌图片 → 千问 VL，可关闭以加速 ingest）
ENABLE_IMAGE_DESCRIPTION = os.getenv("ENABLE_IMAGE_DESCRIPTION", "1") == "1"
# 仅发送这些光栅格式给 VL API；WMF/EMF 等矢量格式无法被解码，直接跳过
QWEN_VL_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}

# ── 本地模型（Ollama，OpenAI 兼容接口）───────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# 键名避开系统环境变量 OLLAMA_MODEL（本机已被占用为模型存储路径）
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4:26b")

# ── 通用问答（硅基流动）───────────────────
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── LLM 输入上下文预算（字符数）─────────────────────────
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

# ── 日志 ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG 会输出各板块候选预览等细粒度日志

# ── 必需环境变量校验 ────────────────────────────────────
_REQUIRED_KEYS = [
    "BGE_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "MYSQL_CONFIG",
]


def validate():
    """启动时校验所有必需配置项已设置。"""
    missing = []
    for key in _REQUIRED_KEYS:
        val = globals().get(key)
        if key == "MYSQL_CONFIG":
            if not val.get("password"):
                missing.append("MYSQL_PASSWORD")
        elif not val:
            missing.append(key)
    if missing:
        raise RuntimeError(
            f"缺失必需的环境变量: {', '.join(missing)}。"
            f"请在 .env 文件或系统环境变量中设置。"
        )
