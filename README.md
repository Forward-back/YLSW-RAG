# YLSW-RAG — 金融支付领域文档智能问答系统

基于 RAG（检索增强生成）的金融支付领域智能问答系统（竞赛项目 `rag-forward-rookie`）。面向约 120 份金融支付语料（产品手册、新闻、产品介绍、法规、报告、表格，中英双语），在 **主办方 30 秒作答硬约束** 下提供高质量问答。

## 系统架构

```
用户问题
  │
  ├─ 0. 实体检测（非 LLM）：产品/机构/别名 → 类型意图
  ├─ 0.6 问题规划器（LLM，一次调用）：异常判定 + 问题标签 + 执行计划
  │       tag: 产品咨询/对比分析/操作指南/政策法规/数据查询/概念解读
  │       type: single / parallel_subq（并行子问）/ multi_hop（多跳 + foreach）
  │
  ├─ 1-3. 四路并行检索
  │       ├─ FAISS HNSW 语义检索（BGE-M3，768 维）
  │       ├─ BM25 词法检索（jieba 分词）
  │       ├─ 元数据精确匹配（产品/机构 → chunks 表元数据字段）
  │       └─ Text-to-SQL（结构化表格查询）
  │
  ├─ 4. RRF 融合（多路共识加分，同 chunk 去重）
  ├─ 5. BGE-Reranker v2-m3 精排（网关/本地双开关）
  │       章节多样性选择 + 源配额（meta≤3/5）+ 邻域补全 + 版本去重
  ├─ 6. Parent-Child 上下文组装（top1 取父块完整上下文，同文档 span 合并）
  │
  └─ 7. LLM 生成（DeepSeek-V4-Flash + 标签动态提示词 + 语言适配 + 拒答保护）
```

## 功能特性

- **多格式入库**：PDF / DOCX / XLSX / Markdown，含 OCR、表格抽取、DOCX 内嵌图片 VL 理解
- **差异化分块**：法规按条款、手册按章节、新闻整篇、Markdown 按标题树 + 标题链前缀
- **Parent-Child 分层**：child（300 token）做检索，parent（1200 token）做上下文，兼顾召回与完整
- **四路混合检索 + RRF 融合**：语义 / 词法 / 元数据 / SQL 多路互补
- **LLM 规划器**：子问题分解、并行检索、多跳推理、foreach 逐实体展开
- **六类问题动态提示词**：按问题标签加载专属提示词与答题规则
- **稳健性设计**：
  - 30s 硬约束下逐段超时预算（规划 5+5s / 精排 5s / 生成 20s），单点挂起自动降级（拒答规则 / RRF 兜底）
  - 网关 keepalive 保温，防 serverless 冷启动（40-60s）
  - HTTP 客户端连接复用 + 分类重试（429 读 Retry-After，5xx 指数退避）
  - 提示词注入防护、虚假拒答防护、时间错标防护
- **可观测**：全链路 trace_id 日志、`timing`/`trace` 响应字段、离线评测集

## 目录结构

```
YLSW_RAG/
├── Data/                      # 语料（MAN 手册 / NWS 新闻 / PDT 产品 / REG 法规 / RPT 报告 / TBL 表格）
│                              #   不入 git 仓库（.gitignore），需自行放入后运行 ingest
├── src/
│   ├── app.py                 # Flask API 入口 + 主查询管线（实体→规划→检索→融合→精排→生成）
│   ├── config.py              # 全局配置（自动加载 .env）
│   ├── http_client.py         # 共享 HTTP 会话 + 分类重试
│   ├── prompt_guard.py        # 提示词注入防护 / 拒答文案
│   ├── models_singleton.py    # 本地 BGE 模型单例（懒加载 + 预热）
│   ├── server_bench.py        # 模型基准 / 端到端延迟测试
│   ├── ingest/                # 入库管线：解析 → 分类 → 分块 → 嵌入 → 入库
│   │   ├── pipeline.py        # 幂等入库（文件哈希去重）+ FAISS 索引构建
│   │   ├── chunker.py         # Parent-Child 差异化分块
│   │   ├── parsers.py         # PDF/DOCX/XLSX/MD 解析
│   │   ├── metadata.py        # 元数据提取（产品/机构/文号/日期）
│   │   └── paddle_ocr.py      # OCR
│   ├── retrieval/             # 检索子系统
│   │   ├── faiss_search.py    # FAISS HNSW 语义检索
│   │   ├── bm25_search.py     # BM25 词法检索（内存索引）
│   │   ├── chunk_store.py     # chunk 解析 / parent 反查 / 邻域
│   │   ├── query_filter.py    # 实体检测 + 元数据精确匹配
│   │   ├── text_to_sql.py     # 自然语言 → SQL
│   │   ├── fusion.py          # RRF 融合 + Reranker 精排 + 章节多样性
│   │   └── planner.py         # LLM 规划器（Plan-and-Solve DAG）
│   ├── llm/generator.py       # 答案生成（网关/本地双通道）
│   └── prompts/               # 六类标签动态提示词（JSON）
├── test/                      # 评测：Stage_1/2/3 评测集、judge、冒烟测试
├── models/                    # 本地 BGE 模型（挂载目录，不进镜像）
├── Dockerfile                 # 应用镜像（torch CPU + FlagEmbedding 可选）
├── docker-compose.yml         # 生产部署（app + 自建 MySQL）
├── docker-compose.local.yml   # 本地开发覆盖（3307 MySQL 容器）
├── entrypoint.sh              # 幂等启动管线：等库 → 建表 → ingest → 索引自愈 → 预热 → 起 API
├── init.sql                   # 表结构
├── OPS.md                     # 运维手册（踩坑记录，强烈建议先读）
└── DEPLOY_TEST_SERVER.md      # 服务器 BGE 混合部署 + 测试手册
```

## 快速开始

### 环境要求

- Docker + Docker Compose（推荐部署方式）
- 或 Python 3.12 + MySQL（裸机开发，见 [OPS.md](OPS.md)）
- 主办方模型网关访问凭证（BGE / DeepSeek / Qwen API Key）

### 1. 配置 .env

在项目根目录创建 `.env`（密钥不进镜像、不入库）：

```ini
# ── MySQL ──
MYSQL_HOST=mysql            # 容器内走 compose DNS；裸机改为 localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=<你的密码>
MYSQL_DATABASE=rag_db

# ── 服务端口（主办方固定问答端口 6151）──
PORT=6151

# ── 主办方模型网关（必填三项）──
BGE_API_KEY=<BGE key>
DEEPSEEK_API_KEY=<DeepSeek key>
QWEN_API_KEY=<Qwen key>
# 网关地址默认已内置（http://180.76.46.222:8289/v1/...），如需覆盖：
# BGE_EMBEDDING_URL / BGE_RERANKER_URL / DEEPSEEK_URL / QWEN_URL

# ── BGE 后端开关（可选，默认 gateway）──
# embedding / reranker 可独立切换 gateway=主办方网关 / local=本地模型推理
# local 模式需挂载 models/ 目录（bge-m3 / bge-reranker-v2-m3）
BGE_EMBED_BACKEND=gateway
BGE_RERANK_BACKEND=gateway
```

### 2. 启动（Docker）

```bash
# 生产部署（自建 MySQL 容器，问答 6151 / MySQL 6152 仅回环）
docker compose up -d --build

# 本地开发（自带 3307 MySQL 容器）
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build

# 观察启动管线：等 MySQL → 建表 → 幂等 ingest（哈希跳过）→ FAISS 索引自愈 → 模型预热 → API
docker compose logs -f app
```

首次启动会自动完成全部入库与索引构建（约 120 份文档、1900 个 chunk、数分钟），完成后：

```bash
curl http://localhost:6151/health          # → {"status":"ok"}
```

### 3. 裸机开发（Windows，可选）

```bash
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m src.ingest.pipeline    # 入库 + 建索引（幂等）
.venv/Scripts/python -m src.app                # 起服务
```

## API

### POST /chat — 问答（竞赛标准接口）

```bash
curl -X POST http://localhost:6151/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "扫码支付的费率是多少？"}'
```

```json
{"answer": "..."}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | string | 必填，用户问题 |
| `timing` | bool | 可选，响应附 `timings`（各阶段耗时 ms） |
| `trace` | bool | 可选，响应附 `trace`（检索链中间数据，评测用） |

### GET /health — 健康检查

返回 `{"status": "ok"}`。容器 healthcheck 同用此端点。

## 检索与生成设计要点

| 环节 | 设计 |
|---|---|
| 分块 | Parent-Child：child 300 token 检索 / parent 1200 token 上下文；法规按条、手册按节、新闻整篇；标题链前缀补上下文 |
| 嵌入 | BGE-M3 截断前 768 维（官方建议，索引内存 -25%），FAISS HNSW（M=32, ef_search=40） |
| 融合 | RRF 四路融合，同 chunk 多路命中加分（共识提升）；SQL 命中固定高分位 |
| 精排 | BGE-Reranker v2-m3；多跳问题对原问 + 各子问多路打分取最大；章节多样性三轮贪心（概述类不整章缺料）；源配额 meta≤3（跨文档≤5）；低分元数据不占配额 |
| 上下文 | top1 每源取 parent 完整上下文，其余 child；同文档 span 相邻合并去重；邻域补全（±5）防条款细节丢失；同文档多版本只留最新版 |
| 规划器 | 一次 LLM 调用出标签+计划；parallel_subq 并行检索；multi_hop 中间答案（summary+items）供 foreach 展开；全失败回退单步管线 |
| 生成 | 标签动态提示词 + 时间核对规则 + 语言适配（中英提问/文档交叉）+ 拒答规则（部分信息也必须作答） |

## 测试与评测

```bash
# 端到端延迟基准（三问覆盖单步/并行子问/多跳 foreach）
python src/server_bench.py e2e

# 模型本地基准
python src/server_bench.py models

# 冒烟测试（本地栈）
python test/smoke_local.py
```

离线评测集见 `test/Stage_1`、`test/Stage_2`、`test/Stage_3`（`full_eval.py` 全量评测 + `answer_judge.py` LLM 裁判），失败案例见 `test/Stage_3/wrong_cases.md`。

## 相关文档

| 文档 | 内容 |
|---|---|
| [OPS.md](OPS.md) | 运维手册：镜像构建、清库重入、启停、已踩坑与解决（必读） |
| [DEPLOY_TEST_SERVER.md](DEPLOY_TEST_SERVER.md) | 服务器 BGE 混合部署 + 测试手册 |

## 注意事项

- **30s 硬约束**：主办方要求单问 30s 内作答。超时预算规划 5+5s / 精排 5s / 生成 20s；网关 serverless 冷启动 40-60s，靠预热 + keepalive（`KEEPALIVE_INTERVAL`）保温
- **索引一致性**：清库后必须同步删除 `faiss_index.bin`（或 force-recreate 容器走自愈），否则向量与 DB 错位、检索全错
- **改分块参数需清库重入**：ingest 幂等只比对文件哈希，改 `chunker.py`/chunk 参数不清库不生效
- **BGE 后端切换需重建索引**：本地与网关向量非同源，切换 `BGE_EMBED_BACKEND` 后必须重建索引
- **密钥安全**：`.env` 已被 `.dockerignore` 排除，勿将密钥提交或打入镜像
