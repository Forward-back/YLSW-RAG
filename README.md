# YLSW-RAG — 金融支付领域文档智能问答系统

基于检索增强生成（RAG）的金融支付领域智能问答系统（竞赛项目 `rag-forward-rookie`）。面向约 140 份金融支付语料（产品手册、新闻、法规、报告、表格，中英双语），在 **30 秒作答硬约束** 下提供高质量问答：检索层文档级命中率 100%，端到端答案正确率约 87%（120 题 LLM 裁判），全链延迟中位 ~8.5s。

## 系统架构

```
用户问题
  │
  ├─ 0. 实体检测（非 LLM）：产品/机构/别名动态枚举 → 类型意图
  ├─ 0.6 规划器 + 变体生成（两路 LLM 并行）
  │       规划器：异常判定（injection 拒答 / chitchat 放行）+ 六类标签 + 执行计划
  │       变体：2-3 条同义查询（仅基于原问题，与规划并行）
  │       type: single / parallel_subq（并行子问）/ multi_hop（多跳 + foreach）
  │
  ├─ 1-3. 四路并行检索
  │       ├─ FAISS HNSW 语义检索（BGE-M3 768 维，批量 embedding，ef_search=192）
  │       ├─ BM25 词法检索（jieba + 文号/版本号 token 保护 + 标题加权）
  │       ├─ 元数据精确匹配（产品/机构/类型 + 产品提及倒排）
  │       └─ Text-to-SQL（模板链 → 列语义索引 → LLM SQL，schema 剪枝 + 语义选表）
  │
  ├─ 4. RRF 融合（多路共识加分，同 chunk 去重）
  ├─ 5. BGE-Reranker v2-m3 精排
  │       多路打分并行（原问 + 各子问取 max）+ 章节多样性三轮贪心
  │       + 源配额（meta≤3/5）+ 邻域补全（±8）+ 同文档多版本去重
  ├─ 6. Parent-Child 上下文组装
  │       top1 父块（1500 字截断）+ 其余 child + span 合并去重
  │       + 多步题文档摘要前置（≤3 篇）+ SQL 命中上下文收紧
  │
  └─ 7. LLM 生成（标签动态提示词 + 要点拆解自查 + 时间线 CoT
           + 数值误差声明 + 语言适配 + 拒答保护）
```

## 核心设计

### 检索层

| 环节 | 设计 |
|---|---|
| 分块 | Parent-Child：child 300 token 检索 / parent 1200 token 上下文；法规按条、手册按节、新闻整篇；标题链前缀补上下文 |
| 嵌入 | BGE-M3 截断前 768 维（官方建议）；支持本地模型 / API 双后端（`BGE_EMBED_BACKEND`） |
| 融合 | RRF 四路，同 chunk 多路命中加分；SQL 命中固定高分位 |
| 精排 | 章节多样性三轮贪心（概述类不整章缺料）+ 源配额 + 邻域补全；多步题多路打分并行 |
| 上下文 | 分层组装（top1 parent/其余 child）+ span 合并 + 版本去重 + 摘要前置（多步题） |
| 降级 | 精排挂起回退 RRF 排序；空融合池直接回未找到；单点故障不阻断主链路 |

### SQL 路（结构化表格）

四层递进，确定性优先：

1. **模板链**（零 LLM）：极值（经度/纬度 max/min）、计数（条件/排序/全表）、代码反查、列举/定点查询
2. **列语义索引**：模糊字段问法（"与金额相关的字段"）词重叠直答
3. **LLM SQL**：关键词 schema 剪枝 + 表描述 embedding 语义选表 + few-shot（GROUP BY 排序/多条件/极值/COUNT DISTINCT）
4. **执行护栏**：SELECT-only 白名单校验、结果行数上限、空结果自动回退 LLM 重试、连接 ping 自愈

### 生成层

- **标签动态提示词**：产品咨询 / 对比分析 / 操作指南 / 政策法规 / 数据查询 / 概念解读
- **要点拆解自查**：多要点问题逐条作答，禁止笼统"资料未包含"
- **时间线 CoT**：多文号/日期问题强制三步推理（逐文列规则 → 判适用 → 结论）
- **数值计算**：完整算式 + 两位小数 + 舍入误差声明
- **拒答安全**：仅 LLM 注入判定可触发拒答；chitchat 放行走正常管线（业务题误判不拦）

## 目录结构

```
YLSW_RAG/
├── Data/                      # 语料（MAN 手册 / NWS 新闻 / PDT 产品 / REG 法规 / RPT 报告 / TBL 表格）
│                              #   不入 git 仓库，需自行放入后运行 ingest
├── src/
│   ├── app.py                 # Flask API 入口 + 主查询管线
│   ├── config.py              # 全局配置（自动加载 .env）
│   ├── http_client.py         # 共享 HTTP 会话 + 分类重试
│   ├── prompt_guard.py        # 提示词注入防护 / 拒答文案
│   ├── models_singleton.py    # 本地 BGE 模型单例（懒加载 + 预热）
│   ├── server_bench.py        # 模型基准 / 端到端延迟测试
│   ├── ingest/                # 入库管线：解析 → 分类 → 分块 → 嵌入 → 入库
│   │   ├── pipeline.py        # 幂等入库（文件哈希去重）+ FAISS 索引构建/自愈
│   │   ├── chunker.py         # Parent-Child 差异化分块
│   │   ├── parsers.py         # PDF/DOCX/XLSX/MD 解析
│   │   ├── metadata.py        # 元数据提取（产品/机构/文号/日期）
│   │   ├── summarizer.py      # 文档要点摘要回填（综述类题上下文前置用）
│   │   └── paddle_ocr.py      # OCR
│   ├── retrieval/
│   │   ├── faiss_search.py    # FAISS HNSW 语义检索（批量 embedding 接口）
│   │   ├── bm25_search.py     # BM25 词法检索（文号 token 保护 + 标题加权）
│   │   ├── chunk_store.py     # chunk 内存镜像 / parent 反查 / 摘要
│   │   ├── query_filter.py    # 实体检测（动态枚举同步）+ 元数据精确匹配
│   │   ├── text_to_sql.py     # Text-to-SQL 引擎（模板链/列索引/语义选表/自愈）
│   │   ├── fusion.py          # RRF 融合 + 精排（多路并行）+ 章节多样性
│   │   └── planner.py         # LLM 规划器（Plan-and-Solve DAG）+ 变体生成
│   ├── llm/generator.py       # 答案生成（流式 + 降级）
│   └── prompts/               # 六类标签动态提示词（JSON）
├── test/                      # 评测（见下）
├── models/                    # 本地 BGE 模型（挂载目录，不进镜像/仓库）
├── Dockerfile                 # 应用镜像（torch CPU + FlagEmbedding）
├── docker-compose.yml         # 生产部署（app + 自建 MySQL）
├── docker-compose.local.yml   # 本地开发覆盖（3307 MySQL 容器）
├── entrypoint.sh              # 幂等启动管线：等库 → 建表 → ingest → 索引自愈 → 预热 → API
├── init.sql                   # 表结构
├── 运维手册.md                # 运维手册（清库/启停/踩坑全集，必读）
└── 部署手册.md                # 服务器部署手册（两镜像形态 / 打包 / 索引重建）
```

## 快速开始

### 环境要求

- Docker + Docker Compose（推荐）
- 或 Python 3.12 + MySQL（裸机开发）
- 模型 API 凭证：主办方网关（或自有 DeepSeek/硅基流动 key）

### 1. 配置 .env

```ini
# ── MySQL ──
MYSQL_HOST=mysql            # 容器内走 compose DNS；裸机改 localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=<密码>
MYSQL_DATABASE=rag_db

PORT=6151                   # 主办方固定问答端口

# ── 模型 API（部署形态：网关 + 本地 embedding）──
BGE_API_KEY=<主办方网关 key>
BGE_EMBEDDING_URL=http://180.76.46.222:8289/v1/embeddings
BGE_RERANKER_URL=http://180.76.46.222:8289/v1/rerank
DEEPSEEK_API_KEY=<主办方网关 key>
DEEPSEEK_URL=http://180.76.46.222:8289/v1/chat/completions
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-V4-Flash
BGE_EMBED_BACKEND=local     # embedding 本地模型（models/ 挂载）
BGE_RERANK_BACKEND=gateway

# ── 30s 硬约束超时预算 ──
PLANNER_TIMEOUT=5
RERANK_TIMEOUT=5
GENERATE_TIMEOUT=20
KEEPALIVE_INTERVAL=480
```

完整配置与服务器部署见 [部署手册.md](部署手册.md)。

### 2. 启动（Docker）

```bash
docker compose up -d --build
docker compose logs -f app     # 首次全量 ingest + 索引构建，数分钟
curl http://localhost:6151/health
```

### 3. 裸机开发（Windows）

```bash
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m src.ingest.pipeline    # 入库 + 建索引（幂等）
.venv/Scripts/python -m src.app                # 起服务
```

## API

### POST /chat

```bash
curl -X POST http://localhost:6151/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "扫码支付的费率是多少？"}'
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | string | 必填 |
| `timing` | bool | 可选，响应附 `timings`（各阶段耗时 ms） |
| `trace` | bool | 可选，响应附 `trace`（检索链中间数据） |

### GET /health

返回 `{"status": "ok"}`，容器 healthcheck 同用。

## 评测体系

| 层级 | 内容 |
|---|---|
| 测试集 | `test/produce_question/qa_testset_120.json`：140 题（120 业务题 + 20 SQL 能力题，L1 精确取值 ~ L5 语义结构梯度） |
| 检索层 | `test/Stage_2/eval_retrieval.py`：47 题 trace，Recall@K / 排序分布 / PR 曲线 / 精排迁移 |
| 答案层 | `test/Stage_3/full_eval.py` 全量采集 + `answer_judge.py` LLM 裁判（文档级参考 + 表数据聚合视图） |
| 延迟 | `src/server_bench.py e2e`；`full_eval` 的 timings 可按阶段统计 |

当前基线（2026-10）：

| 指标 | 值 |
|---|---|
| 检索文档级 hit@10 | 100%（106/106） |
| 答案正确率（judge） | ~87%（正确 104 / 部分 13 / 错误 3，含安全题） |
| 端到端延迟 | 中位 8486ms，p90 9808ms，max 11926ms，超 30s 0 题 |
| 阶段耗时中位 | 规划+变体（并行）1.9s / 检索 0.9s / 精排 1.8s / 生成 1.7s |

## 相关文档

| 文档 | 内容 |
|---|---|
| [架构设计.md](架构设计.md) | 各子系统设计决策：为什么这样做、被否方案、实验数据佐证 |
| [评测说明.md](评测说明.md) | 测试集构造、指标定义、当前基线、历史演进、复现命令 |
| [运维手册.md](运维手册.md) | 镜像构建、清库重入、启停、20 条踩坑记录（必读） |
| [部署手册.md](部署手册.md) | 服务器部署：两镜像形态、交付物打包、从零部署、索引重建、摘要回填 |
| [术语表.md](术语表.md) | 业务术语 + 系统术语速查 |

## 注意事项

- **30s 硬约束**：超时预算规划 5s / 精排 5s / 生成 20s，单点挂起自动降级；网关冷池靠预热 + keepalive 保温
- **清库 / 改分块 / 换 embedding 源的操作红线**：见 运维手册.md §3 与 §5（索引错位、幂等跳过等历史事故与解法）
- **服务器升级代码**：必须重新 build + save + load 镜像，`restart` 用的是旧镜像
- **密钥安全**：`.env` 已被 `.gitignore` / `.dockerignore` 排除，勿将密钥提交或打入镜像
