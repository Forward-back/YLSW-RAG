# RAG 系统运维手册

覆盖：镜像构建、清库入库、服务启停、日志排查，以及历次踩过的坑与解决方式。

## 1. 环境总览

| 组件 | 说明 |
|---|---|
| app 服务 | Flask，问答端口 `.env` 的 `PORT`（默认 6151），接口 `POST /chat`、`GET /health` |
| MySQL ×3 | **生产 = 自建 mysql 容器**（compose 自带 rag-forward-rookie-mysql，宿主机回环 127.0.0.1:6152 发布，root/rag_db）；本地 = **3306 宿主机自装 MySQL**、**3307 compose mysql 容器**（docker-compose.local.yml 自带）。三套数据互相独立，操作前先确认目标库！ |
| 检索 | FAISS HNSW + BM25 + 元数据 + SQL，RRF 融合，精排走主办方模型网关 BGE-Reranker（无本地模型） |
| 生成 | 主办方模型网关（http://180.76.46.222:8289/v1）：DeepSeek-V4-Flash（thinking 关闭，实测单跳 0.8-1.3s）+ PaddleOCR-VL 图片识别（冷启动约 16s，QWEN_TIMEOUT=90） |
| 索引文件 | `faiss_index.bin`：宿主机根目录一份（裸机跑用）、容器内 `/app/faiss_index.bin` 一份（容器跑用），**互不同步** |
| 数据卷 | compose mysql 数据卷 `mysql_data`，`down` 不丢数据，`down -v` 才会删库 |

关键文件：`.env`（配置/密钥）、`Dockerfile`、`docker-compose.yml`（生产）、`docker-compose.local.yml`（本地 MySQL 覆盖）、`entrypoint.sh`（启动管线）、`init.sql`（表结构）、`Data/`（语料）、`src/ingest/pipeline.py`（入库管线）。

**容器启动管线（entrypoint.sh 每次启动都会跑，幂等）**：等 MySQL（30×2s）→ 建库建表 → ingest（按文件哈希跳过已入库）→ 重建/追加 FAISS 索引 → 刷新 product_mentions → 启动 API。

## 2. 镜像构建（建立/重建）

```bash
# 国内网络（当前配置：镜像代理前缀 + 清华 pip 源）
docker compose -f docker-compose.yml -f docker-compose.local.yml build
# 海外网络：把 docker-compose.yml build.args 里的 BASE_IMAGE 代理前缀
# docker.1ms.run/ 去掉（改回 python:3.12-slim）
```

- **改代码后必须重建镜像并重建容器**（见 §4），否则容器跑的还是旧代码
- `.dockerignore` 已排除 `.venv`、`.env`（密钥不进镜像）、`test/`、`faiss_index.bin`（运行时自建）——不要把密钥或虚拟环境打进镜像
- 镜像不含任何本地模型（models/ 走挂载不进镜像）；embedding 可选本地（`BGE_EMBED_BACKEND=local`），重排序固定走 BGE 网关 API（本地 reranker 服务器实测不达标，见 §5.0）

## 3. 清库与重新入库

### 3.1 什么时候必须清库重入

ingest 幂等检测**只比对文件路径 + 内容哈希**（`documents.file_path/file_hash`）。以下情况改了代码也不会生效，必须清库重入：

- 修改了分块逻辑/参数（`src/ingest/chunker.py`、`config.py` 的 `CHILD_CHUNK_SIZE` 等）
- Data/ 换了一批同名文件（哈希变了会自动重入，不需要清库；但旧文件对应的旧文档会残留，需要清）
- 想彻底重建索引/文档

### 3.2 清库（以 3307 容器库为例）

生产库（服务器上）连接：`127.0.0.1:6152` root / .env 的 MYSQL_PASSWORD / 库 rag_db。

```sql
-- 连接：127.0.0.1:3307  root / ABC123  库 fuck_rag
SET FOREIGN_KEY_CHECKS=0;
DELETE FROM documents;         -- 级联清 chunks/parse_log/tables_structured/product_mentions
DELETE FROM schema_registry;   -- 该表外键为 SET NULL，不会级联，需单独清
SET FOREIGN_KEY_CHECKS=1;
```

（documents 与其余表有 `ON DELETE CASCADE`，但显式逐表清更保险，顺序如上。）

**同时必须处理 faiss_index.bin**，否则旧索引向量与 DB 新 chunk 的 vector_id 错位、检索结果全错：

- 容器内跑：清库后 `--force-recreate` 重建 app 容器（新 writable layer，无旧索引），entrypoint 会走"索引缺失 + 有 embedded chunk → 重置 pending → 全量重建"自愈
- 裸机跑：直接删除宿主机根目录的 `faiss_index.bin`

### 3.3 入库方式 A（推荐：容器内自动，与生产路径一致）

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml stop app   # 保留 mysql
# ... 执行 §3.2 清库 ...
docker compose -f docker-compose.yml -f docker-compose.local.yml build      # 新代码进镜像
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --force-recreate app
# entrypoint 自动：重新入库 120 文件 → BGE embedding（约 1900 chunk，几分钟）→ 建索引 → 起 API
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f app   # 观察进度
```

### 3.4 入库方式 B（裸机 venv，适合调试 ingest 代码）

```bash
# 目标为 3307 容器库时用环境变量覆盖 .env；3306 宿主机库则直接用 .env 默认值
MYSQL_PORT=3307 ./.venv/Scripts/python -m src.ingest.pipeline
```

### 3.5 入库验证

```sql
SELECT parse_status, COUNT(*) FROM documents GROUP BY parse_status;   -- 应全部 success
SELECT chunk_level, status, COUNT(*) FROM chunks GROUP BY chunk_level, status;  -- child 全 embedded
SELECT COUNT(*) FROM product_mentions;   -- 非 0
```

## 4. 启动 / 关闭 / 重启服务

```bash
# 本地开发（自带 MySQL 容器，首次启动会自动入库）：app 用 host 网络模式，经宿主机回环 127.0.0.1:3307 访问
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build

# 生产（自建 MySQL 容器 + 桥接网络）：mysql 发布宿主机回环 127.0.0.1:6152（仅回环），app 发布 0.0.0.0:6151
docker compose up -d --build

# 部署前清理（生产，防历史残留占端口）：
docker ps -a                 # 找旧容器（非 rag-forward-rookie-* 的）
docker rm -f <旧容器名>       # 删镜像杀不掉容器！必须 rm 容器本身
ss -tlnp | grep -E '6011|6151|6152'   # 确认目标端口没被旧进程/docker-proxy 孤儿占着

# 关闭：stop 保留容器；down 删除容器但保留 mysql_data 卷；down -v 连数据卷一起删（慎用）
docker compose -f docker-compose.yml -f docker-compose.local.yml stop app
docker compose -f docker-compose.yml -f docker-compose.local.yml down

# 重启（改代码后）：restart 用的是旧镜像！必须 build 后 force-recreate
docker compose -f docker-compose.yml -f docker-compose.local.yml build
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --force-recreate app
```

**日常检查**：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml ps            # 状态（healthcheck 30×10s）
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f app   # 实时日志
curl http://localhost:6151/health                                              # 健康检查
```

**冒烟测试**（git-bash 里 curl 发 JSON 中文容易被转义吞掉，直接用 Python）：

```bash
./.venv/Scripts/python -c "
import requests
r = requests.post('http://localhost:6151/chat', json={'question': '2024年中国经济热词有哪些？'}, timeout=120)
print(r.json()['answer'])"
```

**日志排查**：每条请求日志带 trace_id（如 `[req:56ecc011]`），按 id grep 可串起 重写→检索→RRF→精排→上下文→LLM 全链路；`LOG_LEVEL=DEBUG`（.env）输出候选明细。测试报告落盘在 `test/result/{层级}/`，每问一个 JSON。

## 5. 已踩过的坑与解决方式

0. **本地 BGE 模型模式（2026-09-27 新增，服务器实测后拆双开关）**：`BGE_EMBED_BACKEND` / `BGE_RERANK_BACKEND`（.env，独立控制）切换 embedding/reranker 为本地推理（bge-m3 + bge-reranker-v2-m3，CPU fp32，FlagEmbedding）。模型目录 `./models/`（compose 挂载 `/app/models`，不进镜像）；开发机测试用 `.venv-model`（独立 venv，装 torch CPU + FlagEmbedding，不污染 .venv）。**换 embedding 源必须重建索引**：本地与网关向量非同源，索引重建 = reset embedded → pending + 删 faiss_index.bin（容器 force-recreate 自动触发自愈）。规划器/生成 LLM 不受影响仍走网关。本地模式 ingest 并发降为 1。**服务器实测（48 核，2026-09-27）：编码 73-87ms 达标；精排全本地 20×500字=6.9s / 60=12s / 100=18.6s 远超 5s 判定线（网关 ~1.3s）→ 最终形态 = embedding 本地 + reranker 网关**。torch CPU 推理给多少线程吃多少核，共享宿主机 LOCAL_MODEL_THREADS 默认 24 勿拉满。
1. **本地 jina-reranker 重排序方案被否决（已完全回滚）**：CPU 推理一次精排要分钟级、镜像未配 GPU 透传。结论：本项目重排序固定走 BGE API，不要再往镜像里打包本地模型、不要装 torch/transformers（已从 requirements.txt 与 .venv 移除）。
2. **生成层"虚假拒答"**：上下文只覆盖文档部分条款时，LLM 以"资料不完整/未提供完整框架"为由拒答（prompt 规则 1+5 组合触发）。已修：规则 1 放宽为"部分信息也必须尽力作答，不得以资料不完整为由拒答"；并新增规则 5/7 约束列举/概述类每条解释三句话内。改 prompt 需重建镜像 + force-recreate 生效。
3. **幂等跳过陷阱**：只改 chunker 不清库，重跑 ingest 全部"跳过（已入库）"，新切分逻辑不生效（见 §3.1）。
4. **faiss 索引与 DB 错位**：清库不删索引（或容器不清 writable layer），旧向量配新 vector_id，检索全错（见 §3.2）。
5. **git-bash 的 curl 发 /chat**：JSON 里中文被转义吞掉、收到"请输入您的问题"，改用 Python requests 冒烟。
6. **Docker Desktop 未启动**：docker 命令报 `npipe:////./pipe/dockerDesktopLinuxEngine` 找不到，启动 Docker Desktop 即可。
7. **本机系统代理劫持 localhost**（Clash 127.0.0.1:7897 开着时）：httpx/openai SDK 访问 localhost 会被代理 502，本地 HTTP 服务必须 `trust_env=False` 直连。
8. **系统环境变量 `OLLAMA_MODEL` 被占用**：指向 E:/ollamaModel（模型存储路径），config.py 加载 .env 时系统 env 优先，故本地模型键名改用 `LOCAL_MODEL`。
9. **git-bash 里 `docker run -e VAR=/路径` 被 MSYS 转成 Windows 路径**：加前缀 `MSYS_NO_PATHCONV=1`。
10. **pytorch wheel 国内下载（历史，仅当再次需要 torch 时参考）**：阿里 pytorch-wheels 镜像已失效（404）；官方 CPU 索引直连 ~180KB/s 且易超时；走 Clash 代理（`curl -x http://127.0.0.1:7897`）可达 1.5MB/s。本地 wheel 用 pip 安装时文件名必须保留原始 wheel 名（pip 会校验文件名与包元数据）。
11. **删镜像杀不掉容器/端口**：运行中的容器不依赖磁盘镜像，`docker rmi` 无效，必须 `docker stop`/`docker rm -f` 容器本身。旧版 bridge 部署（ports 端口映射）的 docker-proxy 进程可能在容器被强杀后残留占端口（表现：容器删光端口仍被占）——`ss -tlnp | grep <端口>` 定位 PID 后 kill，或重启 docker daemon。历史部署未固定项目名时旧容器（如 ylsw_rag-app-1）不会被新目录的 `docker compose down` 清掉，需 `docker ps -a` 逐个确认。
12. **硅基流动 serverless 冷启动（首问 45s 之谜）**：实例冷池后首个请求要 40-60s 加载模型；旧 `API_TIMEOUT=30` 下 planner 首试 30s 超时 + 2s 退避 + 热重试 ≈ 35.8s，叠加生成 ≈ 首问总计 45s。已修：`API_TIMEOUT=90` + entrypoint 起 API 前预热 chat/embedding/rerank（best-effort，失败不阻断启动）。注意：冷池约 10-15 分钟无请求会再次冷却——预热只保证部署后立即测评快，评分若长时间间隔单发仍可能撞冷启动（但只会慢一次，不会超时重试浪费）。
13. **托管版 V4-Flash 年份错标（"2025年经济热词"飘逸）**：同代码同提示词 A/B 实测——DeepSeek 官方 API 版两次正确指出"资料仅含2024"；硅基流动托管版（主办方网关同源）两次把 2024 热词标成"2025年经济热词包括…"。原因：托管实例指令遵循弱（检查点/量化/服务参数差异），无视【时间核对规则】。已修：BASE_PROMPT 时间核对规则 2 强化为显式禁令+反例（"严禁出现'2025年…包括/有…'式错标"），修复后网关 6/6 不再错标。若主办方换网关模型版本需回归此题。
14. **托管版误拒答与语言规则漂移（2026-09-25 深度测试发现并修复）**：12 题深度测试揪出——(a) "2026年手续费标准/蓝鲸售后服务"两例整体拒答（实体重叠存在，违反拒答规则 1）；(b) "What is 全民付蓝鲸?" 混合中英问题被 detect_language 误判中文（CJK 占比阈值 0.3 太低）。修复：拒答处理规则 2 加"个别细节缺失必须先给已有内容再说明缺失项"+机械式禁止"抱歉，未找到…"整句话术；时间核对规则加"年份缺失不得整体拒答"；detect_language 阈值 0.3→0.5。回归：误拒答/语言规则/正确拒答/年份错标全部通过。
15. **HTTP 客户端重构（2026-09-25）**：新建 `src/http_client.py`——全局 requests.Session（TCP/TLS 复用，每问省 0.3-0.8s 握手）+ post_json 分类重试（400/401/403/404/422 不重试；429 读 Retry-After，缺失指数退避；5xx/网络异常 2**attempt 退避）。修掉了非 200 无退避 bug（此前三连 429 零间隔连打，日志 13:48:24.437/.787/.941 实锤）。7 个调用点已接入：generator._chat_completion/describe_image、planner、text_to_sql、fusion._call_reranker、faiss_search、pipeline.call_bge_embedding。网关 thinking 参数验证有效（流式 delta 无 reasoning_content）。
16. **"单步问题跳过精排" A/B 实验（2026-09-25，结论：否决）**：`RERANK_SKIP_SINGLE=1`（fusion.rerank 的 skip_api 参数）对单步问题直走 RRF 排序。16 道单文档题实测：B 侧实际跳过 7 题，**reranked 候选与开精排 0/7 一致**——精排把同文档相邻块顶上候选（如 Q03 开精排全 47xx-48xx 区域 vs 跳过混入 40xx 异文档），省 ~1.3s 但真丢精度。开关代码保留（默认 off）供后续实验；压候选数中间方案收益有限（rerank 耗时与文本量线性，真实长文档 ~1.3s，压到 15 条仅省 0.2-0.4s），未采纳。
17. **"规划器与检索并行化（投机预取）"实验（2026-09-25，结论：零收益已回滚）**：先用原问题预取四路检索与规划器 LLM 并发。26 题 7 类实测：质量零回归（注入/闲聊/计算/多文档全正常），但**规划器对几乎所有单步题都改写查询**（日志：检索预取命中 0 次 / 检索重跑 13 次），预取结果永远作废 → 零延迟收益（检索仅 150-550ms 且照旧串行）。已完全回滚为串行版。教训：规划器的单步改写是常态不是例外，任何"绕过规划器"的预取都需先验证改写率。
18. **planner foreach 断裂修复（2026-09-25，重大）**：(a) `_step_query_text` 曾提前把 foreach 占位符替换成 items[0]（逐项展开永远匹配不到，N 个 item 检索同一条查询）——已修为保留 foreach 占位符，负向前瞻清理其他残留；(b) `reject=bool(...)` 对 "false" 字符串误判 True 拒答正常问题——已严格类型收窄；(c) foreach 全空不标 skipped（下游拿空串检索噪声）——已标 skipped+写 summary+传 items；(d) `_call_llm` 响应解析裸奔——已加 try/except + finish_reason=length 截断检查；(e) 中间答案三连修：SQL 候选置顶（语料噪声曾把它挤出可见窗口）、items 在 summary 为空时存活、关键词式查询处理指令 + few-shot 示例（否则托管模型输出空 items 或把问题关键词当实体）。e2e 验证："运营公司邮箱"foreach 题全链路打通（中间答案提取真实公司名 → FOREACH 展开 4 条不同查询 → 无邮箱数据时正确回答"资料中未提供邮箱信息"）。多文档/单文档/注入回归全过。
19. **网关挂起 90s+（2026-09-27 实测）**：同题连问，规划器 LLM 时好时坏（快时 ~1s、坏时 92.4s = API_TIMEOUT 90s 超时 + 退避 + 重试成功）。精排/生成同批请求正常，仅规划器挂起。**主办方 30s 作答硬约束**下已全链路收紧：规划器两段式 `PLANNER_TIMEOUT=5` 短试 + `PLANNER_LONG_TIMEOUT=5` 救援（全失败回退类1不拒答）；精排 `RERANK_TIMEOUT=5` retries=1（挂起回退 RRF 顺序）；生成 `GENERATE_TIMEOUT=20` retries=1（失败=该题失败）。预算账：单点挂起 ≤25s；规划+精排双挂 5+5+5+2=17s。另加 **keepalive 线程**（app.py `_start_keepalive`，`KEEPALIVE_INTERVAL=480`）每 8 分钟 ping 在走网关的端点，防冷池 10-15 分钟再冷却（冷启动 40-60s 在 30s 预算下必失败，唯一解法是保温）。中间答案 LLM 同走 `_call_llm` 自动受保护。**陈旧连接根因**：卡点恒为每问首个网关调用（规划器），疑网关/LB 静默回收空闲连接导致池化连接半死——http_client 已改每请求 `Connection: close`（不再复用连接，代价 ~100-300ms TLS/次，覆盖 OPS #15 的池化收益，环境已变、可靠性优先）。判据：卡住请求日志若带 `HTTP 请求异常 (attempt 1/3)` 警告=超时+重试成功（陈旧连接或网关丢请求）；无警告=网关慢慢回（慢副本）。

## 6. 常用命令速查

| 操作 | 命令 |
|---|---|
| 构建镜像 | `docker compose -f docker-compose.yml -f docker-compose.local.yml build` |
| 启动（本地栈） | `docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build` |
| 启动（生产） | `docker compose up -d --build` |
| 重建容器（代码生效） | `... build && ... up -d --force-recreate app` |
| 查看状态 | `docker compose ... ps` |
| 实时日志 | `docker compose ... logs -f app` |
| 按请求查日志 | `docker compose ... logs app \| grep req:<trace_id>` |
| 清库重入（完整流程） | 见 §3.2 + §3.3 |
| 裸机入库 | `MYSQL_PORT=3307 ./.venv/Scripts/python -m src.ingest.pipeline` |
| 健康检查 | `curl http://localhost:6151/health` |
| 冒烟提问 | Python requests → `POST /chat {"question": "..."}` |
