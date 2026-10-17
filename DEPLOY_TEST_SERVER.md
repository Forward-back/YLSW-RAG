# 服务器部署手册（2026-10-16 v3）

覆盖：部署形态、交付物打包、服务器从零部署、API 通道配置、索引重建、验证与回滚。

## 0. 部署形态（两镜像）

服务器上由 **两个镜像** 组成完整部署（docker-compose.yml 定义两个 service）：

| 镜像 | 来源 | 内容 |
|---|---|---|
| app 镜像（自建） | 本机 `docker build` 后 `docker save` 成 tar 上传 | 全部代码 + torch CPU + FlagEmbedding（本地 BGE 可选）；**models/ 不进镜像**，运行时挂载 `./models:/app/models` |
| MySQL 镜像（官方） | 服务器 `docker compose up` 时自动拉取 `mysql:8.4.11` | 数据库本体，数据落 `mysql_data` 数据卷 |

要点：

- **数据库不在 app 镜像里**：MySQL 独立容器 + 命名卷持久化。`docker compose down` 不丢数据，`down -v` 才删库。
- **端口**：问答 6151（发布 0.0.0.0 供评分方）、MySQL 6152（仅宿主机回环 127.0.0.1，不对外）。
- **启动管线**（entrypoint.sh，每次启动幂等）：等 MySQL → 建库建表 → ingest（文件哈希跳过）→ FAISS 索引自愈重建 → product_mentions 刷新 → 模型预热 → 起 API。
- app 容器走 compose 桥接网络连本栈 `mysql:3306`，不依赖宿主机回环。

## 1. 交付物打包（本机执行，优化完成后重新打包）

```bash
# 1) app 镜像 tar（含最新代码；改过代码必须重新打包）
docker compose build
docker save rag-forward-rookie-app -o rag_app_local.tar

# 2) models 目录 tar（bge-m3 + bge-reranker-v2-m3；仅当本地模型文件变更才需重打）
tar -czf rag_models.tar.gz -C models .

# 3) 同步到服务器（连同 compose 与 .env 模板）
scp rag_app_local.tar rag_models.tar.gz docker-compose.yml user@<服务器IP>:~/rag_project/
```

注意：`.env` 不进镜像、不入 git，服务器上的 .env 手工准备（见 §3）。

## 2. 服务器从零部署清单

```bash
# ① 环境：装 docker + compose 插件，确认与主办方分配端口一致（6151/6152 未被占）
docker --version && docker compose version
ss -tlnp | grep -E '6151|6152'   # 应无监听

# ② 目录布局
mkdir -p ~/rag_project && cd ~/rag_project
# 上传：docker-compose.yml、rag_app_local.tar、rag_models.tar.gz

# ③ 加载镜像与模型
docker load -i rag_app_local.tar
mkdir -p models && tar -xzf rag_models.tar.gz -C models   # tar 顶层无 models/ 前缀，必须 -C

# ④ 准备 .env（在 ~/rag_project/ 下，内容见 §3）

# ⑤ 起服务
docker compose up -d --no-build
docker compose logs -f app   # 观察：等 MySQL → ingest（首次全量，数分钟）→ 索引构建 → 预热 → API 就绪

# ⑥ 健康检查
curl http://localhost:6151/health   # {"status":"ok"}
```

## 3. .env 配置（服务器）

当前 API 通道（2026-10-16 起，不用主办方网关）：

```ini
# ── MySQL（compose 插值注入；app 容器内被覆盖为 mysql:3306）──
MYSQL_HOST=127.0.0.1
MYSQL_PORT=6152
MYSQL_USER=root
MYSQL_PASSWORD=<强密码，自定>
MYSQL_DATABASE=rag_db

PORT=6151

# ── BGE：硅基流动官方 API ──
BGE_API_KEY=<用户硅基流动 key>
BGE_EMBEDDING_URL=https://api.siliconflow.cn/v1/embeddings
BGE_RERANKER_URL=https://api.siliconflow.cn/v1/rerank

# ── LLM：DeepSeek 官方 API ──
DEEPSEEK_API_KEY=<用户 DeepSeek key>
DEEPSEEK_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-v4-flash

# ── 多模态图片理解（ingest 用，仅当需要 DOCX 内嵌图片描述）──
QWEN_API_KEY=<key>
QWEN_URL=<URL>
QWEN_MODEL=PaddlePaddle/PaddleOCR-VL-1.5

# ── 30s 硬约束超时预算 ──
PLANNER_TIMEOUT=5
PLANNER_LONG_TIMEOUT=5
RERANK_TIMEOUT=5
GENERATE_TIMEOUT=20
KEEPALIVE_INTERVAL=480
API_TIMEOUT=90

# ── BGE 后端开关（gateway=走 API；local=本地推理，需 models 挂载）──
BGE_EMBED_BACKEND=gateway
BGE_RERANK_BACKEND=gateway
LOCAL_MODEL_THREADS=24
```

## 4. 索引重建注意事项（重要）

换 embedding 源（如 gateway ↔ local、不同 API 平台）后，DB 里旧向量与新查询向量**非同源**，检索语义路全部失效（自检方式：重嵌任一已知 chunk 检索 top1 应为自身，`self_hit=NO` 即不同源）。

服务器上重建步骤：

```bash
# ① 连接宿主机回环 MySQL（127.0.0.1:6152，root / .env 密码 / rag_db）
UPDATE chunks SET status='pending', vector_id=NULL WHERE chunk_level='child' AND status='embedded';

# ② 删容器内旧索引并重建容器（新 writable layer 无索引，entrypoint 自愈全量重嵌）
docker compose up -d --no-build --force-recreate app
docker compose logs -f app   # 观察 "embedding 进度: x/总数"，全量重嵌后自建索引
```

本机 3306 开发库已于 2026-10-16 用硅基向量重建（1881 chunk / 42s），服务器库若尚未切新通道，部署时必须执行上述重建。

## 5. 文档摘要回填（2026-10-17 新增）

综述类跨文档题的上下文依赖 `documents.doc_summary` 列（检索不参与，组装时前置）。服务器库首部署后回填一次（118 篇约 6-8 分钟，幂等：已有摘要跳过）：

```bash
docker exec rag-forward-rookie-app python -u -m src.ingest.summarizer
```

## 6. 验证与回滚

```bash
# 端到端延迟（容器内或宿主机跑 test）
python src/server_bench.py e2e   # 三问覆盖单步/并行子问/多跳 foreach，总耗时应 < 30s

# 模型基准（容器内）
docker exec rag-forward-rookie-app python -u src/server_bench.py models

# 回滚到旧镜像（若保留了旧 tar）
docker load -i rag_app_local.old.tar && docker compose up -d --no-build --force-recreate app
```

## 6. 常见坑（详见 OPS.md §5）

- 改代码必须重新 build + save + load，`restart` 用的是旧镜像
- 清库不删索引（或容器不清 writable layer）→ 向量与 DB 错位、检索全错
- 改分块参数不清库 → ingest 哈希跳过、新切分不生效
- 网关/API 冷启动：靠 entrypoint 预热 + app 内 keepalive 线程保温
