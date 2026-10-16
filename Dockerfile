# BASE_IMAGE 由 compose build.args 传入镜像代理前缀（国内网络加速），海外网络可去掉
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Python 依赖（先装 faiss 等重依赖，利用 Docker 缓存；清华 pip 源，国内构建加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 本地 BGE 模型推理依赖（BGE_EMBED_BACKEND / BGE_RERANK_BACKEND =local 时启用）。
# torch 走官方 CPU 索引（Linux 轮子 ~200MB，pypi 通用轮 ~900MB）；
# transformers 4.44.2：FlagEmbedding 1.3.3 的依赖上限（>4.45 会冲突）
RUN pip install --no-cache-dir --timeout 300 \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 300 -i https://pypi.tuna.tsinghua.edu.cn/simple \
        transformers==4.44.2 FlagEmbedding==1.3.3 peft

# 代码
COPY src/ ./src/
COPY Data/ ./Data/
COPY init.sql .

# 入口：等待 MySQL → 按需 ingest → 按需重建索引 → 启动 API
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 6151

ENTRYPOINT ["./entrypoint.sh"]
