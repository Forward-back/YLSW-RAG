-- ============================================================
-- RAG 项目 MySQL 表结构初始化脚本（幂等，可重复执行）
-- 数据库由 entrypoint 按环境变量 MYSQL_DATABASE 自动创建，本文件只含建表语句
-- ============================================================

-- ============================================================
-- 1. documents — 文档级元数据
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    file_name       VARCHAR(500) NOT NULL,
    file_type       VARCHAR(20)  NOT NULL COMMENT 'pdf/docx/xlsx/txt/doc',
    file_size       BIGINT,
    file_path       VARCHAR(1000),

    -- 从文件名/内容提取的通用字段
    doc_number      VARCHAR(100)  COMMENT '文号，如银发2015325号',
    org_name        VARCHAR(200)  COMMENT '发布机构',
    doc_date        DATE          COMMENT '文档日期',
    doc_title       VARCHAR(500)  COMMENT '清洗后的标题',
    doc_summary     TEXT          COMMENT '文档要点摘要（综述类题上下文用，严格忠实原文）',
    doc_type        VARCHAR(20)   COMMENT '文档类型: regulation/manual/product/report/news/table/general',
    product_name    VARCHAR(200)  COMMENT '关联产品名，通用表示无特定产品',
    file_hash       VARCHAR(64)   COMMENT '文件 SHA256 哈希，用于幂等检测',

    -- 解析状态
    parse_status    ENUM('pending','parsing','success','failed') DEFAULT 'pending',
    parse_error     TEXT,
    chunk_count     INT DEFAULT 0,

    created_at      DATETIME DEFAULT NOW(),
    updated_at      DATETIME DEFAULT NOW() ON UPDATE NOW(),

    INDEX idx_file_type (file_type),
    INDEX idx_parse_status (parse_status),
    INDEX idx_doc_number (doc_number),
    INDEX idx_org_name (org_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 2. chunks — chunk 文本 + FAISS 映射 + Parent-Child 关联
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_id     INT NOT NULL,
    chunk_index     INT NOT NULL COMMENT '文档内排序序号',
    chunk_text      LONGTEXT NOT NULL,
    chunk_level     ENUM('child','parent') DEFAULT 'child',

    -- Parent-Child
    parent_chunk_id BIGINT COMMENT 'FK→chunks.id，NULL=自身为父chunk',

    -- FAISS 映射
    vector_id       BIGINT COMMENT 'FAISS HNSW 索引中的位置',

    -- 关键词（BGE-M3 sparse / BM25）
    keywords        TEXT,

    -- 溯源码
    position_start  INT COMMENT '原文起始偏移',
    position_end    INT COMMENT '原文终止偏移',

    -- 反范式化元数据（加速检索过滤，避免 JOIN）
    org_name        VARCHAR(200)  COMMENT '发布机构',
    doc_type        VARCHAR(20)   COMMENT '文档类型',
    product_name    VARCHAR(200)  COMMENT '关联产品名',

    -- 状态
    status          ENUM('pending','embedded','failed') DEFAULT 'pending',
    error_msg       TEXT,

    created_at      DATETIME DEFAULT NOW(),
    updated_at      DATETIME DEFAULT NOW() ON UPDATE NOW(),

    INDEX idx_doc (document_id),
    INDEX idx_parent (parent_chunk_id),
    INDEX idx_vector (vector_id),
    INDEX idx_level (chunk_level),
    INDEX idx_org (org_name),
    INDEX idx_doc_type (doc_type),
    INDEX idx_product (product_name),
    FULLTEXT idx_keywords (keywords),

    CONSTRAINT fk_chunks_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 3. tables_structured — XLSX/PDF表格的结构化行数据 (JSON)
-- ============================================================
CREATE TABLE IF NOT EXISTS tables_structured (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_id     INT NOT NULL,
    table_name      VARCHAR(200) NOT NULL,
    sheet_name      VARCHAR(200) COMMENT 'Excel sheet名 / PDF页码',
    row_index       INT NOT NULL COMMENT '行号，保留原始顺序',
    row_data        JSON NOT NULL COMMENT '{"列名":"值",...}',
    created_at      DATETIME DEFAULT NOW(),

    INDEX idx_doc_table (document_id, table_name),
    INDEX idx_table (table_name),

    CONSTRAINT fk_tables_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 4. schema_registry — 结构化表的列信息注册（供 Text-to-SQL 使用）
-- ============================================================
CREATE TABLE IF NOT EXISTS schema_registry (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    table_name      VARCHAR(200) NOT NULL,
    column_name     VARCHAR(200) NOT NULL,
    column_type     ENUM('number','string','date','unknown') DEFAULT 'unknown',
    sample_values   VARCHAR(500) COMMENT '示例值，帮助LLM理解语义',
    alias           VARCHAR(500) COMMENT '同义词，竖线分隔',
    document_id     INT,
    updated_at      DATETIME DEFAULT NOW(),

    UNIQUE KEY uk_table_col (table_name, column_name),
    INDEX idx_table (table_name),

    CONSTRAINT fk_schema_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 5. parse_log — 文件解析管线日志
-- ============================================================
CREATE TABLE IF NOT EXISTS parse_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_id     INT NOT NULL,
    stage           ENUM('parsing','chunking','embedding','indexing','done') NOT NULL,
    status          ENUM('running','success','failed') DEFAULT 'running',
    message         TEXT,
    started_at      DATETIME DEFAULT NOW(),
    finished_at     DATETIME,

    INDEX idx_doc (document_id),
    INDEX idx_status (status),

    CONSTRAINT fk_log_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 6. product_mentions — 产品提及倒排索引
-- ============================================================
CREATE TABLE IF NOT EXISTS product_mentions (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_name    VARCHAR(200) NOT NULL,
    chunk_id        BIGINT NOT NULL,
    document_id     INT NOT NULL,
    created_at      DATETIME DEFAULT NOW(),

    UNIQUE KEY uk_product_chunk (product_name, chunk_id),
    INDEX idx_product (product_name),
    INDEX idx_chunk (chunk_id),
    INDEX idx_doc (document_id),

    CONSTRAINT fk_pm_chunk FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    CONSTRAINT fk_pm_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 7. 增量迁移：Phase 2 新增字段（已有数据库执行）
-- ============================================================
-- ALTER TABLE documents ADD COLUMN doc_type VARCHAR(20) COMMENT '文档类型' AFTER doc_title;
-- ALTER TABLE documents ADD COLUMN product_name VARCHAR(200) COMMENT '关联产品名' AFTER doc_type;
-- ALTER TABLE chunks ADD COLUMN org_name VARCHAR(200) COMMENT '发布机构' AFTER position_end;
-- ALTER TABLE chunks ADD COLUMN doc_type VARCHAR(20) COMMENT '文档类型' AFTER org_name;
-- ALTER TABLE chunks ADD COLUMN product_name VARCHAR(200) COMMENT '关联产品名' AFTER doc_type;
-- ALTER TABLE chunks ADD INDEX idx_org (org_name);
-- ALTER TABLE chunks ADD INDEX idx_doc_type (doc_type);
-- ALTER TABLE chunks ADD INDEX idx_product (product_name);
