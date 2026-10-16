"""
文档解析入库管线主控。扫描 Data/ 目录 → 解析 → 分块 → 入库 → 建索引。
"""

import os
import sys
import re
import time
import json
import hashlib
import logging
import tempfile
import shutil
import requests
import pymysql
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import _CN_DIGIT, setup_windows_console
from http_client import post_json
from config import (
    MYSQL_CONFIG, DATA_DIR, BGE_EMBEDDING_URL, BGE_API_KEY,
    EMBEDDING_DIM, HNSW_M, API_TIMEOUT, API_MAX_RETRIES,
    QWEN_TIMEOUT, QWEN_CONCURRENCY, QWEN_VL_IMAGE_MIME_TYPES,
    ENABLE_IMAGE_DESCRIPTION, BGE_EMBED_BACKEND,
)
from ingest.metadata import extract_metadata
from ingest.parsers import parse_file, TableData
from ingest.chunker import chunk_file, Chunk
from ingest.paddle_ocr import get_paddle_processor
from retrieval.query_filter import build_product_mentions
from retrieval.text_to_sql import COLUMN_ALIASES

setup_windows_console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


# ═══════════════════════════════════════════════════════
# 数据库操作
# ═══════════════════════════════════════════════════════

class DB:
    """MySQL 连接管理。"""

    def __init__(self):
        self.conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
            charset=MYSQL_CONFIG["charset"],
            autocommit=False,
        )

    def file_already_ingested(self, file_path: str) -> bool:
        """检查文件是否已成功入库（按路径 + 内容哈希）。失败记录不算，允许重试。"""
        file_hash = _compute_file_hash(file_path) if os.path.exists(file_path) else None
        if not file_hash:
            return False
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE file_path = %s AND file_hash = %s AND parse_status != 'failed'",
                (os.path.relpath(file_path, os.path.dirname(DATA_DIR)), file_hash),
            )
            return cur.fetchone()[0] > 0

    def remove_failed_document(self, file_path: str) -> list[int]:
        """清除历史失败记录（document + parse_log + 残留 chunks），返回被删 doc_id。"""
        rel = os.path.relpath(file_path, os.path.dirname(DATA_DIR))
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE file_path = %s AND parse_status = 'failed'",
                (rel,),
            )
            ids = [r[0] for r in cur.fetchall()]
            for did in ids:
                cur.execute("DELETE FROM parse_log WHERE document_id = %s", (did,))
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (did,))
                cur.execute("DELETE FROM documents WHERE id = %s", (did,))
            self.conn.commit()
            return ids

    def insert_document(self, file_name: str, file_type: str, file_size: int,
                        file_path: str, doc_number: str, org_name: str,
                        doc_date_str: str, doc_title: str,
                        doc_type: str = "", product_name: str = "") -> int:
        sql = """
            INSERT INTO documents (file_name, file_type, file_size, file_path,
                                   file_hash, doc_number, org_name, doc_date, doc_title,
                                   doc_type, product_name, parse_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'parsing')
        """
        file_hash = _compute_file_hash(os.path.join(
            os.path.dirname(DATA_DIR), file_path
        )) if os.path.exists(os.path.join(os.path.dirname(DATA_DIR), file_path)) else ""
        try:
            parsed_date = _parse_date(doc_date_str) if doc_date_str else None
        except Exception:
            parsed_date = None
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                file_name, file_type, file_size, file_path,
                file_hash, doc_number, org_name, parsed_date, doc_title,
                doc_type, product_name,
            ))
            self.conn.commit()
            return cur.lastrowid

    def update_document_status(self, doc_id: int, status: str, error: str = None,
                               chunk_count: int = None):
        sql = "UPDATE documents SET parse_status=%s, parse_error=%s"
        params = [status, error]
        if chunk_count is not None:
            sql += ", chunk_count=%s"
            params.append(chunk_count)
        sql += " WHERE id=%s"
        params.append(doc_id)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            self.conn.commit()

    def insert_chunk(self, document_id: int, chunk_index: int, chunk_text: str,
                     chunk_level: str, parent_chunk_id: int = None,
                     keywords: str = "", pos_start: int = 0, pos_end: int = 0,
                     org_name: str = "", doc_type: str = "",
                     product_name: str = "") -> int:
        sql = """
            INSERT INTO chunks (document_id, chunk_index, chunk_text, chunk_level,
                                parent_chunk_id, keywords, position_start, position_end,
                                org_name, doc_type, product_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                document_id, chunk_index, chunk_text, chunk_level,
                parent_chunk_id, keywords, pos_start, pos_end,
                org_name, doc_type, product_name,
            ))
            return cur.lastrowid

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def update_chunk_vector_id(self, chunk_id: int, vector_id: int):
        sql = "UPDATE chunks SET vector_id=%s, status='embedded' WHERE id=%s"
        with self.conn.cursor() as cur:
            cur.execute(sql, (vector_id, chunk_id))

    def batch_update_vector_ids(self, id_pairs: list[tuple[int, int]]):
        """批量更新 vector_id，一条 SQL 完成。"""
        sql = "UPDATE chunks SET vector_id = CASE id " \
              + " ".join(f"WHEN {cid} THEN {vid}" for cid, vid in id_pairs) \
              + " END, status = 'embedded' WHERE id IN (" \
              + ",".join(str(cid) for cid, _ in id_pairs) + ")"
        with self.conn.cursor() as cur:
            cur.execute(sql)
            self.conn.commit()

    def insert_table_data(self, document_id: int, table_name: str,
                          sheet_name: str, rows: list):
        sql = """
            INSERT INTO tables_structured (document_id, table_name, sheet_name,
                                           row_index, row_data)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            for idx, row in enumerate(rows):
                cur.execute(sql, (
                    document_id, table_name, sheet_name, idx,
                    json.dumps(row, ensure_ascii=False),
                ))
            self.conn.commit()

    def insert_schema_registry(self, table_name: str, columns: list,
                               sample_rows: list, document_id: int):
        sql = """
            INSERT INTO schema_registry (table_name, column_name, column_type,
                                         sample_values, alias, document_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                sample_values = VALUES(sample_values),
                alias = VALUES(alias),
                updated_at = NOW()
        """
        # 别名单一事实源：retrieval.text_to_sql.COLUMN_ALIASES（表键 → 列名 → 别名列表）
        key = next(
            (k for k in COLUMN_ALIASES if k in table_name or table_name in k),
            None,
        )
        col_aliases = COLUMN_ALIASES.get(key, {}) if key else {}
        with self.conn.cursor() as cur:
            for col in columns:
                col_type = _infer_column_type(col, sample_rows)
                samples = _get_sample_values(col, sample_rows, n=3)
                alias = "、".join(col_aliases.get(col, []))
                cur.execute(sql, (
                    table_name, col, col_type, samples, alias, document_id,
                ))
            self.conn.commit()

    def insert_parse_log(self, document_id: int, stage: str,
                         status: str = "running", message: str = ""):
        sql = """
            INSERT INTO parse_log (document_id, stage, status, message)
            VALUES (%s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (document_id, stage, status, message))
            self.conn.commit()
            return cur.lastrowid

    def update_parse_log(self, log_id: int, status: str, message: str = ""):
        sql = "UPDATE parse_log SET status=%s, message=%s, finished_at=NOW() WHERE id=%s"
        with self.conn.cursor() as cur:
            cur.execute(sql, (status, message, log_id))
            self.conn.commit()

    def get_all_child_chunks(self) -> list:
        sql = ("SELECT id, chunk_text, document_id FROM chunks "
               "WHERE chunk_level='child' AND status='pending'")
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def count_embedded_chunks(self) -> int:
        sql = "SELECT COUNT(*) FROM chunks WHERE chunk_level='child' AND status='embedded'"
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()[0]

    def reset_embedded_to_pending(self):
        """索引文件丢失时自愈：已嵌入 chunk 重置为 pending，等待全量重建。"""
        sql = ("UPDATE chunks SET status='pending', vector_id=NULL "
               "WHERE chunk_level='child' AND status='embedded'")
        with self.conn.cursor() as cur:
            cur.execute(sql)
            self.conn.commit()

    def close(self):
        self.conn.close()


# ── 辅助 ───────────────────────────────────────────────

def _compute_file_hash(file_path: str) -> str:
    """计算文件 SHA256 哈希（用于幂等检测）。"""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _parse_date(date_str: str):
    """尝试将日期字符串解析为 date 对象。"""
    if not date_str or date_str == "00000000":
        return None
    m = re.match(r"(\d{4})\s*年\s*第?\s*([\d一二三四五六七八九十]{1,2})\s*季度", date_str)
    if m:
        y = int(m.group(1))
        q_str = m.group(2)
        for c, a in _CN_DIGIT.items():
            q_str = q_str.replace(c, a)
        q = int(q_str)
        return datetime(y, (q - 1) * 3 + 1, 1).date()
    m = re.match(r"(\d{4})\s*年\s*([\d一二三四五六七八九十]{1,2})\s*月", date_str)
    if m:
        y = int(m.group(1))
        m_str = m.group(2)
        for c, a in _CN_DIGIT.items():
            m_str = m_str.replace(c, a)
        return datetime(y, int(m_str), 1).date()
    m = re.match(r"(\d{8})", date_str)
    if m:
        s = m.group(1)
        y, mo, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        if y == 0 or mo == 0 or mo > 12 or d == 0 or d > 31:
            return None
        return datetime(y, mo, d).date()
    m = re.match(r"(\d{4})(\d{2})00", date_str)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if y == 0 or mo == 0 or mo > 12:
            return None
        return datetime(y, mo, 1).date()
    m = re.match(r"(\d{4})0000", date_str)
    if m:
        y = int(m.group(1))
        if y == 0:
            return None
        return datetime(y, 1, 1).date()
    return None


def _infer_column_type(column_name: str, sample_rows: list) -> str:
    """推断列类型：number / string / date。

    数字串优先判 number；仅当值形如日期（YYYY-MM-DD、YYYY/MM/DD、YYYY年M月D日、8 位数字）
    才判 date——纯长数字串（账号等）不是日期。
    """
    date_like = re.compile(
        r"^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?$|^\d{8}$"
    )
    for row in sample_rows:
        val = str(row.get(column_name, "")).strip()
        if not val:
            continue
        try:
            float(val.replace(",", "").replace("元", "").replace("%", ""))
            return "number"
        except ValueError:
            pass
        if date_like.fullmatch(val):
            return "date"
        return "string"
    return "unknown"


def _get_sample_values(column_name: str, sample_rows: list, n: int = 3) -> str:
    values = [str(r.get(column_name, "")) for r in sample_rows[:n] if r.get(column_name, "")]
    return ", ".join(values[:n])


def _extract_and_describe_images(file_path: str, file_type: str) -> list[str]:
    """从 DOCX 提取图片并调用千问 VL 生成描述。

    - 用 PIL 检测图片真实格式，仅 PNG/JPEG 发送给 VL API（WMF/EMF 无法解码，跳过）
    - 并发调用 + 短超时，避免单张图片拖慢整个文件 ingest
    """
    descriptions = []
    if file_type != "docx":
        return descriptions

    try:
        import base64
        import io
        from PIL import Image
        from docx import Document
        from llm.generator import describe_image

        images = []
        skipped = 0
        d = Document(file_path)
        for rel in d.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                image_data = rel.target_part.blob
                with Image.open(io.BytesIO(image_data)) as img:
                    mime = Image.MIME.get(img.format, "")
                if mime not in QWEN_VL_IMAGE_MIME_TYPES:
                    skipped += 1
                    logger.info(f"  跳过不支持格式图片: {img.format}")
                    continue
                images.append((base64.b64encode(image_data).decode("utf-8"), mime))
            except Exception:
                # PIL 无法识别的格式（如 WMF/EMF）直接跳过
                skipped += 1

        if skipped:
            logger.info(f"  跳过 {skipped} 张无法识别/不支持的图片")

        if not images:
            return descriptions

        logger.info(f"  {len(images)} 张图片待 VL 描述，{QWEN_CONCURRENCY} 线程并发...")
        with ThreadPoolExecutor(max_workers=QWEN_CONCURRENCY) as executor:
            futures = [
                executor.submit(describe_image, b64, mime, QWEN_TIMEOUT)
                for b64, mime in images
            ]
            for i, future in enumerate(futures):
                try:
                    desc = future.result()
                    if desc:
                        descriptions.append(desc)
                except Exception as e:
                    logger.warning(f"  图片 {i+1} 描述失败: {e}")
    except Exception as e:
        logger.warning(f"  图片提取失败: {e}")
    return descriptions


# ═══════════════════════════════════════════════════════
# BGE Embedding API
# ═══════════════════════════════════════════════════════

def call_bge_embedding(texts: list[str]) -> list[list[float]] | None:
    """批量获取 BGE-M3 embedding（网关 API 或本地模型，BGE_EMBED_BACKEND 控制）。"""
    if BGE_EMBED_BACKEND == "local":
        from models_singleton import encode_texts
        return encode_texts(texts, EMBEDDING_DIM)
    resp, err = post_json(
        BGE_EMBEDDING_URL,
        {"input": texts, "model": "BAAI/bge-m3"},
        {
            "Authorization": f"Bearer {BGE_API_KEY}",
            "Content-Type": "application/json",
        },
        API_TIMEOUT,
    )
    if resp is None:
        logger.warning(f"BGE API 调用失败: {err}")
        return None
    if resp.status_code != 200:
        logger.warning(f"BGE API HTTP {resp.status_code}: {resp.text[:150]}")
        return None
    data = resp.json()
    return [item["embedding"][:EMBEDDING_DIM] for item in data["data"]]


EMBEDDING_MIN_CHARS = 20  # BGE-M3 拒绝过短文本，逐条降级时跳过


def _embed_single_fallback(
    items: list[tuple[int, str, int]],
) -> tuple[list[tuple[int, list[float]]], list[tuple[int, int, str]]]:
    """批量调用失败后逐条降级。

    items: [(chunk_id, text, document_id), ...]
    返回: (ok_pairs, failed_pairs)
      ok_pairs: [(chunk_id, embedding), ...]
      failed_pairs: [(chunk_id, document_id, reason), ...]
    """
    ok_pairs: list[tuple[int, list[float]]] = []
    failed_pairs: list[tuple[int, int, str]] = []

    def _one(item):
        cid, text, did = item
        if len(text.strip()) < EMBEDDING_MIN_CHARS:
            return "fail", (cid, did, f"文本过短(<{EMBEDDING_MIN_CHARS}字符)")
        emb = call_bge_embedding([text])
        if emb is not None:
            return "ok", (cid, emb[0])
        return "fail", (cid, did, "API 失败")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for kind, payload in executor.map(_one, items):
            if kind == "ok":
                ok_pairs.append(payload)
            else:
                failed_pairs.append(payload)
    return ok_pairs, failed_pairs


def _log_embedding_failures(db: DB, failed_pairs: list[tuple[int, int, str]]):
    """embedding 失败的 chunk 按 document_id 分组写入 parse_log。"""
    by_doc: dict[int, list[str]] = {}
    for cid, did, reason in failed_pairs:
        by_doc.setdefault(did, []).append(f"chunk {cid}: {reason}")
    for did, msgs in by_doc.items():
        detail = "; ".join(msgs[:10])
        if len(msgs) > 10:
            detail += f"; ...共 {len(msgs)} 条"
        try:
            db.insert_parse_log(did, "embedding", "failed",
                                f"{len(msgs)} 个 chunk embedding 失败: {detail}")
        except Exception as e:
            logger.warning(f"parse_log 写入失败: {e}")


# ═══════════════════════════════════════════════════════
# 单文件处理（用于并行）
# ═══════════════════════════════════════════════════════

def _process_single_file(file_path: str) -> dict:
    """处理单个文件，返回状态。每个线程独立创建 DB 连接。"""
    fname = os.path.basename(file_path)
    ext = os.path.splitext(fname)[1].lower().lstrip(".")
    fsize = os.path.getsize(file_path)
    fpath_rel = os.path.relpath(file_path, os.path.dirname(DATA_DIR))

    db = DB()

    try:
        # 幂等检查（历史失败记录允许重试）
        if db.file_already_ingested(file_path):
            logger.info(f"跳过（已入库）: {fname}")
            return {"file": fname, "status": "skipped"}

        removed = db.remove_failed_document(file_path)
        if removed:
            logger.info(f"清除历史失败记录后重试: {fname} (doc_id={removed})")

        logger.info(f"处理: {fname}")
        meta = extract_metadata(fname)

        doc_id = db.insert_document(
            file_name=fname, file_type=ext, file_size=fsize,
            file_path=fpath_rel, doc_number=meta.doc_number,
            org_name=meta.org_name, doc_date_str=meta.doc_date_str,
            doc_title=meta.doc_title, doc_type=meta.doc_type,
            product_name=meta.product_name,
        )

        log_id = db.insert_parse_log(doc_id, "parsing", "running")

        # PDF：先走标准解析（fitz/pdfplumber），文本太短则判定为扫描件走 PaddleOCR
        if ext == "pdf":
            parsed = parse_file(file_path)
            if len(parsed.full_text.strip()) < 100:
                logger.info(f"  PDF 文本量小，判定为扫描件，启动 PaddleOCR: {fname}")
                try:
                    paddle = get_paddle_processor()
                    ocr_text = paddle.extract_text(file_path)
                    if ocr_text.strip():
                        parsed.full_text = ocr_text
                        parsed.parse_error = None
                        logger.info(f"  PaddleOCR 提取: {len(ocr_text)} 字符")
                except Exception as e:
                    logger.warning(f"  PaddleOCR 失败，使用标准解析结果: {e}")
        else:
            parsed = parse_file(file_path)

        if parsed.parse_error:
            logger.error(f"  解析失败: {parsed.parse_error}")
            db.update_document_status(doc_id, "failed", parsed.parse_error)
            db.update_parse_log(log_id, "failed", parsed.parse_error)
            return {"file": fname, "status": "failed", "error": parsed.parse_error}

        # 图片理解：DOCX 提取图片 + 千问 VL 描述（MIME 过滤 + 并发，可用 ENABLE_IMAGE_DESCRIPTION=0 关闭）
        if ext == "docx" and ENABLE_IMAGE_DESCRIPTION:
            try:
                img_descs = _extract_and_describe_images(file_path, ext)
                if img_descs:
                    parsed.full_text += "\n\n## 图片描述\n" + "\n".join(
                        f"[图片{i+1}]: {d}" for i, d in enumerate(img_descs))
                    logger.info(f"  图片描述: {len(img_descs)} 张")
            except Exception as e:
                logger.warning(f"  图片提取跳过: {e}")

        db.update_parse_log(log_id, "success", f"文本 {len(parsed.full_text)} 字符")
        logger.info(f"  解析完成，文本 {len(parsed.full_text)} 字符，表格 {len(parsed.tables)} 个")

        # 表格入库
        for table in parsed.tables:
            if table.rows:
                db.insert_table_data(doc_id, table.table_name, table.sheet_name, table.rows)
                db.insert_schema_registry(table.table_name, table.columns, table.rows[:5], doc_id)
                logger.info(f"  表格 {table.table_name}: {len(table.rows)} 行 x {len(table.columns)} 列")
            elif table.markdown:
                parsed.full_text += f"\n\n## {table.table_name}\n{table.markdown}\n"

        # Chunking
        db.insert_parse_log(doc_id, "chunking", "running")
        table_mds = [t.markdown for t in parsed.tables if t.markdown]
        chunks = chunk_file(parsed.full_text, fname, table_mds)
        logger.info(f"  分块完成: {len(chunks)} chunks")

        db.insert_parse_log(doc_id, "chunking", "success", f"{len(chunks)} chunks")

        # 存 chunk：先 child 后 parent
        child_obj_to_db_id: dict = {}
        chunk_idx = 0
        for chunk in chunks:
            if chunk.level == "child":
                chunk_db_id = db.insert_chunk(
                    document_id=doc_id, chunk_index=chunk_idx,
                    chunk_text=chunk.text, chunk_level="child",
                    keywords=chunk.keywords, pos_start=chunk.position_start,
                    pos_end=chunk.position_end, org_name=meta.org_name,
                    doc_type=meta.doc_type, product_name=meta.product_name,
                )
                child_obj_to_db_id[id(chunk)] = chunk_db_id
                chunk_idx += 1

        for chunk in chunks:
            if chunk.level == "parent":
                first_child_db_id = None
                if chunk.child_chunks:
                    first_child_db_id = child_obj_to_db_id.get(id(chunk.child_chunks[0]))
                db.insert_chunk(
                    document_id=doc_id, chunk_index=chunk_idx,
                    chunk_text=chunk.text, chunk_level="parent",
                    parent_chunk_id=first_child_db_id,
                    keywords=chunk.keywords, pos_start=chunk.position_start,
                    pos_end=chunk.position_end, org_name=meta.org_name,
                    doc_type=meta.doc_type, product_name=meta.product_name,
                )
                chunk_idx += 1

        db.commit()
        db.update_document_status(doc_id, "success", chunk_count=len(chunks))
        db.insert_parse_log(doc_id, "done", "success", "入库完成")
        logger.info(f"  {fname} 处理完成 ✓")
        return {"file": fname, "status": "success"}

    except Exception as e:
        logger.error(f"  {fname} 处理异常: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass
        return {"file": fname, "status": "error", "error": str(e)}
    finally:
        db.close()


# ═══════════════════════════════════════════════════════
# 主管线
# ═══════════════════════════════════════════════════════

def run_ingest():
    """主入口：扫描 Data/ → 并行解析 → 分块 → 入库 → 建 FAISS 索引。"""
    logger.info("===== 开始文档解析入库管线 =====")

    supported_exts = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".doc"}
    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_exts:
                files.append(os.path.join(root, fname))

    logger.info(f"发现 {len(files)} 个待处理文件")

    # 并行处理文件
    max_workers = min(4, max(1, len(files)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, fp): fp for fp in files}
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "error":
                logger.error(f"  失败: {result['file']} - {result.get('error', '')}")

    # FAISS 索引构建
    db = DB()
    try:
        logger.info("===== 开始 BGE Embedding + FAISS 索引构建 =====")
        _build_faiss_index(db)

        # 构建产品提及倒排索引
        logger.info("===== 构建 product_mentions 倒排索引 =====")
        build_product_mentions(db.conn)

        logger.info("===== Ingest 完成 =====")
    finally:
        db.close()


def _load_faiss_index(index_path: str):
    """加载已有 FAISS 索引（先读 bytes 再走临时文件，绕过中文路径）。不存在返回 None。"""
    import faiss
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        index_bytes = f.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.close()
    try:
        with open(tmp.name, "wb") as f:
            f.write(index_bytes)
        return faiss.read_index(tmp.name)
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def _build_faiss_index(db: DB):
    """从 MySQL 读取 pending child chunks，并发调 BGE API，增量构建 FAISS HNSW 索引。

    增量策略：已有索引文件则加载并追加新向量（vector_id 从 index.ntotal 起），
    避免覆盖旧索引导致已入库 chunk 的 vector_id 错位。
    """
    import numpy as np
    import faiss

    index_dir = os.path.dirname(DATA_DIR)
    index_path = os.path.join(index_dir, "faiss_index.bin")

    index = _load_faiss_index(index_path)
    if index is None:
        if db.count_embedded_chunks() > 0:
            logger.warning("索引文件缺失但存在已嵌入 chunk，重置为 pending 后全量重建")
            db.reset_embedded_to_pending()
        index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M)
        index.hnsw.efConstruction = 200
        base = 0
    else:
        base = index.ntotal
        logger.info(f"加载已有索引: {base} 个向量，新 chunk 将追加其后")

    chunks = db.get_all_child_chunks()
    if not chunks:
        logger.warning("没有待处理的 child chunks")
        return

    logger.info(f"共 {len(chunks)} 个 child chunks 待 embedding")

    batch_size = 128
    batches = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batches.append((
            [c[1] for c in batch],   # texts
            [c[0] for c in batch],   # chunk ids
            [c[2] for c in batch],   # document ids
        ))

    # 本地模式 torch 已吃满线程，并发批只会争抢 CPU，串行更快
    embed_workers = 1 if BGE_EMBED_BACKEND == "local" else 3
    logger.info(f"  分为 {len(batches)} 批，每批 {batch_size}，{embed_workers} 线程并发")

    all_embeddings = []
    chunk_ids = []
    failed_batch_ids: list[tuple[int, list[int], list[int]]] = []  # (batch_idx, ids, doc_ids)

    with ThreadPoolExecutor(max_workers=embed_workers) as executor:
        future_to_batch = {
            executor.submit(call_bge_embedding, texts): (texts, ids, doc_ids, idx)
            for idx, (texts, ids, doc_ids) in enumerate(batches)
        }
        for future in as_completed(future_to_batch):
            texts, ids, doc_ids, idx = future_to_batch[future]
            try:
                embs = future.result()
                if embs is not None:
                    all_embeddings.extend(embs)
                    chunk_ids.extend(ids)
                    if (idx + 1) % 5 == 0 or idx == len(batches) - 1:
                        logger.info(f"  embedding 进度: {len(all_embeddings)}/{len(chunks)}")
                else:
                    failed_batch_ids.append((idx, ids, doc_ids))
                    logger.warning(f"  batch {idx + 1} embedding 失败，已加入重试队列")
            except Exception as e:
                failed_batch_ids.append((idx, ids, doc_ids))
                logger.error(f"  batch {idx + 1} embedding 异常: {e}")

    # 重试失败批次（指数退避），仍失败则逐条降级
    if failed_batch_ids:
        logger.info(f"  重试 {len(failed_batch_ids)} 个失败批次...")
        for bidx, bid_ids, bid_doc_ids in failed_batch_ids:
            bid_texts = [c[1] for c in chunks if c[0] in bid_ids]
            success = False
            for retry in range(3):
                time.sleep(2 ** retry)
                embs = call_bge_embedding(bid_texts)
                if embs is not None:
                    all_embeddings.extend(embs)
                    chunk_ids.extend(bid_ids)
                    success = True
                    break
            if success:
                continue

            # 批内个别坏文本（如过短 OCR 碎片）会导致整批 400，逐条降级定位坏 chunk
            logger.warning(f"  batch {bidx + 1} 批量重试失败，逐条降级重试...")
            ok_pairs, failed_pairs = _embed_single_fallback(
                list(zip(bid_ids, bid_texts, bid_doc_ids))
            )
            for cid, emb in ok_pairs:
                all_embeddings.append(emb)
                chunk_ids.append(cid)
            if failed_pairs:
                _log_embedding_failures(db, failed_pairs)

    if not all_embeddings:
        logger.error("所有 embedding 均失败，无法构建 FAISS 索引")
        return

    vectors = np.array(all_embeddings, dtype=np.float32)
    dim = vectors.shape[1]
    if dim != index.d:
        logger.error(f"embedding 维度 {dim} 与索引维度 {index.d} 不一致，跳过索引更新")
        return

    index.add(vectors)

    logger.info(f"FAISS HNSW 索引更新完成: {index.ntotal} 个向量, {dim} 维")

    # vector_id 从已有向量数之后顺延分配，避免与旧 chunk 的 vector_id 冲突
    id_pairs = [(chunk_id, base + idx) for idx, chunk_id in enumerate(chunk_ids)]
    db.batch_update_vector_ids(id_pairs)
    logger.info(f"  已更新 {len(id_pairs)} 条 vector_id")

    # 持久化索引文件（先写系统临时目录避免中文路径，再 copy2 到目标）
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp_path = tmp.name
    tmp.close()
    try:
        faiss.write_index(index, tmp_path)
        shutil.copy2(tmp_path, index_path)
        logger.info(f"FAISS 索引已保存到: {index_path}")
    except Exception as e:
        logger.error(f"FAISS 索引持久化失败: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    run_ingest()
