"""
FAISS 索引重建脚本 —— 增量构建，复用 ingest 管线的 _build_faiss_index。
"""

import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from core.utils import setup_windows_console
setup_windows_console()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild")

from core.config import MYSQL_CONFIG
from ingest.pipeline import DB, _build_faiss_index


def main():
    db = DB()
    try:
        _build_faiss_index(db)

        # ── 汇总 ──
        with db.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_level='child' AND status='embedded'")
            embedded = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks WHERE chunk_level='child' AND status='pending'")
            pending = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents WHERE parse_status='success'")
            docs_ok = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents WHERE parse_status='failed'")
            docs_fail = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tables_structured")
            tables = cur.fetchone()[0]

        logger.info("=" * 40)
        logger.info(f"文档: {docs_ok} 成功 / {docs_fail} 失败")
        logger.info(f"Chunks (child): {embedded} embedded / {pending} pending")
        logger.info(f"结构化表行数: {tables}")
    finally:
        db.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
