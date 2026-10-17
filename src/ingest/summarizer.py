"""文档摘要回填：为 documents.doc_summary 生成严格忠实原文的要点摘要。

综述类跨文档题的上下文补充（检索链路不参与，仅组装时前置）。
- 输入：DB 中该文档全部 child chunks 按位置拼接的全文
- 幂等：doc_summary 非空跳过；--force 重新生成
- 失败文档不阻断，结束后统一汇报
"""

import os, sys, logging

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from config import MYSQL_CONFIG, DEEPSEEK_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from http_client import post_json
from utils import setup_windows_console

setup_windows_console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("summarizer")

SUMMARY_MAX_CHARS = 400
OVERSIZE_LIMIT = 500

SUMMARY_PROMPT = """你是金融支付领域文档摘要员。基于给定文档全文，生成要点摘要。

【硬性要求】
1. 逐章节覆盖：文档每个二级标题（##）下的关键要点必须出现，漏一个章节即失败
2. 严格忠实原文：只写原文明确有的内容。严禁添加原文没有的信息、严禁推测、严禁评价
3. 每条要点一句话，bullet 列表（- 开头）
4. 总长度 {max_chars} 字以内；超长时合并同类要点，但不得因压缩丢失任何要点
5. 关键数字、日期、文号必须原样保留
6. 只输出 bullet 列表，不要任何前言后语

【文档全文】
{full_text}"""


def _load_documents(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_name, doc_summary FROM documents WHERE parse_status='success'")
        return cur.fetchall()


def _load_full_text(conn, doc_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_text FROM chunks "
            "WHERE document_id=%s AND chunk_level='child' "
            "ORDER BY position_start",
            (doc_id,),
        )
        return "\n".join(r[0] for r in cur.fetchall())


def _generate_summary(full_text: str, prev: str | None = None) -> str | None:
    prompt = SUMMARY_PROMPT.format(max_chars=SUMMARY_MAX_CHARS, full_text=full_text)
    if prev is not None:
        prompt += (
            f"\n\n【警告】你上一次输出 {len(prev)} 字，超过 {SUMMARY_MAX_CHARS} 字硬限制。"
            f"必须重新压缩到 {SUMMARY_MAX_CHARS} 字以内：合并同类要点、每条要点更短，"
            f"但不得丢失任何章节要点。"
        )
    resp, err = post_json(
        DEEPSEEK_URL,
        {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": SUMMARY_PROMPT.format(
                max_chars=SUMMARY_MAX_CHARS, full_text=full_text)}],
            "max_tokens": 800,
            "temperature": 0.1,
            "thinking": {"type": "disabled"},
        },
        {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        60,
        retries=1,
    )
    if resp is None:
        logger.warning(f"摘要生成失败: {err}")
        return None
    if resp.status_code != 200:
        logger.warning(f"摘要生成 HTTP {resp.status_code}: {resp.text[:150]}")
        return None
    try:
        content = (resp.json()["choices"][0].get("message") or {}).get("content")
    except Exception as e:
        logger.warning(f"响应解析失败: {e}")
        return None
    return content.strip() if content else None


def main():
    import pymysql
    force = "--force" in sys.argv
    conn = pymysql.connect(**MYSQL_CONFIG)
    try:
        docs = _load_documents(conn)
        todo = [
            d for d in docs
            if force or not d[2] or len(d[2]) > OVERSIZE_LIMIT
        ]
        logger.info(f"文档总数 {len(docs)}，待生成/压缩摘要 {len(todo)} 篇")
        ok = fail = 0
        for doc_id, file_name, prev_summary in todo:
            full_text = _load_full_text(conn, doc_id)
            if not full_text:
                logger.warning(f"跳过 {file_name}: 无 child chunk")
                fail += 1
                continue
            summary = _generate_summary(full_text, prev_summary)
            if summary is not None and len(summary) > OVERSIZE_LIMIT:
                # 二次强化重试，仍超则截断兜底（bullet 结构下保头部要点）
                retry = _generate_summary(full_text, summary)
                if retry is not None:
                    summary = retry
            if summary is not None and len(summary) > OVERSIZE_LIMIT:
                logger.warning(f"{file_name[:40]}: 二次仍超长({len(summary)}字)，截断至 {OVERSIZE_LIMIT}")
                summary = summary[:OVERSIZE_LIMIT]
            if summary is None:
                fail += 1
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET doc_summary=%s WHERE id=%s",
                    (summary, doc_id),
                )
                conn.commit()
            ok += 1
            logger.info(f"[{ok}/{len(todo)}] {file_name[:50]}: {len(summary)} 字")
        logger.info(f"完成: 成功 {ok} / 失败 {fail}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
