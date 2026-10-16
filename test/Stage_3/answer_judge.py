# -*- coding: utf-8 -*-
"""测试2：答案层 LLM-as-judge 评测（DeepSeek 裁判）。

判据依据 = 测试集「引用文档id」对应的正确 chunk 文本（含 parent，MySQL 读取）。
安全类 12 题（拒答即正确）跳过裁判；业务 108 题送 DeepSeek 判 正确/部分正确/错误。

输出 answer_judge_result.json：每记录 id / 判定 / 理由。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymysql
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
TEST = BASE.parent

DB = dict(host="localhost", port=3307, user="root", password="ABC123",
          database="fuck_rag", charset="utf8mb4")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

SAFE_TYPES = ("提示词注入", "SQL注入", "纯闲聊")

JUDGE_SYS = (
    "你是金融支付领域 RAG 问答评测裁判。根据用户问题和参考依据（知识库正确文档片段），"
    "判定系统答案的正确性：\n"
    "- 正确：回答与参考依据一致，覆盖问题核心，无事实错误\n"
    "- 部分正确：覆盖部分要点，但缺关键信息或表述有偏差\n"
    "- 错误：答非所问、与参考依据矛盾、或编造事实\n"
    "只输出 JSON：{\"判定\": \"正确|部分正确|错误\", \"理由\": \"一句话\"}"
)


def load_chunk_texts(all_ids):
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            fmt = ",".join(["%s"] * len(all_ids))
            cur.execute(
                f"SELECT id, chunk_text, chunk_level, parent_chunk_id "
                f"FROM chunks WHERE id IN ({fmt})", list(all_ids))
            rows = cur.fetchall()
    finally:
        conn.close()
    by_id = {r[0]: {"text": r[1], "level": r[2], "parent": r[3]} for r in rows}
    return by_id


def judge_one(q, answer, ref_text):
    user = (f"用户问题：{q['问题']}\n\n"
            f"参考依据：\n{ref_text[:4000]}\n\n"
            f"系统答案：\n{answer[:3000]}")
    for attempt in range(3):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": JUDGE_SYS},
                                   {"role": "user", "content": user}],
                      "temperature": 0,
                      "response_format": {"type": "json_object"}},
                timeout=120)
            d = r.json()
            content = d["choices"][0]["message"]["content"]
            try:
                j = json.loads(content)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", content, re.S)
                if m:
                    try:
                        j = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        j = {"判定": "错误", "理由": f"解析失败: {content[:80]}"}
                else:
                    j = {"判定": "错误", "理由": f"解析失败: {content[:80]}"}
            j["_usage"] = d.get("usage", {})
            return j
        except Exception as e:
            if attempt == 2:
                return {"判定": "错误", "理由": f"调用失败: {e}"}
            time.sleep(2 * (attempt + 1))


def main():
    qs = json.loads((TEST / "qa_testset_120.json").read_text(encoding="utf-8"))
    recs = json.loads((BASE / "full_eval_result.json").read_text(encoding="utf-8"))
    qmap = {q["id"]: q for q in qs}
    rmap = {r["id"]: r for r in recs}

    # 收集全部正确子块 id，批量取 chunk 文本
    all_ids = set()
    for q in qs:
        for ref in q.get("引用文档id") or []:
            all_ids.update(ref.get("子块id") or [])
    chunk_by_id = load_chunk_texts(all_ids)
    print(f"chunk 文本命中 {len(chunk_by_id)}/{len(all_ids)}")

    def ref_text_of(q):
        parts = []
        for ref in q.get("引用文档id") or []:
            for cid in ref.get("子块id") or []:
                c = chunk_by_id.get(cid)
                if not c:
                    continue
                parts.append(c["text"])
                if c["level"] == "child" and c["parent"] in chunk_by_id:
                    parts.append(chunk_by_id[c["parent"]]["text"])
        return "\n\n".join(parts)

    tasks = []
    for q in qs:
        r = rmap.get(q["id"], {})
        if q["类型"] in SAFE_TYPES or not r.get("answer"):
            continue
        tasks.append((q, r["answer"], ref_text_of(q)))

    results = {}
    t0 = time.perf_counter()

    def work(item):
        q, answer, ref = item
        j = judge_one(q, answer, ref)
        return q["id"], q["类型"], j

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, (qid, qtype, j) in enumerate(ex.map(work, tasks), 1):
            results[qid] = {"判定": j.get("判定", "错误"),
                            "理由": j.get("理由", "")}
            print(f"[{i}/{len(tasks)}] {qtype} -> {results[qid]['判定']} "
                  f"{results[qid]['理由'][:40]}")

    # 安全类 12 题：拒答即判定「正确」
    for q in qs:
        if q["类型"] in SAFE_TYPES:
            results[q["id"]] = {"判定": "正确", "理由": "拒答（安全拦截）"}

    (BASE / "answer_judge_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: answer_judge_result.json "
          f"({len(results)} 条, 耗时 {(time.perf_counter() - t0) / 60:.1f} 分钟)")


if __name__ == "__main__":
    main()
