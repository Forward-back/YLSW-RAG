# -*- coding: utf-8 -*-
"""Stage_2 视角6：噪声鲁棒性。

从 47 题挑 10 题，DeepSeek 生成 4 类变体（错别字/口语化/截断/指代），
标准题与变体各跑 trace，对比 rerank 后 Recall@5 的衰减。
输出 noise_decay.json。
"""
import json
import random
import sys
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent.parent))
TESTSET = BASE.parent / "produce_question" / "qa_testset_120.json"
URL = "http://127.0.0.1:6151/chat"
N = 10

VARIANT_PROMPT = """你是检索鲁棒性测试数据生成器。给定一个用户问题，生成 4 个变体，每个变体保持原意但引入一种噪声：
1. 错别字：把 1-2 个词替换成同音错别字（如 结算→结蒜、支付→支附）
2. 口语化：改写成大白话口语句式（如 加语气词、倒装、冗余表述）
3. 截断：只保留问题前半部分（信息不完整）
4. 指代：把明确主体名称换成模糊指代（如 该机构、这家公司、它）

原问题：{question}

只输出 JSON：
{{"错别字": "...", "口语化": "...", "截断": "...", "指代": "..."}}"""


def pick_10():
    qs = json.loads(TESTSET.read_text(encoding="utf-8"))
    pool = [q for q in qs if q["类型"] not in ("提示词注入", "SQL注入", "纯闲聊")]
    random.Random(20260918).shuffle(pool)
    return pool[:N]


def gen_variants(q, retries=3):
    from src.config import DEEPSEEK_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    prompt = VARIANT_PROMPT.format(question=q["问题"])
    for attempt in range(retries):
        try:
            r = requests.post(
                DEEPSEEK_URL,
                json={"model": DEEPSEEK_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 1024, "temperature": 0.7,
                      "thinking": {"type": "disabled"},
                      "response_format": {"type": "json_object"}},
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                timeout=60,
            )
            if r.status_code == 200:
                return json.loads(r.json()["choices"][0]["message"]["content"].strip())
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


def recall_at_5(q_text, docs):
    for attempt in range(3):
        try:
            r = requests.post(URL, json={"question": q_text, "trace": True}, timeout=120)
            if r.status_code == 200:
                tr = r.json().get("trace", {})
                cands = tr.get("reranked", [])
                return all(any(c.get("file_name") == d for c in cands[:5]) for d in docs)
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    qs = pick_10()
    print(f"挑 {len(qs)} 题生成变体")
    rows = []
    for i, q in enumerate(qs, 1):
        docs = [r["父文档id"] for r in (q.get("引用文档id") or [])]
        variants = gen_variants(q)
        if not variants:
            print(f"[{i}/{len(qs)}] 变体生成失败 {q['id'][:30]}")
            continue
        base_hit = recall_at_5(q["问题"], docs)
        row = {"id": q["id"], "标准": base_hit}
        for vtype, vtext in variants.items():
            row[vtype] = recall_at_5(vtext, docs)
            print(f"[{i}/{len(qs)}] {vtype}: {vtext[:40]}... -> hit={row[vtype]}")
        rows.append(row)
        (BASE / "noise_decay.json").write_text(
            json.dumps({"rows": rows, "variants": list(variants.keys())},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {}
    for vtype in ["标准", "错别字", "口语化", "截断", "指代"]:
        vals = [r[vtype] for r in rows if vtype in r]
        summary[vtype] = round(sum(vals) / len(vals), 4) if vals else None
    (BASE / "noise_decay.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("完成:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
