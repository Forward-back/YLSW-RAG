# -*- coding: utf-8 -*-
"""端到端延迟拆解测试。

从 qa_testset_120.json 按类型比例抽样 50 题（单文档 30 / 多文档 12 / 数值 5 / 注入闲聊 3），
串行逐题 + 10 并发分批请求 POST /chat (timing=true)，
阶段耗时落两个 JSON：serial_latency.json / concurrent_latency.json。
字段为阶段名，值单位 ms（数值）。
"""
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp
import requests

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:6151/chat"
STAGES = [
    "规划改写", "检索_faiss", "检索_bm25", "检索_meta", "检索_sql",
    "RRF融合", "精排rerank", "parent上下文", "LLM生成", "llm首token", "总计",
]


def sample_questions():
    qs = json.loads((BASE.parent / "qa_testset_120.json").read_text(encoding="utf-8"))
    groups = {
        "单文档": 30,
        "多文档-并行": 6,
        "多文档-串行": 6,
        "数值计算": 5,
        "提示词注入": 1,
        "SQL注入": 1,
        "纯闲聊": 1,
    }
    picked = []
    for t, n in groups.items():
        pool = [q for q in qs if q["类型"] == t]
        random.Random(20260918).shuffle(pool)
        picked.extend(pool[:n])
    random.Random(7).shuffle(picked)
    return picked


def normalize(timings):
    return {s: round(float(timings.get(s, 0.0)), 1) for s in STAGES}


def run_serial(qs):
    records = []
    for i, q in enumerate(qs, 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, json={"question": q["问题"], "timing": True}, timeout=120)
            d = r.json()
            rec = {"id": q["id"]}
            rec.update(normalize(d.get("timings", {})))
            rec["总计"] = round((time.perf_counter() - t0) * 1000, 1)
            records.append(rec)
            print(f"[串行 {i}/{len(qs)}] {q['类型']} 总计={rec['总计']}ms")
        except Exception as e:
            print(f"[串行 {i}/{len(qs)}] FAIL {e}")
            rec = {"id": q["id"]}
            rec.update({s: 0.0 for s in STAGES})
            records.append(rec)
    return records


async def fetch_one(session, sem, q):
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(URL, json={"question": q["问题"], "timing": True},
                                    timeout=aiohttp.ClientTimeout(total=180)) as r:
                d = await r.json()
        except Exception as e:
            print(f"[并发] FAIL {q['id'][:40]} {e}")
            d = {}
        rec = {"id": q["id"]}
        rec.update(normalize(d.get("timings", {})))
        if "总计" not in d.get("timings", {}):
            rec["总计"] = round((time.perf_counter() - t0) * 1000, 1)
        return rec


async def run_concurrent(qs, batch=10):
    records = []
    for i in range(0, len(qs), batch):
        chunk = qs[i:i + batch]
        sem = asyncio.Semaphore(batch)
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(fetch_one(session, sem, q) for q in chunk)
            )
        records.extend(results)
        done = min(i + batch, len(qs))
        print(f"[并发 {done}/{len(qs)}] 批次完成")
    return records


def main():
    qs = sample_questions()
    print(f"抽样 {len(qs)} 题: " + json.dumps(
        {t: sum(1 for q in qs if q["类型"] == t) for t in
         ["单文档", "多文档-并行", "多文档-串行", "数值计算", "提示词注入", "SQL注入", "纯闲聊"]},
        ensure_ascii=False))
    serial = run_serial(qs)
    (BASE / "serial_latency.json").write_text(
        json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
    concurrent = asyncio.run(run_concurrent(qs, batch=10))
    (BASE / "concurrent_latency.json").write_text(
        json.dumps(concurrent, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成: serial_latency.json + concurrent_latency.json")


if __name__ == "__main__":
    main()
