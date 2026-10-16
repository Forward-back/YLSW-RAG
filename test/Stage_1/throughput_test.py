# -*- coding: utf-8 -*-
"""吞吐量与并发压测。

asyncio+aiohttp 模拟 10/20/30/40 并发用户，每档持续 60s，
从 50 题抽样池循环取题持续请求 POST /chat。
输出每档：QPS、平均/P50/P95/P99 延迟、错误率 → throughput_result.json。
"""
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:6151/chat"
DURATION = 60
LEVELS = [10, 20, 30, 40, 60, 80, 100]


def load_pool():
    qs = json.loads((BASE.parent / "qa_testset_120.json").read_text(encoding="utf-8"))
    rnd = random.Random(20260918)
    pool = [q["问题"] for q in qs if q["类型"] not in ("提示词注入", "SQL注入")]
    rnd.shuffle(pool)
    return pool[:50]


async def worker(session, pool, latencies, errors, stop_at):
    rnd = random.Random()
    while time.monotonic() < stop_at:
        q = rnd.choice(pool)
        t0 = time.perf_counter()
        try:
            async with session.post(URL, json={"question": q},
                                    timeout=aiohttp.ClientTimeout(total=180)) as r:
                if r.status != 200:
                    errors.append(1)
                    continue
                await r.json()
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors.append(1)


async def run_level(level, pool):
    latencies, errors = [], []
    stop_at = time.monotonic() + DURATION
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(worker(session, pool, latencies, errors, stop_at))
                 for _ in range(level)]
        await asyncio.gather(*tasks)
    latencies.sort()
    n = len(latencies)
    errs = len(errors)
    total_req = n + errs
    qps = total_req / DURATION
    stats = {
        "并发": level,
        "总请求数": total_req,
        "QPS": round(qps, 2),
        "平均(ms)": round(statistics.mean(latencies), 1) if n else 0,
        "P50(ms)": round(latencies[int(n * 0.50)], 1) if n else 0,
        "P95(ms)": round(latencies[int(n * 0.95)], 1) if n else 0,
        "P99(ms)": round(latencies[int(n * 0.99)], 1) if n else 0,
        "错误率(%)": round(errs / total_req * 100, 2) if total_req else 0,
    }
    print(f"并发 {level}: QPS={stats['QPS']} 平均={stats['平均(ms)']}ms "
          f"P50={stats['P50(ms)']} P95={stats['P95(ms)']} P99={stats['P99(ms)']} "
          f"错误率={stats['错误率(%)']}%")
    return stats


async def main():
    pool = load_pool()
    print(f"题池 {len(pool)} 题，每档 {DURATION}s")
    results = []
    for level in LEVELS:
        print(f"--- 并发 {level} 开始 ---")
        results.append(await run_level(level, pool))
    (BASE / "throughput_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成: throughput_result.json")


if __name__ == "__main__":
    asyncio.run(main())
