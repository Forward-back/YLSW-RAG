"""模型 API 可用性探测（走 .env 里的 URL 与 key）。

逐个试调 chat/completions，打印每个模型名变体是否可用。
用法：./.venv/Scripts/python test_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import requests
from src import config as c

KEY = c.DEEPSEEK_API_KEY
BASE = c.DEEPSEEK_URL

CANDIDATES = [
    "deepseek-v4-flash",
    "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-VL-32B-Instruct",
    "PaddlePaddle/PaddleOCR-VL-1.5",
]

headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

for model in CANDIDATES:
    try:
        r = requests.post(
            BASE,
            json={"model": model,
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 8, "enable_thinking": False},
            headers=headers, timeout=30,
        )
        if r.status_code == 200:
            print(f"[OK]   {model}")
        elif r.status_code == 403:
            print(f"[403]  {model}  (白名单未开放，需主办方开通)")
        else:
            print(f"[{r.status_code}] {model}  {r.text[:100]}")
    except Exception as e:
        print(f"[EXC]  {model}  {e}")
