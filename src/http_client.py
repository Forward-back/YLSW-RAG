"""共享 HTTP 客户端：分类重试 + 每请求独立连接。

- 每请求 `Connection: close`：2026-09-27 实测网关/LB 会静默回收空闲连接，
  池化连接变半死后复用会挂到读超时（92.4s 案例，卡点恒为每问首个网关调用=规划器）。
  换每请求新连接（代价 ~100-300ms TLS 握手/次），换确定性。
- 400/401/403/404/422 为永久失败，不重试直接返回
- 429 优先读 Retry-After，缺失则指数退避 2**attempt
- 5xx / 网络异常指数退避 2**attempt
线程安全：requests.Session 底层的 urllib3 连接池线程安全，可配合 ThreadPoolExecutor。
"""
import logging
import time

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.mount("http://", HTTPAdapter(pool_connections=8, pool_maxsize=32))
SESSION.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=32))

# 永久失败状态码：重试无意义（参数/鉴权/资源不存在）
NO_RETRY_STATUS = {400, 401, 403, 404, 422}


def _retry_after_seconds(resp: requests.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return max(1.0, float(ra))
    except ValueError:
        return 2.0


def post_json(url: str, payload: dict, headers: dict, timeout,
              retries: int = 3, **post_kwargs) -> tuple[requests.Response | None, str | None]:
    """POST JSON 并分类重试。返回 (resp, err)：
    resp 非 None 时表示收到了最终 HTTP 响应（200 或不可重试的 4xx 或重试耗尽）；
    resp 为 None 时 err 为最后一次网络异常信息。"""
    last_err = None
    headers = dict(headers)
    headers.setdefault("Connection", "close")
    for attempt in range(retries):
        try:
            resp = SESSION.post(url, json=payload, headers=headers,
                                timeout=timeout, **post_kwargs)
        except Exception as e:
            last_err = str(e)
            logger.warning(f"HTTP 请求异常 (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200 or resp.status_code in NO_RETRY_STATUS:
            return resp, None
        if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
            return resp, None

        wait = _retry_after_seconds(resp)
        if wait is None:
            wait = float(2 ** attempt)
        logger.warning(
            f"HTTP {resp.status_code} (attempt {attempt + 1}/{retries})，"
            f"{wait:.0f}s 后重试: {resp.text[:150]}"
        )
        if attempt < retries - 1:
            time.sleep(wait)
    return None, last_err
