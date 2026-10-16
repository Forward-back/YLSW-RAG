"""
请求级日志追踪：每条日志自动携带 trace_id，并发请求互不混淆。

用法：
    trace_id = start_trace()   # 请求入口（handle_query）
    ... 全链路 logger.info(...) 自动带 [req:trace_id] 前缀 ...
    clear_trace()              # 请求出口（finally）

contextvars 会随 ThreadPoolExecutor.submit 传播到检索子线程，
因此各检索模块日志同样带同一 trace_id。
"""

import logging
import contextvars

_trace_var: contextvars.ContextVar[str] = contextvars.ContextVar("query_trace", default="-")


class TraceFilter(logging.Filter):
    """把当前上下文的 trace_id 注入 LogRecord.trace。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace = _trace_var.get()
        return True


def install_trace_filter():
    """给根 logger 的所有 handler 挂 TraceFilter（app.py 启动时调用一次）。"""
    for handler in logging.getLogger().handlers:
        handler.addFilter(TraceFilter())


def start_trace() -> str:
    import uuid
    tid = uuid.uuid4().hex[:8]
    _trace_var.set(tid)
    return tid


def clear_trace():
    _trace_var.set("-")


def submit_with_trace(executor, fn, *args, **kwargs):
    """ThreadPoolExecutor.submit 不传播 contextvars，显式携带当前上下文提交。"""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)
