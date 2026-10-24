"""项目共享工具模块。"""

import sys


def setup_windows_console():
    """Windows 控制台默认 GBK 代码页，中文日志会乱码。

    将控制台代码页与 stdout/stderr 切到 UTF-8（需在输出任何日志前调用）。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# 中文数字映射
_CN_DIGIT = {"一": "1", "二": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8",
             "九": "9", "十": "10"}
