"""
PaddleOCR 封装：扫描件 PDF / 图片的 OCR 文本提取。
"""

import os
import sys
import logging
from typing import Optional

# PaddleOCR 依赖 wandb，其在 Windows GBK 控制台会因 Unicode 字符崩溃；推理不需要，禁用
os.environ.setdefault("WANDB_MODE", "disabled")

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

logger = logging.getLogger(__name__)


class PaddleOCRProcessor:
    """PaddleOCR 处理器（纯文本提取，不做版面/表格分析）。"""

    def __init__(self):
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
                logger.info("PaddleOCR 初始化完成")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
                raise
        return self._ocr

    def extract_text(self, file_path: str) -> str:
        """对扫描件 PDF 或图片做 OCR，返回纯文本。

        PaddleOCR 3.7 使用 predict() 接口，返回 generator 格式：
        [{rec_texts: [...], rec_scores: [...], ...}, ...]
        """
        try:
            results = self.ocr.predict(file_path)
        except Exception as e:
            logger.error(f"PaddleOCR 文本提取失败: {e}")
            return ""

        text_parts = []
        for page_result in results:
            rec_texts = page_result.get("rec_texts", [])
            for text in rec_texts:
                text_str = str(text) if not isinstance(text, str) else text
                if text_str.strip():
                    text_parts.append(text_str.strip())

        return "\n".join(text_parts)


# ── 全局单例 ───────────────────────────────────────────
_paddle_processor: Optional[PaddleOCRProcessor] = None


def get_paddle_processor() -> PaddleOCRProcessor:
    global _paddle_processor
    if _paddle_processor is None:
        _paddle_processor = PaddleOCRProcessor()
    return _paddle_processor
