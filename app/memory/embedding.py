from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingTextEmbedder:
    """轻量嵌入器：使用哈希向量实现本地可运行嵌入。"""

    def __init__(self, dim: int = 256) -> None:
        # 重要变量：向量维度决定检索空间大小。
        self.dim = max(64, int(dim))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量向量化：输出归一化向量矩阵。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % self.dim
                matrix[row, bucket] += 1.0

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def encode_text(self, text: str) -> np.ndarray:
        """单条向量化：便于搜索与增量写入。"""
        matrix = self.encode_texts([text])
        if matrix.shape[0] == 0:
            return np.zeros((self.dim,), dtype=np.float32)
        return matrix[0]

    def _tokenize(self, text: str) -> list[str]:
        """分词：英文词 + 中文字 + 中文二元片段。"""
        lowered = (text or "").lower()
        tokens: list[str] = re.findall(r"[a-z0-9]+", lowered)
        for seq in re.findall(r"[\u4e00-\u9fff]+", lowered):
            tokens.extend(list(seq))
            if len(seq) >= 2:
                tokens.extend(seq[i : i + 2] for i in range(len(seq) - 1))
        return tokens
