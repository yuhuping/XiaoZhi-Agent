from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    """本地向量存储：按用户+类型维护向量索引。"""

    def __init__(self, index_dir: str, vector_dim: int = 256) -> None:
        # 重要变量：索引目录用于持久化缓存。
        self.index_root = Path(index_dir)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.vector_dim = max(64, int(vector_dim))

    def upsert(self, user_id: str, memory_type: str, memory_id: str, vector: np.ndarray) -> None:
        """写入向量：存在则覆盖，不存在则追加。"""
        ids, matrix = self._load_index(user_id, memory_type)
        vec = self._normalize(vector)
        if vec.shape[0] != self.vector_dim:
            vec = self._resize(vec)

        if memory_id in ids:
            idx = ids.index(memory_id)
            matrix[idx] = vec
        else:
            ids.append(memory_id)
            if matrix.size == 0:
                matrix = np.zeros((0, self.vector_dim), dtype=np.float32)
            matrix = np.vstack([matrix, vec.reshape(1, -1)])

        self._save_index(user_id, memory_type, ids, matrix)

    def remove(self, user_id: str, memory_type: str, memory_id: str) -> None:
        """删除向量：若不存在则忽略。"""
        ids, matrix = self._load_index(user_id, memory_type)
        if memory_id not in ids:
            return
        idx = ids.index(memory_id)
        ids.pop(idx)
        matrix = np.delete(matrix, idx, axis=0)
        self._save_index(user_id, memory_type, ids, matrix)

    def search(
        self,
        user_id: str,
        memory_type: str,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """向量检索：余弦相似度降序返回。"""
        ids, matrix = self._load_index(user_id, memory_type)
        if not ids or matrix.size == 0:
            return []

        query = self._normalize(query_vector)
        if query.shape[0] != self.vector_dim:
            query = self._resize(query)

        scores = matrix @ query
        order = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for idx in order[: max(1, int(top_k))]:
            score = float(scores[idx])
            if score < float(min_score):
                continue
            results.append((ids[int(idx)], score))
        return results

    def clear_user(self, user_id: str) -> None:
        """清空用户索引文件。"""
        prefix = self._normalize_id(user_id)
        for path in self.index_root.glob(f"{prefix}__*.npz"):
            path.unlink(missing_ok=True)
        for path in self.index_root.glob(f"{prefix}__*.ids.json"):
            path.unlink(missing_ok=True)

    def _load_index(self, user_id: str, memory_type: str) -> tuple[list[str], np.ndarray]:
        """加载索引：不存在则返回空索引。"""
        vec_path = self._vec_path(user_id, memory_type)
        ids_path = self._ids_path(user_id, memory_type)
        if not vec_path.exists() or not ids_path.exists():
            return [], np.zeros((0, self.vector_dim), dtype=np.float32)

        try:
            payload = np.load(vec_path)
            matrix = payload["vectors"].astype(np.float32)
        except Exception:
            matrix = np.zeros((0, self.vector_dim), dtype=np.float32)

        try:
            ids_raw = json.loads(ids_path.read_text(encoding="utf-8"))
            ids = [str(x) for x in ids_raw] if isinstance(ids_raw, list) else []
        except Exception:
            ids = []

        if matrix.ndim != 2:
            matrix = np.zeros((0, self.vector_dim), dtype=np.float32)
        if matrix.shape[1] != self.vector_dim:
            matrix = self._resize_matrix(matrix)
        if matrix.shape[0] != len(ids):
            min_rows = min(matrix.shape[0], len(ids))
            matrix = matrix[:min_rows]
            ids = ids[:min_rows]
        return ids, matrix

    def _save_index(self, user_id: str, memory_type: str, ids: list[str], matrix: np.ndarray) -> None:
        """保存索引：向量和ID分离持久化。"""
        vec_path = self._vec_path(user_id, memory_type)
        ids_path = self._ids_path(user_id, memory_type)
        np.savez(vec_path, vectors=matrix.astype(np.float32))
        ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    def _vec_path(self, user_id: str, memory_type: str) -> Path:
        """生成向量文件路径。"""
        return self.index_root / f"{self._normalize_id(user_id)}__{memory_type}.npz"

    def _ids_path(self, user_id: str, memory_type: str) -> Path:
        """生成ID文件路径。"""
        return self.index_root / f"{self._normalize_id(user_id)}__{memory_type}.ids.json"

    def _normalize_id(self, value: str) -> str:
        """路径安全化。"""
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """向量归一化。"""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return np.zeros((self.vector_dim,), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return np.zeros((vec.shape[0],), dtype=np.float32)
        return vec / norm

    def _resize(self, vector: np.ndarray) -> np.ndarray:
        """调整向量维度。"""
        vec = np.zeros((self.vector_dim,), dtype=np.float32)
        copy_len = min(self.vector_dim, vector.shape[0])
        vec[:copy_len] = vector[:copy_len]
        return self._normalize(vec)

    def _resize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """调整矩阵维度。"""
        if matrix.size == 0:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        resized = np.zeros((matrix.shape[0], self.vector_dim), dtype=np.float32)
        copy_len = min(self.vector_dim, matrix.shape[1])
        resized[:, :copy_len] = matrix[:, :copy_len]
        norms = np.linalg.norm(resized, axis=1, keepdims=True)
        return resized / np.maximum(norms, 1e-12)
