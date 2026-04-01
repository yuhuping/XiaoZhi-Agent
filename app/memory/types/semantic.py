from __future__ import annotations

from typing import Any

from app.memory.base import MemoryItem
from app.memory.types.persistent_base import PersistentMemoryBase


class SemanticMemory(PersistentMemoryBase):
    """语义记忆：强调抽象知识与实体关系。"""

    memory_type = "semantic"

    def add(self, item: MemoryItem) -> str:
        """添加语义记忆：自动提取实体缓存到metadata。"""
        metadata = dict(item.metadata)
        entities = self._extract_entities(item.content)
        metadata.setdefault("entities", entities)
        item.metadata = metadata
        return super().add(item)

    def _score(self, item: MemoryItem, vec_score: float, query: str, **kwargs: Any) -> float:
        """语义评分：(向量0.7 + 图关系0.3) * 重要性权重。"""
        query_entities = set(self._extract_entities(query))
        item_entities = set(self._extract_entities_from_item(item))

        graph_score = 0.0
        if query_entities and item_entities:
            graph_score = len(query_entities & item_entities) / max(1, len(query_entities | item_entities))

        base_relevance = float(vec_score) * 0.7 + graph_score * 0.3
        importance_weight = 0.8 + item.importance * 0.4
        return base_relevance * importance_weight

    def _extract_entities(self, text: str) -> list[str]:
        """抽取实体：使用嵌入分词器做轻量实体候选。"""
        tokens = self.embedder._tokenize(text)
        dedup: list[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token not in dedup:
                dedup.append(token)
        return dedup[:20]

    def _extract_entities_from_item(self, item: MemoryItem) -> list[str]:
        """读取实体：优先metadata，缺失则即时抽取。"""
        entities = item.metadata.get("entities")
        if isinstance(entities, list):
            return [str(x) for x in entities if str(x).strip()]
        return self._extract_entities(item.content)
