from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from openai import OpenAI

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

import os
# =========================
# 原型项目配置（直接填写）
# =========================
OPENAI_EMBEDDING_MODEL = "hunyuan-embedding"
OPENAI_EMBEDDING_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
OPENAI_EMBEDDING_API_KEY = os.getenv("RAG_embedding_model_key") or ""

OPENAI_EMBEDDING_BATCH_SIZE = 32
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[0] / "data" / "rag_index"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    source_type: str
    page: int | None
    text: str
    length: int
    term_freq: Counter[str]


@dataclass(frozen=True)
class SourceDocument:
    source: str
    source_type: str
    page: int | None
    text: str


class LocalKnowledgeRetriever:
    """本地 RAG 检索器（FAISS + OpenAI Embedding，首次建库，后续复用缓存）"""

    def __init__(
        self,
        kg_dir: str,
        auto_bootstrap: bool = True,
        chunk_size: int = 520,
        chunk_overlap: int = 80,
        auto_refresh_interval_seconds: int = 2,  # 仅保留兼容，不再使用
    ) -> None:
        self._ensure_faiss_available()
        self.root = Path(kg_dir)
        self.auto_bootstrap = auto_bootstrap
        self.chunk_size = max(120, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 10))
        self.auto_refresh_interval_seconds = auto_refresh_interval_seconds

        self.OPENAI_EMBEDDING_API_KEY = os.getenv("RAG_embedding_model_key") or ""

        self._lock = RLock()
        self.embedding_available = True
        self.embedding_error: str | None = None
        self.embedding_dim = 1536
        self._openai_client: OpenAI | None = None

        self.chunks: list[Chunk] = []
        self._faiss_index: Any = None

        self.index_dir = DEFAULT_INDEX_DIR
        self._index_key = self._build_index_key(self.root, OPENAI_EMBEDDING_MODEL)
        self._faiss_index_path = self.index_dir / f"{self._index_key}.faiss.index"
        self._chunks_path = self.index_dir / f"{self._index_key}.chunks.json"
        self._meta_path = self.index_dir / f"{self._index_key}.meta.json"

        self._index_status: dict[str, Any] = {
            "kg_dir": str(self.root),
            "files_total": 0,
            "pdf_files": 0,
            "txt_files": 0,
            "chunks_total": 0,
            "last_indexed_at": None,
            "backend": "faiss",
            "index_dir": str(self.index_dir),
            "loaded_from_cache": False,
            "embedding_backend": "openai",
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_available": True,
            "embedding_error": "",
        }

        self._setup_embedding()
        self._ensure_kg_dir()
        self._initialize_index()

    @classmethod
    def from_kg_dir(
        cls,
        kg_dir: str,
        auto_bootstrap: bool = True,
        chunk_size: int = 520,
        chunk_overlap: int = 80,
        auto_refresh_interval_seconds: int = 2,
    ) -> "LocalKnowledgeRetriever":
        return cls(
            kg_dir=kg_dir,
            auto_bootstrap=auto_bootstrap,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_refresh_interval_seconds=auto_refresh_interval_seconds,
        )

    def retrieve(self, query: str, top_k: int, min_score: float) -> list[dict[str, object]]:
        """向量检索入口：保持返回字段兼容 basic_tools"""
        print(f'self.embedding_available:{self.embedding_available}')
        if not self.embedding_available:
            return []
        query = (query or "").strip()
        if not query:
            return []
        if not self.chunks or self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        print(f'self._faiss_index.ntotal:{self._faiss_index.ntotal}')

        query_vec = self._embed_texts([query])
        if query_vec.shape[0] == 0:
            return []

        k = max(1, min(top_k, len(self.chunks)))
        scores, indices = self._faiss_index.search(query_vec, k)

        results: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if float(score) < min_score:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "page": chunk.page,
                    "score": round(float(score), 4),
                    "snippet": self._truncate(chunk.text, 280),
                }
            )
        return results

    def get_index_status(self) -> dict[str, Any]:
        return dict(self._index_status)

    def force_refresh(self) -> dict[str, Any]:
        """显式重建索引（KG改动时手动调用）"""
        if not self.embedding_available:
            logger.warning("embedding unavailable, skip force_refresh: %s", self.embedding_error or "unknown")
            return dict(self._index_status)
        with self._lock:
            self._build_and_persist_index()
        return dict(self._index_status)

    def _initialize_index(self) -> None:
        if not self.embedding_available:
            logger.warning("embedding unavailable, skip index init: %s", self.embedding_error or "unknown")
            return
        with self._lock:
            if self._load_cached_index():
                return
            self._build_and_persist_index()

    def _load_cached_index(self) -> bool:
        """加载缓存索引；只校验关键元数据，不做KG自动刷新"""
        if not (self._faiss_index_path.exists() and self._chunks_path.exists() and self._meta_path.exists()):
            return False
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if str(meta.get("kg_dir", "")) != str(self.root.resolve()):
                return False
            if str(meta.get("embedding_model", "")) != OPENAI_EMBEDDING_MODEL:
                return False

            chunk_payload = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            if not isinstance(chunk_payload, list):
                return False
            chunks = self._deserialize_chunks(chunk_payload)

            index = faiss.read_index(str(self._faiss_index_path))
            if int(index.ntotal) != len(chunks):
                return False

            self._faiss_index = index
            self.chunks = chunks
            self.embedding_dim = int(index.d)
            self._index_status = self._build_index_status(
                files_total=int(meta.get("files_total", 0)),
                pdf_files=int(meta.get("pdf_files", 0)),
                txt_files=int(meta.get("txt_files", 0)),
                chunks_total=len(chunks),
                last_indexed_at=str(meta.get("last_indexed_at") or ""),
                loaded_from_cache=True,
            )
            return True
        except Exception as exc:
            logger.warning("加载缓存失败，转为重建索引: %s", exc)
            return False

    def _build_and_persist_index(self) -> None:
        """首次构建或强制重建索引"""
        if not self.embedding_available:
            logger.warning("embedding unavailable, skip build index: %s", self.embedding_error or "unknown")
            return
        files = self._collect_corpus_files(self.root)
        docs = self._load_documents(self.root, files)
        chunks = self._build_chunks(docs, self.chunk_size, self.chunk_overlap)
        index = self._build_faiss_index(chunks)

        suffixes = [f.suffix.lower() for f in files]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if int(index.ntotal) != len(chunks):
            logger.warning(
                "index build incomplete, skip persist (index_rows=%s, chunks=%s)",
                int(index.ntotal),
                len(chunks),
            )
            self._set_embedding_unavailable("embedding/index build failed")
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunks = []
            self._index_status = self._build_index_status(
                files_total=len(files),
                pdf_files=sum(1 for s in suffixes if s == ".pdf"),
                txt_files=sum(1 for s in suffixes if s == ".txt"),
                chunks_total=0,
                last_indexed_at=now,
                loaded_from_cache=False,
            )
            return

        meta = {
            "kg_dir": str(self.root.resolve()),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_dim": self.embedding_dim,
            "files_total": len(files),
            "pdf_files": sum(1 for s in suffixes if s == ".pdf"),
            "txt_files": sum(1 for s in suffixes if s == ".txt"),
            "chunks_total": len(chunks),
            "last_indexed_at": now,
        }

        self._persist_index(index, chunks, meta)
        self._faiss_index = index
        self.chunks = chunks
        self._index_status = self._build_index_status(
            files_total=meta["files_total"],
            pdf_files=meta["pdf_files"],
            txt_files=meta["txt_files"],
            chunks_total=meta["chunks_total"],
            last_indexed_at=meta["last_indexed_at"],
            loaded_from_cache=False,
        )

    def _build_faiss_index(self, chunks: list[Chunk]) -> Any:
        if not chunks:
            return faiss.IndexFlatIP(self.embedding_dim)
        matrix = self._embed_texts([chunk.text for chunk in chunks]).astype(np.float32)
        if matrix.shape[0] == 0:
            return faiss.IndexFlatIP(self.embedding_dim)
        if matrix.ndim != 2:
            self._set_embedding_unavailable("embedding result shape invalid")
            return faiss.IndexFlatIP(self.embedding_dim)
        self.embedding_dim = int(matrix.shape[1])
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(matrix)
        return index

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not self.embedding_available or self._openai_client is None:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), OPENAI_EMBEDDING_BATCH_SIZE):
                batch = texts[start : start + OPENAI_EMBEDDING_BATCH_SIZE]
                response = self._openai_client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                vectors.extend(item.embedding for item in response.data)
        except Exception as exc:
            self._set_embedding_unavailable(f"embedding request failed: {exc}")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] == 0:
            self._set_embedding_unavailable("embedding response invalid")
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _probe_embedding_service(self) -> int | None:
        """启动探活：请求 embedding 接口，失败仅记录 warning 并降级"""
        try:
            sample = self._embed_texts(["启动连接检查"])
        except Exception as exc:
            self._set_embedding_unavailable(f"embedding probe failed: {exc}")
            return None
        if sample.ndim != 2 or sample.shape[0] == 0 or sample.shape[1] == 0:
            self._set_embedding_unavailable("embedding probe returned empty vector")
            return None
        return int(sample.shape[1])

    def _persist_index(self, index: Any, chunks: list[Chunk], meta: dict[str, Any]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        idx_tmp = self._faiss_index_path.with_suffix(self._faiss_index_path.suffix + ".tmp")
        chunks_tmp = self._chunks_path.with_suffix(self._chunks_path.suffix + ".tmp")
        meta_tmp = self._meta_path.with_suffix(self._meta_path.suffix + ".tmp")

        faiss.write_index(index, str(idx_tmp))
        chunks_tmp.write_text(json.dumps(self._serialize_chunks(chunks), ensure_ascii=False), encoding="utf-8")
        meta_tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        idx_tmp.replace(self._faiss_index_path)
        chunks_tmp.replace(self._chunks_path)
        meta_tmp.replace(self._meta_path)

    def _build_index_status(
        self,
        files_total: int,
        pdf_files: int,
        txt_files: int,
        chunks_total: int,
        last_indexed_at: str,
        loaded_from_cache: bool,
    ) -> dict[str, Any]:
        return {
            "kg_dir": str(self.root),
            "files_total": files_total,
            "pdf_files": pdf_files,
            "txt_files": txt_files,
            "chunks_total": chunks_total,
            "last_indexed_at": last_indexed_at,
            "backend": "faiss",
            "index_dir": str(self.index_dir),
            "loaded_from_cache": loaded_from_cache,
            "embedding_backend": "openai",
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_available": self.embedding_available,
            "embedding_error": self.embedding_error or "",
        }

    def _serialize_chunks(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "source_type": c.source_type,
                "page": c.page,
                "text": c.text,
                "length": c.length,
            }
            for c in chunks
        ]

    def _deserialize_chunks(self, payload: list[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for item in payload:
            text = str(item.get("text") or "")
            tokens = self._tokenize(text)
            chunks.append(
                Chunk(
                    chunk_id=str(item.get("chunk_id") or ""),
                    source=str(item.get("source") or ""),
                    source_type=str(item.get("source_type") or "txt"),
                    page=(int(item["page"]) if item.get("page") is not None else None),
                    text=text,
                    length=int(item.get("length") or len(tokens)),
                    term_freq=Counter(tokens),
                )
            )
        return chunks

    def _setup_embedding(self) -> None:
        if not self._validate_openai_config():
            return
        try:
            self._openai_client = OpenAI(
                api_key=self.OPENAI_EMBEDDING_API_KEY,
                base_url=OPENAI_EMBEDDING_BASE_URL,
            )
        except Exception as exc:
            self._set_embedding_unavailable(f"init openai client failed: {exc}")
            return

        dim = self._probe_embedding_service()
        if dim is not None:
            self.embedding_dim = dim

    def _validate_openai_config(self) -> bool:
        if not OPENAI_EMBEDDING_MODEL.strip():
            self._set_embedding_unavailable("OPENAI_EMBEDDING_MODEL is empty")
            return False
        if not OPENAI_EMBEDDING_BASE_URL.strip():
            self._set_embedding_unavailable("OPENAI_EMBEDDING_BASE_URL is empty")
            return False
        if not self.OPENAI_EMBEDDING_API_KEY.strip():
            # print(f'warring, OPENAI_EMBEDDING_API_KEY is empty')
            self._set_embedding_unavailable("OPENAI_EMBEDDING_API_KEY is empty")
            return False
        return True

    def _set_embedding_unavailable(self, reason: str) -> None:
        logger.warning("embedding unavailable: %s", reason)
        self.embedding_available = False
        self.embedding_error = reason
        self._index_status["embedding_available"] = False
        self._index_status["embedding_error"] = reason

    def _ensure_kg_dir(self) -> None:
        if self.root.exists():
            return
        if not self.auto_bootstrap:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        bootstrap = self.root / "bootstrap_science.txt"
        bootstrap.write_text(
            (
                "Apple is a fruit. It grows on trees and can be red or green.\n"
                "The moon does not make its own light. We see sunlight reflected by the moon.\n"
                "Plants need water, sunlight, and air to grow.\n"
            ),
            encoding="utf-8",
        )
        logger.info("KG 目录不存在，已创建 bootstrap 文件: %s", bootstrap)

    @staticmethod
    def _collect_corpus_files(root: Path) -> list[Path]:
        if not root.exists():
            return []
        files: list[Path] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}:
                files.append(path)
        return files

    @classmethod
    def _load_documents(cls, root: Path, files: list[Path]) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for path in files:
            source = str(path.relative_to(root))
            suffix = path.suffix.lower()
            if suffix == ".txt":
                text = cls._clean_text(path.read_text(encoding="utf-8", errors="ignore"))
                if text:
                    docs.append(SourceDocument(source=source, source_type="txt", page=None, text=text))
            if suffix == ".pdf":
                for page_no, page_text in cls._read_pdf_pages(path):
                    text = cls._clean_text(page_text)
                    if text:
                        docs.append(SourceDocument(source=source, source_type="pdf", page=page_no, text=text))
        return docs

    @staticmethod
    def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            logger.warning("pypdf 未安装，跳过 PDF: %s", path)
            return []
        try:
            reader = PdfReader(str(path))
            return [(idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)]
        except Exception as exc:
            logger.warning("PDF 解析失败 %s: %s", path, exc)
            return []

    @classmethod
    def _build_chunks(cls, docs: list[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            start = 0
            chunk_no = 0
            while start < len(doc.text):
                end = min(len(doc.text), start + chunk_size)
                raw = doc.text[start:end].strip(" \n\r\t,.;:!?")
                if raw:
                    tokens = cls._tokenize(raw)
                    if tokens:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc.source}:p{doc.page or 0}:{chunk_no}",
                                source=doc.source,
                                source_type=doc.source_type,
                                page=doc.page,
                                text=raw,
                                length=len(tokens),
                                term_freq=Counter(tokens),
                            )
                        )
                        chunk_no += 1
                if end >= len(doc.text):
                    break
                start = max(end - overlap, start + 1)
        return chunks

    @staticmethod
    def _build_index_key(root: Path, model: str) -> str:
        seed = f"{root.resolve()}|openai|{model}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    def _ensure_faiss_available(self) -> None:
        if faiss is not None:
            return
        raise RuntimeError("faiss-cpu 未安装，无法使用向量检索") from _FAISS_IMPORT_ERROR

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        lowered = text.lower()
        tokens: list[str] = re.findall(r"[a-z0-9]+", lowered)
        for seq in re.findall(r"[\u4e00-\u9fff]+", lowered):
            if len(seq) <= 4:
                tokens.append(seq)
            tokens.extend(list(seq))
            if len(seq) >= 2:
                tokens.extend(seq[i : i + 2] for i in range(len(seq) - 1))
        return tokens

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _truncate(text: str, size: int) -> str:
        if len(text) <= size:
            return text
        return text[: size - 3].rstrip() + "..."
