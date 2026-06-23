"""Literature retriever (vector + optional BM25 hybrid)."""

import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .vector_store import get_vector_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_literature_db_path(collection_name: str) -> Optional[str]:
    bundled_root = PROJECT_ROOT / "knowledge_base" / "vector_db"
    if collection_name == "literature":
        candidate = bundled_root / "vector_db_new_para" / "literature"
    else:
        candidate = bundled_root / collection_name
    return str(candidate) if candidate.exists() else None


def _ensemble_retriever_cls() -> Type:
    """Return EnsembleRetriever without importing langchain.retrievers package __init__."""
    try:
        from langchain.retrievers import EnsembleRetriever

        return EnsembleRetriever
    except Exception:
        import langchain as _lc

        ensemble_path = Path(_lc.__file__).resolve().parent / "retrievers" / "ensemble.py"
        if not ensemble_path.is_file():
            raise ImportError(f"Cannot load EnsembleRetriever: missing {ensemble_path}") from None
        name = "langchain.retrievers._ensemble_bootstrap"
        spec = importlib.util.spec_from_file_location(name, ensemble_path)
        if spec is None or spec.loader is None:
            raise ImportError("importlib could not load langchain retrievers/ensemble.py") from None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.EnsembleRetriever


class LiteratureRetriever:
    def __init__(
        self,
        collection_name: str = "literature",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = (
            persist_directory
            or os.getenv("LITERATURE_DB_PATH")
            or _default_literature_db_path(collection_name)
        )
        self.vector_store = None
        self._hybrid_retriever = None
        self._hybrid_enabled = False

    def _ensure_vector_store(self):
        if not self.persist_directory:
            raise FileNotFoundError(
                "Literature DB is not configured. Set LITERATURE_DB_PATH to an external vector-store directory."
            )
        if self.vector_store is None:
            self.vector_store = get_vector_store(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )

    def _ensure_hybrid_retriever(self, top_k: int):
        self._ensure_vector_store()
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": max(1, top_k * 2)})

        bm25_path = Path(self.persist_directory) / "bm25_retriever.pkl"
        if not bm25_path.exists():
            self._hybrid_enabled = False
            self._hybrid_retriever = vector_retriever
            return

        try:
            with bm25_path.open("rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = max(1, top_k * 2)
            ensemble_cls = _ensemble_retriever_cls()
            self._hybrid_retriever = ensemble_cls(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7],
            )
            self._hybrid_enabled = True
        except Exception:
            self._hybrid_enabled = False
            self._hybrid_retriever = vector_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve literature snippets for a query."""
        if not query.strip():
            return []

        try:
            self._ensure_hybrid_retriever(top_k)
            if self._hybrid_enabled:
                docs = self._hybrid_retriever.invoke(query)
                if not docs:
                    return []
                docs_with_scores = [(doc, 0.0) for doc in docs[:top_k]]
            else:
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

            if not docs_with_scores:
                return []

            results = []
            for doc, score in docs_with_scores:
                similarity = max(0.0, 1.0 - float(score))
                if similarity_threshold and similarity < similarity_threshold:
                    continue
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "distance": float(score),
                        "similarity": similarity,
                    }
                )
            return results
        except Exception as exc:
            print(f"[WARN] Literature retrieval failed: {exc}")
            return []

    def retrieve_formatted(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
        max_chars: int = 400,
    ) -> str:
        """Return a compact snippet string for tool observations."""
        results = self.retrieve(query, top_k, similarity_threshold)
        if not results:
            return "No literature matched."

        lines = [f"Retrieved {len(results)} literature snippets:"]
        for result in results:
            metadata = result.get("metadata") or {}
            source = metadata.get("source") or metadata.get("title") or "unknown"
            snippet = result["content"]
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            lines.append(f"[Source: {source}] {snippet}")
        return "\n".join(lines)
