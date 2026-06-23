"""Minimal FAISS vector-store helpers used by the public repository."""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_literature_db_path(collection_name: str) -> Optional[Path]:
    bundled_root = PROJECT_ROOT / "knowledge_base" / "vector_db"
    if collection_name == "literature":
        candidate = bundled_root / "vector_db_new_para" / "literature"
    else:
        candidate = bundled_root / collection_name
    return candidate if candidate.exists() else None


def _resolve_persist_directory(persist_directory: str | Path) -> Path:
    """Resolve a persist directory to an absolute path."""
    path = Path(persist_directory)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def init_vector_store(
    collection_name: str,
    persist_directory: str,
    embedding_model: Optional[Embeddings] = None,
) -> FAISS:
    """Initialize a FAISS vector store."""
    if embedding_model is None:
        from .embeddings import get_embedding_model

        embedding_model = get_embedding_model()

    persist_path = _resolve_persist_directory(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    index_path = persist_path / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            str(persist_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    dummy_doc = Document(
        page_content="Placeholder document for initializing an empty vector store.",
        metadata={"type": "placeholder"},
    )
    vector_store = FAISS.from_documents([dummy_doc], embedding_model)
    vector_store.save_local(str(persist_path))
    return vector_store


def get_vector_store(
    collection_name: str = "literature",
    persist_directory: Optional[str] = None,
) -> FAISS:
    """Get a FAISS vector store."""
    from .embeddings import get_embedding_model

    if persist_directory is None:
        default_path = _default_literature_db_path(collection_name)
        if default_path is None:
            raise FileNotFoundError(
                "No bundled literature vector store is available. Set LITERATURE_DB_PATH to an external store."
            )
        persist_directory = str(default_path)

    persist_path = _resolve_persist_directory(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()
    index_path = persist_path / "index.faiss"
    if index_path.exists():
        return FAISS.load_local(
            str(persist_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    return init_vector_store(
        collection_name=collection_name,
        persist_directory=str(persist_path),
        embedding_model=embedding_model,
    )


def add_documents(
    vector_store: FAISS,
    documents: List[Document],
    persist_directory: str,
) -> FAISS:
    """Add documents to a FAISS vector store and persist it."""
    if documents:
        persist_path = _resolve_persist_directory(persist_directory)
        vector_store.add_documents(documents)
        vector_store.save_local(str(persist_path))
    return vector_store
