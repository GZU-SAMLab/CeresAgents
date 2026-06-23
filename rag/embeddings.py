"""Embedding model factory for RAG."""

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)


def get_embedding_model(
    model_name: str = None,
    api_key: str = None,
    device: str = "cuda",
) -> Embeddings:
    """Return a local HuggingFace embeddings instance."""
    del api_key
    resolved_model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    print(f"Loading local embedding model: {resolved_model_name} on {device}...")
    return HuggingFaceEmbeddings(
        model_name=resolved_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
