from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "level3_pdf_chunks_v1"


def _to_2d_float_list(values: Any) -> list[list[float]]:
    """
    Convert a NumPy array / list / tensor-like embedding into
    the 2D list format expected by ChromaDB.
    """
    array = np.asarray(values, dtype=np.float32)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"Expected a 1D or 2D embedding array, received shape {array.shape}."
        )

    return array.tolist()


def get_chroma_collection():
    """
    Open or create a persistent ChromaDB collection.

    We provide embeddings ourselves because the project already has a working
    embedding pipeline in src/embeddings.py.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Current Chroma versions use `configuration`.
    # The fallback supports older installed Chroma versions.
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )


def create_pdf_hash(file_bytes: bytes) -> str:
    """Create a stable identity for the actual PDF contents."""
    return hashlib.sha256(file_bytes).hexdigest()


def create_index_id(
    pdf_hash: str,
    chunk_size: int,
    overlap: int,
    embedding_model_id: str,
) -> str:
    """
    The same PDF processed with different chunk settings or embedding models
    is treated as a different index version.
    """
    raw_key = (
        f"{pdf_hash}|chunk={chunk_size}|overlap={overlap}|"
        f"embedding={embedding_model_id}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def index_exists(index_id: str) -> bool:
    collection = get_chroma_collection()

    result = collection.get(
        where={"index_id": index_id},
        limit=1,
        include=["metadatas"],
    )

    return len(result.get("ids", [])) > 0


def store_chunks(
    *,
    ids: list[str],
    chunks: list[str],
    embeddings: Any,
    metadatas: list[dict],
) -> None:
    """
    Store or update chunks, precomputed embeddings, and metadata.
    Stable IDs make the operation idempotent.
    """
    if not (len(ids) == len(chunks) == len(metadatas)):
        raise ValueError(
            "ids, chunks, and metadatas must contain the same number of items."
        )

    embedding_rows = _to_2d_float_list(embeddings)

    if len(embedding_rows) != len(chunks):
        raise ValueError(
            "Number of embeddings must match the number of chunks."
        )

    collection = get_chroma_collection()
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embedding_rows,
        metadatas=metadatas,
    )


def list_indexed_versions() -> list[dict]:
    """
    Return one catalog entry per indexed PDF/configuration version.
    """
    collection = get_chroma_collection()
    result = collection.get(include=["metadatas"])

    versions: dict[str, dict] = {}

    for metadata in result.get("metadatas", []):
        if not metadata:
            continue

        index_id = metadata.get("index_id")
        if not index_id or index_id in versions:
            continue

        versions[index_id] = {
            "index_id": index_id,
            "source": metadata.get("source", "Unknown PDF"),
            "pdf_hash": metadata.get("pdf_hash", ""),
            "chunk_size": metadata.get("chunk_size", "Unknown"),
            "overlap": metadata.get("overlap", "Unknown"),
            "embedding_model": metadata.get(
                "embedding_model", "Unknown embedding model"
            ),
        }

    return sorted(
        versions.values(),
        key=lambda item: (
            str(item["source"]).lower(),
            str(item["chunk_size"]),
            str(item["overlap"]),
        ),
    )


def count_records(where_filter: dict | None = None) -> int:
    collection = get_chroma_collection()

    if where_filter is None:
        return collection.count()

    result = collection.get(
        where=where_filter,
        include=["metadatas"],
    )
    return len(result.get("ids", []))


def search_chroma(
    *,
    query_embedding: Any,
    top_k: int,
    where_filter: dict | None = None,
) -> list[dict]:
    """
    Query persistent embeddings and return the same core fields that the
    Level 2 answer-generation code expects.
    """
    available_records = count_records(where_filter)

    if available_records == 0:
        return []

    effective_top_k = min(top_k, available_records)
    collection = get_chroma_collection()

    query_arguments = {
        "query_embeddings": _to_2d_float_list(query_embedding),
        "n_results": effective_top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter is not None:
        query_arguments["where"] = where_filter

    result = collection.query(**query_arguments)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved_chunks = []

    for rank, document in enumerate(documents):
        metadata = metadatas[rank] if rank < len(metadatas) else {}
        distance = distances[rank] if rank < len(distances) else None

        # For cosine space, Chroma distance = 1 - cosine similarity.
        similarity_score = 1.0 - float(distance) if distance is not None else 0.0

        retrieved_chunks.append(
            {
                "chunk_text": document,
                "chunk_index": int(metadata.get("chunk_id", rank)),
                "similarity_score": similarity_score,
                "distance": float(distance) if distance is not None else None,
                "metadata": metadata,
            }
        )

    return retrieved_chunks
