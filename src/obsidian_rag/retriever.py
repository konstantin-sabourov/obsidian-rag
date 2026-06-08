"""
Semantic retrieval from ChromaDB. No LLM calls here.

retrieve() → list of RetrievedChunk (text + metadata + score)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from loguru import logger

from .config import Settings, get_settings


@dataclass
class RetrievedChunk:
    text: str
    score: float
    note_title: str
    file_path: str
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        return self.note_title or self.file_path or "unknown"


def _parse_csv_field(val: str | None) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def retrieve(
    query: str,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """
    Embed query with nomic-embed-text, retrieve top-k chunks from ChromaDB.
    Returns chunks sorted by relevance (highest score first).
    """
    if settings is None:
        settings = get_settings()

    k = top_k or settings.retrieval.top_k

    logger.debug(f"Retrieving top-{k} chunks for query: {query!r}")
    logger.debug(f"Embedding model: {settings.embedding.model}")

    embed_model = OllamaEmbedding(
        model_name=settings.embedding.model,
        base_url=settings.embedding.ollama_base_url,
    )
    query_embedding = embed_model.get_query_embedding(query)

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — run 'obsidian-rag index' first")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity score (1 = identical)
        score = 1.0 - dist
        chunks.append(RetrievedChunk(
            text=doc,
            score=score,
            note_title=meta.get("note_title", ""),
            file_path=meta.get("file_path", ""),
            tags=_parse_csv_field(meta.get("tags")),
            wikilinks=_parse_csv_field(meta.get("wikilinks")),
            metadata=meta,
        ))

    logger.debug(
        f"Retrieved {len(chunks)} chunks, top score: {chunks[0].score:.3f}"
        if chunks
        else "No chunks retrieved"
    )
    return chunks