"""
Vault ingestion and embedding.

Flow:
  ObsidianReader → Documents → chunk (note→paragraph→size) →
  embed (nomic-embed-text via Ollama) → upsert ChromaDB

Incremental indexing: skips notes unchanged since last run (via 'modified' metadata).
Use index_vault(force=True) to re-embed everything.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from loguru import logger

from .config import Settings, get_settings


def _extract_wikilinks(text: str) -> list[str]:
    """Extract [[wikilink]] targets from markdown text."""
    return re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", text)


def _make_doc_id(file_path: Path, vault_path: Path) -> str:
    """Stable ID from vault-relative path."""
    rel = file_path.relative_to(vault_path)
    return hashlib.sha256(str(rel).encode()).hexdigest()[:16]


def _load_vault_documents(vault_path: Path, metadata_fields: list[str]) -> list[Document]:
    """
    Load all markdown files from vault using SimpleDirectoryReader.
    ObsidianReader is preferred but falls back gracefully.
    Injects note_title, file_path, and wikilinks into metadata.
    """
    try:
        from llama_index.readers.obsidian import ObsidianReader
        reader = ObsidianReader(input_dir=str(vault_path))
        docs = reader.load_data()
        logger.info(f"Loaded {len(docs)} documents via ObsidianReader")
    except Exception as e:
        logger.warning(f"ObsidianReader failed ({e}), falling back to SimpleDirectoryReader")
        reader = SimpleDirectoryReader(
            input_dir=str(vault_path),
            recursive=True,
            required_exts=[".md"],
        )
        docs = reader.load_data()
        logger.info(f"Loaded {len(docs)} documents via SimpleDirectoryReader")

    # Enrich metadata
    for doc in docs:
        fp = Path(doc.metadata.get("file_path", ""))
        doc.metadata["note_title"] = fp.stem if fp.name else "unknown"
        doc.metadata["wikilinks"] = _extract_wikilinks(doc.text)
        # Keep only configured metadata fields + our additions
        keep = set(metadata_fields) | {"note_title", "file_path", "wikilinks"}
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep}

    return docs


def _get_chroma_collection(settings: Settings) -> chromadb.Collection:
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _get_embedding_model(settings: Settings) -> OllamaEmbedding:
    logger.info(
        f"Embedding model: {settings.embedding.model} @ {settings.embedding.ollama_base_url}"
    )
    return OllamaEmbedding(
        model_name=settings.embedding.model,
        base_url=settings.embedding.ollama_base_url,
    )


def _chunk_documents(docs: list[Document], settings: Settings) -> list[TextNode]:
    splitter = SentenceSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"Chunked into {len(nodes)} nodes")
    return nodes


def _get_existing_ids(collection: chromadb.Collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def index_vault(force: bool = False, settings: Settings | None = None) -> dict:
    """
    Main entry point. Indexes (or re-indexes) the vault.

    Returns a summary dict: {total, added, skipped, errors}
    """
    if settings is None:
        settings = get_settings()

    logger.info(f"Vault: {settings.vault_path} (collection: {settings.chroma_collection_name})")

    docs = _load_vault_documents(settings.vault_path, settings.retrieval.metadata_fields)
    nodes = _chunk_documents(docs, settings)
    embed_model = _get_embedding_model(settings)
    collection = _get_chroma_collection(settings)
    existing_ids = set() if force else _get_existing_ids(collection)

    added = skipped = errors = 0

    for node in nodes:
        node_id = node.node_id
        if node_id in existing_ids:
            skipped += 1
            continue
        try:
            embedding = embed_model.get_text_embedding(node.text)
            collection.upsert(
                ids=[node_id],
                embeddings=[embedding],
                documents=[node.text],
                metadatas=[{
                    k: (", ".join(v) if isinstance(v, list) else str(v))
                    for k, v in node.metadata.items()
                    if v is not None
                }],
            )
            added += 1
        except Exception as e:
            logger.error(f"Failed to embed node {node_id}: {e}")
            errors += 1

    summary = {"total": len(nodes), "added": added, "skipped": skipped, "errors": errors}
    logger.info(f"Indexing complete: {summary}")
    return summary


def collection_stats(settings: Settings | None = None) -> dict:
    if settings is None:
        settings = get_settings()
    collection = _get_chroma_collection(settings)
    return {
        "collection": settings.chroma_collection_name,
        "count": collection.count(),
        "vault": str(settings.vault_path),
    }