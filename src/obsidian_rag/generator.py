"""
Prompt construction and LLM call via Ollama.
Always logs which model is used.
Returns answer text + list of cited source notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage
from loguru import logger

from .config import Settings, get_settings
from .retriever import RetrievedChunk

_SYSTEM_PROMPT = """\
You are a knowledgeable assistant with access to the user's personal Obsidian notes.
Answer questions using ONLY the context provided from those notes.
If the context does not contain enough information to answer, say so clearly.
Always be specific and technical — the user is an expert engineer with a PhD in physics.
Cite the source note(s) by name when you use information from them.
"""

_RAG_TEMPLATE = """\
## Context from your notes

{context_block}

---

## Question

{query}

## Answer
"""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[str]       # deduplicated note titles
    model_used: str
    chunks_used: int


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] **{chunk.source_label}**"
        if chunk.tags:
            header += f"  (tags: {', '.join(chunk.tags)})"
        parts.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(parts)


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: set[str] = set()
    sources = []
    for c in chunks:
        label = c.source_label
        if label and label not in seen:  # Only add non-empty labels
            seen.add(label)
            sources.append(label)
    return sources


def generate(
    query: str,
    chunks: list[RetrievedChunk],
    conversation_history: Optional[list[dict]] = None,
    settings: Optional[Settings] = None,
) -> GeneratorResponse:
    """
    Generate an answer from retrieved chunks.

    conversation_history: list of {"role": "user"|"assistant", "content": str}
    for multi-turn chat mode.
    """
    if settings is None:
        settings = get_settings()

    model = settings.llm.model
    if settings.logging.show_model_on_query:
        logger.info(f"LLM: {model} | Embedding: {settings.embedding.model} | Chunks: {len(chunks)}")

    llm = Ollama(
        model=model,
        base_url=settings.llm.ollama_base_url,
        temperature=settings.llm.temperature,
        context_window=settings.llm.context_window,
        request_timeout=settings.llm.request_timeout,
        system_prompt=_SYSTEM_PROMPT,
    )

    if not chunks:
        answer = (
            "I couldn't find relevant information in your notes for this query. "
            "Try re-indexing with `obsidian-rag index` or rephrasing your question."
        )
        return GeneratorResponse(answer=answer, sources=[], model_used=model, chunks_used=0)

    context_block = _build_context_block(chunks)
    prompt = _RAG_TEMPLATE.format(context_block=context_block, query=query)

    # Build message list for chat mode
    messages = []
    if conversation_history:
        # Convert dict history to ChatMessage objects
        for msg in conversation_history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
    messages.append(ChatMessage(role="user", content=prompt))

    response = llm.chat(messages)
    answer = response.message.content.strip()
    sources = _dedupe_sources(chunks)

    return GeneratorResponse(
        answer=answer,
        sources=sources,
        model_used=model,
        chunks_used=len(chunks),
    )