# obsidian-rag

A fully local RAG (Retrieval-Augmented Generation) pipeline over an Obsidian vault.
Runs entirely on your machine via Ollama — no cloud APIs, no data leaves your device.

## Features

- Ingests Obsidian markdown with frontmatter, tags, and wikilink awareness
- Embeds with `nomic-embed-text` via Ollama
- Persists vectors in ChromaDB (local, on-disk)
- Queries via local LLM (Qwen2.5 32B, Llama 3.3 70B, or any Ollama model)
- CLI interface with source-note citations on every answer
- Config-driven model selection — swap models without touching code
- Designed for reproducibility: vault path is external config, repo is vault-agnostic

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- [Ollama](https://ollama.com) running locally
- Required Ollama models pulled:

  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:32b        # daily driver
  ollama pull mistral-nemo       # fast iteration / dev
  # optional, best quality:
  ollama pull llama3.3:70b-instruct-q4_K_M
  ```

- Obsidian vault with [Local REST API plugin](https://github.com/coddingtonbear/obsidian-local-rest-api) installed and enabled

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/obsidian-rag
cd obsidian-rag

# Copy and edit config
cp .env.example .env
# Edit .env: set VAULT_PATH and optionally OBSIDIAN_API_KEY

# Index your vault (first run: takes a few minutes)
uv run obsidian-rag index

# Query
uv run obsidian-rag query "what did I know about library prep QC?"

# Interactive mode
uv run obsidian-rag chat
```

## Configuration

All config lives in two files:

### `.env` — paths and secrets

```bash
VAULT_PATH=/Users/you/path/to/your/vault
OBSIDIAN_API_PORT=27123
OBSIDIAN_API_KEY=your_key_if_set   # from Local REST API plugin settings
CHROMA_PATH=./data/chroma
```

### `config.yaml` — model and retrieval settings

```yaml
embedding:
  model: nomic-embed-text
  ollama_base_url: http://localhost:11434

llm:
  model: qwen2.5:32b               # change to mistral-nemo for speed
  ollama_base_url: http://localhost:11434
  temperature: 0.1
  context_window: 8192

retrieval:
  top_k: 6                         # chunks to retrieve per query
  chunk_size: 512
  chunk_overlap: 64
  metadata_fields:                 # frontmatter fields to preserve
    - tags
    - aliases
    - created
    - modified

logging:
  show_model_on_query: true        # always log which model answered
  show_sources: true               # cite source notes in output
```

## CLI Reference

```bash
# Index / re-index vault
uv run obsidian-rag index [--force]        # --force re-embeds everything

# Single query
uv run obsidian-rag query "your question"

# Interactive chat session
uv run obsidian-rag chat

# Show what's loaded in Ollama right now
uv run obsidian-rag status

# Inspect indexed notes
uv run obsidian-rag inspect --query "library prep"
```

## Project Structure

```code
obsidian-rag/
├── pyproject.toml
├── uv.lock
├── config.yaml
├── .env.example
├── README.md
├── CLAUDE.md                  # instructions for Claude Code
├── src/
│   └── obsidian_rag/
│       ├── __init__.py
│       ├── config.py          # pydantic-settings config loader
│       ├── indexer.py         # vault ingestion + embedding
│       ├── retriever.py       # semantic search + optional rerank
│       ├── generator.py       # prompt construction + LLM call
│       └── cli.py             # typer CLI entry point
├── data/
│   └── chroma/                # gitignored, persistent vector store
├── tests/
│   ├── test_indexer.py
│   └── test_retriever.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Monitoring

```bash
# Which models are loaded in Ollama memory right now
ollama ps

# Live Ollama server log
tail -f ~/.ollama/logs/server.log

# ChromaDB collection stats
uv run obsidian-rag status
```

## Extending

- **Different vault:** change `VAULT_PATH` in `.env`, run `index --force`
- **Different LLM:** change `llm.model` in `config.yaml`, no code changes
- **Add reranking:** set `retrieval.reranker: bge-reranker-v2-m3` in config (pulls via Ollama)
- **Cloud deploy:** swap ChromaDB for Qdrant Cloud, Ollama for any OpenAI-compatible endpoint

## License

MIT
