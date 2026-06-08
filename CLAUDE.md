# CLAUDE.md — Instructions for Claude Code

This file tells Claude Code how to work in this repository.

## Project Purpose

Local RAG pipeline over an Obsidian vault. Everything runs on-device via Ollama.
No cloud APIs are used by default. The user has an M4 Max Mac Studio with 64 GB RAM.

## Key Design Constraints

- **Local-only by default.** Never introduce OpenAI, Anthropic, or other cloud API calls
  without an explicit opt-in flag and a comment explaining the cost implication.
- **Vault-agnostic.** Vault path comes from `.env`, never hardcoded. The repo must work
  on any vault by changing config only.
- **Model-agnostic.** All model names come from `config.yaml`. Never hardcode a model name
  in source code — always read from `settings`.
- **uv for everything.** Use `uv add` to add dependencies, never `pip install`.
  Do not use `poetry`, `pipenv`, or bare `pip`.
- **Always log which model is answering** when `logging.show_model_on_query` is true.

## Preferred Stack

| Concern | Library | Notes |
|---|---|---|
| Vault ingestion | `llama-index-readers-obsidian` | Handles frontmatter, wikilinks |
| Embeddings | `llama-index-embeddings-ollama` | `nomic-embed-text` by default |
| Vector store | `chromadb` | Local persistent, no server needed |
| LLM | `llama-index-llms-ollama` | `qwen2.5:32b` by default |
| Config | `pydantic-settings` + `pyyaml` | `.env` + `config.yaml` |
| CLI | `typer` + `rich` | Clean terminal UX |
| Testing | `pytest` | Keep tests fast, mock Ollama calls |

## Ollama Models Available on This Machine

```
nomic-embed-text      # embeddings — always use this
mistral-nemo          # fast, use for dev/testing
qwen2.5:32b           # daily driver, good quality/speed balance
llama3.3:70b-instruct-q4_K_M  # best quality, slower
```

Check what's running: `ollama ps`
Check server log: `tail -f ~/.ollama/logs/server.log`

## Code Style

- Python 3.11+, type annotations everywhere
- Pydantic models for any structured data passed between modules
- `loguru` preferred over stdlib `logging` (add as dep if needed)
- Functions over classes where state isn't needed
- Each module (`indexer`, `retriever`, `generator`) should be independently importable
  and testable without a running Ollama instance (mock the client in tests)

## Module Responsibilities

### `config.py`
- Load `.env` via pydantic-settings `BaseSettings`
- Load `config.yaml` and merge into a single `Settings` object
- Expose a `get_settings()` function (cached with `lru_cache`)
- Never read env vars or yaml anywhere else in the codebase

### `indexer.py`
- Read vault using `ObsidianReader`
- Chunk documents respecting note boundaries first, then paragraph, then size
- Preserve these metadata fields per chunk: `note_title`, `tags`, `aliases`,
  `file_path`, `created`, `modified`, `wikilinks` (extracted from content)
- Embed with Ollama `nomic-embed-text`
- Upsert into ChromaDB collection named after the vault basename
- Support incremental re-index: skip notes whose `modified` timestamp hasn't changed
- `index --force` re-embeds everything

### `retriever.py`
- Take a query string, embed it, retrieve top-k chunks from ChromaDB
- Return chunks with their metadata (source note title + path)
- Optional: cross-encoder rerank if configured
- Never call the LLM here

### `generator.py`
- Take retrieved chunks + original query
- Build a prompt that includes source note titles
- Call Ollama LLM
- Return answer + list of source notes cited
- Log model name used on every call

### `cli.py`
- `index` command: runs indexer, shows progress bar via `rich`
- `query` command: single shot Q&A, prints answer + sources
- `chat` command: interactive REPL, maintains conversation history
- `status` command: shows ChromaDB collection stats + `ollama ps` output
- `inspect` command: shows what chunks would be retrieved for a query, without generating

## Common Tasks for Claude Code

### Adding a new dependency
```bash
uv add <package>
# then update pyproject.toml comment if the dep needs explanation
```

### Running tests
```bash
uv run pytest tests/ -v
```

### Running the CLI during development
```bash
uv run obsidian-rag status
uv run obsidian-rag index --force
uv run obsidian-rag query "test question"
```

### Checking Ollama
```bash
ollama ps                          # loaded models
ollama list                        # installed models
curl http://localhost:11434/api/tags  # programmatic check
```

## Things to Avoid

- Do not use `langchain` — LlamaIndex is the chosen framework
- Do not use `faiss` directly — ChromaDB handles persistence cleanly
- Do not add async complexity unless there's a clear need — this is a CLI tool
- Do not hardcode any paths, model names, or API keys
- Do not add a Gradio/Streamlit UI unless explicitly asked — keep it CLI-first
- Do not call `ollama pull` programmatically — models are assumed to be pre-pulled

## Future Extension Points (don't build yet, just keep in mind)

- Swap ChromaDB → Qdrant for cloud deploy
- Swap Ollama LLM → OpenAI-compatible endpoint (same LlamaIndex interface)
- Add a Dataview-query pre-filter before semantic search
- MCP server wrapper so this can be used as a Claude tool
- GitHub Actions CI: lint (ruff) + tests (pytest, mocked Ollama)