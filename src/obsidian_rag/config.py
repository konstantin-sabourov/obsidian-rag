"""
Central config loader. Single source of truth for all settings.
Reads .env (via pydantic-settings) and config.yaml, merges into Settings.
Use get_settings() everywhere — never read env vars or yaml directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"


def _load_yaml() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


class EmbeddingConfig(BaseSettings):
    model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"


class LLMConfig(BaseSettings):
    model: str = "qwen2.5:32b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    context_window: int = 8192
    request_timeout: float = 120.0


class RetrievalConfig(BaseSettings):
    top_k: int = 6
    chunk_size: int = 512
    chunk_overlap: int = 64
    metadata_fields: list[str] = ["tags", "aliases", "created", "modified"]
    reranker: Optional[str] = None


class LoggingConfig(BaseSettings):
    show_model_on_query: bool = True
    show_sources: bool = True
    level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # From .env
    vault_path: Path = Field(..., description="Path to Obsidian vault")
    chroma_path: Path = Field(Path("./data/chroma"))
    obsidian_api_port: int = 27123
    obsidian_api_key: str = ""

    # Allow env override of model names
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None

    # Nested config (populated from yaml in get_settings)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("vault_path")
    @classmethod
    def vault_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"VAULT_PATH does not exist: {v}")
        return v.expanduser().resolve()

    @property
    def vault_name(self) -> str:
        return self.vault_path.name

    @property
    def chroma_collection_name(self) -> str:
        # Sanitize vault name for use as ChromaDB collection name
        return self.vault_name.lower().replace(" ", "_").replace("-", "_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    yaml_cfg = _load_yaml()

    # Build nested config objects from yaml
    emb_cfg = EmbeddingConfig(**(yaml_cfg.get("embedding", {})))
    llm_cfg = LLMConfig(**(yaml_cfg.get("llm", {})))
    ret_cfg = RetrievalConfig(**(yaml_cfg.get("retrieval", {})))
    log_cfg = LoggingConfig(**(yaml_cfg.get("logging", {})))

    settings = Settings(
        embedding=emb_cfg,
        llm=llm_cfg,
        retrieval=ret_cfg,
        logging=log_cfg,
    )

    # Env var overrides for model names
    if settings.embedding_model:
        settings.embedding.model = settings.embedding_model
    if settings.llm_model:
        settings.llm.model = settings.llm_model

    return settings