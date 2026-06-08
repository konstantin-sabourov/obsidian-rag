"""
CLI entry point. All commands live here.

Commands:
  index    — ingest and embed vault (incremental by default)
  query    — single-shot Q&A
  chat     — interactive REPL with conversation history
  status   — show ChromaDB stats + Ollama loaded models
  inspect  — show retrieved chunks for a query (no LLM call)
"""
from __future__ import annotations

import subprocess
import sys

import typer
from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Local RAG over your Obsidian vault via Ollama.")
console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def _print_answer(response, show_sources: bool = True) -> None:
    rprint(Panel(response.answer, title="Answer", border_style="green"))
    if show_sources and response.sources:
        rprint(f"\n[dim]Sources:[/dim] {', '.join(f'[[{s}]]' for s in response.sources)}")
    rprint(f"[dim]Model: {response.model_used} | Chunks used: {response.chunks_used}[/dim]\n")


@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Re-embed all notes, ignoring cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest and embed vault into ChromaDB. Incremental by default."""
    from .config import get_settings
    from .indexer import index_vault

    settings = get_settings()
    _setup_logging("DEBUG" if verbose else settings.logging.level)

    rprint(f"[bold]Vault:[/bold] {settings.vault_path}")
    rprint(f"[bold]Collection:[/bold] {settings.chroma_collection_name}")
    rprint(f"[bold]Embedding model:[/bold] {settings.embedding.model}")
    if force:
        rprint("[yellow]--force: re-embedding all notes[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing vault...", total=None)
        summary = index_vault(force=force, settings=settings)
        progress.update(task, completed=True)

    table = Table(title="Indexing Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    for k, v in summary.items():
        table.add_row(k.capitalize(), str(v))
    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int | None = typer.Option(None, "--top-k", "-k", help="Number of chunks to retrieve"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Single-shot question answering over your vault."""
    from .config import get_settings
    from .generator import generate
    from .retriever import retrieve

    settings = get_settings()
    _setup_logging("DEBUG" if verbose else settings.logging.level)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as p:
        t = p.add_task("Retrieving...", total=None)
        chunks = retrieve(question, top_k=top_k, settings=settings)
        p.update(t, description="Generating answer...")
        response = generate(question, chunks, settings=settings)
        p.update(t, completed=True)

    _print_answer(response, show_sources=settings.logging.show_sources)


@app.command()
def chat(
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive chat session with conversation history."""
    from .config import get_settings
    from .generator import generate
    from .retriever import retrieve

    settings = get_settings()
    _setup_logging("DEBUG" if verbose else settings.logging.level)

    rprint(Panel(
        f"[bold]Obsidian RAG Chat[/bold]\n"
        f"Vault: {settings.vault_path.name} | LLM: {settings.llm.model}\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to end. [bold]/clear[/bold] to reset history.",
        border_style="blue",
    ))

    history: list[dict] = []

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            rprint("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            rprint("[dim]Goodbye.[/dim]")
            break
        if user_input == "/clear":
            history.clear()
            rprint("[dim]History cleared.[/dim]")
            continue

        with Progress(
            SpinnerColumn(),
            TextColumn("Thinking..."),
            console=console,
            transient=True,
        ) as p:
            p.add_task("", total=None)
            chunks = retrieve(user_input, top_k=top_k, settings=settings)
            response = generate(user_input, chunks, conversation_history=history, settings=settings)

        _print_answer(response, show_sources=settings.logging.show_sources)

        # Maintain history for context
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response.answer})


@app.command()
def status() -> None:
    """Show ChromaDB collection stats and currently loaded Ollama models."""
    from .config import get_settings
    from .indexer import collection_stats

    settings = get_settings()

    # ChromaDB stats
    try:
        stats = collection_stats(settings)
        table = Table(title="ChromaDB Collection")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for k, v in stats.items():
            table.add_row(k.capitalize(), str(v))
        console.print(table)
    except Exception as e:
        rprint(f"[red]ChromaDB error:[/red] {e}")

    # Ollama status
    rprint("\n[bold]Ollama loaded models:[/bold]")
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        rprint(result.stdout or "[dim](none running)[/dim]")
    except FileNotFoundError:
        rprint("[red]ollama not found in PATH[/red]")

    rprint("\n[bold]Config:[/bold]")
    rprint(f"  Embedding model: {settings.embedding.model}")
    rprint(f"  LLM model:       {settings.llm.model}")
    rprint(f"  Vault:           {settings.vault_path}")
    rprint(f"  ChromaDB:        {settings.chroma_path}")


@app.command()
def inspect(
    query_text: str = typer.Argument(..., help="Query to inspect retrieval for"),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
) -> None:
    """Show retrieved chunks for a query without calling the LLM."""
    from .config import get_settings
    from .retriever import retrieve

    settings = get_settings()
    _setup_logging(settings.logging.level)

    chunks = retrieve(query_text, top_k=top_k, settings=settings)

    if not chunks:
        rprint("[yellow]No chunks retrieved. Is the vault indexed?[/yellow]")
        return

    for i, chunk in enumerate(chunks, 1):
        rprint(Panel(
            chunk.text,
            title=f"[{i}] {chunk.source_label}  (score: {chunk.score:.3f})"
            + (f"  tags: {', '.join(chunk.tags)}" if chunk.tags else ""),
            border_style="dim",
        ))