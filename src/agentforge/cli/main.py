"""AgentForge CLI — single entry point for the full LLM workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="agentforge",
    help="End-to-end LLM fine-tuning, serving, and agentic workflow for Apple Silicon.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

dataset_app = typer.Typer(help="Dataset management commands.", no_args_is_help=True)
agent_app = typer.Typer(help="Run agentic tasks against the local model.", no_args_is_help=True)

app.add_typer(dataset_app, name="dataset")
app.add_typer(agent_app, name="agent")

console = Console()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Show system memory, Metal GPU stats, and current configuration."""
    from agentforge.config import get_config
    import psutil

    cfg = get_config(config)

    # System memory
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024**3
    used_gb = mem.used / 1024**3
    avail_gb = mem.available / 1024**3

    # Metal memory (MLX)
    metal_active = metal_peak = None
    try:
        import mlx.core as mx
        metal_active = mx.metal.get_active_memory() / 1024**3
        metal_peak = mx.metal.get_peak_memory() / 1024**3
    except Exception:
        pass

    table = Table(title="AgentForge Status", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Base model", cfg.model.base_model)
    table.add_row("Adapter path", cfg.model.adapter_path or "[dim]not set[/dim]")
    table.add_row("Fused model path", cfg.model.fused_model_path)
    table.add_row("System RAM total", f"{total_gb:.1f} GB")
    table.add_row("System RAM used", f"{used_gb:.1f} GB")
    table.add_row("System RAM available", f"{avail_gb:.1f} GB")

    if metal_active is not None:
        table.add_row("Metal active memory", f"{metal_active:.2f} GB")
        table.add_row("Metal peak memory", f"{metal_peak:.2f} GB")
    else:
        table.add_row("Metal memory", "[dim]MLX not loaded[/dim]")

    table.add_row("Training data dir", cfg.training.data_path)
    table.add_row("Training output dir", cfg.training.output_dir)
    table.add_row("Server", f"{cfg.server.host}:{cfg.server.port}")
    table.add_row("Agent API base", cfg.agent.api_base)

    console.print(table)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

@dataset_app.command("pull")
def dataset_pull(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    hermes_only: bool = typer.Option(False, "--hermes-only"),
    toolbench_only: bool = typer.Option(False, "--toolbench-only"),
    custom: Optional[Path] = typer.Option(None, "--custom", help="Path to custom JSON/JSONL file"),
):
    """Download and normalize datasets to ./data (mlx_lm-compatible JSONL)."""
    from agentforge.config import get_config
    from agentforge.datasets.processors.hermes import HermesProcessor
    from agentforge.datasets.processors.toolbench import ToolBenchProcessor
    from agentforge.datasets.processors.custom import CustomProcessor

    cfg = get_config(config)
    output_dir = Path(cfg.datasets.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processors_to_run = []

    for source in cfg.datasets.sources:
        if not source.enabled:
            continue
        if hermes_only and source.name != "hermes":
            continue
        if toolbench_only and source.name != "toolbench":
            continue

        if source.name == "hermes":
            processors_to_run.append(
                HermesProcessor(source.hf_repo, source.max_samples, cfg.datasets.train_split)
            )
        elif source.name == "toolbench":
            processors_to_run.append(
                ToolBenchProcessor(source.hf_repo, source.subset, source.max_samples, cfg.datasets.train_split)
            )

    if custom:
        processors_to_run.append(CustomProcessor(custom, cfg.datasets.train_split))

    if not processors_to_run:
        console.print("[yellow]No datasets configured or selected.[/yellow]")
        raise typer.Exit(1)

    for processor in processors_to_run:
        console.print(Panel(f"Processing [bold]{processor.name}[/bold]", style="cyan"))
        processor.process_and_save(output_dir)

    console.print(f"\n[green]✓[/green] Datasets saved to [bold]{output_dir}[/bold]")
    console.print("Run [bold]agentforge train[/bold] to start fine-tuning.")


@dataset_app.command("inspect")
def dataset_inspect(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    n: int = typer.Option(3, "--n", help="Number of examples to show"),
):
    """Show sample entries from the processed dataset."""
    import json
    from agentforge.config import get_config

    cfg = get_config(config)
    data_dir = Path(cfg.training.data_path)
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        console.print(f"[red]No train.jsonl found in {data_dir}. Run `agentforge dataset pull` first.[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]Sample from {train_file}[/bold] (first {n} rows)", style="cyan"))
    with open(train_file) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            example = json.loads(line)
            console.print_json(data=example)
            console.print()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    resume: bool = typer.Option(False, "--resume", help="Resume from latest checkpoint"),
):
    """LoRA fine-tune the model using MLX-LM. Data must exist in ./data."""
    from agentforge.config import get_config
    from agentforge.finetune.trainer.lora_trainer import LoRATrainer

    cfg = get_config(config)
    data_path = Path(cfg.training.data_path)

    if not (data_path / "train.jsonl").exists():
        console.print(f"[red]No train.jsonl in {data_path}. Run `agentforge dataset pull` first.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"Training [bold]{cfg.model.base_model}[/bold]\n"
        f"Data: {data_path}  |  Iterations: {cfg.training.num_iterations}  |  Batch: {cfg.training.batch_size}",
        title="AgentForge Train",
        style="cyan",
    ))

    trainer = LoRATrainer()
    adapter_path = trainer.train(cfg, resume=resume)

    console.print(f"\n[green]✓[/green] Adapter saved to [bold]{adapter_path}[/bold]")
    console.print(f"Update [bold]configs/config.yaml[/bold] → [italic]model.adapter_path: \"{adapter_path}\"[/italic]")
    console.print("Then run [bold]agentforge fuse[/bold] or [bold]agentforge serve --adapter[/bold]")


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------

@app.command()
def fuse(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    adapter: Optional[Path] = typer.Option(None, "--adapter", help="Override adapter path from config"),
    output: Optional[Path] = typer.Option(None, "--output", help="Override output path from config"),
):
    """Merge LoRA adapter weights into the base model."""
    from agentforge.config import get_config
    from agentforge.finetune.trainer.lora_trainer import LoRATrainer

    cfg = get_config(config)
    adapter_path = adapter or (Path(cfg.model.adapter_path) if cfg.model.adapter_path else None)
    output_path = output or Path(cfg.model.fused_model_path)

    if adapter_path is None:
        console.print("[red]No adapter path. Set model.adapter_path in config or pass --adapter.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"Fusing [bold]{adapter_path}[/bold] into [bold]{cfg.model.base_model}[/bold]\n"
        f"Output: {output_path}",
        title="AgentForge Fuse",
        style="cyan",
    ))

    trainer = LoRATrainer()
    trainer.fuse(cfg.model.base_model, adapter_path, output_path)

    console.print(f"\n[green]✓[/green] Fused model saved to [bold]{output_path}[/bold]")
    console.print(f"Update [bold]configs/config.yaml[/bold] → [italic]server.model_path: \"{output_path}\"[/italic]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model path"),
    adapter: Optional[Path] = typer.Option(None, "--adapter", help="Load loose adapter (no fusion needed)"),
):
    """Start the OpenAI-compatible inference server (FastAPI + MLX-LM)."""
    import uvicorn
    from agentforge.config import get_config
    from agentforge.serve.api.server import create_app

    cfg = get_config(config)
    _host = host or cfg.server.host
    _port = port or cfg.server.port
    model_path = model or cfg.server.model_path or cfg.model.base_model
    adapter_path = str(adapter) if adapter else cfg.server.adapter_path or cfg.model.adapter_path

    console.print(Panel(
        f"Model: [bold]{model_path}[/bold]\n"
        f"Adapter: {adapter_path or '[dim]none[/dim]'}\n"
        f"Listening on [bold]http://{_host}:{_port}[/bold]",
        title="AgentForge Serve",
        style="cyan",
    ))

    fastapi_app = create_app(model_path=model_path, adapter_path=adapter_path, cfg=cfg.server)
    uvicorn.run(fastapi_app, host=_host, port=_port, log_level="info")


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------

@agent_app.command("run")
def agent_run(
    prompt: str = typer.Argument(..., help="The task or question for the agent"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    tools: Optional[str] = typer.Option(None, "--tools", help="Comma-separated tool names to enable"),
):
    """Run the LangChain agent against the local inference server."""
    from agentforge.config import get_config
    from agentforge.agents.agent import AgentForgeAgent

    cfg = get_config(config)

    if tools:
        cfg.agent.tools = [t.strip() for t in tools.split(",")]

    console.print(Panel(
        f"[bold]Prompt:[/bold] {prompt}\n"
        f"Tools: {', '.join(cfg.agent.tools)}\n"
        f"API: {cfg.agent.api_base}",
        title="AgentForge Agent",
        style="cyan",
    ))

    agent = AgentForgeAgent(cfg)
    result = agent.run(prompt)

    console.print(Panel(result, title="[green]Result[/green]", style="green"))


if __name__ == "__main__":
    app()
