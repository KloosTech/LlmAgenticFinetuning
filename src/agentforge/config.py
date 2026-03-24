"""Configuration loading and validation for AgentForge."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    base_model: str = "mlx-community/Mistral-7B-Instruct-v0.3-8bit"
    adapter_path: Optional[str] = None
    fused_model_path: str = "./models/fused"


class TrainingConfig(BaseModel):
    output_dir: str = "./adapters"
    num_layers: int = 32
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_iterations: int = 1000
    steps_per_eval: int = 100
    save_every: int = 200
    grad_checkpoint: bool = True
    lora_rank: int = 16
    lora_scale: float = 20.0
    data_path: str = "./data"


class DatasetSource(BaseModel):
    name: str
    enabled: bool = True
    hf_repo: str
    subset: Optional[str] = None
    max_samples: int = 5000


class DatasetsConfig(BaseModel):
    output_dir: str = "./data"
    train_split: float = 0.95
    sources: List[DatasetSource] = Field(default_factory=list)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    max_tokens: int = 2048
    temp: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class AgentConfig(BaseModel):
    api_base: str = "http://127.0.0.1:8080/v1"
    model: str = "local"
    max_iterations: int = 10
    verbose: bool = True
    tools: List[str] = Field(default_factory=lambda: ["web_search", "shell", "python_repl", "read_file"])


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(path: Path | str) -> AppConfig:
    """Load and validate configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw or {})


def get_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Find and load config.yaml, searching from CWD upward."""
    if config_path:
        return load_config(config_path)

    # Search for configs/config.yaml starting from CWD
    search = Path.cwd()
    for _ in range(5):  # walk up at most 5 levels
        candidate = search / "configs" / "config.yaml"
        if candidate.exists():
            return load_config(candidate)
        if search.parent == search:
            break
        search = search.parent

    # Fall back to defaults if no config found
    return AppConfig()
