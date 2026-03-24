"""Base class for dataset processors."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List


class BaseProcessor(ABC):
    """Abstract base for all dataset processors.

    Each processor downloads a dataset from HuggingFace (or local disk),
    normalises each example into the standard chat format expected by
    ``mlx_lm.lora``, and writes ``train.jsonl`` / ``valid.jsonl`` to disk.

    Expected output format per line::

        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """

    def __init__(self, train_split: float = 0.95) -> None:
        self.train_split = train_split

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in log messages."""

    @abstractmethod
    def iter_examples(self) -> Iterator[dict]:
        """Yield raw examples from the source dataset."""

    @abstractmethod
    def to_chat_format(self, example: dict) -> dict | None:
        """Convert a raw example to ``{"messages": [...]}``.

        Return ``None`` to skip the example.
        """

    def process_and_save(self, output_dir: Path) -> tuple[int, int]:
        """Process all examples and write train/valid JSONL files.

        Returns:
            (train_count, valid_count)
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

        output_dir.mkdir(parents=True, exist_ok=True)

        examples: List[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Loading {self.name}…", total=None)
            for raw in self.iter_examples():
                converted = self.to_chat_format(raw)
                if converted is not None:
                    examples.append(converted)
                progress.advance(task)

        # Shuffle before splitting
        random.shuffle(examples)
        split_idx = int(len(examples) * self.train_split)
        train_examples = examples[:split_idx]
        valid_examples = examples[split_idx:]

        # Append to existing files so multiple processors merge their data
        train_file = output_dir / "train.jsonl"
        valid_file = output_dir / "valid.jsonl"

        _append_jsonl(train_file, train_examples)
        _append_jsonl(valid_file, valid_examples)

        return len(train_examples), len(valid_examples)


def _append_jsonl(path: Path, examples: List[dict]) -> None:
    with open(path, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
