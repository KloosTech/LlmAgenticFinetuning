"""Processor for custom local JSON / JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from agentforge.datasets.processors.base import BaseProcessor


class CustomProcessor(BaseProcessor):
    """Load and normalise a user-supplied JSON or JSONL file.

    Supported input formats:

    1. Already in chat format::

        {"messages": [{"role": "user", "content": "..."}, ...]}

    2. Instruction / response pairs::

        {"instruction": "...", "response": "..."}

        or::

        {"input": "...", "output": "..."}

    3. Plain prompt / completion pairs::

        {"prompt": "...", "completion": "..."}
    """

    def __init__(self, path: Path | str, train_split: float = 0.95) -> None:
        super().__init__(train_split)
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"custom:{self.path.name}"

    def iter_examples(self) -> Iterator[dict]:
        suffix = self.path.suffix.lower()
        with open(self.path) as f:
            if suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            else:
                data = json.load(f)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data

    def to_chat_format(self, example: dict) -> Optional[dict]:
        # Already in messages format
        if "messages" in example:
            msgs = example["messages"]
            if isinstance(msgs, list) and len(msgs) >= 2:
                return {"messages": msgs}
            return None

        # instruction / response
        instruction = example.get("instruction") or example.get("input") or example.get("prompt", "")
        response = example.get("response") or example.get("output") or example.get("completion", "")
        instruction = str(instruction).strip()
        response = str(response).strip()

        if not instruction or not response:
            return None

        messages = []
        system = example.get("system", example.get("system_prompt", "")).strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})

        return {"messages": messages}
