"""Processor for the OpenHermes-2.5 dataset (ShareGPT format)."""

from __future__ import annotations

from typing import Iterator, Optional

from agentforge.datasets.processors.base import BaseProcessor

# ShareGPT role → standard chat role
_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}


class HermesProcessor(BaseProcessor):
    """Normalise ``teknium/OpenHermes-2.5`` into standard chat JSONL.

    Each row in OpenHermes-2.5 has a ``conversations`` field — a list of
    ``{"from": "<role>", "value": "<text>"}`` dicts in ShareGPT format.
    """

    def __init__(
        self,
        hf_repo: str = "teknium/OpenHermes-2.5",
        max_samples: int = 5000,
        train_split: float = 0.95,
    ) -> None:
        super().__init__(train_split)
        self.hf_repo = hf_repo
        self.max_samples = max_samples

    @property
    def name(self) -> str:
        return "OpenHermes-2.5"

    def iter_examples(self) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split="train", streaming=False)
        for i, row in enumerate(ds):
            if i >= self.max_samples:
                break
            yield row

    def to_chat_format(self, example: dict) -> Optional[dict]:
        conversations = example.get("conversations", [])
        if not conversations:
            return None

        messages = []
        for turn in conversations:
            role_raw = turn.get("from", "")
            role = _ROLE_MAP.get(role_raw, role_raw)
            content = turn.get("value", "").strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})

        if len(messages) < 2:
            return None

        return {"messages": messages}
