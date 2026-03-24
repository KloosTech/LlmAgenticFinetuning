"""Processor for the ToolBench dataset."""

from __future__ import annotations

import json
from typing import Iterator, Optional

from agentforge.datasets.processors.base import BaseProcessor


class ToolBenchProcessor(BaseProcessor):
    """Normalise ``ToolBench/ToolBench`` into standard chat JSONL.

    ToolBench rows contain an ``conversations`` field similar to ShareGPT,
    or fall back to ``instruction`` / ``output`` fields.
    """

    def __init__(
        self,
        hf_repo: str = "ToolBench/ToolBench",
        subset: Optional[str] = "G1_instruction",
        max_samples: int = 2000,
        train_split: float = 0.95,
    ) -> None:
        super().__init__(train_split)
        self.hf_repo = hf_repo
        self.subset = subset
        self.max_samples = max_samples

    @property
    def name(self) -> str:
        return f"ToolBench/{self.subset or 'default'}"

    def iter_examples(self) -> Iterator[dict]:
        from datasets import load_dataset

        kwargs: dict = {"split": "train", "streaming": False}
        if self.subset:
            kwargs["name"] = self.subset

        try:
            ds = load_dataset(self.hf_repo, **kwargs)
        except Exception:
            # Some ToolBench subsets require trust_remote_code
            kwargs["trust_remote_code"] = True
            ds = load_dataset(self.hf_repo, **kwargs)

        for i, row in enumerate(ds):
            if i >= self.max_samples:
                break
            yield row

    def to_chat_format(self, example: dict) -> Optional[dict]:
        # Try conversations field first (ShareGPT-like)
        conversations = example.get("conversations")
        if conversations:
            messages = []
            for turn in conversations:
                role_raw = turn.get("from", turn.get("role", ""))
                role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role_raw, role_raw)
                content = turn.get("value", turn.get("content", "")).strip()
                if content:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 2:
                return {"messages": messages}

        # Fall back to instruction / output fields
        instruction = example.get("instruction", "").strip()
        output = example.get("output", example.get("response", "")).strip()
        if instruction and output:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            # Prepend system prompt if available
            system = example.get("system_prompt", "").strip()
            if system:
                messages.insert(0, {"role": "system", "content": system})
            return {"messages": messages}

        return None
