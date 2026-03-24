"""LoRA fine-tuning and adapter fusion wrapper around MLX-LM."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from agentforge.config import AppConfig

console = Console()


class LoRATrainer:
    """Wraps ``mlx_lm.lora`` and ``mlx_lm.fuse`` with a friendlier interface.

    MLX-LM's training API is invoked via its Python module entrypoint rather
    than via shell, so we get proper error propagation and stdout capture.
    """

    def train(self, config: AppConfig, resume: bool = False) -> Path:
        """Run LoRA fine-tuning and return the path to the saved adapter dir.

        Calls ``mlx_lm.lora`` with arguments derived from ``config``.
        """
        args = self._build_train_args(config, resume)
        console.print(f"[dim]mlx_lm.lora args: {' '.join(args)}[/dim]")

        # Run in the same process via importlib so we get live output
        try:
            from mlx_lm import lora as mlx_lora  # type: ignore[import]
            # mlx_lm.lora.main() accepts sys.argv-style argument list
            import mlx_lm.lora as _lora_mod

            original_argv = sys.argv[:]
            sys.argv = ["mlx_lm.lora"] + args
            try:
                _lora_mod.main()
            finally:
                sys.argv = original_argv
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"mlx_lm.lora exited with code {e.code}") from e

        adapter_dir = Path(config.training.output_dir)
        return adapter_dir

    def fuse(
        self,
        base_model: str,
        adapter_path: Path | str,
        output_path: Path | str,
    ) -> None:
        """Merge LoRA adapter weights into the base model and save to disk.

        Calls ``mlx_lm.fuse`` with the supplied paths.
        """
        import mlx_lm.fuse as _fuse_mod  # type: ignore[import]

        args = [
            "--model", str(base_model),
            "--adapter-path", str(adapter_path),
            "--save-path", str(output_path),
            "--de-quantize",  # restore full precision after fusing quantised model
        ]

        console.print(f"[dim]mlx_lm.fuse args: {' '.join(args)}[/dim]")

        original_argv = sys.argv[:]
        sys.argv = ["mlx_lm.fuse"] + args
        try:
            _fuse_mod.main()
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"mlx_lm.fuse exited with code {e.code}") from e
        finally:
            sys.argv = original_argv

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_train_args(self, config: AppConfig, resume: bool) -> list[str]:
        tc = config.training
        args = [
            "--model", config.model.base_model,
            "--train",
            "--data", tc.data_path,
            "--adapter-path", tc.output_dir,
            "--batch-size", str(tc.batch_size),
            "--learning-rate", str(tc.learning_rate),
            "--iters", str(tc.num_iterations),
            "--steps-per-eval", str(tc.steps_per_eval),
            "--save-every", str(tc.save_every),
            "--num-layers", str(tc.num_layers),
            "--lora-rank", str(tc.lora_rank),
            "--lora-scale", str(tc.lora_scale),
        ]
        if tc.grad_checkpoint:
            args.append("--grad-checkpoint")
        if resume:
            args.append("--resume-adapter-file")
            args.append(str(Path(tc.output_dir) / "adapters.safetensors"))
        return args
