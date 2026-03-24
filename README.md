# AgentForge

End-to-end LLM fine-tuning, serving, and agentic workflow — built for **Apple Silicon M1/M2/M3 Max**.

Powered by [MLX-LM](https://github.com/ml-explore/mlx-lm). One CLI, one config file.

```
agentforge dataset pull → agentforge train → agentforge fuse → agentforge serve → agentforge agent run "task"
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- ~8 GB free unified memory for the default 7B model (32 GB recommended for training)

## Install

```bash
git clone <this-repo>
cd LlmAgenticFinetuning
pip install -e .
```

## Quick Start

```bash
# 1. Check your setup
agentforge status

# 2. Download and normalise datasets
agentforge dataset pull

# 3. Fine-tune with LoRA
agentforge train

# 4. (Optional) Fuse adapter into base model for faster inference
agentforge fuse

# 5. Start the OpenAI-compatible server
agentforge serve

# 6. Run the agent
agentforge agent run "Summarise the latest MLX release notes from the web"
```

## Configuration

All settings live in `configs/config.yaml`. Edit once, run everywhere.

```yaml
model:
  base_model: "mlx-community/Mistral-7B-Instruct-v0.3-8bit"  # ~8 GB Metal memory
  adapter_path: null          # set after training
  fused_model_path: "./models/fused"

training:
  num_layers: 32              # all layers on M1 Max 32GB
  batch_size: 8
  num_iterations: 1000
  data_path: "./data"

server:
  host: "127.0.0.1"
  port: 8080
  max_tokens: 2048

agent:
  api_base: "http://127.0.0.1:8080/v1"
  tools: [web_search, shell, python_repl, read_file]
```

## CLI Reference

### `agentforge status`
Prints Metal GPU memory usage, system RAM, and the active configuration.

### `agentforge dataset pull`
Downloads OpenHermes-2.5 and ToolBench from HuggingFace, normalises them to
`mlx_lm`-compatible JSONL, and writes `./data/train.jsonl` + `./data/valid.jsonl`.

```bash
agentforge dataset pull --hermes-only       # only OpenHermes
agentforge dataset pull --toolbench-only
agentforge dataset pull --custom ./my_examples.jsonl  # add your own data
```

### `agentforge dataset inspect`
Shows sample entries from the processed dataset.

### `agentforge train`
Runs LoRA fine-tuning via `mlx_lm.lora`. Progress is printed to stdout.

```bash
agentforge train
agentforge train --resume   # continue from last checkpoint
```

After training, set `model.adapter_path` in `configs/config.yaml`.

### `agentforge fuse`
Merges the LoRA adapter into the base model weights for faster inference.

```bash
agentforge fuse
agentforge fuse --adapter ./adapters --output ./models/my-model
```

### `agentforge serve`
Starts a FastAPI server at `http://127.0.0.1:8080`. Exposes:
- `GET  /health`
- `GET  /v1/models`
- `POST /v1/chat/completions`  (OpenAI-compatible, supports `stream: true`)

```bash
agentforge serve                         # use config defaults
agentforge serve --model ./models/fused  # serve a fused model
agentforge serve --adapter ./adapters    # serve base + loose adapter
agentforge serve --port 11434            # custom port
```

Compatible with any OpenAI client:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}]}'
```

### `agentforge agent run`
Runs a LangChain ReAct agent against the local server.

```bash
agentforge agent run "What is the current Bitcoin price?"
agentforge agent run "Read ./data/train.jsonl and count the examples"
agentforge agent run "Write a Python script that sorts a list" --tools python_repl
```

## Custom Dataset Format

Pass any of these formats to `--custom`:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "response": "..."}
{"prompt": "...", "completion": "..."}
```

## Memory Usage (M1 Max, 32 GB)

| Stage | Metal Memory |
|-------|-------------|
| Inference (8-bit 7B) | ~8 GB |
| Training (8-bit 7B, LoRA all layers, batch 8) | ~18–22 GB |
| Fused model (bf16) | ~14 GB |

## Architecture

```
src/agentforge/
├── config.py              # Pydantic config loader
├── cli/main.py            # Typer CLI entry point
├── datasets/processors/   # HF dataset downloaders + normalizers
│   ├── base.py            # BaseProcessor ABC
│   ├── hermes.py          # OpenHermes-2.5 (ShareGPT format)
│   ├── toolbench.py       # ToolBench G1_instruction
│   └── custom.py          # local JSON/JSONL files
├── finetune/trainer/
│   └── lora_trainer.py    # mlx_lm.lora wrapper
├── serve/api/
│   └── server.py          # FastAPI + MLX-LM inference
└── agents/
    ├── tools.py            # LangChain tools
    └── agent.py            # ReAct agent
```
