# Model-Quantizer

This project downloads configurable LLMs, applies multiple quantization methods, saves reloadable artifacts, writes a dedicated logfile for every `(model, quantizer)` run, and runs interactive inference against local raw snapshots or local quantized artifacts.

## Features

- Config-driven model registry using Hugging Face identifiers
- Modular pipeline with separate downloader, quantizers, artifact writer, and runner
- Manual symmetric `int8` weight-only quantization
  - per-tensor
  - per-channel
- Manual grouped symmetric `int4` weight-only quantization
- Structured outputs under:
  - `models/raw/`
  - `models/quantized/<model>/<method>/`
  - `logs/`
  - `artifacts/metadata/`
- Reload support for the manual quantized artifacts through `QuantizedArtifactLoader`
- Automated benchmark evaluation for:
  - HellaSwag multiple-choice accuracy
  - MMLU multiple-choice accuracy
  - structured JSON summaries and per-example prediction logs
- Interactive inference for:
  - stateless `one-shot` prompts
  - remembered-history `conversational` chat
  - local raw snapshots or saved quantized artifacts
  - latency and throughput metrics such as TTFT and tokens/sec

## Project Layout

```text
.
├── config/
│   └── default.yaml
├── main.py
├── model_quantizer/
│   ├── artifacts/
│   ├── download/
│   ├── pipeline/
│   ├── quantization/
│   └── utils/
├── models/
│   ├── raw/
│   └── quantized/
├── artifacts/
│   └── metadata/
└── logs/
```

## Requirements

- Python 3.10+
- PyTorch-compatible environment
- Hugging Face access for gated models such as Gemma and Llama
- Enough host RAM / cloud capacity for the selected checkpoint

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you are using gated Hugging Face repositories, provide a token first.

Option 1: create a local `.env` file in the project root from the example template:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
HF_TOKEN=your_token_here
```

Option 2: export it in your shell:

```bash
export HF_TOKEN=your_token_here
```

## Configuration

The default configuration lives at [config/default.yaml](/Users/jacksal1/Desktop/Model%20Quantizer/Model-Quantizer/config/default.yaml).

It defines:

- available models and their Hugging Face ids
- available quantizers and their parameters
- available benchmark tasks and dataset sources
- output directories
- runtime options such as shard size and default device
- inference defaults such as:
  - the injected system prompt used before the first user query
  - default chat mode
  - generation parameters like `max_new_tokens`, `temperature`, and `top_p`

Supported aliases in the default config:

- `gemma-2-27b`
- `gemma-2-9b`
- `llama-3-8b`
- `mistral-7b`
- `phi-3`
- `mixtral-8x7b` (disabled by default)

## Usage

List what is available:

```bash
python main.py --list-models
python main.py --list-quantizers
python main.py --list-benchmarks
```

Run a focused subset:

```bash
python main.py --models mistral-7b phi-3 --quantizers int8_per_tensor int4_grouped --device cuda:0
```

Run every enabled combination from the config:

```bash
python main.py --all-models --all-quantizers --device auto
```

Use a custom config:

```bash
python main.py --config config/default.yaml --models llama-3-8b --quantizers int4_grouped
```

Run one stateless prompt against a local raw model snapshot:

```bash
python main.py --run-model phi-3 --model-source raw --chat-mode one-shot --prompt "Write a haiku about quantization."
```

Run conversational chat against a quantized artifact:

```bash
python main.py --run-model phi-3 --model-source quantized --artifact-quantizer int4_grouped --chat-mode conversational
```

Override the injected system prompt for a single session:

```bash
python main.py --run-model phi-3 --artifact-quantizer int4_grouped --system-prompt "Answer like a terse code reviewer."
```

Run automated benchmark evaluation for one model and two quantizers:

```bash
python main.py --run-benchmarks --models phi-3 --quantizers int8_per_channel int4_grouped --benchmarks hellaswag mmlu --device cuda:0
```

Run a smaller smoke test while developing:

```bash
python main.py --run-benchmarks --models phi-3 --quantizers int4_grouped --benchmark-limit 25 --device cuda:0
```

## Quantization Notes

The manual quantizers are intentionally educational.

In [model_quantizer/quantization/int8.py](/Users/jacksal1/Desktop/Model%20Quantizer/Model-Quantizer/model_quantizer/quantization/int8.py), the code shows:

- how symmetric scaling is computed from the largest absolute value
- why zero stays exact in symmetric quantization
- how rounding and clipping map floats into signed `int8`
- how per-tensor and per-channel scaling differ
- how scales are stored for later reconstruction

In [model_quantizer/quantization/int4.py](/Users/jacksal1/Desktop/Model%20Quantizer/Model-Quantizer/model_quantizer/quantization/int4.py), the code shows:

- how grouped quantization reduces error versus one global scale
- how each group gets its own symmetric scale
- how signed `int4` values are packed two-per-byte
- how packed storage is later unpacked and dequantized

## Output Format

For manual quantizers, each artifact directory contains:

- tokenizer/config files for the original model family
- generation/support files needed to reload the model later
- sharded `safetensors` files containing:
  - quantized weights
  - scales
  - passthrough non-quantized tensors
- `quantization_manifest.json` with:
  - quantization parameters
  - tensor-level metadata
  - shard layout
  - size summary
  - runtime info

## Reloading Manual Artifacts

```python
from pathlib import Path

from model_quantizer.artifacts.loader import QuantizedArtifactLoader

artifact_dir = Path("models/quantized/mistral-7b/int8_per_channel")
model = QuantizedArtifactLoader.load_model(artifact_dir, device="cpu")
```

If you only need the reconstructed weights:

```python
from pathlib import Path

from model_quantizer.artifacts.loader import QuantizedArtifactLoader

state_dict = QuantizedArtifactLoader.load_state_dict(
    Path("models/quantized/mistral-7b/int4_grouped")
)
```

The saved artifacts are intended to be directly reusable for evaluation. Each
quantized directory stores enough tokenizer/config/supporting metadata for
`QuantizedArtifactLoader.load_model(...)` to rebuild a standard local
Transformers model before scoring benchmarks.

## Interactive Inference

The inference runtime loads only local files from:

- `models/raw/<model>/`
- `models/quantized/<model>/<quantizer>/`

This is intentional: inference never downloads a checkpoint, calls an API, or
uses a pre-quantized model from Hugging Face at runtime. If a local raw snapshot
or quantized artifact is missing, the command fails instead of fetching it.

Conversation behavior:

- `one-shot`: one prompt in, one response out, no remembered history
- `conversational`: a small REPL that remembers previous turns until `/reset` or `/exit`

Useful commands:

```bash
python main.py --run-model phi-3 --model-source raw --chat-mode conversational
python main.py --run-model phi-3 --model-source raw --chat-mode one-shot --prompt "Explain grouped int4 quantization."
python main.py --run-model phi-3 --artifact-quantizer int4_grouped --chat-mode conversational
python main.py --run-model phi-3 --artifact-quantizer int4_grouped --chat-mode one-shot --prompt "Explain grouped int4 quantization."
python main.py --run-model phi-3 --artifact-quantizer int8_per_channel
```

After each response, the runtime prints metrics including:

- prompt token count
- generated token count
- time to first token
- total completion time
- decode throughput
- peak CUDA memory when running on GPU

## Benchmark Evaluation

Benchmark mode runs deterministic multiple-choice evaluation against local raw
snapshots and local quantized artifacts. The default config enables:

- `hellaswag` on the validation split
- `mmlu` on the `all` subject config using the test split

Benchmark mode writes:

- one summary JSON per `(model, variant, benchmark)`
- one JSONL file with per-example predictions and likelihood scores

Outputs are stored under:

- `artifacts/benchmarks/<model>/<variant>/`
- `artifacts/datasets/` for cached benchmark datasets

Useful commands:

```bash
python main.py --run-benchmarks --models phi-3 --quantizers int4_grouped
python main.py --run-benchmarks --models phi-3 --quantizers int4_grouped --benchmark-limit 50
python main.py --run-benchmarks --models phi-3 --quantizers int8_per_tensor int8_per_channel int4_grouped --benchmarks hellaswag mmlu
python main.py --run-benchmarks --models phi-3 --quantizers int4_grouped --no-raw-baseline
```

Scoring method:

- HellaSwag: average log-likelihood of each candidate ending conditioned on the prompt
- MMLU: average log-likelihood of answer labels `A/B/C/D` conditioned on the question stem and options

This keeps evaluation reproducible across raw and quantized variants without
relying on manual prompt inspection.

## Logging

Every `(model, quantizer)` pair gets its own logfile under `logs/`.

Each logfile includes:

- timestamps
- model and method names
- device information
- quantization parameters
- progress checkpoints
- size information
- warnings/errors
- total runtime

## Limitations

- The manual quantizers prioritize clarity and reloadable artifacts over optimized inference kernels.
- Very large checkpoints can still require substantial RAM even though artifacts are written in shards.
- Quantized benchmark evaluation reconstructs a standard dense model from the saved low-bit artifact before inference, so task-accuracy comparisons are valid but runtime numbers are not equivalent to specialized low-bit kernels such as GPTQ/AWQ runtimes.
