# Model-Quantizer

This project downloads configured Hugging Face causal LMs, applies a set of manual quantization methods, evaluates raw and quantized variants on benchmark tasks, and cleans up quantized artifacts once benchmarking is complete.

The repo is intentionally focused on two workflows:

1. quantize configured models
2. benchmark them on Hugging Face dataset tasks

Interactive chat and one-off inference are no longer part of the project.

## Features

- Config-driven model registry using Hugging Face identifiers
- Manual weight-only quantization methods:
  - symmetric `int8` per-tensor
  - symmetric `int8` per-channel
  - symmetric grouped `int4`
- Sharded `safetensors` artifact writing for large checkpoints
- Reload support for saved quantized artifacts during evaluation
- Automated benchmark evaluation for Hugging Face dataset tasks:
  - HellaSwag
  - MMLU
- Structured JSON summaries and per-example prediction logs
- Per-run logfiles for every `(model, quantizer)` pair
- Automatic deletion of quantized artifacts after a successful full benchmark pass
- Standalone cleanup command for already-benchmarked quantized artifacts
- Docker setup for GPU-backed batch runs on Vast.ai

## Project Layout

```text
.
├── config/
│   └── default.yaml
├── docker/
│   └── run_pipeline_and_benchmarks.sh
├── main.py
├── model_quantizer/
│   ├── artifacts/
│   ├── benchmarks/
│   ├── download/
│   ├── pipeline/
│   ├── quantization/
│   └── runtime/
├── models/
│   ├── raw/
│   └── quantized/
├── artifacts/
│   ├── benchmarks/
│   ├── datasets/
│   └── metadata/
└── logs/
```

## Requirements

- Python 3.10+
- PyTorch-compatible environment
- Enough disk and RAM for the selected checkpoints
- Hugging Face access for gated models such as Gemma and Llama

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you are using gated Hugging Face repositories, set a token:

```bash
export HF_TOKEN=your_token_here
```

You can also place `HF_TOKEN=...` in a local `.env` file in the repo root.

## Configuration

The default configuration lives in [config/default.yaml](/mnt/c/Users/aljac/Desktop/Model-Quantizer/config/default.yaml:1).

It defines:

- output directories
- runtime defaults such as shard size and device selection
- benchmark tasks and dataset sources
- whether successful full benchmark runs should delete quantized artifacts
- model aliases and Hugging Face ids
- quantizer methods and their parameters

Default model aliases:

- `gemma-2-27b`
- `gemma-2-9b`
- `llama-3-8b`
- `mistral-7b`
- `phi-3`
- `mixtral-8x7b` disabled by default

## Usage

List configured options:

```bash
python main.py --list-models
python main.py --list-quantizers
python main.py --list-benchmarks
```

Run quantization for a focused subset:

```bash
python main.py --models mistral-7b phi-3 --quantizers int8_per_tensor int4_grouped --device cuda:0
```

Run quantization for every enabled combination:

```bash
python main.py --all-models --all-quantizers --device auto
```

Run benchmarks for one model across two quantizers:

```bash
python main.py --run-benchmarks --models phi-3 --quantizers int8_per_channel int4_grouped --benchmarks hellaswag mmlu --device cuda:0
```

Run a smaller benchmark smoke test:

```bash
python main.py --run-benchmarks --models phi-3 --quantizers int4_grouped --benchmark-limit 25 --device cuda:0
```

Remove local quantized artifacts that already have successful full benchmark summaries:

```bash
python main.py --cleanup-benchmarked-quantized
```

## Benchmark Behavior

Benchmark mode evaluates local raw snapshots and local quantized artifacts. The default config uses Hugging Face datasets for:

- `hellaswag` validation
- `mmlu` test with `dataset_config: all`

Scoring method:

- HellaSwag: normalized log-likelihood of each ending continuation
- MMLU: normalized log-likelihood of answer labels `A/B/C/D`

Outputs are written under:

- `artifacts/benchmarks/<model>/<variant>/`
- `artifacts/datasets/` for dataset cache

Each benchmark run writes:

- one summary JSON per `(model, variant, benchmark)`
- one JSONL file with per-example predictions

## Quantized Artifact Cleanup

The default config enables automatic cleanup after benchmarking.

Cleanup happens only when all of the following are true:

- the evaluated variant is quantized
- every enabled benchmark in the config was selected
- every selected benchmark finished successfully
- the run was not limited with `--benchmark-limit`

When cleanup runs, the repo deletes only `models/quantized/<model>/<quantizer>/`.
Raw snapshots under `models/raw/` are preserved.

## Quantization Notes

The manual quantizers are educational and reloadable, not kernel-optimized.

- [model_quantizer/quantization/int8.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/quantization/int8.py:39) implements symmetric int8 weight-only quantization with per-tensor and per-channel scaling.
- [model_quantizer/quantization/int4.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/quantization/int4.py:40) implements grouped symmetric int4 quantization with packed 4-bit storage.
- [model_quantizer/artifacts/loader.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/artifacts/loader.py:107) reconstructs a dense Transformers model from the saved low-bit artifact for evaluation.

Because the loader dequantizes back into a standard dense model, benchmark accuracy comparisons are valid, but runtime is not representative of specialized low-bit serving stacks such as GPTQ or AWQ runtimes.

## Docker and Vast.ai

The repo includes:

- [Dockerfile](/mnt/c/Users/aljac/Desktop/Model-Quantizer/Dockerfile:1)
- [.dockerignore](/mnt/c/Users/aljac/Desktop/Model-Quantizer/.dockerignore:1)
- [docker/run_pipeline_and_benchmarks.sh](/mnt/c/Users/aljac/Desktop/Model-Quantizer/docker/run_pipeline_and_benchmarks.sh:1)

Build the image:

```bash
docker build -t model-quantizer .
```

Run a batch job locally or on Vast.ai:

```bash
docker run --gpus all --rm \
  -e HF_TOKEN=your_token_here \
  -e MQ_MODELS="phi-3 mistral-7b" \
  -e MQ_QUANTIZERS="int8_per_tensor int4_grouped" \
  -e MQ_BENCHMARKS="hellaswag mmlu" \
  -e MQ_DEVICE="cuda:0" \
  -v "$(pwd)/models:/workspace/app/models" \
  -v "$(pwd)/artifacts:/workspace/app/artifacts" \
  -v "$(pwd)/logs:/workspace/app/logs" \
  model-quantizer
```

Useful container environment variables:

- `MQ_CONFIG` defaults to `config/default.yaml`
- `MQ_MODELS` space-separated model aliases
- `MQ_QUANTIZERS` space-separated quantizer aliases
- `MQ_BENCHMARKS` optional space-separated benchmark names
- `MQ_DEVICE` defaults to `auto`
- `MQ_ALL_MODELS` defaults to `0`
- `MQ_ALL_QUANTIZERS` defaults to `0`
- `MQ_BENCHMARK_LIMIT` optional smoke-test limit
- `MQ_NO_RAW_BASELINE=1` to skip raw baselines

For Vast.ai, use the same image and pass the environment variables in the instance template or startup command. Mount persistent storage for `models/`, `artifacts/`, and `logs/` so downloads and benchmark results survive instance restarts.
