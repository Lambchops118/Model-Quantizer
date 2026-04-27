#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${MQ_CONFIG:-config/default.yaml}"
DEVICE="${MQ_DEVICE:-auto}"

build_selection_args() {
  local kind="$1"
  local all_var="$2"
  local names_var="$3"

  local -n out_ref="$kind"
  out_ref=()

  if [[ "${!all_var:-0}" == "1" ]]; then
    if [[ "$kind" == "model_args" ]]; then
      out_ref+=(--all-models)
    else
      out_ref+=(--all-quantizers)
    fi
    return
  fi

  if [[ -z "${!names_var:-}" ]]; then
    echo "Missing ${names_var}. Set ${all_var}=1 or provide a space-separated ${names_var} list." >&2
    exit 1
  fi

  read -r -a names <<<"${!names_var}"
  if [[ "$kind" == "model_args" ]]; then
    out_ref+=(--models "${names[@]}")
  else
    out_ref+=(--quantizers "${names[@]}")
  fi
}

model_args=()
quantizer_args=()
benchmark_args=()

build_selection_args model_args MQ_ALL_MODELS MQ_MODELS
build_selection_args quantizer_args MQ_ALL_QUANTIZERS MQ_QUANTIZERS

if [[ -n "${MQ_BENCHMARKS:-}" ]]; then
  read -r -a benchmarks <<<"${MQ_BENCHMARKS}"
  benchmark_args+=(--benchmarks "${benchmarks[@]}")
fi

if [[ -n "${MQ_BENCHMARK_LIMIT:-}" ]]; then
  benchmark_args+=(--benchmark-limit "${MQ_BENCHMARK_LIMIT}")
fi

if [[ "${MQ_NO_RAW_BASELINE:-0}" == "1" ]]; then
  benchmark_args+=(--no-raw-baseline)
fi

python main.py \
  --config "${CONFIG_PATH}" \
  "${model_args[@]}" \
  "${quantizer_args[@]}" \
  --device "${DEVICE}"

python main.py \
  --config "${CONFIG_PATH}" \
  --run-benchmarks \
  "${model_args[@]}" \
  "${quantizer_args[@]}" \
  "${benchmark_args[@]}" \
  --device "${DEVICE}"
