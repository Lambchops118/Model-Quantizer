FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache

WORKDIR /workspace/app

COPY pyproject.toml README.md main.py ./
COPY config ./config
COPY model_quantizer ./model_quantizer
COPY docker/run_pipeline_and_benchmarks.sh /usr/local/bin/run-pipeline-and-benchmarks

RUN chmod +x /usr/local/bin/run-pipeline-and-benchmarks \
    && python -m pip install --upgrade pip \
    && python -m pip install -e .

RUN mkdir -p \
    /workspace/app/models/raw \
    /workspace/app/models/quantized \
    /workspace/app/artifacts/benchmarks \
    /workspace/app/artifacts/datasets \
    /workspace/app/artifacts/metadata \
    /workspace/app/logs

CMD ["/usr/local/bin/run-pipeline-and-benchmarks"]
