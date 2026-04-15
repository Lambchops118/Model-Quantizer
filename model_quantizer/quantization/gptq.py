"""GPTQ integration using the official Transformers + GPTQModel path."""

from __future__ import annotations

from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from model_quantizer.quantization.base import BaseQuantizer, QuantizationContext


class GPTQQuantizer(BaseQuantizer):
    """Wraps Hugging Face GPTQ quantization for reloadable local artifacts."""

    def quantize(self, context: QuantizationContext) -> Dict[str, Any]:
        options = context.quantizer_config.options
        tokenizer_model = options.get("tokenizer_model") or str(context.raw_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=context.model_config.trust_remote_code,
        )

        gptq_config = GPTQConfig(
            bits=int(options.get("bits", 4)),
            dataset=options.get("dataset", "c4"),
            tokenizer=tokenizer,
            group_size=int(options.get("group_size", 128)),
            desc_act=bool(options.get("desc_act", False)),
            sym=bool(options.get("sym", True)),
            true_sequential=bool(options.get("true_sequential", True)),
            batch_size=int(options.get("batch_size", 1)),
            damp_percent=float(options.get("damp_percent", 0.1)),
        )

        context.logger.info("Starting GPTQ quantization with options=%s", options)
        model = AutoModelForCausalLM.from_pretrained(
            context.raw_model_dir,
            device_map=context.runtime_config.gptq_device_map,
            torch_dtype="auto",
            trust_remote_code=context.model_config.trust_remote_code,
            quantization_config=gptq_config,
        )

        try:
            model = model.to("cpu")
        except Exception as exc:  # pragma: no cover - depends on runtime backend
            context.logger.warning("Unable to move GPTQ model to CPU before save: %s", exc)

        model.save_pretrained(context.output_dir)
        tokenizer.save_pretrained(context.output_dir)

        size_bytes = sum(path.stat().st_size for path in context.output_dir.rglob("*") if path.is_file())
        return {
            "quantizer_name": context.quantizer_config.name,
            "quantizer_type": context.quantizer_config.type,
            "source_model": {
                "name": context.model_config.name,
                "hf_id": context.model_config.hf_id,
                "raw_model_dir": str(context.raw_model_dir),
                "trust_remote_code": context.model_config.trust_remote_code,
            },
            "quantization_parameters": dict(options),
            "artifact_shards": [],
            "tensor_records": [],
            "size_summary": {
                "original_size_bytes": 0,
                "quantized_size_bytes": size_bytes,
            },
            "notes": [
                "This artifact was saved through Transformers save_pretrained().",
                "Reload it with AutoModelForCausalLM.from_pretrained(<artifact_dir>, device_map='auto').",
            ],
        }
