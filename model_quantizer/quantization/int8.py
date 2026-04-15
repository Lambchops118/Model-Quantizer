"""Manual symmetric int8 weight-only quantization.

This module intentionally shows the math in a straightforward way instead of
hiding it behind a library call. The implementation saves quantized weights and
their scales so they can be reloaded and dequantized later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from model_quantizer.artifacts.manager import ShardedTensorWriter
from model_quantizer.quantization.base import BaseQuantizer, QuantizationContext
from model_quantizer.quantization.common import (
    count_checkpoint_tensors,
    iter_source_tensors,
    maybe_cast_passthrough_tensor,
    tensor_nbytes,
)


INT8_QMAX = 127


@dataclass(frozen=True)
class Int8TensorPayload:
    """The stored tensors and metadata for one quantized weight matrix."""

    qweight: torch.Tensor
    scales: torch.Tensor
    original_size_bytes: int
    quantized_size_bytes: int
    metadata: Dict[str, Any]


class Int8Quantizer(BaseQuantizer):
    """Symmetric int8 weight-only quantization with two granularities."""

    def quantize(self, context: QuantizationContext) -> Dict[str, Any]:
        granularity = str(context.quantizer_config.options["granularity"])
        channel_axis = int(context.quantizer_config.options.get("channel_axis", 0))
        quant_device = self.resolve_quantization_device(context)
        linear_weight_names = set(self.discover_linear_weight_names(context))
        total_tensors = count_checkpoint_tensors(context.raw_model_dir)

        self.save_supporting_files(context)
        writer = ShardedTensorWriter(
            context.output_dir,
            max_shard_size_bytes=context.runtime_config.max_shard_size_mb * 1024 * 1024,
        )

        tensor_records = []
        original_size_bytes = 0
        quantized_size_bytes = 0

        context.logger.info(
            "Starting manual int8 quantization with granularity=%s, channel_axis=%s, device=%s",
            granularity,
            channel_axis,
            quant_device,
        )

        for index, source_tensor in enumerate(iter_source_tensors(context.raw_model_dir), start=1):
            tensor = source_tensor.tensor
            original_size_bytes += tensor_nbytes(tensor)

            if source_tensor.name in linear_weight_names:
                payload = self._quantize_weight(
                    weight=tensor,
                    granularity=granularity,
                    channel_axis=channel_axis,
                    device=quant_device,
                )
                qweight_key = f"quantized::{source_tensor.name}::qweight"
                scale_key = f"quantized::{source_tensor.name}::scale"
                writer.add_tensor(qweight_key, payload.qweight)
                writer.add_tensor(scale_key, payload.scales)
                quantized_size_bytes += payload.quantized_size_bytes

                tensor_records.append(
                    {
                        "name": source_tensor.name,
                        "kind": "int8_weight",
                        "scheme": "symmetric_int8",
                        "granularity": granularity,
                        "channel_axis": channel_axis,
                        "original_shape": list(tensor.shape),
                        "original_dtype": str(tensor.dtype).replace("torch.", ""),
                        "qweight_key": qweight_key,
                        "scale_key": scale_key,
                        "source_file": source_tensor.source_file,
                        **payload.metadata,
                    }
                )
            else:
                storage_tensor = maybe_cast_passthrough_tensor(
                    tensor,
                    context.runtime_config.passthrough_float_dtype,
                )
                writer.add_tensor(source_tensor.name, storage_tensor)
                quantized_size_bytes += tensor_nbytes(storage_tensor)
                tensor_records.append(
                    {
                        "name": source_tensor.name,
                        "kind": "passthrough",
                        "storage_key": source_tensor.name,
                        "original_shape": list(tensor.shape),
                        "original_dtype": str(tensor.dtype).replace("torch.", ""),
                        "stored_dtype": str(storage_tensor.dtype).replace("torch.", ""),
                        "source_file": source_tensor.source_file,
                    }
                )

            if index % 25 == 0 or index == total_tensors:
                context.logger.info("Processed %s/%s tensors", index, total_tensors)

        artifact_shards = writer.finalize()
        return {
            "quantizer_name": context.quantizer_config.name,
            "quantizer_type": context.quantizer_config.type,
            "source_model": {
                "name": context.model_config.name,
                "hf_id": context.model_config.hf_id,
                "raw_model_dir": str(context.raw_model_dir),
                "trust_remote_code": context.model_config.trust_remote_code,
            },
            "quantization_parameters": {
                "granularity": granularity,
                "channel_axis": channel_axis,
                "compute_device": str(quant_device),
            },
            "artifact_shards": artifact_shards,
            "tensor_records": tensor_records,
            "size_summary": {
                "original_size_bytes": original_size_bytes,
                "quantized_size_bytes": quantized_size_bytes,
            },
        }

    def _quantize_weight(
        self,
        weight: torch.Tensor,
        granularity: str,
        channel_axis: int,
        device: torch.device,
    ) -> Int8TensorPayload:
        """Quantize a weight tensor with symmetric int8 arithmetic.

        Symmetric quantization means zero stays exactly representable and the same
        scale is used for positive and negative values. For int8 we use the signed
        range [-127, 127], so the scale is chosen from the largest absolute value:

            scale = max(abs(weight)) / 127

        The quantized integers are then:

            q = round(weight / scale)

        and clipped into the valid int8 range. During reconstruction we multiply
        back by the scale.
        """

        working = weight.detach().to(device=device, dtype=torch.float32)

        # Per-tensor uses a single scale for the entire weight matrix.
        if granularity == "per_tensor":
            max_abs = working.abs().amax().clamp(min=1e-8)
            scales = (max_abs / INT8_QMAX).reshape(1)
            scaled = working / scales.item()
        elif granularity == "per_channel":
            # Per-channel computes one scale per slice along `channel_axis`.
            # For Linear weights, axis 0 corresponds to output channels / rows.
            reduce_dims = tuple(dim for dim in range(working.ndim) if dim != channel_axis)
            max_abs = working.abs().amax(dim=reduce_dims).clamp(min=1e-8)
            scales = max_abs / INT8_QMAX

            # Reshape the scale vector so PyTorch can broadcast it across the
            # remaining dimensions during division and reconstruction.
            view_shape = [1] * working.ndim
            view_shape[channel_axis] = working.shape[channel_axis]
            scaled = working / scales.view(*view_shape)
        else:
            raise ValueError(f"Unsupported int8 granularity: {granularity}")

        # Rounding converts the scaled real values to the nearest representable
        # integer, and clipping prevents overflow outside the signed int8 range.
        qweight = torch.round(scaled).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8).cpu()
        scales = scales.to(torch.float32).cpu()

        return Int8TensorPayload(
            qweight=qweight,
            scales=scales,
            original_size_bytes=tensor_nbytes(weight),
            quantized_size_bytes=tensor_nbytes(qweight) + tensor_nbytes(scales),
            metadata={
                "scale_shape": list(scales.shape),
                "scale_dtype": str(scales.dtype).replace("torch.", ""),
            },
        )
