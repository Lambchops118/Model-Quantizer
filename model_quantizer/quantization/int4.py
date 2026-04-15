"""Manual grouped symmetric int4 weight-only quantization.

Grouped int4 quantization saves memory more aggressively than int8 by using only
4 bits per weight. The tradeoff is higher error, so we store one scale per small
group of values instead of one scale for the whole tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch

from model_quantizer.artifacts.manager import ShardedTensorWriter
from model_quantizer.quantization.base import BaseQuantizer, QuantizationContext
from model_quantizer.quantization.common import (
    count_checkpoint_tensors,
    iter_source_tensors,
    maybe_cast_passthrough_tensor,
    tensor_nbytes,
)


INT4_QMAX = 7
INT4_STORAGE_ZERO_POINT = 8


@dataclass(frozen=True)
class Int4TensorPayload:
    """The stored tensors and metadata for one grouped int4 weight matrix."""

    packed_weights: torch.Tensor
    scales: torch.Tensor
    quantized_size_bytes: int
    metadata: Dict[str, Any]


class Int4GroupedQuantizer(BaseQuantizer):
    """Grouped symmetric int4 quantization for Linear weights."""

    def quantize(self, context: QuantizationContext) -> Dict[str, Any]:
        group_size = int(context.quantizer_config.options.get("group_size", 128))
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
            "Starting manual grouped int4 quantization with group_size=%s, device=%s",
            group_size,
            quant_device,
        )

        for index, source_tensor in enumerate(iter_source_tensors(context.raw_model_dir), start=1):
            tensor = source_tensor.tensor
            original_size_bytes += tensor_nbytes(tensor)

            if source_tensor.name in linear_weight_names:
                payload = self._quantize_weight(
                    weight=tensor,
                    group_size=group_size,
                    device=quant_device,
                )
                packed_key = f"quantized::{source_tensor.name}::packed_weight"
                scale_key = f"quantized::{source_tensor.name}::scale"
                writer.add_tensor(packed_key, payload.packed_weights)
                writer.add_tensor(scale_key, payload.scales)
                quantized_size_bytes += payload.quantized_size_bytes

                tensor_records.append(
                    {
                        "name": source_tensor.name,
                        "kind": "int4_weight",
                        "scheme": "symmetric_int4_grouped",
                        "group_size": group_size,
                        "storage_zero_point": INT4_STORAGE_ZERO_POINT,
                        "original_shape": list(tensor.shape),
                        "original_numel": tensor.numel(),
                        "original_dtype": str(tensor.dtype).replace("torch.", ""),
                        "packed_weight_key": packed_key,
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
                "group_size": group_size,
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
        group_size: int,
        device: torch.device,
    ) -> Int4TensorPayload:
        """Quantize a matrix row-by-row into grouped symmetric int4.

        Grouping splits each row into small blocks. Every block gets its own
        scale:

            scale_g = max(abs(group_values)) / 7

        That local scale lowers error versus using a single scale for an entire
        large matrix. After scaling, values are rounded and clipped to [-7, 7].
        We then pack two signed 4-bit values into one uint8 byte to cut storage
        in half relative to int8.
        """

        original_shape = tuple(weight.shape)
        if len(original_shape) < 2:
            raise ValueError("Grouped int4 quantization expects at least a 2D weight tensor.")

        matrix = weight.detach().to(device=device, dtype=torch.float32).reshape(original_shape[0], -1)
        rows, columns = matrix.shape
        padded_columns = int(math.ceil(columns / group_size) * group_size)
        if padded_columns != columns:
            pad_amount = padded_columns - columns
            matrix = torch.nn.functional.pad(matrix, (0, pad_amount))

        groups_per_row = padded_columns // group_size
        grouped = matrix.view(rows, groups_per_row, group_size)

        # Each group gets its own symmetric scale from the largest magnitude.
        scales = grouped.abs().amax(dim=2).clamp(min=1e-8) / INT4_QMAX

        # Divide by the per-group scale, then round and clip into the signed
        # int4 range [-7, 7]. We keep the range symmetric around zero.
        scaled = grouped / scales.unsqueeze(-1)
        quantized = torch.round(scaled).clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8)

        # Nibbles are stored as unsigned values, so we shift the signed range by
        # +8. Zero becomes 8, positive values become 9..15, and negative values
        # become 1..7. The 0 code is unused because we chose the symmetric range.
        unsigned = (quantized.to(torch.int16) + INT4_STORAGE_ZERO_POINT).to(torch.uint8).reshape(-1)
        if unsigned.numel() % 2 != 0:
            unsigned = torch.cat((unsigned, torch.zeros(1, dtype=torch.uint8, device=unsigned.device)))

        low = unsigned[0::2]
        high = unsigned[1::2] << 4
        packed = (low | high).cpu()
        scales = scales.to(torch.float32).cpu()

        return Int4TensorPayload(
            packed_weights=packed,
            scales=scales,
            quantized_size_bytes=tensor_nbytes(packed) + tensor_nbytes(scales),
            metadata={
                "rows": rows,
                "columns": columns,
                "padded_columns": padded_columns,
                "groups_per_row": groups_per_row,
                "scale_shape": list(scales.shape),
                "scale_dtype": str(scales.dtype).replace("torch.", ""),
            },
        )
