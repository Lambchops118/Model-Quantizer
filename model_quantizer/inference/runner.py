"""Interactive inference runtime for local raw and quantized models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from model_quantizer.artifacts.loader import QuantizedArtifactLoader, load_normalized_model_config
from model_quantizer.configuration import ModelConfig, ProjectConfig
from model_quantizer.download.downloader import ModelDownloader
from model_quantizer.utils.device import resolve_compute_device, resolve_torch_dtype
from model_quantizer.utils.filesystem import sanitize_name


@dataclass(frozen=True)
class InferenceRequest:
    """Resolved CLI inputs for a model inference session."""

    model_name: str
    source: str
    quantizer_name: Optional[str]
    mode: str
    device: str
    prompt: Optional[str]
    system_prompt: Optional[str]
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    repetition_penalty: float


@dataclass(frozen=True)
class LoadedModelBundle:
    """All runtime objects needed for generation."""

    model_name: str
    source: str
    source_path: Path
    resolved_device: torch.device
    load_seconds: float
    tokenizer: Any
    model: AutoModelForCausalLM


@dataclass(frozen=True)
class GenerationMetrics:
    """Useful latency and throughput metrics for one generation turn."""

    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    prompt_characters: int
    generated_characters: int
    time_to_first_token_seconds: Optional[float]
    completion_seconds: float
    decode_seconds: Optional[float]
    tokens_per_second: Optional[float]
    decode_tokens_per_second: Optional[float]
    max_cuda_memory_allocated_gb: Optional[float]
    max_cuda_memory_reserved_gb: Optional[float]


@dataclass(frozen=True)
class GenerationResponse:
    """Generated text plus metrics."""

    text: str
    metrics: GenerationMetrics


class MetricsTextStreamer(TextStreamer):
    """Text streamer that prints tokens live while capturing timing metadata."""

    def __init__(self, tokenizer: Any) -> None:
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.first_token_at: Optional[float] = None
        self.chunks: List[str] = []

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Write each finalized text chunk to stdout and store it."""

        if text:
            if self.first_token_at is None:
                self.first_token_at = time.perf_counter()
            print(text, end="", flush=True)
            self.chunks.append(text)


class InferenceRunner:
    """Loads local raw snapshots or local quantized artifacts and chats with them."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.downloader = ModelDownloader(
            config.paths.raw_models_dir,
            prefer_safetensors=config.runtime.prefer_safetensors,
        )

    def run(self, request: InferenceRequest) -> None:
        """Execute one-shot or conversational inference."""

        bundle = self.load_model_bundle(request)
        self._print_load_summary(bundle)
        if request.mode == "one-shot":
            self._run_one_shot(bundle, request)
            return
        self._run_conversational(bundle, request)

    def load_model_bundle(self, request: InferenceRequest) -> LoadedModelBundle:
        """Load a local model bundle for inference."""

        try:
            model_config = self.config.models[request.model_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.config.models))
            raise ValueError(
                f"Unknown model '{request.model_name}'. Available models: {available}"
            ) from exc

        resolved_device = resolve_compute_device(request.device)
        load_start = time.perf_counter()

        if request.source == "raw":
            source_path, tokenizer, model = self._load_raw_model(model_config, resolved_device)
        elif request.source == "quantized":
            if not request.quantizer_name:
                raise ValueError(
                    "--artifact-quantizer is required when --model-source=quantized."
                )
            source_path, tokenizer, model = self._load_quantized_model(
                request.model_name,
                request.quantizer_name,
                resolved_device,
            )
        else:
            raise ValueError(f"Unsupported inference source '{request.source}'.")

        self._prepare_tokenizer(tokenizer)
        model.config.use_cache = False
        model.eval()

        return LoadedModelBundle(
            model_name=request.model_name,
            source=request.source,
            source_path=source_path,
            resolved_device=resolved_device,
            load_seconds=time.perf_counter() - load_start,
            tokenizer=tokenizer,
            model=model,
        )

    def _run_one_shot(self, bundle: LoadedModelBundle, request: InferenceRequest) -> None:
        """Generate one response without persisting history."""

        user_prompt = request.prompt
        if user_prompt is None:
            user_prompt = input("User> ").strip()
        if not user_prompt:
            raise ValueError("A non-empty prompt is required for one-shot mode.")

        messages = self._compose_messages(request.system_prompt, [], user_prompt)
        print("Assistant> ", end="", flush=True)
        response = self.generate(bundle, messages, request)
        print()
        self._print_metrics(response.metrics)

    def _run_conversational(self, bundle: LoadedModelBundle, request: InferenceRequest) -> None:
        """Launch a basic remembered-history chatbot loop."""

        history: List[Dict[str, str]] = []
        print("Conversational mode. Use /reset to clear history and /exit to stop.")

        while True:
            user_prompt = input("User> ").strip()
            if not user_prompt:
                continue
            if user_prompt in {"/exit", "/quit"}:
                break
            if user_prompt == "/reset":
                history.clear()
                print("Conversation history cleared.")
                continue

            messages = self._compose_messages(request.system_prompt, history, user_prompt)
            print("Assistant> ", end="", flush=True)
            response = self.generate(bundle, messages, request)
            print()
            self._print_metrics(response.metrics)

            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": response.text})

    def generate(
        self,
        bundle: LoadedModelBundle,
        messages: List[Dict[str, str]],
        request: InferenceRequest,
    ) -> GenerationResponse:
        """Generate a response while streaming tokens and collecting metrics."""

        prompt_text = self._render_prompt(bundle.tokenizer, messages)
        encoded = bundle.tokenizer(prompt_text, return_tensors="pt")
        model_inputs = {name: tensor.to(bundle.resolved_device) for name, tensor in encoded.items()}
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])

        streamer = MetricsTextStreamer(bundle.tokenizer)
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "use_cache": False,
            "pad_token_id": bundle.tokenizer.pad_token_id,
            "eos_token_id": bundle.tokenizer.eos_token_id,
            "streamer": streamer,
            "return_dict_in_generate": True,
        }
        if not request.do_sample:
            generation_kwargs.pop("top_p", None)
            generation_kwargs.pop("temperature", None)

        cuda_index = self._cuda_index(bundle.resolved_device)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
            torch.cuda.reset_peak_memory_stats(cuda_index)

        start = time.perf_counter()
        with torch.inference_mode():
            output = bundle.model.generate(**generation_kwargs)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)

        finished_at = time.perf_counter()
        generated_tokens = int(output.sequences.shape[-1] - prompt_tokens)
        decode_seconds = (
            finished_at - streamer.first_token_at if streamer.first_token_at is not None else None
        )
        completion_seconds = finished_at - start
        tokens_per_second = (
            generated_tokens / completion_seconds if completion_seconds > 0 and generated_tokens > 0 else None
        )
        decode_tokens_per_second = (
            generated_tokens / decode_seconds
            if decode_seconds and decode_seconds > 0 and generated_tokens > 0
            else None
        )
        generated_text = "".join(streamer.chunks)

        max_allocated = None
        max_reserved = None
        if cuda_index is not None:
            max_allocated = round(torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 3), 3)
            max_reserved = round(torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 3), 3)

        return GenerationResponse(
            text=generated_text,
            metrics=GenerationMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
                prompt_characters=len(prompt_text),
                generated_characters=len(generated_text),
                time_to_first_token_seconds=(
                    streamer.first_token_at - start
                    if streamer.first_token_at is not None
                    else None
                ),
                completion_seconds=completion_seconds,
                decode_seconds=decode_seconds,
                tokens_per_second=tokens_per_second,
                decode_tokens_per_second=decode_tokens_per_second,
                max_cuda_memory_allocated_gb=max_allocated,
                max_cuda_memory_reserved_gb=max_reserved,
            ),
        )

    def _load_raw_model(
        self,
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM]:
        """Load an already-downloaded raw checkpoint from local storage only."""

        downloaded = self.downloader.require_local_snapshot(model_config)
        tokenizer = AutoTokenizer.from_pretrained(
            downloaded.local_path,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=True,
        )
        config = load_normalized_model_config(
            downloaded.local_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        torch_dtype = self._resolve_inference_dtype(model_config, resolved_device)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": model_config.trust_remote_code,
            "config": config,
            "dtype": torch_dtype,
            "local_files_only": True,
        }
        if getattr(config, "model_type", None) == "phi3":
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            downloaded.local_path,
            **model_kwargs,
        )
        return downloaded.local_path, tokenizer, model.to(resolved_device)

    def _load_quantized_model(
        self,
        model_name: str,
        quantizer_name: str,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM]:
        """Load a quantized artifact reconstructed into a normal HF model."""

        artifact_dir = (
            self.config.paths.quantized_models_dir
            / sanitize_name(model_name)
            / sanitize_name(quantizer_name)
        )
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Quantized artifact directory not found: {artifact_dir}. "
                "Run the quantization pipeline first."
            )

        manifest = QuantizedArtifactLoader.load_manifest(artifact_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
            local_files_only=True,
        )
        model = QuantizedArtifactLoader.load_model(artifact_dir, device=str(resolved_device))
        if resolved_device.type == "cpu":
            first_param = next(model.parameters(), None)
            if first_param is not None and first_param.dtype == torch.float16:
                model = model.to(dtype=torch.float32)
        return artifact_dir, tokenizer, model

    @staticmethod
    def _compose_messages(
        system_prompt: Optional[str],
        history: List[Dict[str, str]],
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """Build the canonical message list for a generation turn."""

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
        """Render messages through a chat template when available."""

        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines: List[str] = []
        role_names = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }
        for message in messages:
            role = role_names.get(message["role"], message["role"].capitalize())
            lines.append(f"{role}: {message['content'].strip()}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    @staticmethod
    def _prepare_tokenizer(tokenizer: Any) -> None:
        """Fill missing tokenizer generation metadata."""

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _resolve_inference_dtype(
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> Optional[torch.dtype]:
        """Prefer safer defaults when raw checkpoints are run on CPU."""

        dtype = resolve_torch_dtype(model_config.torch_dtype)
        if resolved_device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype

    @staticmethod
    def _cuda_index(device: torch.device) -> Optional[int]:
        """Return a usable CUDA index when the active device is CUDA."""

        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()

    @staticmethod
    def _print_load_summary(bundle: LoadedModelBundle) -> None:
        """Print where the model came from and how long loading took."""

        print(
            f"Loaded local {bundle.source} model '{bundle.model_name}' from {bundle.source_path} "
            f"on {bundle.resolved_device} in {bundle.load_seconds:.2f}s."
        )

    @staticmethod
    def _print_metrics(metrics: GenerationMetrics) -> None:
        """Print a compact metrics block after each response."""

        print("Metrics:")
        print(f"  prompt_tokens={metrics.prompt_tokens}")
        print(f"  generated_tokens={metrics.generated_tokens}")
        print(f"  total_tokens={metrics.total_tokens}")
        print(f"  prompt_characters={metrics.prompt_characters}")
        print(f"  generated_characters={metrics.generated_characters}")
        if metrics.time_to_first_token_seconds is not None:
            print(f"  time_to_first_token_seconds={metrics.time_to_first_token_seconds:.3f}")
        print(f"  completion_seconds={metrics.completion_seconds:.3f}")
        if metrics.decode_seconds is not None:
            print(f"  decode_seconds={metrics.decode_seconds:.3f}")
        if metrics.tokens_per_second is not None:
            print(f"  tokens_per_second={metrics.tokens_per_second:.3f}")
        if metrics.decode_tokens_per_second is not None:
            print(f"  decode_tokens_per_second={metrics.decode_tokens_per_second:.3f}")
        if metrics.max_cuda_memory_allocated_gb is not None:
            print(f"  max_cuda_memory_allocated_gb={metrics.max_cuda_memory_allocated_gb:.3f}")
        if metrics.max_cuda_memory_reserved_gb is not None:
            print(f"  max_cuda_memory_reserved_gb={metrics.max_cuda_memory_reserved_gb:.3f}")
