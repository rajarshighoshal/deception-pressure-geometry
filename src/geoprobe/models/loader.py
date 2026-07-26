from __future__ import annotations

import gc
import importlib.util
import os

from geoprobe.models.registry import get_model_meta, resolve_model_name
from geoprobe.paths import ensure_hf_env

# Resolve the persistent cache before transformers reads HF_HOME at import time.
ensure_hf_env()

import torch  # noqa: E402  (imported after ensure_hf_env so HF_HOME is set before transformers reads it)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf_model(model_name_or_key: str, device: str | None = None, dtype: torch.dtype | None = None):
    device = device or choose_device()
    model_name = resolve_model_name(model_name_or_key)
    registry_meta = get_model_meta(model_name_or_key)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default precision matches the GPU/MPS inference convention; callers may pin an explicit dtype
    # (e.g. float16 for MLX<->PyTorch parity). CPU stays float32 since half is unsupported/slow there.
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16
    elif device == "cpu" and dtype is torch.float16:
        dtype = torch.float32
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    quantization = os.environ.get("GEOPROBE_HF_QUANTIZATION", "").strip().lower()
    if quantization:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("GEOPROBE_HF_QUANTIZATION needs bitsandbytes installed")
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("GEOPROBE_HF_QUANTIZATION must be 4bit or 8bit")

    device_map = os.environ.get("GEOPROBE_HF_DEVICE_MAP")
    if device == "cuda" and importlib.util.find_spec("accelerate") is not None:
        model_kwargs["device_map"] = device_map or {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    meta = registry_meta | {
        "name": model_name,
        "num_layers": getattr(model.config, "num_hidden_layers", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_params": sum(param.numel() for param in model.parameters()),
        "device": device,
    }
    return model, tokenizer, meta


def format_chat_prompt(prompt: str, tokenizer, model_meta: dict) -> str:
    if not model_meta.get("instruct"):
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
