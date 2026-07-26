from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]


MODEL_REGISTRY = {
    "qwen_0p5b_instruct": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "mlx_name": "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
        "family": "qwen2",
        "params": "0.5B",
        "instruct": True,
    },
    "qwen_1p5b_instruct": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "mlx_name": "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        "family": "qwen2",
        "params": "1.5B",
        "instruct": True,
    },
    "qwen_7b_instruct": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "mlx_name": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        "family": "qwen2",
        "params": "7B",
        "instruct": True,
    },
    "qwen_14b_instruct": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "mlx_name": "mlx-community/Qwen2.5-14B-Instruct-8bit",
        "family": "qwen2",
        "params": "14B",
        "instruct": True,
    },
    "qwen_72b_instruct": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "mlx_name": "mlx-community/Qwen2.5-72B-Instruct-8bit",
        "family": "qwen2",
        "params": "72B",
        "instruct": True,
    },
    "llama3_8b_instruct": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "llama",
        "params": "8B",
        "instruct": True,
        "gated": True,
    },
    "llama3_70b_instruct": {
        "name": "meta-llama/Meta-Llama-3-70B-Instruct",
        "family": "llama",
        "params": "70B",
        "instruct": True,
        "gated": True,
    },
    "llama31_8b_instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "local_mlx_name": "data/cache/models/llama31_8b_instruct_fp16",
        "family": "llama",
        "params": "8B",
        "instruct": True,
        "gated": True,
        "sae_layer": 19,  # Goodfire open SAE target layer (huggingface.co/Goodfire)
    },
    "llama33_70b_instruct": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "family": "llama",
        "params": "70B",
        "instruct": True,
        "gated": True,
        "sae_layer": 50,  # Goodfire open SAE target layer (huggingface.co/Goodfire)
    },
    "deepseek_7b_instruct": {
        "name": "deepseek-ai/deepseek-llm-7b-chat",
        "family": "llama",
        "params": "7B",
        "instruct": True,
    },
    "tiny_gpt2": {
        "name": "sshleifer/tiny-gpt2",
        "family": "gpt2",
        "params": "tiny",
    },
}


def resolve_model_name(model_name_or_key: str) -> str:
    return MODEL_REGISTRY.get(model_name_or_key, {}).get("name", model_name_or_key)


def resolve_mlx_model_name(model_name_or_key: str) -> str | None:
    if model_name_or_key in MODEL_REGISTRY:
        meta = MODEL_REGISTRY[model_name_or_key]
        local_name = meta.get("local_mlx_name")
        if local_name:
            local_path = Path(local_name)
            if not local_path.is_absolute():
                local_path = REPO_ROOT / local_path
            if local_path.exists():
                return str(local_path)
        return meta.get("mlx_name")
    return None


def resolve_backend(model_name_or_key: str, requested: str = "auto") -> str:
    if requested not in {"auto", "hf", "mlx"}:
        raise ValueError(f"unknown backend: {requested!r}")
    if requested != "auto":
        return requested
    if sys.platform == "darwin" and resolve_mlx_model_name(model_name_or_key):
        return "mlx"
    return "hf"


def get_model_meta(model_name_or_key: str) -> dict:
    if model_name_or_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_key] | {"key": model_name_or_key}
    return {"key": model_name_or_key, "name": model_name_or_key}
