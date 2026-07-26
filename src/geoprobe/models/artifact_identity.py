"""Content identities for local Hugging Face model and tokenizer artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from geoprobe.io import file_sha256


_TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tiktoken.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weight_paths(root: Path) -> list[Path]:
    candidates = (
        ("model.safetensors.index.json", "model.safetensors"),
        ("pytorch_model.bin.index.json", "pytorch_model.bin"),
    )
    for index_name, single_name in candidates:
        index_path = root / index_name
        if index_path.is_file():
            payload = json.loads(index_path.read_text())
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid weight map in {index_path}")
            shard_names = sorted({str(value) for value in weight_map.values()})
            paths = [index_path, *(root / name for name in shard_names)]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise ValueError(f"model weight artifact is incomplete: {missing[:3]}")
            return paths
        single_path = root / single_name
        if single_path.is_file():
            return [single_path]
    raise ValueError(f"no supported Hugging Face weight artifact in {root}")


def _records(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return records


def _support_paths(root: Path) -> list[Path]:
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model artifact is missing config.json: {root}")
    paths = [config_path]
    generation_config = root / "generation_config.json"
    if generation_config.is_file():
        paths.append(generation_config)
    paths.extend(sorted(root.rglob("*.py")))
    tokenizer_paths = [root / name for name in sorted(_TOKENIZER_FILES) if (root / name).is_file()]
    tokenizer_paths.extend(sorted((root / "chat_templates").glob("*.jinja")))
    lexical = {
        "tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "tiktoken.model",
        "vocab.json", "vocab.txt",
    }
    if not any(path.name in lexical for path in tokenizer_paths):
        raise ValueError(f"model artifact is missing tokenizer vocabulary/model files: {root}")
    return [*paths, *tokenizer_paths]


def _is_weight_record(record: dict[str, Any]) -> bool:
    path = str(record["path"])
    return (
        path.endswith((".safetensors", ".bin"))
        or path in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    )


def fingerprint_local_hf_artifact(model_dir: Path) -> dict[str, Any]:
    """Hash every local file that can define loaded weights, config, or tokenization."""
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"exact artifact identity requires a local model directory: {model_dir}")
    weight_paths = _weight_paths(root)
    support_paths = _support_paths(root)

    weights = _records(root, weight_paths)
    support = _records(root, support_paths)
    model = [record for record in support if (
        record["path"] in {"config.json", "generation_config.json"}
        or record["path"].endswith(".py")
    )]
    tokenizer = [record for record in support if record not in model]
    all_records = sorted(
        {record["path"]: record for record in [*weights, *model, *tokenizer]}.values(),
        key=lambda record: record["path"],
    )
    return {
        "schema_version": 1,
        "kind": "local_hf_artifact_identity",
        "artifact_sha256": _canonical_sha256(all_records),
        "weights_sha256": _canonical_sha256(weights),
        "model_config_sha256": _canonical_sha256(model),
        "tokenizer_sha256": _canonical_sha256(tokenizer),
        "files": all_records,
        "file_count": len(all_records),
        "total_bytes": sum(int(record["bytes"]) for record in all_records),
    }


def validate_local_hf_support_identity(
    model_support_dir: Path, identity: dict[str, Any]
) -> None:
    """Bind a local tokenizer/config mirror to a separately verified weight identity."""
    validate_local_hf_artifact_identity(identity)
    root = model_support_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"exact artifact support requires a local directory: {model_support_dir}")
    actual = _records(root, _support_paths(root))
    expected = [record for record in identity["files"] if not _is_weight_record(record)]
    if actual != expected:
        raise ValueError("local model tokenizer/config support differs from artifact identity")


def validate_local_hf_artifact_identity(identity: dict[str, Any]) -> None:
    if identity.get("schema_version") != 1 or identity.get("kind") != "local_hf_artifact_identity":
        raise ValueError("invalid local Hugging Face artifact identity")
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model artifact identity has no files")
    paths = [record.get("path") for record in files]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("model artifact identity file inventory is not canonical")
    for record in files:
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid model artifact file record")
        record_path = Path(str(record["path"]))
        if (
            not isinstance(record["path"], str)
            or record_path.is_absolute()
            or ".." in record_path.parts
        ):
            raise ValueError("model artifact paths must be relative")
        digest = str(record["sha256"])
        if int(record["bytes"]) < 0 or len(digest) != 64:
            raise ValueError("invalid model artifact size or digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("invalid model artifact size or digest") from error
    if identity.get("artifact_sha256") != _canonical_sha256(files):
        raise ValueError("model artifact aggregate digest is invalid")
    if int(identity.get("file_count", -1)) != len(files):
        raise ValueError("model artifact file count is invalid")
    if int(identity.get("total_bytes", -1)) != sum(int(record["bytes"]) for record in files):
        raise ValueError("model artifact byte count is invalid")
    weights = [record for record in files if _is_weight_record(record)]
    model = [
        record for record in files
        if record["path"] in {"config.json", "generation_config.json"}
        or record["path"].endswith(".py")
    ]
    tokenizer = [
        record for record in files
        if record["path"] in _TOKENIZER_FILES
        or record["path"].startswith("chat_templates/") and record["path"].endswith(".jinja")
    ]
    if not weights or not model or not tokenizer:
        raise ValueError("model artifact identity omits weights, config, or tokenizer files")
    if identity.get("weights_sha256") != _canonical_sha256(weights):
        raise ValueError("model artifact weight digest is invalid")
    if identity.get("model_config_sha256") != _canonical_sha256(model):
        raise ValueError("model artifact config digest is invalid")
    if identity.get("tokenizer_sha256") != _canonical_sha256(tokenizer):
        raise ValueError("model artifact tokenizer digest is invalid")


def validate_local_hf_artifact_readability(
    model_dir: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    """Check a frozen model inventory without hashing its large weight files."""
    validate_local_hf_artifact_identity(identity)
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"model artifact directory is missing: {model_dir}")
    checked = 0
    for record in identity["files"]:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"model artifact file is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"model artifact file size differs: {path}")
        with path.open("rb") as handle:
            if not handle.read(64):
                raise ValueError(f"model artifact file is unreadable: {path}")
        checked += 1
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model artifact config is unreadable") from error
    if not isinstance(config, dict) or not config:
        raise ValueError("model artifact config is empty")
    return {
        "status": "pass",
        "kind": "local_hf_artifact_readability",
        "files_checked": checked,
        "artifact_sha256": identity["artifact_sha256"],
    }


__all__ = [
    "fingerprint_local_hf_artifact",
    "validate_local_hf_artifact_identity",
    "validate_local_hf_artifact_readability",
    "validate_local_hf_support_identity",
]
