"""Run the frozen exact-prefix pre-status causal replay plan once per vector root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence


for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from geoprobe.models.artifact_identity import (  # noqa: E402
    fingerprint_local_hf_artifact,
    validate_local_hf_artifact_identity,
)
from geoprobe.models.loader import load_hf_model  # noqa: E402
from geoprobe.models.relational_structured_action import (  # noqa: E402
    TOKENIZER_ARTIFACT_SHA256,
    canonical_json_sha256,
    FIELD_SPECS,
    STATUS_FIELD,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.provenance import git_provenance  # noqa: E402
from geoprobe.runtime.relational_pre_status_causal_runner import (  # noqa: E402
    load_causal_replay_inputs,
    run_exact_prefix_causal_replay,
    validate_causal_replay_output_paths,
)


FROZEN_REPLAY_PLAN_LEDGER_SHA256 = (
    "1970535a607bf650982aaa829e8ca9dedcae6824ce0b8d6c2fe7c44c3fd43732"
)
FROZEN_MODEL_ARTIFACT_SHA256 = (
    "8071d53a4509c0404328b791800ba79657556490b276b8383e1e8b2f0f63e104"
)
FROZEN_TOKENIZER_SHA256 = (
    "e901ac02d4fd9ca87689c4399c57897d03f261b9671f01808f9127614dc50b4c"
)
FROZEN_ROOT_COUNT = 402
FROZEN_EVENT_COUNT = 656
FROZEN_SOURCE_MANIFEST_KIND: Final = "relational_structured_action_rollout_manifest"
FROZEN_SOURCE_CONTRACT_KIND: Final = "relational_structured_action_rollout_contract"
FROZEN_SOURCE_MANIFEST_SCHEMA_VERSION: Final = 2
FROZEN_SOURCE_CONTRACT_SCHEMA_VERSION: Final = 2
FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256: Final = TOKENIZER_ARTIFACT_SHA256


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument("--vector-bank-root", type=Path, required=True)
    parser.add_argument("--source-rollout-manifest", type=Path, required=True)
    parser.add_argument("--expected-root-count", type=int, required=True)
    parser.add_argument("--expected-event-count", type=int, required=True)
    parser.add_argument("--expected-replay-plan-ledger-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--status-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-artifact-identity", type=Path)
    parser.add_argument("--expected-model-artifact-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args(argv)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _read_identity(path: Path) -> Mapping[str, Any]:
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("--model-artifact-identity is not valid UTF-8 JSON") from error
    if not isinstance(identity, dict):
        raise ValueError("--model-artifact-identity must be a JSON object")
    validate_local_hf_artifact_identity(identity)
    return identity


def _read_finite_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not finite UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_contract_body(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"schema_version", "kind", "contract_sha256"}
    }


def _validate_contract_tokenizer_binding(
    contract_tokenizer: Mapping[str, Any],
    *,
    model_identity: Mapping[str, Any] | None,
) -> Mapping[str, str]:
    if not isinstance(contract_tokenizer, Mapping):
        raise ValueError("source rollout contract tokenizer artifacts must be mapped")
    frozen_expected = FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256
    if {str(name) for name in contract_tokenizer} != set(frozen_expected):
        raise ValueError("source rollout contract tokenizer artifacts differ from the frozen expectation")
    for name, expected in frozen_expected.items():
        if _sha(contract_tokenizer.get(name), f"source rollout contract tokenizer SHA-256 for {name}") != expected:
            raise ValueError("source rollout contract tokenizer artifacts differ from the frozen expectation")
    if model_identity is None:
        return {name: str(contract_tokenizer[name]) for name in sorted(frozen_expected)}
    files = model_identity.get("files")
    if not isinstance(files, list):
        raise ValueError("validated model identity must include file records")
    identity_by_path: dict[str, str] = {}
    for record in files:
        if not isinstance(record, Mapping):
            continue
        path = record.get("path")
        digest = record.get("sha256")
        if isinstance(path, str):
            identity_by_path[path] = _sha(digest, f"validated model file SHA-256 for {path}")
    for name, expected in frozen_expected.items():
        if identity_by_path.get(name) != expected:
            raise ValueError("source rollout contract tokenizer artifacts do not match validated model identity")
    return {name: str(contract_tokenizer[name]) for name in sorted(frozen_expected)}


def _validate_source_rollout_manifest(
    source_bindings: Mapping[str, Any],
    source_manifest_path: Path,
    model_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(source_bindings, Mapping):
        raise ValueError("replay-plan source-rollout bindings are invalid")
    rows_binding = source_bindings.get("rows")
    manifest_binding = source_bindings.get("rollout_manifest")
    if not isinstance(rows_binding, Mapping) or not isinstance(manifest_binding, Mapping):
        raise ValueError("replay-plan source-rollout bindings are invalid")
    rows_sha = _sha(rows_binding.get("sha256"), "source rollout rows SHA-256")
    manifest_sha = _sha(manifest_binding.get("sha256"), "source rollout manifest SHA-256")
    manifest_bytes = manifest_binding.get("bytes")
    if not isinstance(manifest_bytes, int) or isinstance(manifest_bytes, bool) or manifest_bytes < 0:
        raise ValueError("replay-plan source-rollout manifest bytes are invalid")
    manifest_path = source_manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("source-rollout manifest is not a file")
    if manifest_path.stat().st_size != manifest_bytes:
        raise ValueError("source-rollout manifest physical binding is invalid")
    physical_manifest_sha = file_sha256(manifest_path)
    if physical_manifest_sha != manifest_sha:
        raise ValueError("source-rollout manifest physical binding is invalid")
    manifest = _read_finite_json(manifest_path, "source-rollout manifest")
    if (
        manifest.get("schema_version") != FROZEN_SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != FROZEN_SOURCE_MANIFEST_KIND
        or manifest.get("status") != "success"
    ):
        raise ValueError("source-rollout manifest is not a successful structured-action v2 manifest")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("source-rollout manifest output is missing")
    output_rows_sha = _sha(
        output.get("rows_sha256"),
        "source-rollout manifest output rows SHA-256",
    )
    if output_rows_sha != rows_sha:
        raise ValueError("source-rollout manifest rows hash is not bound to replay-plan rows")
    contract = manifest.get("rollout_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("source-rollout manifest contract is missing")
    if contract.get("schema_version") != FROZEN_SOURCE_CONTRACT_SCHEMA_VERSION or contract.get("kind") != FROZEN_SOURCE_CONTRACT_KIND:
        raise ValueError("source-rollout contract is not a structured-action rollout contract")
    model_artifact_sha = _sha(
        contract.get("model_artifact_sha256"),
        "source-rollout contract model artifact SHA-256",
    )
    if model_artifact_sha != FROZEN_MODEL_ARTIFACT_SHA256:
        raise ValueError("source-rollout contract does not reference the frozen source model")
    expected_contract_sha = _sha(
        contract.get("contract_sha256"),
        "source-rollout contract hash",
    )
    if canonical_json_sha256(_canonical_contract_body(contract)) != expected_contract_sha:
        raise ValueError("source-rollout contract canonical hash is invalid")
    tokenizer_map = _validate_contract_tokenizer_binding(
        contract.get("tokenizer_artifact_sha256"),
        model_identity=model_identity,
    )
    return {
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "sha256": physical_manifest_sha,
        "bytes": manifest_bytes,
        "rows_sha256": rows_sha,
        "rollout_contract": {
            "schema_version": contract.get("schema_version"),
            "kind": contract.get("kind"),
            "contract_sha256": expected_contract_sha,
            "model_artifact_sha256": model_artifact_sha,
            "tokenizer_artifact_sha256": dict(tokenizer_map),
        },
    }


def _model_provenance(
    args: argparse.Namespace,
    *,
    model_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    required = (
        args.model, args.model_artifact_identity,
        args.expected_model_artifact_sha256, args.expected_tokenizer_sha256,
    )
    if any(value is None for value in required):
        raise ValueError("a live causal run requires local --model and exact model/tokenizer identity inputs")
    if args.device != "cuda" or args.dtype != "bfloat16":
        raise ValueError("the frozen causal runner requires CUDA BF16")
    if (
        _sha(
            args.expected_model_artifact_sha256,
            "expected model artifact SHA-256",
        )
        != FROZEN_MODEL_ARTIFACT_SHA256
        or _sha(
            args.expected_tokenizer_sha256,
            "expected tokenizer SHA-256",
        )
        != FROZEN_TOKENIZER_SHA256
    ):
        raise ValueError(
            "model/tokenizer expectations differ from the frozen source-model trust root"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen causal runner requires an available CUDA device")
    identity = model_identity if model_identity is not None else _read_identity(args.model_artifact_identity)
    if identity["artifact_sha256"] != FROZEN_MODEL_ARTIFACT_SHA256:
        raise ValueError("model identity differs from --expected-model-artifact-sha256")
    if identity["tokenizer_sha256"] != FROZEN_TOKENIZER_SHA256:
        raise ValueError("model identity differs from --expected-tokenizer-sha256")
    observed = fingerprint_local_hf_artifact(args.model)
    if observed != identity:
        raise ValueError("local model/tokenizer files differ from supplied exact identity")
    return {
        "local_model_path": str(args.model.resolve()),
        "artifact_identity": dict(identity),
        "artifact_sha256": identity["artifact_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
    }


def _reject_runtime_overrides() -> None:
    forbidden = {
        name: os.environ.get(name, "").strip()
        for name in ("GEOPROBE_HF_QUANTIZATION", "GEOPROBE_HF_DEVICE_MAP")
        if os.environ.get(name, "").strip()
    }
    if forbidden:
        raise ValueError(
            "the frozen causal run forbids quantization and device-map overrides"
        )


def _assert_effective_cuda_bf16(model: Any) -> Mapping[str, Any]:
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("loaded causal model has no parameters")
    devices = sorted({parameter.device.type for parameter in parameters})
    device_indices = sorted(
        {
            int(parameter.device.index)
            for parameter in parameters
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )
    floating_dtypes = sorted(
        {str(parameter.dtype) for parameter in parameters if parameter.is_floating_point()}
    )
    if (
        devices != ["cuda"]
        or len(device_indices) != 1
        or floating_dtypes != [str(torch.bfloat16)]
    ):
        raise RuntimeError(
            "loaded causal model is not wholly resident on CUDA with BF16 parameters"
        )
    if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    ):
        raise RuntimeError("the frozen causal model must not be quantized")
    return {
        "parameter_devices": devices,
        "parameter_cuda_device_indices": device_indices,
        "floating_parameter_dtypes": floating_dtypes,
        "quantized": False,
    }


def _cuda_runtime_identity(device_index: int) -> Mapping[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    raw_device_uuid = getattr(properties, "uuid", None)
    device_uuid = "" if raw_device_uuid is None else str(raw_device_uuid).strip()
    if not device_uuid:
        raise RuntimeError("CUDA device UUID is unavailable; cross-pod-safe resume is disabled")
    return {
        "device_index": int(device_index),
        "device_uuid": device_uuid,
        "device_name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "torch_compiled_cuda": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _validate_status_tokenizer(tokenizer: Any) -> None:
    spec = FIELD_SPECS[STATUS_FIELD]
    prefix_ids = tuple(
        int(token)
        for token in tokenizer.encode(spec.prefix_text, add_special_tokens=False)
    )
    if prefix_ids != spec.prefix_token_ids:
        raise ValueError("runtime tokenizer differs from the frozen Status prefix")
    for candidate in spec.candidates:
        contextual = tuple(
            int(token)
            for token in tokenizer.encode(
                spec.prefix_text + candidate.decoded_exact,
                add_special_tokens=False,
            )
        )
        decoded = tokenizer.decode(
            [candidate.raw_token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if contextual != spec.prefix_token_ids + (candidate.raw_token_id,) or decoded != candidate.decoded_exact:
            raise ValueError(
                f"runtime tokenizer differs from frozen {candidate.mapped_action} token"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if (
        args.expected_root_count != FROZEN_ROOT_COUNT
        or args.expected_event_count != FROZEN_EVENT_COUNT
    ):
        raise ValueError("root/event counts differ from the frozen causal inventory")
    if args.beta != 1.0:
        raise ValueError("the frozen causal run requires --beta 1.0")
    ledger, works, input_hashes = load_causal_replay_inputs(args.replay_plan, args.vector_bank_root)
    expected_plan_sha = _sha(
        args.expected_replay_plan_ledger_sha256,
        "expected replay-plan ledger SHA-256",
    )
    validated_model_identity = None
    if not args.validate_only and args.model_artifact_identity is not None:
        validated_model_identity = _read_identity(args.model_artifact_identity)
    source_rollout_manifest = _validate_source_rollout_manifest(
        ledger.get("rollout_binding"),
        args.source_rollout_manifest,
        model_identity=validated_model_identity,
    )
    input_hashes = {
        **dict(input_hashes),
        "source_rollout_manifest": source_rollout_manifest,
    }
    if (
        expected_plan_sha != FROZEN_REPLAY_PLAN_LEDGER_SHA256
        or ledger.get("ledger_sha256") != FROZEN_REPLAY_PLAN_LEDGER_SHA256
    ):
        raise ValueError("replay plan differs from the frozen causal trust root")
    if len(works) != args.expected_root_count or sum(len(work.events) for work in works) != args.expected_event_count:
        raise ValueError("validated replay-plan counts differ from CLI assertions")
    if args.validate_only:
        if any(value is not None for value in (args.out, args.manifest_out, args.status_out)):
            raise ValueError("--validate-only creates no output; omit output paths")
        print(json.dumps({"status": "validated", "root_count": len(works), "event_count": sum(len(work.events) for work in works), "replay_plan_ledger_sha256": ledger["ledger_sha256"]}, sort_keys=True))
        return 0
    if args.out is None:
        raise ValueError("a live causal run requires --out")
    manifest_out = args.manifest_out or args.out.with_suffix(args.out.suffix + ".manifest.json")
    status_out = args.status_out or args.out.parent / "STATUS"
    validate_causal_replay_output_paths(
        args.out, manifest_out, status_out, resume=args.resume
    )
    _reject_runtime_overrides()
    provenance = _model_provenance(args, model_identity=validated_model_identity)
    model, tokenizer, loaded_meta = load_hf_model(str(args.model), device="cuda", dtype=torch.bfloat16)
    effective_runtime = _assert_effective_cuda_bf16(model)
    cuda_runtime = _cuda_runtime_identity(
        int(effective_runtime["parameter_cuda_device_indices"][0])
    )
    _validate_status_tokenizer(tokenizer)
    runtime = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__,
        "loader_meta": loaded_meta,
        "effective_model_runtime": effective_runtime,
        "cuda_runtime": cuda_runtime,
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT / "src/geoprobe/runtime/relational_pre_status_causal_runner.py",
                REPO_ROOT / "src/geoprobe/models/relational_causal_replay.py",
                REPO_ROOT / "src/geoprobe/models/hf_capture.py",
                REPO_ROOT / "src/geoprobe/control/relational_pre_status_causal.py",
            ]
        ),
        "capture_hooks_enabled": False,
    }
    manifest = run_exact_prefix_causal_replay(
        works, model=model, tokenizer=tokenizer, out=args.out, manifest_out=manifest_out,
        status_out=status_out, beta=args.beta, expected_root_count=args.expected_root_count,
        expected_event_count=args.expected_event_count, input_hashes=input_hashes,
        model_provenance=provenance, runtime_provenance=runtime, resume=args.resume,
    )
    manifest["argv"] = list(
        sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    )
    temporary_manifest = manifest_out.with_name(
        f".{manifest_out.name}.tmp.{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_manifest, manifest_out)
    print(json.dumps({"status": manifest["status"], "out": str(args.out.resolve()), "completed_rows": manifest["completed_rows"], "expected_row_count": manifest["expected_row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
