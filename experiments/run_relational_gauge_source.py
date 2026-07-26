"""Run the frozen one-step four-layer relational gauge controller on source roots."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any


for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from experiments import run_relational_pre_status_causal as causal_cli  # noqa: E402
from geoprobe.control.relational_gauge_controller import (  # noqa: E402
    GaugeControllerObservation,
)
from geoprobe.data.relational_pre_status_rooted_star_store import (  # noqa: E402
    build_relational_pre_status_rooted_star_index,
)
from geoprobe.eval.relational_gauge_controller_artifact import (  # noqa: E402
    load_fold_gauge_controller_artifact,
    validate_relational_gauge_controller_manifest,
)
from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (  # noqa: E402
    load_relational_pre_status_rooted_graph_artifacts,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.models.loader import load_hf_model  # noqa: E402
from geoprobe.models.relational_capture import set_eager_attention  # noqa: E402
from geoprobe.models.relational_gauge_replay import (  # noqa: E402
    SOURCE_GAUGE_ARM_ORDER,
    build_source_gauge_controllers,
)
from geoprobe.provenance import git_provenance  # noqa: E402
from geoprobe.runtime.relational_gauge_replay_observation import (  # noqa: E402
    CaptureBoundPreStatusGaugeObservationBuilder,
)
from geoprobe.runtime.relational_gauge_source_replay import (  # noqa: E402
    build_frozen_pre_status_gauge_replay_rows,
)
from geoprobe.runtime.relational_gauge_source_runner import (  # noqa: E402
    REQUIRED_EXECUTION_SOURCE_NAMES,
    join_gauge_source_root_works,
    run_exact_prefix_gauge_source_replay,
)
from geoprobe.runtime.relational_pre_status_causal_runner import (  # noqa: E402
    load_causal_replay_inputs,
)


FROZEN_ROOT_COUNT = 402
FROZEN_EVENT_COUNT = 656
FROZEN_VIEW = "intervention_masked_action_free"
EXECUTION_SOURCE_RELATIVE_PATHS = {
    "causal_sampler": "src/geoprobe/models/relational_causal_replay.py",
    "cli": "experiments/run_relational_gauge_source.py",
    "controller_artifact": "src/geoprobe/eval/relational_gauge_controller_artifact.py",
    "gauge_controller": "src/geoprobe/control/relational_gauge_controller.py",
    "gauge_replay": "src/geoprobe/models/relational_gauge_replay.py",
    "hf_capture": "src/geoprobe/models/hf_capture.py",
    "hf_interface": "src/geoprobe/models/interface.py",
    "horizontal_lift": "src/geoprobe/control/relational_horizontal_lift.py",
    "intrinsic_risk_field": "src/geoprobe/control/relational_intrinsic_risk_field.py",
    "replay_observation": "src/geoprobe/runtime/relational_gauge_replay_observation.py",
    "runner": "src/geoprobe/runtime/relational_gauge_source_runner.py",
    "source_replay": "src/geoprobe/runtime/relational_gauge_source_replay.py",
    "step_capture": "src/geoprobe/models/relational_step_capture.py",
    "structured_action_sampler": (
        "src/geoprobe/models/relational_structured_action_rollout.py"
    ),
}


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument("--vector-bank-root", type=Path, required=True)
    parser.add_argument("--source-rollout-manifest", type=Path, required=True)
    parser.add_argument("--rooted-star-bank", type=Path, required=True)
    parser.add_argument("--rooted-graph-artifact-root", type=Path, required=True)
    parser.add_argument("--controller-artifact-root", type=Path, required=True)
    parser.add_argument("--expected-replay-plan-ledger-sha256", required=True)
    parser.add_argument("--expected-rooted-star-manifest-sha256", required=True)
    parser.add_argument("--expected-rooted-graph-manifest-sha256", required=True)
    parser.add_argument("--expected-controller-manifest-sha256", required=True)
    parser.add_argument("--expected-controller-internal-sha256", required=True)
    parser.add_argument("--expected-root-count", type=int, default=FROZEN_ROOT_COUNT)
    parser.add_argument("--expected-event-count", type=int, default=FROZEN_EVENT_COUNT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-artifact-identity", type=Path)
    parser.add_argument("--expected-model-artifact-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--status-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("controller manifest is not UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("controller manifest must be an object")
    return value


def _execution_source_paths() -> dict[str, Path]:
    return {
        name: (REPO_ROOT / relative).resolve()
        for name, relative in EXECUTION_SOURCE_RELATIVE_PATHS.items()
    }


def _execution_source_sha256() -> dict[str, str]:
    paths = _execution_source_paths()
    if (
        set(paths) != REQUIRED_EXECUTION_SOURCE_NAMES
        or any(not path.is_file() for path in paths.values())
    ):
        raise ValueError("gauge execution source inventory is incomplete")
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _validated_fold_artifact_root(
    controller_root: Path,
    controller_manifest: Mapping[str, Any],
    fold: str,
) -> Path:
    folds = controller_manifest.get("fold_artifacts")
    row = folds.get(fold) if isinstance(folds, Mapping) else None
    if not isinstance(row, Mapping) or set(row) != {
        "path",
        "sha256",
        "tensor_path",
        "tensor_sha256",
    }:
        raise ValueError(f"controller manifest fold binding is invalid: {fold}")
    expected_bundle = f"{fold}/bundle.json"
    expected_tensor = f"{fold}/controller.safetensors"
    if row.get("path") != expected_bundle or row.get("tensor_path") != expected_tensor:
        raise ValueError(f"controller manifest fold paths changed: {fold}")
    bundle_sha = _sha256(row.get("sha256"), name=f"{fold} bundle SHA-256")
    tensor_sha = _sha256(
        row.get("tensor_sha256"), name=f"{fold} tensor SHA-256"
    )
    root = controller_root.resolve()
    bundle_path = root / expected_bundle
    tensor_path = root / expected_tensor
    if (
        not bundle_path.is_file()
        or not tensor_path.is_file()
        or file_sha256(bundle_path) != bundle_sha
        or file_sha256(tensor_path) != tensor_sha
    ):
        raise ValueError(f"controller fold files differ from top-level manifest: {fold}")
    return root / fold


def _validate_static_inputs(
    args: argparse.Namespace,
    *,
    validated_model_identity: Mapping[str, Any] | None,
) -> tuple[
    Mapping[str, Any],
    tuple[Any, ...],
    Mapping[str, Any],
    Any,
    Any,
    Mapping[str, tuple[Any, ...]],
    tuple[Any, ...],
    Mapping[str, Any],
    str,
]:
    ledger, causal_works, input_hashes = load_causal_replay_inputs(
        args.replay_plan, args.vector_bank_root
    )
    expected_plan = _sha256(
        args.expected_replay_plan_ledger_sha256,
        name="expected replay-plan ledger SHA-256",
    )
    if ledger.get("ledger_sha256") != expected_plan:
        raise ValueError("replay plan differs from its expected ledger SHA-256")
    source_rollout = causal_cli._validate_source_rollout_manifest(
        ledger.get("rollout_binding"),
        args.source_rollout_manifest,
        model_identity=validated_model_identity,
    )

    index = build_relational_pre_status_rooted_star_index(args.rooted_star_bank)
    expected_stars = _sha256(
        args.expected_rooted_star_manifest_sha256,
        name="expected rooted-star manifest SHA-256",
    )
    if index.manifest_sha256 != expected_stars:
        raise ValueError("rooted-star bank differs from its expected manifest")

    graph_root = args.rooted_graph_artifact_root.resolve()
    graph_manifest_path = graph_root / "manifest.json"
    expected_graph = _sha256(
        args.expected_rooted_graph_manifest_sha256,
        name="expected rooted-graph manifest SHA-256",
    )
    if file_sha256(graph_manifest_path) != expected_graph:
        raise ValueError("rooted-graph artifact differs from its expected manifest")
    graphs = load_relational_pre_status_rooted_graph_artifacts(graph_root)

    controller_root = args.controller_artifact_root.resolve()
    controller_manifest_path = controller_root / "manifest.json"
    controller_physical = file_sha256(controller_manifest_path)
    expected_controller = _sha256(
        args.expected_controller_manifest_sha256,
        name="expected controller manifest SHA-256",
    )
    if controller_physical != expected_controller:
        raise ValueError("gauge-controller artifact differs from its expected manifest")
    controller_manifest = _read_manifest(controller_manifest_path)
    validate_relational_gauge_controller_manifest(controller_manifest)
    expected_internal = _sha256(
        args.expected_controller_internal_sha256,
        name="expected controller internal SHA-256",
    )
    if controller_manifest.get("manifest_sha256") != expected_internal:
        raise ValueError("gauge-controller internal manifest hash changed")
    binding = controller_manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("gauge-controller input binding is absent")
    star_binding = binding.get("rooted_star_manifest")
    graph_binding = binding.get("rooted_graph_manifest")
    if (
        not isinstance(star_binding, Mapping)
        or star_binding.get("sha256") != expected_stars
        or not isinstance(graph_binding, Mapping)
        or graph_binding.get("sha256") != expected_graph
    ):
        raise ValueError("gauge-controller input bindings differ from supplied geometry")

    replay_rows_by_fold = build_frozen_pre_status_gauge_replay_rows(
        index, causal_works, view=FROZEN_VIEW
    )
    joined = join_gauge_source_root_works(causal_works, replay_rows_by_fold)
    if (
        len(joined) != args.expected_root_count
        or sum(len(work.causal.events) for work in joined)
        != args.expected_event_count
    ):
        raise ValueError("joined gauge replay counts differ from CLI assertions")
    resolved_hashes = {
        **dict(input_hashes),
        "source_rollout_manifest": source_rollout,
        "rooted_star_manifest": expected_stars,
        "rooted_graph_manifest": expected_graph,
        "controller_manifest": {
            "physical_sha256": controller_physical,
            "internal_sha256": expected_internal,
        },
    }
    return (
        ledger,
        causal_works,
        resolved_hashes,
        index,
        graphs,
        replay_rows_by_fold,
        joined,
        controller_manifest,
        controller_physical,
    )


def _resource_loader(
    *,
    graphs: Any,
    replay_rows_by_fold: Mapping[str, tuple[Any, ...]],
    controller_root: Path,
    controller_manifest: Mapping[str, Any],
    controller_artifact_sha256: str,
):
    def load(fold: str):
        graph = graphs.fold_graph(FROZEN_VIEW, fold, "joint")
        fold_root = _validated_fold_artifact_root(
            controller_root, controller_manifest, fold
        )
        bundle = load_fold_gauge_controller_artifact(
            fold_root,
            graph,
        )
        builder = CaptureBoundPreStatusGaugeObservationBuilder(
            bundle=bundle,
            rows=replay_rows_by_fold[fold],
            controller_artifact_sha256=controller_artifact_sha256,
        )
        return builder, build_source_gauge_controllers(bundle)

    return load


def _validate_controller_inventory(
    joined: Sequence[Any],
    *,
    load_resources: Any,
) -> Mapping[str, int]:
    counts: Counter[str] = Counter()
    by_fold: dict[str, list[Any]] = {}
    for work in joined:
        by_fold.setdefault(work.causal.family_fold, []).append(work)
    for fold, works in sorted(by_fold.items()):
        builder, controllers = load_resources(fold)
        if tuple(controllers) != SOURCE_GAUGE_ARM_ORDER:
            raise ValueError("source gauge controller arm order changed")
        for work in works:
            row = work.replay
            builder.validate_backend_row(row.row_id, row.expected_token_ids)
            query = builder.bundle.held_out_queries[row.root_id]
            observation = GaugeControllerObservation(
                observation_id=f"offline-plan:{row.root_id}",
                chart=builder.bundle.atlas.get_chart(query.chart_id),
                query=query,
                nuisance_key=row.nuisance_key,
                randomization_key=f"sealed-root:{row.root_id}",
            )
            proposal = controllers["gauge_geodesic"].propose(
                observation,
                controllers["gauge_geodesic"].initial_state(),
            )
            counts[proposal.status] += 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    if (
        args.expected_root_count != FROZEN_ROOT_COUNT
        or args.expected_event_count != FROZEN_EVENT_COUNT
    ):
        raise ValueError("root/event counts differ from the frozen source inventory")
    validated_model_identity = None
    if not args.validate_only and args.model_artifact_identity is not None:
        validated_model_identity = causal_cli._read_identity(
            args.model_artifact_identity
        )
    (
        ledger,
        _causal_works,
        input_hashes,
        _index,
        graphs,
        replay_rows_by_fold,
        joined,
        controller_manifest,
        controller_physical,
    ) = _validate_static_inputs(
        args,
        validated_model_identity=validated_model_identity,
    )
    load_resources = _resource_loader(
        graphs=graphs,
        replay_rows_by_fold=replay_rows_by_fold,
        controller_root=args.controller_artifact_root.resolve(),
        controller_manifest=controller_manifest,
        controller_artifact_sha256=controller_physical,
    )
    if args.validate_only:
        if any(value is not None for value in (args.out, args.manifest_out, args.status_out)):
            raise ValueError("--validate-only creates no output")
        status_counts = _validate_controller_inventory(
            joined, load_resources=load_resources
        )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "root_count": len(joined),
                    "event_count": sum(len(work.causal.events) for work in joined),
                    "expected_row_count": args.expected_event_count
                    * len(SOURCE_GAUGE_ARM_ORDER),
                    "proposal_status_counts": status_counts,
                    "replay_plan_ledger_sha256": ledger["ledger_sha256"],
                    "controller_manifest_sha256": controller_physical,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.out is None:
        raise ValueError("a live gauge replay requires --out")
    manifest_out = args.manifest_out or args.out.with_suffix(
        args.out.suffix + ".manifest.json"
    )
    status_out = args.status_out or args.out.parent / "STATUS"
    causal_cli._reject_runtime_overrides()
    model_provenance = causal_cli._model_provenance(
        args, model_identity=validated_model_identity
    )
    model, tokenizer, loaded_meta = load_hf_model(
        str(args.model), device="cuda", dtype=torch.bfloat16
    )
    set_eager_attention(model)
    effective_runtime = causal_cli._assert_effective_cuda_bf16(model)
    device_index = int(effective_runtime["parameter_cuda_device_indices"][0])
    causal_cli._validate_status_tokenizer(tokenizer)
    execution_sources = _execution_source_sha256()
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "loader_meta": loaded_meta,
        "effective_model_runtime": effective_runtime,
        "cuda_runtime": causal_cli._cuda_runtime_identity(device_index),
        "attention_implementation": model.config._attn_implementation,
        "capture_hooks_enabled": True,
        "git": git_provenance(list(_execution_source_paths().values())),
    }
    manifest = run_exact_prefix_gauge_source_replay(
        joined,
        model=model,
        tokenizer=tokenizer,
        fold_resource_loader=load_resources,
        out=args.out,
        manifest_out=manifest_out,
        status_out=status_out,
        controller_artifact_sha256=controller_physical,
        expected_root_count=args.expected_root_count,
        expected_event_count=args.expected_event_count,
        input_hashes=input_hashes,
        model_provenance=model_provenance,
        runtime_provenance=runtime,
        execution_source_sha256=execution_sources,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "out": str(args.out.resolve()),
                "completed_rows": manifest["completed_rows"],
                "expected_row_count": manifest["expected_row_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
