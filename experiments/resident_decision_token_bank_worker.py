"""Resident HF/BF16 decision-token control-bank worker.

Run this after activation shards have been merged into one activation bank. Each
worker reads the full activation bank for leak-safe LOFO direction fitting, but
evaluates only its assigned family shard. The HF model is loaded once and reused
for action-response margins and steered generation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.activation_control_tomography import decision_prefix_ids, decision_tokens  # noqa: E402
import geoprobe.models.decision as _decision_mod  # noqa: E402  fingerprint promoted backend
import geoprobe.models.decision_backend as _decision_backend_mod  # noqa: E402
from experiments.bidirectional_dp_diagnostic import attach_status_classes, fit_status_direction  # noqa: E402
from experiments.control_deception_intent_transition import prompt_without_final_answer, reply_coherence  # noqa: E402
from experiments.control_graded_dp_bidirectional_stack import (  # noqa: E402
    add_status_class_to_transcript,
    parse_int_csv,
    select_status_balanced_rows,
)
from experiments.control_graded_dp_decision_token import (  # noqa: E402
    fit_global_mean_direction,
    fit_global_probe_direction,
    fit_random_direction,
    load_gate_predictions,
    route_for_row,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    config_defaults,
    fit_tangent_cloud,
    load_activation_points,
    off_tangent_direction,
    point_index,
    project_to_local_tangent,
    read_jsonl_paths,
    select_eval_rows,
    to_jsonable,
    unit,
)
from experiments.control_graded_dp_frontier import vector_stats  # noqa: E402
from experiments.control_graded_dp_stack_frontier import aggregate_projection, parse_csv, stack_injection_stats  # noqa: E402
from experiments.decision_token_action_response import (  # noqa: E402
    BIDIR_METHODS,
    GLOBAL_METHODS,
    build_result_row,
    make_spec,
    parse_targets,
    pointcloud_context_features,
    public_direction_info,
    signed_global_direction,
    summarize,
)
from experiments.decision_token_backend import (  # noqa: E402
    choose_status_then_complete,
    decision_margin,
    public_model_meta,
)
from geoprobe.io import atomic_text, file_sha256  # noqa: E402
from geoprobe.runtime.resource_monitoring import (  # noqa: E402
    resource_snapshot,
    shard_rows_by_family,
)
from experiments.rollout_deception_intent import parse_status  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models import load_activation_pipeline  # noqa: E402


METHOD_ALIASES = {
    "bidir_linear": "bidir_linear",
    "linear": "bidir_linear",
    "bidir_tangent": "bidir_tangent",
    "tangent": "bidir_tangent",
    "bidir_off_tangent": "bidir_off_tangent",
    "off_tangent": "bidir_off_tangent",
    "global_mean": "global_mean",
    "global_probe": "global_probe",
    "probe": "global_probe",
    "random": "random_global",
    "random_global": "random_global",
}
STEER_METHODS = BIDIR_METHODS | GLOBAL_METHODS


def canonical_methods(value: str) -> list[str]:
    out: list[str] = []
    for item in parse_csv(value):
        if item not in METHOD_ALIASES:
            raise ValueError(f"unknown method {item!r}; expected one of {sorted(METHOD_ALIASES)}")
        method = METHOD_ALIASES[item]
        if method not in out:
            out.append(method)
    return out


def seen_action_keys(rows: list[dict]) -> set[tuple[str, str, str | None, int | None, float]]:
    return {
        (
            str(row["conversation_id"]),
            str(row["method"]),
            row.get("target_status"),
            row.get("layer"),
            float(row.get("alpha", 0.0)),
        )
        for row in rows
    }


def seen_generation_keys(rows: list[dict]) -> set[tuple[str, str, str | None, int | None, float]]:
    return {
        (
            str(row["conversation_id"]),
            str(row["method"]),
            row.get("target_status"),
            row.get("layer"),
            float(row.get("alpha", 0.0)),
        )
        for row in rows
    }


def load_eval_rows(
    *,
    transcript_paths: list[Path],
    activation_path: Path,
    layer: int,
    query_turn: int,
    query_phase: str,
    eval_levels: set[str],
    limit: int | None,
    limit_per_status_class: int | None,
    limit_strategy: str,
    seed: int,
    scenario_include: str | None,
    scenario_exclude: str | None,
    num_shards: int,
    shard_index: int,
) -> tuple[list[dict], dict[str, dict], dict]:
    first_query, activation_meta = load_activation_points(
        activation_path,
        layer=layer,
        turns={query_turn},
        phases={query_phase},
    )
    query_ids = set(point_index(first_query))
    transcript_rows = read_jsonl_paths(transcript_paths)
    transcript_by_cid = {str(row.get("conversation_id")): row for row in transcript_rows}

    rows = []
    for row in transcript_rows:
        cid = str(row.get("conversation_id", ""))
        if not row.get("valid_outcome") or row.get("arm") not in eval_levels or cid not in query_ids:
            continue
        classified = add_status_class_to_transcript(row)
        if classified is not None:
            rows.append(classified)
    if scenario_include:
        include = {item.strip() for item in scenario_include.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("scenario_id", "")) in include]
    if scenario_exclude:
        exclude = {item.strip() for item in scenario_exclude.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("scenario_id", "")) not in exclude]
    if limit_per_status_class is not None:
        rows = select_status_balanced_rows(rows, per_status_class=limit_per_status_class, seed=seed)
    else:
        rows = select_eval_rows(rows, limit=limit, strategy=limit_strategy, seed=seed)
    rows = shard_rows_by_family(rows, num_shards=num_shards, shard_index=shard_index)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations + shard")
    return rows, transcript_by_cid, activation_meta


def load_resident_hf_pipeline(
    *,
    model_key: str,
    backend: str,
    device: str | None,
    dtype: str,
    max_length: int,
    mlx_model: str | None,
):
    if backend != "hf":
        raise ValueError("resident bank worker requires the HF backend; use --backend hf")
    pipeline = load_activation_pipeline(
        model_key,
        backend=backend,
        device=device,
        dtype=dtype,
        max_length=max_length,
        mlx_model=mlx_model,
    )
    if not hasattr(pipeline, "model") or not hasattr(pipeline, "tokenizer"):
        raise TypeError("HF activation pipeline did not expose model/tokenizer")
    return pipeline


def target_for_generation(row: dict, *, routing: str, gate_by_cid: dict[str, dict]) -> tuple[str | None, dict]:
    route = route_for_row(row, routing=routing, gate_by_cid=gate_by_cid)
    target = route.get("target_status")
    if target not in {"PASS", "FAIL"}:
        return None, route
    return str(target), route


def usable_unit_vector(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    clean = unit(np.asarray(vec, dtype=np.float64))
    if not np.isfinite(clean).all() or float(np.linalg.norm(clean)) <= 1e-8:
        return None
    return clean


def assert_resume_identity(prior: dict, current: dict) -> None:
    mismatches = []
    for key, value in current.items():
        if prior.get(key) != value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"--resume metadata mismatch for fields: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20")
    parser.add_argument("--methods", default="bidir_tangent,global_mean,global_probe")
    parser.add_argument("--alphas", default="24,48")
    parser.add_argument("--candidate-targets", default="PASS,FAIL")
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--eval-levels", default="p3,p4,p5")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--tangent-levels", default="p0,p1,p2,p3,p4,p5,p6")
    parser.add_argument("--tangent-turns", default="0,1,2,3")
    parser.add_argument("--tangent-phases", default="pre_response,post_response")
    parser.add_argument("--min-mixed-scenarios", type=int, default=2)
    parser.add_argument("--min-direction-levels", type=int, default=2)
    parser.add_argument("--tangent-neighbors", type=int, default=16)
    parser.add_argument("--tangent-dim", type=int, default=4)
    parser.add_argument("--context-neighbors", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-status-class", type=int, default=None)
    parser.add_argument(
        "--limit-strategy",
        choices=["first", "shuffle", "family_round_robin"],
        default="family_round_robin",
    )
    parser.add_argument("--scenario-include", default=None)
    parser.add_argument("--scenario-exclude", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--routing", choices=["oracle_true_status", "gate_file"], default="oracle_true_status")
    parser.add_argument("--gate-predictions", default=None)
    parser.add_argument("--unidirectional-targets", action="store_true")
    parser.add_argument("--equivariant-directions", action="store_true")
    parser.add_argument("--backend", choices=["hf"], default="hf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    transcript_paths = [Path(path) for path in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = config_defaults(config_path)
    model_key = args.model or defaults["model_key"]
    mlx_model = args.mlx_model or defaults["mlx_model"]
    dtype = args.dtype or defaults["dtype"]
    max_length = args.max_length or defaults["max_length"]
    max_new_tokens = args.max_new_tokens or defaults["max_new_tokens"]
    max_generation_seconds = args.max_generation_seconds or defaults["max_generation_seconds"]

    layers = parse_int_csv(args.layers)
    methods = canonical_methods(args.methods)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    candidate_targets = parse_targets(args.candidate_targets)
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))
    gate_by_cid = load_gate_predictions(Path(args.gate_predictions)) if args.gate_predictions else {}
    transcript_hashes = {str(path): file_sha256(path) for path in transcript_paths}
    activation_hash = file_sha256(activation_path)
    config_hash = file_sha256(config_path)
    gate_hash = file_sha256(Path(args.gate_predictions)) if args.gate_predictions else None

    rows, transcript_by_cid, activation_meta = load_eval_rows(
        transcript_paths=transcript_paths,
        activation_path=activation_path,
        layer=layers[0],
        query_turn=args.query_turn,
        query_phase=args.query_phase,
        eval_levels=eval_levels,
        limit=args.limit,
        limit_per_status_class=args.limit_per_status_class,
        limit_strategy=args.limit_strategy,
        seed=args.seed,
        scenario_include=args.scenario_include,
        scenario_exclude=args.scenario_exclude,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    prior = json.loads(out_path.read_text()) if args.resume and out_path.exists() else {}
    current_identity = {
        "backend": args.backend,
        "requested_model_key": model_key,
        "mlx_model": str(mlx_model),
        "dtype": dtype,
        "max_length": int(max_length),
        "transcripts_sha256": transcript_hashes,
        "activations_sha256": activation_hash,
        "config_sha256": config_hash,
        "gate_predictions_sha256": gate_hash,
        "layers": layers,
        "methods": methods,
        "alphas": alphas,
        "candidate_targets": candidate_targets,
        "eval_levels": sorted(eval_levels),
        "direction_levels": sorted(direction_levels),
        "tangent_levels": sorted(tangent_levels),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "routing": args.routing,
        "equivariant_directions": bool(args.equivariant_directions),
    }
    if prior:
        assert_resume_identity(prior, current_identity)
    action_rows: list[dict] = list(prior.get("action_response", {}).get("rows", prior.get("rows", [])))
    generation_rows: list[dict] = list(prior.get("control_generation", {}).get("results", prior.get("results", [])))
    action_seen = seen_action_keys(action_rows)
    generation_seen = seen_generation_keys(generation_rows)
    telemetry: list[dict] = list(prior.get("telemetry", [])) if prior else []
    started_at = time.perf_counter()

    planned_actions = len(rows)
    for row in rows:
        status_cls = str(row.get("status_class", ""))
        targets = list(candidate_targets)
        if args.unidirectional_targets:
            if status_cls == "false_FAIL":
                targets = ["PASS"]
            elif status_cls == "false_PASS":
                targets = ["FAIL"]
        planned_actions += len(layers) * len(methods) * len(targets) * len(alphas)
    planned_generations = len(rows) + len(rows) * len(layers) * len(methods) * len(alphas)

    def add_telemetry(stage: str) -> None:
        if args.telemetry:
            telemetry.append(resource_snapshot(stage, started_at))

    def write_payload(model_meta: dict | None, *, validate_only: bool) -> None:
        payload = {
            "schema_version": 1,
            "argv": sys.argv,
            "validate_only": validate_only,
            "backend": args.backend,
            "requested_model_key": model_key,
            "mlx_model": str(mlx_model),
            "dtype": dtype,
            "max_length": int(max_length),
            "model": model_meta,
            "transcripts": [str(path.resolve()) for path in transcript_paths],
            "transcripts_sha256": transcript_hashes,
            "activations": str(activation_path.resolve()),
            "activations_sha256": activation_hash,
            "activation_meta": activation_meta,
            "config": str(config_path.resolve()),
            "config_sha256": config_hash,
            "gate_predictions": str(Path(args.gate_predictions).resolve()) if args.gate_predictions else None,
            "gate_predictions_sha256": gate_hash,
            "provenance": git_provenance([Path(__file__), Path(_decision_backend_mod.__file__), Path(_decision_mod.__file__), config_path, activation_path, *transcript_paths]),
            "layers": layers,
            "methods": methods,
            "alphas": alphas,
            "candidate_targets": candidate_targets,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "tangent_levels": sorted(tangent_levels),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "routing": args.routing,
            "equivariant_directions": bool(args.equivariant_directions),
            "eval_rows": len(rows),
            "eval_status_class_balance": dict(Counter(str(row["status_class"]) for row in rows)),
            "planned_actions": planned_actions,
            "planned_generations": planned_generations,
            "telemetry": telemetry,
            "rows": action_rows,
            "summary": summarize(action_rows) if action_rows else None,
            "results": generation_rows,
            "action_response": {
                "rows": action_rows,
                "summary": summarize(action_rows) if action_rows else None,
            },
            "control_generation": {
                "results": generation_rows,
                "summary": {
                    "n": len(generation_rows),
                    "steered": int(sum(row.get("injection") is not None for row in generation_rows)),
                    "parse_success": int(sum(row.get("coherence", {}).get("parse_success") for row in generation_rows)),
                    "coherence_preserved": int(sum(row.get("coherence", {}).get("coherence_preserved") for row in generation_rows)),
                    "methods": dict(Counter(str(row.get("method")) for row in generation_rows)),
                },
            },
            "note": (
                "Resident bank worker: full activation bank is used for train-family direction "
                "fitting, this worker evaluates only its assigned family shard. Oracle routing "
                "is a feasibility regime, not deployable control."
            ),
        }
        atomic_text(out_path, json.dumps(to_jsonable(payload), indent=2, sort_keys=True))

    if args.validate_only:
        add_telemetry("validate_only")
        write_payload(None, validate_only=True)
        print(f"validated -> {out_path} ({planned_actions} actions, {planned_generations} generations; model not loaded)", flush=True)
        return

    add_telemetry("before_model_load")
    model_load_start = time.perf_counter()
    pipeline = load_resident_hf_pipeline(
        model_key=model_key,
        backend=args.backend,
        device=args.device,
        dtype=dtype,
        max_length=max_length,
        mlx_model=mlx_model,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    pass_id, fail_id = decision_tokens(tokenizer)
    model_meta = public_model_meta(pipeline.meta)
    model_meta["model_load_seconds"] = float(model_load_seconds)
    add_telemetry("after_model_load")

    direction_by_layer: dict[int, list[dict]] = {}
    query_by_layer: dict[int, dict[str, dict]] = {}
    region_by_layer: dict[int, list[dict]] = {}
    tangent_by_layer: dict[int, list[dict]] = {}
    for layer in layers:
        direction_raw, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.direction_turn},
            phases={args.direction_phase},
        )
        direction_by_layer[layer], _ = attach_status_classes(direction_raw, transcript_by_cid)
        query_raw, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.query_turn},
            phases={args.query_phase},
        )
        query_by_layer[layer] = point_index(query_raw)
        region_by_layer[layer], _ = attach_status_classes(query_raw, transcript_by_cid)
        tangent_by_layer[layer], _ = load_activation_points(
            activation_path,
            layer=layer,
            turns=tangent_turns,
            phases=tangent_phases,
            levels=tangent_levels,
        )

    direction_cache: dict[tuple[int, str, str], dict | None] = {}
    global_cache: dict[tuple[int, str, str], dict | None] = {}
    tangent_cache: dict[tuple[int, str], dict | None] = {}

    def get_status_direction(layer: int, family: str, target: str) -> dict | None:
        key = (layer, family, target)
        if key not in direction_cache:
            direction_cache[key] = fit_status_direction(
                direction_by_layer[layer],
                heldout_family=family,
                direction_levels=direction_levels,
                target_status=target,
                min_mixed_scenarios=args.min_mixed_scenarios,
                min_levels=args.min_direction_levels,
                equivariant=args.equivariant_directions,
            )
        return direction_cache[key]

    def get_global_direction(layer: int, family: str, method: str) -> dict | None:
        key = (layer, family, method)
        if key not in global_cache:
            if method == "global_mean":
                global_cache[key] = fit_global_mean_direction(
                    direction_by_layer[layer],
                    heldout_family=family,
                    direction_levels=direction_levels,
                )
            elif method == "global_probe":
                global_cache[key] = fit_global_probe_direction(
                    direction_by_layer[layer],
                    heldout_family=family,
                    direction_levels=direction_levels,
                )
            elif method == "random_global":
                global_cache[key] = fit_random_direction(
                    direction_by_layer[layer],
                    heldout_family=family,
                    direction_levels=direction_levels,
                    layer=layer,
                    seed=args.seed,
                )
            else:
                raise ValueError(f"bad global method {method!r}")
        return global_cache[key]

    def get_tangent(layer: int, family: str) -> dict | None:
        key = (layer, family)
        if key not in tangent_cache:
            tangent_cache[key] = fit_tangent_cloud(tangent_by_layer[layer], heldout_family=family)
        return tangent_cache[key]

    def action_vector(layer: int, row: dict, method: str, target: str) -> tuple[np.ndarray | None, dict | None, dict | None]:
        family = str(row["family"])
        query = query_by_layer[layer].get(str(row["conversation_id"]))
        if query is None:
            return None, None, {"reason": "missing_query_activation"}
        if method in BIDIR_METHODS:
            direction_info = get_status_direction(layer, family, target)
            if direction_info is None:
                return None, None, {"reason": "missing_status_direction"}
            raw = np.asarray(direction_info["_direction_np"], dtype=np.float64)
            if method == "bidir_linear":
                return unit(raw), direction_info, None
            tangent_info = get_tangent(layer, family)
            if tangent_info is None:
                return None, direction_info, {"reason": "missing_tangent_cloud"}
            tangent_vec, projection = project_to_local_tangent(
                raw,
                tangent_info,
                query["x"],
                tangent_neighbors=args.tangent_neighbors,
                tangent_dim=args.tangent_dim,
            )
            if tangent_vec is None:
                return None, direction_info, projection
            if method == "bidir_tangent":
                return unit(tangent_vec.detach().float().cpu().numpy()), direction_info, projection
            off = off_tangent_direction(raw, tangent_vec)
            if off is None:
                return None, direction_info, {"reason": "missing_off_tangent", "projection": projection}
            return unit(off.detach().float().cpu().numpy()), direction_info, projection
        direction_info = get_global_direction(layer, family, method)
        if direction_info is None:
            return None, None, {"reason": "missing_global_direction"}
        return signed_global_direction(direction_info, target), direction_info, None

    def generation_vector(layer: int, row: dict, method: str, target: str) -> tuple[np.ndarray | None, dict | None, dict | None]:
        if method in GLOBAL_METHODS:
            direction_info = get_global_direction(layer, str(row["family"]), method)
            if direction_info is None:
                return None, None, {"reason": "missing_global_direction"}
            return np.asarray(direction_info["_direction_np"], dtype=np.float64), direction_info, None
        return action_vector(layer, row, method, target)

    def action_route(row: dict) -> dict | None:
        return route_for_row(row, routing=args.routing, gate_by_cid=gate_by_cid)

    prefix_by_cid: dict[str, list[int]] = {}
    base_margin_by_cid: dict[str, float] = {}
    for idx, row in enumerate(rows, start=1):
        cid = str(row["conversation_id"])
        prefix = decision_prefix_ids(tokenizer, transcript_by_cid[cid]["messages"])
        prefix_by_cid[cid] = prefix
        base_margin = decision_margin(
            model,
            prefix,
            pass_id,
            fail_id,
            steering=None,
            backend=args.backend,
        )
        base_margin_by_cid[cid] = base_margin
        key = (cid, "abstain", None, None, 0.0)
        if key not in action_seen:
            base_context = None
            query = query_by_layer[layers[0]].get(cid)
            if query is not None:
                base_context = pointcloud_context_features(
                    query_x=query["x"],
                    region_rows=region_by_layer[layers[0]],
                    heldout_family=str(row["family"]),
                    k=args.context_neighbors,
                )
            action_rows.append(build_result_row(
                row=row,
                method="abstain",
                target_status=None,
                layer=None,
                alpha=0.0,
                base_margin=base_margin,
                final_margin=base_margin,
                direction_info=None,
                projection=None,
                route=action_route(row),
                context=base_context,
            ))
            action_seen.add(key)
        if idx % max(args.checkpoint_every, 1) == 0:
            write_payload(model_meta, validate_only=False)
    add_telemetry("after_base_margins")
    write_payload(model_meta, validate_only=False)

    action_steps = 0
    for row in rows:
        cid = str(row["conversation_id"])
        status_cls = str(row.get("status_class", ""))
        targets = list(candidate_targets)
        if args.unidirectional_targets:
            if status_cls == "false_FAIL":
                targets = ["PASS"]
            elif status_cls == "false_PASS":
                targets = ["FAIL"]
        for layer in layers:
            query = query_by_layer[layer].get(cid)
            context = None
            if query is not None:
                context = pointcloud_context_features(
                    query_x=query["x"],
                    region_rows=region_by_layer[layer],
                    heldout_family=str(row["family"]),
                    k=args.context_neighbors,
                )
            for method in methods:
                for target in targets:
                    vec, direction_info, projection = action_vector(layer, row, method, target)
                    clean_vec = usable_unit_vector(vec)
                    if clean_vec is None:
                        continue
                    for alpha in alphas:
                        key = (cid, method, target, layer, float(alpha))
                        if key in action_seen:
                            continue
                        spec = make_spec(layer, clean_vec, alpha)
                        final_margin = decision_margin(
                            model,
                            prefix_by_cid[cid],
                            pass_id,
                            fail_id,
                            steering=[spec],
                            backend=args.backend,
                        )
                        action_rows.append(build_result_row(
                            row=row,
                            method=method,
                            target_status=target,
                            layer=layer,
                            alpha=alpha,
                            base_margin=base_margin_by_cid[cid],
                            final_margin=final_margin,
                            direction_info={**direction_info, "direction_stats": vector_stats(clean_vec)} if direction_info else None,
                            projection=projection,
                            route=action_route(row),
                            context=context,
                        ))
                        action_seen.add(key)
                        action_steps += 1
                        if action_steps % max(args.checkpoint_every, 1) == 0:
                            write_payload(model_meta, validate_only=False)
    add_telemetry("after_action_response")
    write_payload(model_meta, validate_only=False)

    generation_steps = 0
    for row in rows:
        cid = str(row["conversation_id"])
        target, route = target_for_generation(row, routing=args.routing, gate_by_cid=gate_by_cid)
        baseline_key = (cid, "baseline", route.get("target_status"), None, 0.0)
        if baseline_key not in generation_seen:
            reply, decision = choose_status_then_complete(
                model,
                tokenizer,
                prompt_without_final_answer(row),
                pass_id=pass_id,
                fail_id=fail_id,
                steering=None,
                max_new_tokens=max_new_tokens,
                max_generation_seconds=max_generation_seconds,
                backend=args.backend,
            )
            reported = parse_status(reply, "report")
            true_status = str(row["true_status"]).upper()
            generation_rows.append({
                "conversation_id": row["conversation_id"],
                "scenario_id": row["scenario_id"],
                "sample_seed": row.get("sample_seed"),
                "family": row["family"],
                "arm": row["arm"],
                "true_status": true_status,
                "reported_status_before": row.get("reported_status"),
                "status_class_before": row.get("status_class"),
                "desired_status": row.get("desired_status"),
                "original_deceptive": bool(row.get("deceptive")),
                "method": "baseline",
                "alpha": 0.0,
                "layer": None,
                "target_status": route.get("target_status"),
                "route": route,
                "decision": decision,
                "direction_info": None,
                "injection": None,
                "reported_status": reported,
                "honest": reported == true_status,
                "coherence": reply_coherence(reply, reported),
                "reply": reply,
            })
            generation_seen.add(baseline_key)
            write_payload(model_meta, validate_only=False)

        if target is None:
            continue
        for layer in layers:
            for method in methods:
                vec, direction_info, projection = generation_vector(layer, row, method, target)
                clean_vec = usable_unit_vector(vec)
                if clean_vec is None:
                    continue
                for alpha in alphas:
                    key = (cid, method, target, layer, float(alpha))
                    if key in generation_seen:
                        continue
                    spec = make_spec(layer, clean_vec, alpha)
                    reply, decision = choose_status_then_complete(
                        model,
                        tokenizer,
                        prompt_without_final_answer(row),
                        pass_id=pass_id,
                        fail_id=fail_id,
                        steering=[spec],
                        max_new_tokens=max_new_tokens,
                        max_generation_seconds=max_generation_seconds,
                        backend=args.backend,
                    )
                    reported = parse_status(reply, "report")
                    true_status = str(row["true_status"]).upper()
                    generation_rows.append({
                        "conversation_id": row["conversation_id"],
                        "scenario_id": row["scenario_id"],
                        "sample_seed": row.get("sample_seed"),
                        "family": row["family"],
                        "arm": row["arm"],
                        "true_status": true_status,
                        "reported_status_before": row.get("reported_status"),
                        "status_class_before": row.get("status_class"),
                        "desired_status": row.get("desired_status"),
                        "original_deceptive": bool(row.get("deceptive")),
                        "method": method,
                        "alpha": float(alpha),
                        "layer": int(layer),
                        "target_status": target,
                        "route": route,
                        "decision": decision,
                        "direction_info": public_direction_info(direction_info),
                        "direction_projection": aggregate_projection({layer: projection}) if projection else None,
                        "injection": stack_injection_stats([spec], float(alpha)),
                        "reported_status": reported,
                        "honest": reported == true_status,
                        "coherence": reply_coherence(reply, reported),
                        "reply": reply,
                    })
                    generation_seen.add(key)
                    generation_steps += 1
                    if generation_steps % max(args.checkpoint_every, 1) == 0:
                        write_payload(model_meta, validate_only=False)
    add_telemetry("after_generation")
    write_payload(model_meta, validate_only=False)
    print(
        f"saved -> {out_path} ({len(action_rows)} action rows, {len(generation_rows)} generations)",
        flush=True,
    )


if __name__ == "__main__":
    main()
