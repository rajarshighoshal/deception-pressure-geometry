"""Export option-3 local control-flow vectors as a steering-spec bank.

The option-3 proxy in ``powered150_local_control_flow.py`` maps a predicted
local flow back to old candidate actions. This exporter emits the predicted
flow direction itself as a normal steering-spec bank so the GPU scorer can
measure the new action directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometric_chart_atlas_selector import load_state_vectors  # noqa: E402
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402  fingerprint promoted load_state_vectors
from experiments.powered150_activation_base_space_scan import stratified_limit  # noqa: E402
from experiments.powered150_local_control_flow import filter_layers, route_target  # noqa: E402
from experiments.precompute_steering_specs import SPEC_SCHEMA_VERSION, peak_rss_gb, shard_stem  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.text.parse import parse_int_csv  # noqa: E402
from geoprobe.control.action_response import (  # noqa: E402
    file_sha256,
    grouped_by_conversation,
    load_action_response,
)
from geoprobe.control.local_control_flow import (  # noqa: E402
    BoundaryDoseConfig,
    CausalDoseConfig,
    CausalGainDoseEstimator,
    LinearBoundaryDoseEstimator,
    LocalControlFlowConfig,
    LocalControlFlowEstimator,
    causal_dose_context_row,
)
from geoprobe.control.dense_dose_response import (  # noqa: E402
    DenseDoseConfig,
    DenseDoseResponseModel,
    curve_from_context_row,
    extract_dose_curves,
)
from geoprobe.control.local_deformation_field import SteeringSpecLookup  # noqa: E402


DEFAULT_ROOT = Path("results/powered150/run_20260629_142532")
DEFAULT_ACTION_RESPONSE = Path("results/spec_scored/powered150_broad_full/run_20260701_161044/action_response.json")
DEFAULT_ACTIVATIONS = DEFAULT_ROOT / "merged/turns.pt"
DEFAULT_BASE_BANK = Path("results/spec_banks/powered150_broad")
DEFAULT_OUT = Path("results/spec_banks/powered150_local_control_flow_l16_boundary")
METHOD_NAME = "local_control_flow"


def vector_stats(vec: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vec, dtype=np.float64)
    return {
        "norm": float(np.linalg.norm(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def representative_row(group: list[dict], *, layer: int, target: str) -> dict:
    for row in group:
        if row.get("layer") is not None and int(row["layer"]) == int(layer) and str(row.get("target_status")) == target:
            return dict(row)
    base = dict(group[0])
    base.update({
        "method": METHOD_NAME,
        "target_status": target,
        "layer": int(layer),
        "alpha": 0.0,
    })
    return base


def dense_dose_context_row(base_row: dict, dense_row: dict | None, causal_row: dict, causal: Any | None) -> dict:
    """Use the dense-grid context shape while refreshing train-derived local-flow features."""
    out = dict(dense_row) if dense_row is not None else dict(base_row)
    for key in (
        "pc_n_train",
        "pc_neighbor_distance_mean",
        "pc_neighbor_distance_min",
        "pc_neighbor_distance_max",
        "pc_predicted_flow_norm",
        "pc_boundary_alpha",
        "pc_boundary_alpha_raw",
        "pc_boundary_directional_slope",
        "pc_boundary_reason",
    ):
        if key in causal_row:
            out[key] = causal_row[key]
    if causal is not None:
        out.update({
            "pc_causal_alpha": causal.alpha,
            "pc_causal_alpha_raw": causal.raw_alpha,
            "pc_causal_predicted_gain": causal.predicted_gain,
            "pc_causal_required_margin_gain": causal.required_margin_gain,
            "pc_causal_reason": causal.reason,
        })
    return out


def identity_from_base(base_manifest: dict, *, layers: list[int], seed: int) -> dict:
    identity = dict(base_manifest["identity"])
    identity.update({
        "spec_schema_version": SPEC_SCHEMA_VERSION,
        "layers": [int(layer) for layer in layers],
        "methods": [METHOD_NAME],
        "candidate_targets": ["PASS", "FAIL"],
        "unidirectional_targets": True,
        "seed": int(seed),
    })
    return identity


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(args.out)
    specs_dir = out_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    action_rows, action_meta = load_action_response(args.action_response)
    grouped = grouped_by_conversation(action_rows)
    records = {
        cid: {
            "conversation_id": cid,
            "scenario_id": group[0].get("scenario_id"),
            "family": group[0].get("family"),
            "status_class": group[0].get("status_class"),
            "route_action": group[0].get("route_action"),
        }
        for cid, group in grouped.items()
    }
    keep_cids = stratified_limit(records, max_cids=args.max_cids, seed=args.seed)
    if keep_cids != set(records):
        grouped = {cid: group for cid, group in grouped.items() if cid in keep_cids}
        action_rows = [row for group in grouped.values() for row in group]
        records = {cid: records[cid] for cid in grouped}

    layers = parse_int_csv(args.layers)
    state_vectors, activation_meta = load_state_vectors(
        args.activations,
        layers=layers,
        query_turn=args.query_turn,
        query_phase=args.query_phase,
    )
    base_manifest = json.loads((args.base_steering_spec_bank / "manifest.json").read_text())
    base_lookup = SteeringSpecLookup(args.base_steering_spec_bank)
    identity = identity_from_base(base_manifest, layers=layers, seed=args.seed)
    config = LocalControlFlowConfig(
        top_k=args.top_k,
        neighbor_pool_multiplier=args.neighbor_pool_multiplier,
        geometry_dim=args.geometry_dim,
        objective="reward",
        target_route_only=True,
        use_local_neighbors=True,
        seed=args.seed,
    )
    boundary_config = BoundaryDoseConfig(
        target_margin=args.boundary_target_margin,
        max_alpha=args.boundary_max_alpha,
        min_directional_slope=args.boundary_min_directional_slope,
        ridge_alpha=args.boundary_ridge_alpha,
    )
    causal_rows: list[dict] = []
    causal_meta: dict[str, Any] | None = None
    if args.causal_dose_action_response is not None:
        causal_rows, causal_meta = load_action_response(args.causal_dose_action_response)
    causal_config = CausalDoseConfig(
        target_margin=args.causal_target_margin,
        max_alpha=args.causal_max_alpha,
        min_gain=args.causal_min_gain,
        ridge_alpha=args.causal_ridge_alpha,
        fixed_alphas=tuple(float(x) for x in args.causal_fixed_alphas),
    )
    dense_rows: list[dict] = []
    dense_meta: dict[str, Any] | None = None
    dense_curves = []
    if args.dense_dose_action_response is not None:
        dense_rows, dense_meta = load_action_response(args.dense_dose_action_response)
        dense_curves = extract_dose_curves(dense_rows, method=METHOD_NAME)
    dense_curve_by_key = {curve.key: curve for curve in dense_curves}
    dense_config = DenseDoseConfig(
        target=args.dense_target,
        prediction_mode=args.dense_prediction_mode,
        target_margin=args.dense_target_margin,
        safety_quantile=args.dense_safety_quantile,
        ridge_alpha=args.dense_ridge_alpha,
        max_alpha=args.dense_max_alpha,
        calibration_folds=args.dense_calibration_folds,
    )

    families = sorted({str(record["family"]) for record in records.values()})
    shards = []
    total_specs = 0
    skipped: list[dict[str, Any]] = []
    for family in families:
        heldout = {family}
        train_rows = filter_layers([row for row in action_rows if str(row["family"]) not in heldout], set(layers))
        estimator = LocalControlFlowEstimator(base_lookup, config).fit(train_rows, state_vectors)
        boundary_estimator = LinearBoundaryDoseEstimator(boundary_config).fit(train_rows, state_vectors)
        causal_estimator = None
        if causal_rows:
            train_causal_rows = [row for row in causal_rows if str(row.get("family")) not in heldout]
            causal_estimator = CausalGainDoseEstimator(causal_config).fit(train_causal_rows)
        dense_estimator = None
        if dense_curves:
            train_dense_curves = [curve for curve in dense_curves if str(curve.family) not in heldout]
            dense_estimator = DenseDoseResponseModel(dense_config).fit(train_dense_curves)
        for layer in layers:
            vectors = []
            meta = []
            for cid, group in sorted(grouped.items()):
                if str(group[0].get("family")) != family:
                    continue
                target = route_target(group[0])
                if target is None:
                    skipped.append({"conversation_id": cid, "family": family, "layer": layer, "reason": "missing_route_target"})
                    continue
                rep = representative_row(group, layer=layer, target=target)
                prediction = estimator.predict(rep, state_vectors, exclude_same_state=False)
                clean = unit(prediction.vector)
                if float(np.linalg.norm(clean)) <= 1e-12:
                    skipped.append({"conversation_id": cid, "family": family, "layer": layer, "reason": "zero_predicted_flow"})
                    continue
                boundary = boundary_estimator.alpha_for(rep, state_vectors, clean)
                causal_feature_row = causal_dose_context_row(rep, prediction, boundary)
                causal = causal_estimator.alpha_for(causal_feature_row) if causal_estimator is not None else None
                dense_curve = dense_curve_by_key.get((str(cid), int(layer), METHOD_NAME, target))
                dense_context = dense_dose_context_row(
                    causal_feature_row,
                    None if dense_curve is None else dense_curve.representative_row(),
                    causal_feature_row,
                    causal,
                )
                dense = (
                    dense_estimator.predict(curve_from_context_row(dense_context, method=METHOD_NAME))
                    if dense_estimator is not None
                    else None
                )
                vectors.append(clean.astype(np.float32))
                meta.append({
                    "conversation_id": str(cid),
                    "scenario_id": group[0].get("scenario_id"),
                    "family": family,
                    "status_class": group[0].get("status_class"),
                    "true_status": group[0].get("true_status"),
                    "layer": int(layer),
                    "method": METHOD_NAME,
                    "target_status": target,
                    "direction_info": {
                        "method": METHOD_NAME,
                        "direction_convention": "unit(predicted_scaled_local_control_flow)",
                        "predicted_flow_norm": prediction.confidence,
                        "support_count": prediction.support_count,
                        "neighbor_state_ids": prediction.neighbor_state_ids[:8],
                        "neighbor_distances": prediction.neighbor_distances[:8],
                        "geometry_dim": int(args.geometry_dim),
                        "top_k": int(args.top_k),
                        "neighbor_pool_multiplier": int(args.neighbor_pool_multiplier),
                        "heldout_family": family,
                        "objective": "reward",
                        "boundary_alpha": boundary.alpha,
                        "boundary_alpha_raw": boundary.raw_alpha,
                        "boundary_current_margin": boundary.current_margin,
                        "boundary_predicted_margin": boundary.predicted_margin,
                        "boundary_directional_slope": boundary.directional_slope,
                        "boundary_signed_current_margin": boundary.signed_current_margin,
                        "boundary_required_margin_gain": boundary.required_margin_gain,
                        "boundary_reason": boundary.reason,
                        "boundary_target_margin": float(boundary_config.target_margin),
                        "boundary_max_alpha": float(boundary_config.max_alpha),
                        "boundary_detector_train_count": boundary_estimator.n_train_,
                        "causal_alpha": None if causal is None else causal.alpha,
                        "causal_alpha_raw": None if causal is None else causal.raw_alpha,
                        "causal_predicted_gain": None if causal is None else causal.predicted_gain,
                        "causal_current_margin": None if causal is None else causal.current_margin,
                        "causal_signed_current_margin": None if causal is None else causal.signed_current_margin,
                        "causal_required_margin_gain": None if causal is None else causal.required_margin_gain,
                        "causal_reason": None if causal is None else causal.reason,
                        "causal_target_margin": float(causal_config.target_margin),
                        "causal_max_alpha": float(causal_config.max_alpha),
                        "causal_gain_train_count": 0 if causal is None else causal.n_train,
                        "causal_gain_model": "ridge_log_effective_gain_from_train_fixed_alpha_crossing",
                        "dense_alpha": None if dense is None else dense.alpha,
                        "dense_alpha_raw": None if dense is None else dense.raw_alpha,
                        "dense_alpha_anchor": None if dense is None else dense.anchor_alpha,
                        "dense_alpha_multiplier": None if dense is None else dense.multiplier,
                        "dense_reason": None if dense is None else dense.reason,
                        "dense_target": dense_config.target,
                        "dense_prediction_mode": dense_config.prediction_mode,
                        "dense_target_margin": float(dense_config.target_margin),
                        "dense_safety_quantile": float(dense_config.safety_quantile),
                        "dense_ridge_alpha": float(dense_config.ridge_alpha),
                        "dense_max_alpha": float(dense_config.max_alpha),
                        "dense_calibration_folds": int(dense_config.calibration_folds),
                        "dense_dose_train_count": 0 if dense_estimator is None else dense_estimator.n_train_,
                        "dense_dose_calibration_count": 0 if dense_estimator is None else dense_estimator.n_calibration_,
                        "dense_dose_calibration_mode": None if dense_estimator is None else dense_estimator.calibration_mode_,
                        "note": "Predicted vector was trained leave-one-family-out from action-response labels; scorer alpha sweep supplies intervention magnitude.",
                    },
                    "direction_stats": vector_stats(clean),
                    "projection": None,
                    "context": {
                        "pc_n_train": prediction.support_count,
                        "pc_neighbor_distance_mean": float(np.mean(prediction.neighbor_distances)) if prediction.neighbor_distances else None,
                        "pc_neighbor_distance_min": float(np.min(prediction.neighbor_distances)) if prediction.neighbor_distances else None,
                        "pc_neighbor_distance_max": float(np.max(prediction.neighbor_distances)) if prediction.neighbor_distances else None,
                        "pc_predicted_flow_norm": prediction.confidence,
                        "pc_boundary_alpha": boundary.alpha,
                        "pc_boundary_alpha_raw": boundary.raw_alpha,
                        "pc_boundary_directional_slope": boundary.directional_slope,
                        "pc_boundary_reason": boundary.reason,
                        "pc_causal_alpha": None if causal is None else causal.alpha,
                        "pc_causal_alpha_raw": None if causal is None else causal.raw_alpha,
                        "pc_causal_predicted_gain": None if causal is None else causal.predicted_gain,
                        "pc_causal_required_margin_gain": None if causal is None else causal.required_margin_gain,
                        "pc_causal_reason": None if causal is None else causal.reason,
                        "pc_dense_alpha": None if dense is None else dense.alpha,
                        "pc_dense_alpha_raw": None if dense is None else dense.raw_alpha,
                        "pc_dense_alpha_anchor": None if dense is None else dense.anchor_alpha,
                        "pc_dense_alpha_multiplier": None if dense is None else dense.multiplier,
                        "pc_dense_reason": None if dense is None else dense.reason,
                    },
                })
            if not vectors:
                continue
            stem = shard_stem(layer, f"{family}_{METHOD_NAME}")
            np.savez_compressed(specs_dir / f"{stem}.npz", vectors=np.vstack(vectors))
            with (specs_dir / f"{stem}.jsonl").open("w") as handle:
                for row in meta:
                    handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
            shards.append({"layer": int(layer), "family": family, "stem": stem, "n_specs": len(vectors)})
            total_specs += len(vectors)
            print(f"[local-flow-specs] family={family} layer={layer} specs={len(vectors)}", flush=True)

    spec_cids = sorted({
        str(json.loads(line)["conversation_id"])
        for shard in shards
        for line in (specs_dir / f"{shard['stem']}.jsonl").read_text().splitlines()
    })
    missing_cids = sorted(set(grouped) - set(spec_cids))
    if missing_cids:
        raise SystemExit(f"missing local-flow specs for {len(missing_cids)} cids, examples={missing_cids[:5]}")
    manifest = {
        "spec_schema_version": SPEC_SCHEMA_VERSION,
        "kind": "steering_spec_bank",
        "identity": identity,
        "provenance": git_provenance([Path(__file__), Path(_bank_mod.__file__), args.action_response, args.activations, args.base_steering_spec_bank / "manifest.json"]),
        "argv": sys.argv,
        "source_action_response": str(args.action_response),
        "source_action_response_sha256": args.action_response_sha256 or file_sha256(args.action_response),
        "source_action_response_meta": action_meta,
        "source_activation_meta": activation_meta,
        "source_base_steering_spec_bank": str(args.base_steering_spec_bank),
        "source_base_steering_spec_manifest_sha256": file_sha256(args.base_steering_spec_bank / "manifest.json"),
        "local_control_flow_config": jsonable(config.__dict__),
        "boundary_dose_config": jsonable(boundary_config.__dict__),
        "causal_dose_config": jsonable(causal_config.__dict__) if causal_rows else None,
        "dense_dose_config": jsonable(dense_config.__dict__) if dense_rows else None,
        "source_causal_dose_action_response": str(args.causal_dose_action_response) if args.causal_dose_action_response else None,
        "source_causal_dose_action_response_sha256": (
            file_sha256(args.causal_dose_action_response) if args.causal_dose_action_response else None
        ),
        "source_causal_dose_action_response_meta": causal_meta,
        "source_dense_dose_action_response": str(args.dense_dose_action_response) if args.dense_dose_action_response else None,
        "source_dense_dose_action_response_sha256": (
            file_sha256(args.dense_dose_action_response) if args.dense_dose_action_response else None
        ),
        "source_dense_dose_action_response_meta": dense_meta,
        "n_eval_rows": len(grouped),
        "n_eval_cids": len(spec_cids),
        "eval_conversation_ids": spec_cids,
        "n_families": len(families),
        "families": families,
        "n_specs": total_specs,
        "shards": shards,
        "skipped_specs": skipped,
        "duration_sec": round(time.perf_counter() - started, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "note": (
            "Option-3 generated steering bank. Vectors are unit-normalized predicted "
            "local control-flow directions. Metadata includes predicted_flow_norm, "
            "boundary_alpha, optional causal_alpha, and optional dense_alpha; use score_steering_specs.py "
            "alpha tokens to test magnitude."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")
    print(
        f"[local-flow-specs] DONE specs={total_specs} cids={len(spec_cids)} "
        f"duration={manifest['duration_sec']}s -> {out_dir}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-response", type=Path, default=DEFAULT_ACTION_RESPONSE)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--base-steering-spec-bank", type=Path, default=DEFAULT_BASE_BANK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", default="16")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--geometry-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--neighbor-pool-multiplier", type=int, default=4)
    parser.add_argument("--boundary-target-margin", type=float, default=0.0)
    parser.add_argument("--boundary-max-alpha", type=float, default=128.0)
    parser.add_argument("--boundary-min-directional-slope", type=float, default=1e-6)
    parser.add_argument("--boundary-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--causal-dose-action-response", type=Path, default=None)
    parser.add_argument("--causal-target-margin", type=float, default=0.0)
    parser.add_argument("--causal-max-alpha", type=float, default=128.0)
    parser.add_argument("--causal-min-gain", type=float, default=1e-4)
    parser.add_argument("--causal-ridge-alpha", type=float, default=3.0)
    parser.add_argument("--causal-fixed-alphas", type=float, nargs="+", default=[48.0, 72.0, 96.0])
    parser.add_argument("--dense-dose-action-response", type=Path, default=None)
    parser.add_argument("--dense-target", choices=["stable_crossing", "first_crossing", "best_reward"], default="first_crossing")
    parser.add_argument("--dense-prediction-mode", choices=["raw_alpha", "boundary_ratio", "causal_ratio"], default="raw_alpha")
    parser.add_argument("--dense-target-margin", type=float, default=0.0)
    parser.add_argument("--dense-safety-quantile", type=float, default=1.0)
    parser.add_argument("--dense-ridge-alpha", type=float, default=3.0)
    parser.add_argument("--dense-max-alpha", type=float, default=128.0)
    parser.add_argument("--dense-calibration-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--max-cids", type=int, default=0)
    parser.add_argument("--action-response-sha256", default=None)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
