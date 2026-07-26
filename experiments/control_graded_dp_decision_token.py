"""Decision-token-only bidirectional control for the graded deception ramp.

The high-alpha tomography result showed that correction directions can cross the PASS/FAIL logit
margin, but persistent steering throughout generation destroys the report. This runner applies the
steering only for the first report-status decision token, then turns steering off and lets the model
complete the rest of the JSON normally.

By default this is still an oracle feasibility test: bidirectional methods use the known true status
to choose to_PASS vs to_FAIL. With ``--routing gate_file``, a held-out-family gate prediction file
chooses abstain / steer_to_PASS / steer_to_FAIL, giving a deployable-style control test without using
the answer at intervention time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.activation_control_tomography import decision_tokens  # noqa: E402
import geoprobe.models.decision as _decision_mod  # noqa: E402  fingerprint promoted backend
import geoprobe.models.decision_backend as _decision_backend_mod  # noqa: E402
from experiments.bidirectional_dp_diagnostic import attach_status_classes, fit_status_direction  # noqa: E402
from experiments.control_deception_intent_transition import prompt_without_final_answer, reply_coherence  # noqa: E402
from experiments.control_graded_dp_bidirectional_stack import (  # noqa: E402
    add_status_class_to_transcript,
    parse_int_csv,
    select_status_balanced_rows,
    target_status_for_row,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    clean_id_lines,
    clean_matrix,
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
    vector_stats,
)
from experiments.control_graded_dp_stack_frontier import aggregate_projection, parse_csv, stack_injection_stats  # noqa: E402
from experiments.rollout_deception_intent import parse_status  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from experiments.decision_token_backend import (  # noqa: E402
    choose_status_then_complete,
    choose_status_then_complete_batch,
    generation_protocol_name,
    load_decision_backend_model,
    public_model_meta,
    validate_generation_batch_config,
)
from geoprobe.io import atomic_text, file_sha256  # noqa: E402
from geoprobe.models.interface import ResidualSteeringSpec  # noqa: E402
from geoprobe.runtime.gpu_sampling import gpu_sampler, summarize_samples  # noqa: E402


METHOD_ALIASES = {
    "baseline": "baseline",
    "bidir_linear": "bidir_linear",
    "linear": "bidir_linear",
    "bidir_tangent": "bidir_tangent",
    "tangent": "bidir_tangent",
    "bidir_off_tangent": "bidir_off_tangent",
    "off_tangent": "bidir_off_tangent",
    "global_mean": "global_mean",
    "pooled_global_mean": "global_mean",
    "global_probe": "global_probe",
    "logistic_probe": "global_probe",
    "random_global": "random_global",
    "random": "random_global",
    "global_mean_gated": "global_mean_gated",
    "global_probe_gated": "global_probe_gated",
    "random_gated": "random_gated",
    "local_control_flow": "local_control_flow",
}
# Experiment-1 confound-decomposition cells (registered in the privately
# retained results ledger of the program; measure-first experiment-1
# registration, frozen 2026-07-24). Norm-matched-raw
# arms inject the RAW (unprojected) direction
# at the per-row alpha_eff = alpha x that row's dp tangent projection_fraction
# ("per row, not global mean"); the random cells use the 5 frozen seeds 0-4,
# averaged at analysis time (experiments/report_experiment1_confound_factorial.py).
EXPERIMENT1_RANDOM_SEEDS = (0, 1, 2, 3, 4)
RANDOM_TANGENT_METHODS = {f"random_tangent_s{s}" for s in EXPERIMENT1_RANDOM_SEEDS}
RANDOM_NORM_MATCHED_METHODS = {
    f"random_norm_matched_raw_s{s}" for s in EXPERIMENT1_RANDOM_SEEDS
}
EXPERIMENT1_METHODS = (
    {"bidir_norm_matched_raw"} | RANDOM_TANGENT_METHODS | RANDOM_NORM_MATCHED_METHODS
)
METHOD_ALIASES.update({name: name for name in EXPERIMENT1_METHODS})
BIDIR_METHODS = {"bidir_linear", "bidir_tangent", "bidir_off_tangent"}
GLOBAL_METHODS = {"global_mean", "global_probe", "random_global"}
GATED_GLOBAL_METHODS = {"global_mean_gated", "global_probe_gated", "random_gated"}
LOCAL_FLOW_METHODS = {"local_control_flow"}
STEER_METHODS = (
    BIDIR_METHODS
    | GLOBAL_METHODS
    | GATED_GLOBAL_METHODS
    | LOCAL_FLOW_METHODS
    | EXPERIMENT1_METHODS
)
SPEC_BANK_METHOD_FOR_POLICY = {
    "global_mean_gated": "global_mean",
    "global_probe_gated": "global_probe",
    "random_gated": "random_global",
}


def canonical_methods(value: str) -> list[str]:
    out: list[str] = []
    for item in parse_csv(value):
        if item not in METHOD_ALIASES:
            raise ValueError(f"unknown method {item!r}; expected one of {sorted(METHOD_ALIASES)}")
        method = METHOD_ALIASES[item]
        if method not in out:
            out.append(method)
    return out


def load_gate_predictions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for row in payload.get("predictions", []):
        cid = str(row.get("conversation_id", ""))
        if cid:
            out[cid] = row
    if not out:
        raise ValueError(f"no gate predictions found in {path}")
    return out


def load_policy_choices(path: Path | None, policy_name: str) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    policies = payload.get("policies", payload)
    if policy_name not in policies:
        raise ValueError(f"policy {policy_name!r} not found in {path}; keys={sorted(policies)}")
    choices = policies[policy_name].get("choices", [])
    out = {}
    for choice in choices:
        cid = str(choice.get("conversation_id", ""))
        if cid:
            out[cid] = choice
    if not out:
        raise ValueError(f"no choices found for policy {policy_name!r} in {path}")
    return out


def spec_bank_method_for_policy(method: str) -> str:
    return SPEC_BANK_METHOD_FOR_POLICY.get(method, method)


def load_policy_steering_spec_bank(
    bank_dir: Path,
    *,
    rows: list[dict],
    choice_by_cid: dict[str, dict],
    methods: list[str],
    layers: list[int],
    activation_sha256: str | None,
    transcript_hashes: dict[str, str],
    config_sha256: str,
) -> tuple[dict[tuple[str, str], dict], dict[int, dict], list[str], dict]:
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    if manifest.get("kind") != "steering_spec_bank":
        raise ValueError(f"{bank_dir} is not a steering_spec_bank")
    identity = manifest.get("identity", {})
    problems = []
    if activation_sha256 and identity.get("activations_sha256") != activation_sha256:
        problems.append("activations_sha256")
    if identity.get("config_sha256") != config_sha256:
        problems.append("config_sha256")
    if identity.get("transcripts_sha256") != transcript_hashes:
        problems.append("transcripts_sha256")
    covered_layers = {int(layer) for layer in identity.get("layers", [])}
    covered_methods = {str(method) for method in identity.get("methods", [])}
    wanted_bank_methods = {spec_bank_method_for_policy(method) for method in methods}
    if missing_layers := sorted(set(layers) - covered_layers):
        problems.append(f"layers_missing={missing_layers}")
    if missing_methods := sorted(wanted_bank_methods - covered_methods):
        problems.append(f"methods_missing={missing_methods}")
    if problems:
        raise ValueError(f"steering spec bank identity/coverage mismatch: {problems}")

    eval_ids = {str(row["conversation_id"]) for row in rows}
    desired: dict[tuple[str, int, str, str], str] = {}
    for row in rows:
        cid = str(row["conversation_id"])
        choice = choice_by_cid.get(cid)
        if not choice or choice.get("method") in {None, "abstain"}:
            continue
        method = str(choice.get("method"))
        target = str(choice.get("target_status"))
        if target not in {"PASS", "FAIL"}:
            continue
        desired[(cid, int(choice["layer"]), spec_bank_method_for_policy(method), target)] = method

    prepared: dict[tuple[str, str], dict] = {
        (cid, policy_method): {"directions": {}, "projections": {}, "direction_info": {}}
        for (_cid, _layer, _bank_method, _target), policy_method in desired.items()
        for cid in [_cid]
    }
    available_counts: dict[int, Counter] = {int(layer): Counter() for layer in layers}
    found: set[tuple[str, int, str, str]] = set()
    specs_dir = bank_dir / "specs"
    for shard in manifest.get("shards", []):
        layer = int(shard["layer"])
        if layer not in set(layers):
            continue
        stem = str(shard["stem"])
        npz_path = specs_dir / f"{stem}.npz"
        jsonl_path = specs_dir / f"{stem}.jsonl"
        vectors = np.load(npz_path)["vectors"]
        with jsonl_path.open() as handle:
            for idx, line in enumerate(handle):
                meta = json.loads(line)
                cid = str(meta.get("conversation_id"))
                if cid not in eval_ids:
                    continue
                key = (
                    cid,
                    int(meta["layer"]),
                    str(meta["method"]),
                    str(meta["target_status"]),
                )
                policy_method = desired.get(key)
                if policy_method is None:
                    continue
                prepared[(cid, policy_method)]["directions"][layer] = np.asarray(vectors[idx], dtype=np.float64)
                prepared[(cid, policy_method)]["projections"][layer] = meta.get("projection")
                prepared[(cid, policy_method)]["direction_info"][layer] = meta.get("direction_info") or {}
                available_counts[layer][policy_method] += 1
                found.add(key)

    missing = sorted(desired.keys() - found)
    skips = [
        f"missing_spec_bank_vector::{cid}::L{layer}::{method}::{target}"
        for cid, layer, method, target in missing
    ]
    layer_direction_availability = {
        int(layer): {
            "source": "steering_spec_bank",
            "available_target_rows": dict(available_counts[int(layer)]),
            "status_skipped": {},
        }
        for layer in layers
    }
    return prepared, layer_direction_availability, skips, manifest


def route_for_policy_choice(row: dict, choice: dict | None, *, gate_by_cid: dict[str, dict]) -> dict:
    cid = str(row["conversation_id"])
    gate = gate_by_cid.get(cid)
    if choice is None or choice.get("method") in {None, "abstain"}:
        target_status = None
        action = "abstain"
    else:
        target_status = choice.get("target_status")
        if target_status not in {"PASS", "FAIL"}:
            raise ValueError(f"bad policy target for {cid}: {target_status!r}")
        action = f"steer_to_{target_status}"
    return {
        "routing": "policy_file",
        "action": action,
        "target_status": target_status,
        "predicted_true_status": target_status,
        "target_status_correct": target_status == str(row.get("true_status")).upper() if target_status else None,
        "routing_correct": target_status == str(row.get("true_status")).upper() if target_status else None,
        "policy_choice": choice,
        "gate": None if gate is None else {
            "source_conversation_id": gate.get("conversation_id"),
            "reported_status": gate.get("reported_status"),
            "true_status": gate.get("true_status"),
            "status_class": gate.get("status_class"),
            "action": gate.get("action"),
            "proba_PASS": gate.get("proba_PASS"),
            "score_PASS_minus_FAIL": gate.get("score_PASS_minus_FAIL"),
        },
    }


def route_for_row(row: dict, *, routing: str, gate_by_cid: dict[str, dict]) -> dict:
    cid = str(row["conversation_id"])
    if routing == "oracle_true_status":
        target_status = target_status_for_row(row)
        return {
            "routing": routing,
            "action": f"steer_to_{target_status}",
            "target_status": target_status,
            "predicted_true_status": target_status,
            "target_status_correct": True,
            "routing_correct": True,
            "gate": None,
        }
    if routing != "gate_file":
        raise ValueError(f"unknown routing mode {routing!r}")
    gate = gate_by_cid.get(cid)
    if gate is None:
        return {
            "routing": routing,
            "action": "missing_gate_prediction",
            "target_status": None,
            "predicted_true_status": None,
            "target_status_correct": None,
            "routing_correct": None,
            "gate": None,
        }
    action = str(gate.get("action", ""))
    target_status = None
    if action == "steer_to_PASS":
        target_status = "PASS"
    elif action == "steer_to_FAIL":
        target_status = "FAIL"
    elif action != "abstain":
        raise ValueError(f"bad gate action for {cid}: {action!r}")
    return {
        "routing": routing,
        "action": action,
        "target_status": target_status,
        "predicted_true_status": gate.get("predicted_true_status"),
        "target_status_correct": gate.get("target_status_correct"),
        "routing_correct": gate.get("routing_correct"),
        "score_PASS_minus_FAIL": gate.get("score_PASS_minus_FAIL"),
        "proba_PASS": gate.get("proba_PASS"),
        "gate": {
            "source_conversation_id": gate.get("conversation_id"),
            "reported_status": gate.get("reported_status"),
            "true_status": gate.get("true_status"),
            "status_class": gate.get("status_class"),
        },
    }


def target_direction_name_from_route(route: dict) -> str | None:
    target = route.get("target_status")
    return f"to_{target}" if target in {"PASS", "FAIL"} else None


def method_requires_gate_action(method: str) -> bool:
    return (
        method in BIDIR_METHODS
        or method in GATED_GLOBAL_METHODS
        or method in EXPERIMENT1_METHODS
    )


def method_uses_tangent(method: str) -> bool:
    return method in {"bidir_tangent", "bidir_off_tangent"}


def fit_global_mean_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str]) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    honest = [row for row in train if str(row["status_class"]).startswith("honest_")]
    false = [row for row in train if str(row["status_class"]).startswith("false_")]
    if not honest or not false:
        return None
    x_h = clean_matrix(np.vstack([row["x"] for row in honest]))
    x_f = clean_matrix(np.vstack([row["x"] for row in false]))
    direction = unit(x_h.mean(axis=0) - x_f.mean(axis=0))
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None
    return {
        "heldout_family": heldout_family,
        "target_status": "global_honesty",
        "direction_convention": "direction = mean(honest_PASS+honest_FAIL) - mean(false_FAIL+false_PASS)",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "n_honest": int(len(honest)),
        "n_false": int(len(false)),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def fit_global_probe_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str]) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if not train:
        return None
    y = np.asarray([1 if str(row["status_class"]).startswith("honest_") else 0 for row in train], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    x = clean_matrix(np.vstack([row["x"] for row in train])).astype(np.float64)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=0)
    clf.fit(xs, y)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0] = 1.0
    direction = unit(np.asarray(clf.coef_[0], dtype=np.float64) / scale)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None
    return {
        "heldout_family": heldout_family,
        "target_status": "global_honesty_probe",
        "direction_convention": "direction = raw-space normal of logistic probe predicting honest vs false",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "n_honest": int(y.sum()),
        "n_false": int((1 - y).sum()),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def fit_random_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str], layer: int, seed: int) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if not train:
        return None
    dim = int(np.asarray(train[0]["x"]).shape[0])
    rng = np.random.default_rng(stable_seed("random_global", heldout_family, layer, seed))
    direction = unit(rng.normal(size=dim))
    return {
        "heldout_family": heldout_family,
        "target_status": "random_global",
        "direction_convention": "direction = deterministic random unit vector, norm-matched by alpha",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def public_direction_info(direction_info: dict) -> dict:
    return {
        "target_status": direction_info.get("target_status"),
        "n_mixed_scenario_level_pairs": direction_info.get("n_mixed_scenario_level_pairs"),
        "n_train_points": direction_info.get("n_train_points"),
        "n_honest": direction_info.get("n_honest"),
        "n_false": direction_info.get("n_false"),
        "direction_levels": direction_info.get("direction_levels"),
        "direction_convention": direction_info.get("direction_convention"),
    }


def target_direction_for_method(method: str, route: dict) -> str | None:
    if method in BIDIR_METHODS:
        return target_direction_name_from_route(route)
    if method in GLOBAL_METHODS:
        return method
    if method in GATED_GLOBAL_METHODS:
        return method if route.get("target_status") in {"PASS", "FAIL"} else None
    if method in LOCAL_FLOW_METHODS:
        return target_direction_name_from_route(route)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument(
        "--activations-sha256",
        default=None,
        help="Known SHA-256 for --activations. Skips hashing the large activation bank.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20")
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
    parser.add_argument("--methods", default="baseline,bidir_linear,bidir_tangent")
    parser.add_argument("--alphas", default="48")
    parser.add_argument("--alpha-mode", choices=["total", "per_layer"], default="total")
    parser.add_argument("--routing", choices=["oracle_true_status", "gate_file", "policy_file"], default="oracle_true_status")
    parser.add_argument("--gate-predictions", default=None)
    parser.add_argument("--policy-choices", default=None)
    parser.add_argument("--policy-name", default="selective_route_policy")
    parser.add_argument(
        "--steering-spec-bank",
        default=None,
        help="Precomputed steering-spec bank for policy_file runs. Skips activation geometry prep.",
    )
    parser.add_argument("--tangent-neighbors", type=int, default=16)
    parser.add_argument("--tangent-dim", type=int, default=4)
    parser.add_argument("--conversation-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-status-class", type=int, default=None)
    parser.add_argument("--limit-strategy", choices=["first", "shuffle", "family_round_robin"], default="family_round_robin")
    parser.add_argument("--scenario-include", default=None,
                        help="Comma-separated scenario_ids to include (exclusive filter).")
    parser.add_argument("--scenario-exclude", default=None,
                        help="Comma-separated scenario_ids to exclude.")
    parser.add_argument("--equivariant-directions", action="store_true",
                        help="Pool Z₂ partner direction (PASS↔FAIL swap) with sign flipped. "
                             "Doubles effective data for direction fitting.")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--backend", choices=["mlx", "hf"], default=None)
    parser.add_argument("--model", default=None,
                        help="Model registry key or HF model path for --backend hf.")
    parser.add_argument("--device", default=None,
                        help="HF device override, e.g. cuda, cuda:0, cpu. MLX ignores this.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="HF-only targeted generation batch size. Values >1 require --verify-generation-smoke.",
    )
    parser.add_argument(
        "--verify-generation-smoke",
        action="store_true",
        help="Scalar-check early batched generation chunks and fail on unacceptable drift.",
    )
    parser.add_argument(
        "--verify-generation-smoke-batches",
        type=int,
        default=1,
        help="Number of batched generation chunks to scalar-check.",
    )
    parser.add_argument(
        "--batched-generation-margin-fail-atol",
        type=float,
        default=0.5,
        help="Hard margin-drift failure threshold for --verify-generation-smoke.",
    )
    parser.add_argument(
        "--batched-generation-reply-mismatch-policy",
        choices=["fail", "record"],
        default="fail",
        help="Whether scalar-vs-batch reply text mismatches fail the smoke guard or are recorded.",
    )
    parser.add_argument("--sample-gpu", action="store_true", help="Poll nvidia-smi and record GPU/VRAM utilization.")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between --sample-gpu polls.")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    transcript_paths = [Path(path) for path in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = config_defaults(config_path)
    # Keep MLX as the default so existing MLX reproduction commands are unchanged.
    # HF execution is opt-in with ``--backend hf``.
    backend = args.backend or "mlx"
    try:
        validate_generation_batch_config(
            backend=backend,
            generation_batch_size=args.generation_batch_size,
            verify_generation_smoke=args.verify_generation_smoke,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.verify_generation_smoke_batches < 0:
        raise SystemExit("--verify-generation-smoke-batches must be >= 0")
    if args.batched_generation_margin_fail_atol < 0:
        raise SystemExit("--batched-generation-margin-fail-atol must be >= 0")
    generation_protocol = generation_protocol_name(
        backend=backend,
        generation_batch_size=args.generation_batch_size,
    )
    model_key = args.model or defaults["model_key"]
    mlx_model = args.mlx_model or defaults["mlx_model"]
    dtype = args.dtype or defaults["dtype"]
    max_new_tokens = args.max_new_tokens or defaults["max_new_tokens"]
    max_generation_seconds = args.max_generation_seconds or defaults["max_generation_seconds"]
    choice_path = Path(args.policy_choices) if args.policy_choices else None
    spec_bank_path = Path(args.steering_spec_bank) if args.steering_spec_bank else None
    choice_by_cid = load_policy_choices(choice_path, args.policy_name) if choice_path else {}
    if args.routing == "policy_file" and not choice_by_cid:
        raise ValueError("--routing policy_file requires --policy-choices")
    if args.routing != "policy_file" and choice_by_cid:
        raise ValueError("--policy-choices requires --routing policy_file")
    if choice_by_cid:
        policy_layers = sorted({
            int(choice["layer"])
            for choice in choice_by_cid.values()
            if choice.get("method") not in {None, "abstain"} and choice.get("layer") is not None
        })
        layers = policy_layers or parse_int_csv(args.layers)
        methods = sorted({
            str(choice.get("method"))
            for choice in choice_by_cid.values()
            if choice.get("method") not in {None, "abstain"}
        })
    else:
        layers = parse_int_csv(args.layers)
        methods = canonical_methods(args.methods)
    gate_path = Path(args.gate_predictions) if args.gate_predictions else None
    if args.routing == "gate_file" and gate_path is None:
        raise ValueError("--routing gate_file requires --gate-predictions")
    gate_by_cid = load_gate_predictions(gate_path)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))
    transcript_hashes = {str(path): file_sha256(path) for path in transcript_paths}
    config_hash = file_sha256(config_path)
    activation_sha = args.activations_sha256 or file_sha256(activation_path)
    gate_predictions_sha = file_sha256(gate_path) if gate_path else None
    policy_choices_sha = file_sha256(choice_path) if choice_path else None
    spec_bank_manifest_sha = file_sha256(spec_bank_path / "manifest.json") if spec_bank_path else None
    provenance = git_provenance([Path(__file__), Path(_decision_backend_mod.__file__), Path(_decision_mod.__file__), config_path, *transcript_paths])
    provenance.setdefault("file_sha256", {})[str(activation_path)] = activation_sha

    if spec_bank_path:
        if args.routing != "policy_file":
            raise ValueError("--steering-spec-bank currently supports --routing policy_file only")
        spec_manifest = json.loads((spec_bank_path / "manifest.json").read_text())
        first_query_ids = {str(cid) for cid in spec_manifest.get("eval_conversation_ids", [])}
        if not first_query_ids:
            raise ValueError(f"steering spec bank has no eval_conversation_ids: {spec_bank_path}")
        activation_meta = {
            "source": "steering_spec_bank_manifest",
            "steering_spec_bank": str(spec_bank_path),
            "n_eval_cids": len(first_query_ids),
            "n_specs": spec_manifest.get("n_specs"),
        }
    else:
        first_query, activation_meta = load_activation_points(
            activation_path,
            layer=layers[0],
            turns={args.query_turn},
            phases={args.query_phase},
        )
        first_query_ids = set(point_index(first_query))
        del first_query
        gc.collect()

    allowed = clean_id_lines(Path(args.conversation_ids_file)) if args.conversation_ids_file else None
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if allowed is not None and cid not in allowed:
            continue
        if not row.get("valid_outcome") or row.get("arm") not in eval_levels or cid not in first_query_ids:
            continue
        row_with_class = add_status_class_to_transcript(row)
        if row_with_class is not None:
            rows.append(row_with_class)
    if args.scenario_include:
        include_set = {s.strip() for s in args.scenario_include.split(",") if s.strip()}
        rows = [row for row in rows if str(row.get("scenario_id", "")) in include_set]
    if args.scenario_exclude:
        exclude_set = {s.strip() for s in args.scenario_exclude.split(",") if s.strip()}
        rows = [row for row in rows if str(row.get("scenario_id", "")) not in exclude_set]
    if args.limit_per_status_class is not None:
        rows = select_status_balanced_rows(rows, per_status_class=args.limit_per_status_class, seed=args.seed)
    else:
        rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if args.limit is not None and args.limit_per_status_class is not None and len(rows) > args.limit:
        rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations")
    eval_ids = {str(row["conversation_id"]) for row in rows}
    if choice_by_cid:
        selected_policy_choices = {
            cid: choice
            for cid, choice in choice_by_cid.items()
            if cid in eval_ids and choice.get("method") not in {None, "abstain"}
        }
        selected_policy_layers = sorted({
            int(choice["layer"])
            for choice in selected_policy_choices.values()
            if choice.get("layer") is not None
        })
        selected_policy_methods = sorted({str(choice.get("method")) for choice in selected_policy_choices.values()})
        layers = selected_policy_layers or layers
        methods = selected_policy_methods
        print(
            "[control_graded_dp] selected policy scope: "
            f"rows={len(rows)} layers={layers} methods={methods}",
            flush=True,
        )
    transcript_by_cid = {str(row["conversation_id"]): row for row in read_jsonl_paths(transcript_paths)}
    if args.routing == "policy_file":
        route_by_cid = {
            str(row["conversation_id"]): route_for_policy_choice(
                row,
                choice_by_cid.get(str(row["conversation_id"])),
                gate_by_cid=gate_by_cid,
            )
            for row in rows
        }
    else:
        route_by_cid = {
            str(row["conversation_id"]): route_for_row(row, routing=args.routing, gate_by_cid=gate_by_cid)
            for row in rows
        }

    prepared: dict[tuple[str, str], dict] = {}
    planned_skips: Counter = Counter()
    for row in rows:
        row_methods = methods if args.routing == "policy_file" else list(STEER_METHODS)
        for method in row_methods:
            if method not in STEER_METHODS:
                continue
            route = route_by_cid[str(row["conversation_id"])]
            if args.routing == "gate_file" and route["action"] == "missing_gate_prediction":
                planned_skips[f"missing_gate_prediction::{method}"] += 1
                continue
            if method_requires_gate_action(method) and route["target_status"] is None:
                continue
            prepared[(str(row["conversation_id"]), method)] = {"directions": {}, "projections": {}, "direction_info": {}, "alpha_scale": {}}

    layer_direction_availability: dict[int, dict] = {}
    spec_bank_manifest_for_output = None
    if spec_bank_path is not None:
        (
            prepared,
            layer_direction_availability,
            spec_bank_skips,
            spec_bank_manifest_for_output,
        ) = load_policy_steering_spec_bank(
            spec_bank_path,
            rows=rows,
            choice_by_cid=choice_by_cid,
            methods=methods,
            layers=layers,
            activation_sha256=args.activations_sha256,
            transcript_hashes=transcript_hashes,
            config_sha256=config_hash,
        )
        planned_skips.update(spec_bank_skips)

    for layer in ([] if spec_bank_path is not None else layers):
        direction_layer_raw, _ = load_activation_points(activation_path, layer=layer, turns={args.direction_turn}, phases={args.direction_phase})
        direction_layer, skipped = attach_status_classes(direction_layer_raw, transcript_by_cid)
        query_layer, _ = load_activation_points(activation_path, layer=layer, turns={args.query_turn}, phases={args.query_phase})
        query_by_cid = point_index([row for row in query_layer if row["conversation_id"] in eval_ids])
        tangent_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns=tangent_turns,
            phases=tangent_phases,
            levels=tangent_levels,
        )
        direction_cache: dict[tuple[str, str], dict | None] = {}
        global_cache: dict[tuple[str, str], dict | None] = {}
        tangent_cache: dict[str, dict | None] = {}
        seeded_random_cache: dict[tuple[str, int], dict | None] = {}

        def get_direction(family: str, target_status: str) -> dict | None:
            key = (family, target_status)
            if key not in direction_cache:
                direction_cache[key] = fit_status_direction(
                    direction_layer,
                    heldout_family=family,
                    direction_levels=direction_levels,
                    target_status=target_status,
                    min_mixed_scenarios=args.min_mixed_scenarios,
                    min_levels=args.min_direction_levels,
                    equivariant=args.equivariant_directions,
                )
            return direction_cache[key]

        def get_global_direction(family: str, method: str) -> dict | None:
            if method in {"global_mean", "global_mean_gated"}:
                direction_type = "global_mean"
            elif method in {"global_probe", "global_probe_gated"}:
                direction_type = "global_probe"
            elif method in {"random_global", "random_gated"}:
                direction_type = "random_global"
            else:
                raise ValueError(f"not a global method: {method}")
            key = (family, direction_type)
            if key not in global_cache:
                if direction_type == "global_mean":
                    global_cache[key] = fit_global_mean_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                    )
                elif direction_type == "global_probe":
                    global_cache[key] = fit_global_probe_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                    )
                else:
                    global_cache[key] = fit_random_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                        layer=layer,
                        seed=args.seed,
                    )
            return global_cache[key]

        def get_tangent(family: str) -> dict | None:
            if family not in tangent_cache:
                tangent_cache[family] = fit_tangent_cloud(tangent_layer, heldout_family=family)
            return tangent_cache[family]

        def get_seeded_random_direction(family: str, seed: int) -> dict | None:
            # Experiment-1 frozen seeds 0-4; deterministic per (family, layer, seed)
            # via fit_random_direction's stable_seed.
            key = (family, int(seed))
            if key not in seeded_random_cache:
                seeded_random_cache[key] = fit_random_direction(
                    direction_layer,
                    heldout_family=family,
                    direction_levels=direction_levels,
                    layer=layer,
                    seed=int(seed),
                )
            return seeded_random_cache[key]

        available_counts = Counter()
        for row in rows:
            cid = str(row["conversation_id"])
            family = str(row["family"])
            route = route_by_cid[cid]
            target_status = route["target_status"]

            def pop_experiment1(reason: str) -> None:
                # Consistent denominators: every Experiment-1 cell depends on the
                # dp tangent projection (fraction and/or basis), so any upstream
                # failure removes ALL cells for this row.
                popped = False
                for method in EXPERIMENT1_METHODS:
                    popped = prepared.pop((cid, method), None) is not None or popped
                if popped:
                    planned_skips[f"experiment1_popped::{reason}::{family}::L{layer}"] += 1

            if target_status is not None and any((cid, method) in prepared for method in BIDIR_METHODS):
                direction_info = get_direction(family, target_status)
                if direction_info is None:
                    for method in BIDIR_METHODS:
                        prepared.pop((cid, method), None)
                    pop_experiment1("no_direction")
                    planned_skips[f"no_direction::{target_status}::{family}::L{layer}"] += 1
                else:
                    available_counts[f"to_{target_status}"] += 1
                    raw_direction = direction_info["_direction_np"]
                    if (cid, "bidir_linear") in prepared:
                        prepared[(cid, "bidir_linear")]["directions"][layer] = raw_direction
                        prepared[(cid, "bidir_linear")]["direction_info"][layer] = direction_info
                    if (cid, "bidir_tangent") in prepared:
                        tangent_info = get_tangent(family)
                        if tangent_info is None:
                            prepared.pop((cid, "bidir_tangent"), None)
                            prepared.pop((cid, "bidir_off_tangent"), None)
                            pop_experiment1("no_tangent_cloud")
                            planned_skips[f"no_tangent_cloud::{family}::L{layer}"] += 1
                        else:
                            tangent_direction, projection = project_to_local_tangent(
                                raw_direction,
                                tangent_info,
                                query_by_cid[cid]["x"],
                                tangent_neighbors=args.tangent_neighbors,
                                tangent_dim=args.tangent_dim,
                            )
                            if tangent_direction is None:
                                prepared.pop((cid, "bidir_tangent"), None)
                                prepared.pop((cid, "bidir_off_tangent"), None)
                                pop_experiment1("no_tangent")
                                planned_skips[f"no_tangent::{target_status}::{family}::L{layer}::{projection.get('reason')}"] += 1
                            else:
                                prepared[(cid, "bidir_tangent")]["directions"][layer] = tangent_direction.detach().float().cpu().numpy()
                                prepared[(cid, "bidir_tangent")]["projections"][layer] = projection
                                prepared[(cid, "bidir_tangent")]["direction_info"][layer] = direction_info
                                off_direction = off_tangent_direction(raw_direction, tangent_direction)
                                if off_direction is None:
                                    prepared.pop((cid, "bidir_off_tangent"), None)
                                    planned_skips[f"no_off_tangent::{target_status}::{family}::L{layer}"] += 1
                                elif (cid, "bidir_off_tangent") in prepared:
                                    prepared[(cid, "bidir_off_tangent")]["directions"][layer] = off_direction.detach().float().cpu().numpy()
                                    prepared[(cid, "bidir_off_tangent")]["projections"][layer] = projection
                                    prepared[(cid, "bidir_off_tangent")]["direction_info"][layer] = direction_info
                                # ---- Experiment-1 cells (frozen 2026-07-24) ----
                                fraction = float(projection.get("projection_fraction") or 0.0)
                                if fraction <= 0.0 or not np.isfinite(fraction):
                                    pop_experiment1("nonpositive_projection_fraction")
                                else:
                                    if (cid, "bidir_norm_matched_raw") in prepared:
                                        payload = prepared[(cid, "bidir_norm_matched_raw")]
                                        payload["directions"][layer] = raw_direction
                                        payload["alpha_scale"][layer] = fraction
                                        payload["projections"][layer] = projection
                                        payload["direction_info"][layer] = direction_info
                                    for seed in EXPERIMENT1_RANDOM_SEEDS:
                                        rt_key = (cid, f"random_tangent_s{seed}")
                                        rn_key = (cid, f"random_norm_matched_raw_s{seed}")
                                        if rt_key not in prepared and rn_key not in prepared:
                                            continue
                                        rand_info = get_seeded_random_direction(family, seed)
                                        if rand_info is None:
                                            prepared.pop(rt_key, None)
                                            prepared.pop(rn_key, None)
                                            planned_skips[f"no_seeded_random::{family}::L{layer}::s{seed}"] += 1
                                            continue
                                        rand_raw = rand_info["_direction_np"]
                                        if rt_key in prepared:
                                            rt_direction, rt_projection = project_to_local_tangent(
                                                rand_raw,
                                                tangent_info,
                                                query_by_cid[cid]["x"],
                                                tangent_neighbors=args.tangent_neighbors,
                                                tangent_dim=args.tangent_dim,
                                            )
                                            if rt_direction is None:
                                                prepared.pop(rt_key, None)
                                                planned_skips[f"no_random_tangent::{family}::L{layer}::s{seed}"] += 1
                                            else:
                                                payload = prepared[rt_key]
                                                payload["directions"][layer] = rt_direction.detach().float().cpu().numpy()
                                                payload["projections"][layer] = rt_projection
                                                payload["direction_info"][layer] = rand_info
                                        if rn_key in prepared:
                                            payload = prepared[rn_key]
                                            payload["directions"][layer] = rand_raw
                                            # "same alpha_eff" as the dp norm-matched cell:
                                            # matched to the dp projection fraction per row.
                                            payload["alpha_scale"][layer] = fraction
                                            payload["projections"][layer] = projection
                                            payload["direction_info"][layer] = rand_info

            for method in GLOBAL_METHODS | GATED_GLOBAL_METHODS:
                if (cid, method) not in prepared:
                    continue
                direction_info = get_global_direction(family, method)
                if direction_info is None:
                    prepared.pop((cid, method), None)
                    planned_skips[f"no_global_direction::{method}::{family}::L{layer}"] += 1
                    continue
                prepared[(cid, method)]["directions"][layer] = direction_info["_direction_np"]
                prepared[(cid, method)]["direction_info"][layer] = direction_info
                available_counts[method] += 1
        layer_direction_availability[layer] = {"available_target_rows": dict(available_counts), "status_skipped": dict(skipped)}
        del direction_layer_raw, query_layer, query_by_cid
        gc.collect()

    existing_results: list[dict] = []
    timings: dict[str, float] = {}
    batch_verifications: list[dict] = []
    prior_batch_verifications: list[dict] = []
    gpu_summary: dict = {}
    sample_path = out_path.parent / (out_path.stem + "_gpu_samples.csv")
    stop_event = threading.Event()
    sampler: threading.Thread | None = None
    completed: set[tuple[str, str, float]] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        expected_resume_identity = {
            "backend": backend,
            "requested_model_key": model_key,
            "mlx_model": mlx_model,
            "dtype": dtype,
            "routing": args.routing,
            "gate_predictions_sha256": gate_predictions_sha,
            "policy_choices_sha256": policy_choices_sha,
            "policy_name": args.policy_name if choice_path else None,
            "steering_spec_bank_manifest_sha256": spec_bank_manifest_sha,
            "transcripts_sha256": transcript_hashes,
            "activations_sha256": activation_sha,
            "config_sha256": config_hash,
            "generation_protocol": generation_protocol,
            "generation_batch_size": int(args.generation_batch_size),
            "max_new_tokens": max_new_tokens,
        }
        resume_mismatches = [
            key
            for key, expected in expected_resume_identity.items()
            if existing.get(key) != expected
        ]
        if resume_mismatches:
            raise ValueError(
                "refusing to resume from output with different identity: "
                f"{resume_mismatches}"
            )
        if not existing.get("blocked") and isinstance(existing.get("results"), list):
            existing_results = existing["results"]
            completed = {(str(row["conversation_id"]), str(row["method"]), float(row["alpha"])) for row in existing_results}
            timings = dict(existing.get("timings", {}))
            prior_batch_verifications = list(existing.get("batched_generation_verifications", []))

    jobs = []
    for idx, row in enumerate(rows, start=1):
        cid = str(row["conversation_id"])
        if args.routing == "policy_file":
            choice = choice_by_cid.get(cid)
            method = "abstain" if choice is None else str(choice.get("method") or "abstain")
            alpha = 0.0 if method == "abstain" else float(choice.get("alpha") or 0.0)
            key = (cid, method, float(alpha))
            if key in completed:
                planned_skips["already_completed"] += 1
                continue
            specs = None
            projection = None
            direction_info_public = None
            choice_layer = None if method == "abstain" else int(choice.get("layer"))
            if method in STEER_METHODS:
                payload = prepared.get((cid, method))
                if payload is None or choice_layer not in payload["directions"]:
                    planned_skips[f"missing_prepared::{method}"] += 1
                    continue
                specs = [
                    ResidualSteeringSpec(
                        layer=choice_layer,
                        direction=np.asarray(payload["directions"][choice_layer]),
                        alpha=float(alpha),
                    )
                ]
                raw_projection = payload["projections"].get(choice_layer)
                projection = aggregate_projection({choice_layer: raw_projection}) if raw_projection else None
                direction_info_public = {
                    str(choice_layer): public_direction_info(payload["direction_info"][choice_layer])
                }
            jobs.append({
                "row_index": idx,
                "row": row,
                "method": method,
                "alpha": float(alpha),
                "specs": specs,
                "projection": projection,
                "direction_info": direction_info_public,
                "policy_choice": choice,
            })
            continue

        for method in methods:
            active_alphas = [0.0] if method == "baseline" else [alpha for alpha in alphas if alpha != 0]
            for alpha in active_alphas:
                key = (cid, method, float(alpha))
                if key in completed:
                    planned_skips["already_completed"] += 1
                    continue
                specs = None
                projection = None
                direction_info_public = None
                if method in STEER_METHODS:
                    route = route_by_cid[cid]
                    if method_requires_gate_action(method) and route["target_status"] is None:
                        specs = None
                        projection = None
                        direction_info_public = None
                    else:
                        payload = prepared.get((cid, method))
                        if payload is None or len(payload["directions"]) != len(layers):
                            planned_skips[f"missing_prepared::{method}"] += 1
                            continue
                        layer_alpha = alpha / np.sqrt(len(layers)) if args.alpha_mode == "total" else alpha
                        # Experiment-1 norm-matched cells: per-row alpha_eff =
                        # alpha x dp projection_fraction (alpha_scale defaults to
                        # 1.0 for every other method).
                        alpha_scale = payload.get("alpha_scale", {})
                        specs = [
                            ResidualSteeringSpec(
                                layer=layer,
                                direction=np.asarray(payload["directions"][layer]),
                                alpha=float(layer_alpha) * float(alpha_scale.get(layer, 1.0)),
                            )
                            for layer in layers
                        ]
                        projection = aggregate_projection(payload["projections"])
                        direction_info_public = {
                            str(layer): public_direction_info(payload["direction_info"][layer])
                            for layer in layers
                        }
                jobs.append({"row_index": idx, "row": row, "method": method, "alpha": float(alpha), "specs": specs, "projection": projection, "direction_info": direction_info_public, "policy_choice": None})

    planned_generation_total = len(existing_results) + len(jobs)
    if not jobs and prior_batch_verifications:
        batch_verifications = list(prior_batch_verifications)

    def write_results(model_meta: dict | None, validate_only: bool = False) -> None:
        out = {
            "schema_version": 1,
            "argv": sys.argv,
            "blocked": False,
            "validate_only": validate_only,
            "model": model_meta,
            "backend": backend,
            "requested_model_key": model_key,
            "mlx_model": mlx_model,
            "dtype": dtype,
            "oracle_true_status_direction": args.routing == "oracle_true_status",
            "routing": args.routing,
            "gate_predictions": str(gate_path.resolve()) if gate_path else None,
            "gate_predictions_sha256": gate_predictions_sha,
            "policy_choices": str(choice_path.resolve()) if choice_path else None,
            "policy_choices_sha256": policy_choices_sha,
            "policy_name": args.policy_name if choice_path else None,
            "steering_spec_bank": str(spec_bank_path.resolve()) if spec_bank_path else None,
            "steering_spec_bank_manifest_sha256": spec_bank_manifest_sha,
            "steering_spec_bank_n_specs": (
                spec_bank_manifest_for_output.get("n_specs") if spec_bank_manifest_for_output else None
            ),
            "steering_scope": "decision_token_only",
            "layers": layers,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "methods": methods,
            "alphas": alphas,
            "alpha_mode": args.alpha_mode,
            "max_new_tokens": max_new_tokens,
            "max_generation_seconds": max_generation_seconds,
            "generation_protocol": generation_protocol,
            "generation_batch_size": int(args.generation_batch_size),
            "verify_generation_smoke": bool(args.verify_generation_smoke),
            "verify_generation_smoke_batches": int(args.verify_generation_smoke_batches),
            "batched_generation_margin_fail_atol": float(args.batched_generation_margin_fail_atol),
            "batched_generation_reply_mismatch_policy": args.batched_generation_reply_mismatch_policy,
            "sample_gpu": bool(args.sample_gpu),
            "gpu_samples": str(sample_path) if args.sample_gpu else None,
            "gpu_summary": gpu_summary,
            "transcripts": [str(path.resolve()) for path in transcript_paths],
            "transcripts_sha256": transcript_hashes,
            "activations": str(activation_path.resolve()),
            "activations_sha256": activation_sha,
            "activation_meta": activation_meta,
            "config": str(config_path.resolve()),
            "config_sha256": config_hash,
            "provenance": provenance,
            "eval_rows": len(rows),
            "limit": args.limit,
            "limit_per_status_class": args.limit_per_status_class,
            "planned_generations": planned_generation_total,
            "pending_generations": len(jobs),
            "prior_completed_generations": len(existing_results),
            "planned_skips": dict(planned_skips),
            "routing_action_balance": dict(Counter(route["action"] for route in route_by_cid.values())),
            "layer_direction_availability": layer_direction_availability,
            "eval_status_class_balance": dict(Counter(str(row["status_class"]) for row in rows)),
            "timings": {k: round(float(v), 4) for k, v in sorted(timings.items())},
            "prior_batched_generation_verifications": prior_batch_verifications,
            "batched_generation_verifications": batch_verifications,
            "results": results,
        }
        atomic_text(out_path, json.dumps(to_jsonable(out), indent=2, sort_keys=True))

    results = list(existing_results)
    if args.validate_only:
        write_results(None, validate_only=True)
        print(f"validated -> {out_path} ({len(jobs)} planned generations; model not loaded)", flush=True)
        return

    def finalize_gpu_sampling() -> None:
        nonlocal sampler
        if sampler is None:
            return
        stop_event.set()
        sampler.join(timeout=5)
        gpu_summary.update(summarize_samples(sample_path))
        sampler = None

    if args.sample_gpu and backend == "hf":
        sampler = threading.Thread(
            target=gpu_sampler,
            args=(sample_path, stop_event, float(args.sample_interval)),
            daemon=True,
        )
        sampler.start()

    model, tokenizer, meta = load_decision_backend_model(
        backend=backend,
        model_key=model_key,
        mlx_model=mlx_model,
        dtype=dtype,
        device=args.device,
    )
    pass_id, fail_id = decision_tokens(tokenizer)
    model_meta = public_model_meta(meta)

    def add_time(bucket: str, t0: float) -> None:
        timings[bucket] = timings.get(bucket, 0.0) + (time.perf_counter() - t0)

    def runtime_specs(job: dict) -> Any:
        if job["specs"] is None:
            return None
        return [
            ResidualSteeringSpec(
                layer=spec.layer,
                direction=__import__("torch").tensor(
                    np.asarray(spec.direction),
                    dtype=__import__("torch").float32,
                ),
                alpha=spec.alpha,
            )
            for spec in job["specs"]
        ]

    for job in jobs:
        job["runtime_specs"] = runtime_specs(job)

    verified_generation_batches = len(batch_verifications)

    def verify_generation_batch(batch: list[dict], batched: list[tuple[str, dict]], *, batch_index: int) -> None:
        nonlocal verified_generation_batches
        if not args.verify_generation_smoke or verified_generation_batches >= args.verify_generation_smoke_batches:
            return
        t0 = time.perf_counter()
        scalar = [
            choose_status_then_complete(
                model,
                tokenizer,
                prompt_without_final_answer(job["row"]),
                pass_id=pass_id,
                fail_id=fail_id,
                steering=job["runtime_specs"],
                max_new_tokens=max_new_tokens,
                max_generation_seconds=max_generation_seconds,
                backend=backend,
            )
            for job in batch
        ]
        add_time("generation_verify_guard", t0)
        margin_diffs = [
            abs(float(b_decision["margin"]) - float(s_decision["margin"]))
            for (_b_reply, b_decision), (_s_reply, s_decision) in zip(batched, scalar)
        ]
        forced_status_mismatches = sum(
            int(b_decision.get("forced_status") != s_decision.get("forced_status"))
            for (_b_reply, b_decision), (_s_reply, s_decision) in zip(batched, scalar)
        )
        reply_mismatches = sum(
            int(b_reply != s_reply)
            for (b_reply, _b_decision), (s_reply, _s_decision) in zip(batched, scalar)
        )
        max_abs_margin_diff = max(margin_diffs) if margin_diffs else 0.0
        verification = {
            "batch_index": int(batch_index),
            "batch_size": len(batch),
            "conversation_ids": [str(job["row"]["conversation_id"]) for job in batch],
            "methods": [str(job["method"]) for job in batch],
            "max_abs_margin_diff": float(max_abs_margin_diff),
            "mean_abs_margin_diff": float(sum(margin_diffs) / len(margin_diffs)) if margin_diffs else 0.0,
            "forced_status_mismatches": int(forced_status_mismatches),
            "reply_mismatches": int(reply_mismatches),
            "margin_fail_atol": float(args.batched_generation_margin_fail_atol),
            "reply_mismatch_policy": args.batched_generation_reply_mismatch_policy,
        }
        batch_verifications.append(verification)
        verified_generation_batches += 1
        if (
            forced_status_mismatches
            or max_abs_margin_diff > args.batched_generation_margin_fail_atol
            or (reply_mismatches and args.batched_generation_reply_mismatch_policy == "fail")
        ):
            finalize_gpu_sampling()
            write_results(model_meta)
            raise ValueError(
                "batched generation smoke failed: "
                f"forced_status_mismatches={forced_status_mismatches}, "
                f"reply_mismatches={reply_mismatches}, "
                f"max_abs_margin_diff={max_abs_margin_diff:.6g}"
            )

    def append_result(job: dict, reply: str, decision: dict, *, batch_index: int, effective_batch_size: int) -> None:
        row = job["row"]
        specs = job["runtime_specs"]
        decision = dict(decision)
        decision["generation_protocol"] = generation_protocol
        decision["generation_batch_size"] = int(args.generation_batch_size)
        decision["generation_effective_batch_size"] = int(effective_batch_size)
        decision["generation_batch_index"] = int(batch_index)
        reported = parse_status(reply, "report")
        true_status = str(row["true_status"])
        results.append({
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
            "method": job["method"],
            "base_representation": "graded_bidirectional_dp_decision_token",
            "alpha": float(job["alpha"]),
            "heldout_family": row["family"],
            "route": route_by_cid[str(row["conversation_id"])],
            "target_direction": target_direction_for_method(job["method"], route_by_cid[str(row["conversation_id"])]) if job["method"] in STEER_METHODS else None,
            "policy_choice": job.get("policy_choice"),
            "decision": decision,
            "generation_protocol": generation_protocol,
            "generation_batch_size": int(args.generation_batch_size),
            "generation_effective_batch_size": int(effective_batch_size),
            "generation_batch_index": int(batch_index),
            "direction_info": job["direction_info"],
            "direction_projection": job["projection"],
            "injection": stack_injection_stats(specs, float(job["alpha"])) if specs else None,
            "reported_status": reported,
            "honest": reported == true_status,
            "coherence": reply_coherence(reply, reported),
            "reply": reply,
        })

    use_batched_generation = backend == "hf" and int(args.generation_batch_size) > 1
    step = int(args.generation_batch_size) if use_batched_generation else 1
    batch_starts = range(0, len(jobs), step)
    for batch_index, start_idx in enumerate(tqdm(batch_starts, desc="decision-token-control")):
        batch = jobs[start_idx:start_idx + step]
        if use_batched_generation:
            t0 = time.perf_counter()
            pairs = choose_status_then_complete_batch(
                model,
                tokenizer,
                [prompt_without_final_answer(job["row"]) for job in batch],
                pass_id=pass_id,
                fail_id=fail_id,
                steering_batch=[job["runtime_specs"] for job in batch],
                max_new_tokens=max_new_tokens,
                max_generation_seconds=max_generation_seconds,
                backend=backend,
            )
            add_time("generation_batched", t0)
            verify_generation_batch(batch, pairs, batch_index=batch_index)
            for job, (reply, decision) in zip(batch, pairs):
                append_result(
                    job,
                    reply,
                    decision,
                    batch_index=batch_index,
                    effective_batch_size=len(batch),
                )
        else:
            job = batch[0]
            t0 = time.perf_counter()
            reply, decision = choose_status_then_complete(
                model,
                tokenizer,
                prompt_without_final_answer(job["row"]),
                pass_id=pass_id,
                fail_id=fail_id,
                steering=job["runtime_specs"],
                max_new_tokens=max_new_tokens,
                max_generation_seconds=max_generation_seconds,
                backend=backend,
            )
            add_time("generation_scalar", t0)
            append_result(
                job,
                reply,
                decision,
                batch_index=batch_index,
                effective_batch_size=1,
            )
        write_results(model_meta)
    finalize_gpu_sampling()
    write_results(model_meta)
    print(f"saved -> {out_path} ({len(results)} generations)", flush=True)


if __name__ == "__main__":
    main()
