"""Raw-logit response diagnostics for the pre-status causal replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
import json
from pathlib import Path
from typing import Any

import numpy as np

from geoprobe.control.relational_pre_status_causal import CAUSAL_ARM_ORDER
from geoprobe.eval.relational_post_commitment_transport_metrics import (
    scenario_cluster_bootstrap_ci,
)
from geoprobe.eval.relational_pre_status_causal_report import (
    validate_causal_runner_artifacts,
    validate_relational_pre_status_causal_report,
)
from geoprobe.io import file_sha256
from geoprobe.provenance import git_provenance


SCHEMA_VERSION = 1
REPORT_KIND = "relational_pre_status_causal_response_diagnostic_report_v1"
REPORT_JSON_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "report.md"
REPO_ROOT = Path(__file__).resolve().parents[3]
# The registered diagnostic-protocol document is privately retained with the
# program's run ledgers; it is pinned by the SHA-256 below and is not part of
# the public tree, so no file on disk is re-hashed at report-build time.
DIAGNOSTIC_PROTOCOL_DOC_SHA256 = "3b3d3d7d90b05f6a09add19485f5307b170919064dc86b14fa9ec698146052bd"

DEFAULT_INPUT_RUN_DIR = (
    REPO_ROOT / "results/relational_geometry/pre_status_causal_replay_a100_v3_20260721T212949Z"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/relational_geometry/pre_status_causal_response_diagnostic_v1_20260722"
)
DEFAULT_RUNNER_JSONL = DEFAULT_INPUT_RUN_DIR / "causal_rows.jsonl"
DEFAULT_SOURCE_MANIFEST = DEFAULT_INPUT_RUN_DIR / "causal_manifest.json"
DEFAULT_REGISTERED_CAUSAL_REPORT = (
    REPO_ROOT
    / "results/relational_geometry/pre_status_causal_effects_recovered_v1_20260722/report.json"
)

DEFAULT_RUNNER_JSONL_SHA256 = "4addc698c9817ca9339f6364fc7a94c5bba17e81243b40acf91b01ae0f1e7afe"
DEFAULT_SOURCE_MANIFEST_SHA256 = "172b083e73925c76abb4c3ae9835bd8d1a5169615229c4281e2027265b8e77e9"
DEFAULT_REGISTERED_CAUSAL_REPORT_SHA256 = (
    "bd56b141b10f41d15457896301c633bc74daba450967f6beac807c36d28c082a"
)
DEFAULT_REGISTERED_CAUSAL_REPORT_INTERNAL_SHA256 = (
    "930823be2af08e430ad34dc82004d42bffee50ab1d6e996e8dcb377564dbc241"
)

EXPECTED_RUNNER_ROW_COUNT = 4_592
BOOTSTRAP_SEED = 20_260_722
BOOTSTRAP_RESAMPLES = 2_000

MEASURE_NAMES = (
    "full_reach",
    "generic_reach",
    "specific_origin",
    "specific_after_generic",
    "signed_s_secant",
    "even_curvature",
    "ts_interaction",
    "matched_random_contrast",
    "local_full_minus_global",
)

_OPPOSITE_STATUS = {"PASS": "FAIL", "FAIL": "PASS"}


class RelationalPreStatusCausalResponseError(ValueError):
    """Raised when the response diagnostic input contract is violated."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusCausalResponseError(
            "value is not finite canonical JSON"
        ) from error


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RelationalPreStatusCausalResponseError(f"{label} must be a lowercase SHA-256")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RelationalPreStatusCausalResponseError(f"{label} must be a lowercase SHA-256")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RelationalPreStatusCausalResponseError(f"{label} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise RelationalPreStatusCausalResponseError(f"{label} must be finite")
    return number


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPreStatusCausalResponseError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusCausalResponseError(f"{label} must be an object")
    return value


def _raw_margin(row: Mapping[str, Any]) -> float:
    logits = _mapping(row["status_logits"], "status_logits")
    true_status = _string(row["true_status"], "true_status")
    if true_status not in _OPPOSITE_STATUS:
        raise RelationalPreStatusCausalResponseError("true_status must be PASS or FAIL")
    opposite = _OPPOSITE_STATUS[true_status]
    return _number(logits[true_status], "true_status logit") - _number(
        logits[opposite], "opposite status logit"
    )


def _row_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "state_id",
        "family",
        "family_fold",
        "turn_index",
        "scenario_id",
        "true_status",
        "desired_status",
        "knowledge_status",
        "knowledge_correct",
        "status_logits",
        "prefix",
        "beta",
        "actuation_layer",
        "actuation_vector_sha256",
        "hook_layers",
        "vector_tensor_row_index",
        "vector_source_hashes",
        "arm_vector_sha256",
        "bundle_hash",
        "bundle_metadata_hash",
        "arm_vector_l2_norm",
        "model",
        "runtime",
    )
    return {name: row[name] for name in fields}


def _collapse_rows_by_root(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_by_root: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        rows_by_root[str(row["root_id"])][str(row["arm"])].append(row)

    root_blocks: list[dict[str, Any]] = []
    event_seed_counts: dict[str, int] = {}

    for root_id, rows_by_arm in sorted(rows_by_root.items()):
        if set(rows_by_arm) != set(CAUSAL_ARM_ORDER):
            raise RelationalPreStatusCausalResponseError(
                f"root {root_id} does not expose all causal arms"
            )
        seed_counts = {arm: len(values) for arm, values in rows_by_arm.items()}
        if len(set(seed_counts.values())) != 1:
            raise RelationalPreStatusCausalResponseError(
                f"root {root_id} has inconsistent seed replication across arms"
            )
        seed_count = next(iter(seed_counts.values()))
        event_seed_counts[root_id] = seed_count

        representative_by_arm: dict[str, Mapping[str, Any]] = {}
        for arm in CAUSAL_ARM_ORDER:
            records = rows_by_arm[arm]
            base = _row_signature(records[0])
            for record in records[1:]:
                if _row_signature(record) != base:
                    raise RelationalPreStatusCausalResponseError(
                        f"root {root_id}, arm {arm} records differ across event seeds"
                    )
            representative_by_arm[arm] = records[0]

        for arm in CAUSAL_ARM_ORDER[1:]:
            for name in (
                "state_id",
                "family",
                "family_fold",
                "turn_index",
                "scenario_id",
                "true_status",
                "desired_status",
                "knowledge_status",
                "knowledge_correct",
            ):
                if (
                    representative_by_arm[arm][name]
                    != representative_by_arm[CAUSAL_ARM_ORDER[0]][name]
                ):
                    raise RelationalPreStatusCausalResponseError(
                        f"root {root_id} has arm-wise changes to root invariants"
                    )
            if representative_by_arm[arm]["root_id"] != root_id:  # defensive, already keyed
                raise RelationalPreStatusCausalResponseError(
                    f"root {root_id} row arm {arm} has mismatched root_id"
                )

        root_reference_row = representative_by_arm[CAUSAL_ARM_ORDER[0]]

        m_noop = _raw_margin(representative_by_arm["noop"])
        m_generic_t = _raw_margin(representative_by_arm["generic_t"])
        m_specific_s = _raw_margin(representative_by_arm["specific_s"])
        m_full_h = _raw_margin(representative_by_arm["full_h"])
        m_fixed_global_h = _raw_margin(representative_by_arm["fixed_global_h"])
        m_generic_minus_s = _raw_margin(representative_by_arm["generic_minus_s"])
        m_generic_plus_random_s = _raw_margin(representative_by_arm["generic_plus_random_s"])

        root_blocks.append(
            {
                "root_id": root_id,
                "state_id": _string(root_reference_row["state_id"], "state_id"),
                "family": _string(root_reference_row["family"], "family"),
                "family_fold": _string(root_reference_row["family_fold"], "family_fold"),
                "turn_index": int(root_reference_row["turn_index"]),
                "scenario_id": _string(root_reference_row["scenario_id"], "scenario_id"),
                "knowledge_correct": bool(root_reference_row["knowledge_correct"]),
                "event_seed_count": int(seed_count),
                "m_noop": m_noop,
                "m_generic_t": m_generic_t,
                "m_specific_s": m_specific_s,
                "m_full_h": m_full_h,
                "m_fixed_global_h": m_fixed_global_h,
                "m_generic_minus_s": m_generic_minus_s,
                "m_generic_plus_random_s": m_generic_plus_random_s,
            }
        )

        record = root_blocks[-1]
        record["full_reach"] = record["m_full_h"] - record["m_noop"]
        record["generic_reach"] = record["m_generic_t"] - record["m_noop"]
        record["specific_origin"] = record["m_specific_s"] - record["m_noop"]
        record["specific_after_generic"] = record["m_full_h"] - record["m_generic_t"]
        record["signed_s_secant"] = (record["m_full_h"] - record["m_generic_minus_s"]) / 2.0
        record["even_curvature"] = (
            record["m_full_h"] - 2.0 * record["m_generic_t"] + record["m_generic_minus_s"]
        )
        record["ts_interaction"] = (
            record["m_full_h"] - record["m_generic_t"] - record["m_specific_s"] + record["m_noop"]
        )
        record["matched_random_contrast"] = record["m_full_h"] - record["m_generic_plus_random_s"]
        record["local_full_minus_global"] = record["m_full_h"] - record["m_fixed_global_h"]

    return root_blocks, event_seed_counts


def _scalar_summary(values: Sequence[float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "positive_fraction": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "q10": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    q10, q25, q50, q75, q90 = np.quantile(array, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    return {
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
    }


def _measure_summary(
    rows: Sequence[Mapping[str, Any]],
    measure_name: str,
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    values = [_number(row[measure_name], measure_name) for row in rows]
    summary = _scalar_summary(values)
    if values:
        bootstrap_rows = [
            {
                "scenario_id": str(row["scenario_id"]),
                "difference": float(row[measure_name]),
            }
            for row in rows
        ]
        bootstrap = scenario_cluster_bootstrap_ci(
            bootstrap_rows,
            seed=seed,
            resamples=resamples,
            confidence=confidence,
        )
        summary["bootstrap"] = {
            "method": bootstrap.get("method", "paired scenario-cluster bootstrap percentile"),
            "seed": bootstrap["seed"],
            "resamples": bootstrap["resamples"],
            "confidence": bootstrap["confidence"],
            "point": bootstrap["point"],
            "lower": float(bootstrap["interval"][0]),
            "upper": float(bootstrap["interval"][1]),
        }
    else:
        summary["bootstrap"] = {
            "method": "paired scenario-cluster bootstrap percentile",
            "seed": int(seed),
            "resamples": int(resamples),
            "confidence": confidence,
            "point": 0.0,
            "lower": 0.0,
            "upper": 0.0,
        }
    return summary


def _boundary_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "m_noop_quantiles": {"q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0},
            "m_generic_t_quantiles": {
                "q10": 0.0,
                "q25": 0.0,
                "q50": 0.0,
                "q75": 0.0,
                "q90": 0.0,
            },
            "m_full_h_quantiles": {
                "q10": 0.0,
                "q25": 0.0,
                "q50": 0.0,
                "q75": 0.0,
                "q90": 0.0,
            },
            "negative_count": 0,
            "negative_fraction": 0.0,
            "negative_crossings": {
                "full_h": {"count": 0, "fraction": 0.0},
                "generic_t": {"count": 0, "fraction": 0.0},
                "fixed_global_h": {"count": 0, "fraction": 0.0},
            },
            "intervention_increases": {
                "full_h": {"count": 0, "fraction": 0.0},
                "generic_t": {"count": 0, "fraction": 0.0},
                "fixed_global_h": {"count": 0, "fraction": 0.0},
            },
        }

    m_noop = [_number(row["m_noop"], "m_noop") for row in rows]
    m_generic_t = [_number(row["m_generic_t"], "m_generic_t") for row in rows]
    m_full_h = [_number(row["m_full_h"], "m_full_h") for row in rows]
    negatives = [row for row in rows if _number(row["m_noop"], "m_noop") < 0.0]
    negative_count = len(negatives)
    negative_cross_full = sum(1 for row in negatives if row["m_full_h"] >= 0.0)
    negative_cross_generic = sum(1 for row in negatives if row["m_generic_t"] >= 0.0)
    negative_cross_global = sum(1 for row in negatives if row["m_fixed_global_h"] >= 0.0)

    inc_full = sum(1 for row in rows if row["m_full_h"] > row["m_noop"])
    inc_generic = sum(1 for row in rows if row["m_generic_t"] > row["m_noop"])
    inc_global = sum(1 for row in rows if row["m_fixed_global_h"] > row["m_noop"])

    row_count = len(rows)
    return {
        "count": row_count,
        "m_noop_quantiles": _quantiles(m_noop),
        "m_generic_t_quantiles": _quantiles(m_generic_t),
        "m_full_h_quantiles": _quantiles(m_full_h),
        "negative_count": negative_count,
        "negative_fraction": negative_count / row_count,
        "negative_crossings": {
            "full_h": {
                "count": negative_cross_full,
                "fraction": negative_cross_full / negative_count if negative_count else 0.0,
            },
            "generic_t": {
                "count": negative_cross_generic,
                "fraction": negative_cross_generic / negative_count if negative_count else 0.0,
            },
            "fixed_global_h": {
                "count": negative_cross_global,
                "fraction": negative_cross_global / negative_count if negative_count else 0.0,
            },
        },
        "intervention_increases": {
            "full_h": {"count": inc_full, "fraction": inc_full / row_count},
            "generic_t": {
                "count": inc_generic,
                "fraction": inc_generic / row_count,
            },
            "fixed_global_h": {
                "count": inc_global,
                "fraction": inc_global / row_count,
            },
        },
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)

    def summarize_bucket(bucket_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "root_count": len(bucket_rows),
            "full_reach": _scalar_summary(
                [_number(r["full_reach"], "full_reach") for r in bucket_rows]
            ),
            "generic_reach": _scalar_summary(
                [_number(r["generic_reach"], "generic_reach") for r in bucket_rows]
            ),
            "m_noop": _scalar_summary([_number(r["m_noop"], "m_noop") for r in bucket_rows]),
            "m_generic_t": _scalar_summary(
                [_number(r["m_generic_t"], "m_generic_t") for r in bucket_rows]
            ),
            "m_full_h": _scalar_summary([_number(r["m_full_h"], "m_full_h") for r in bucket_rows]),
        }

    return {name: summarize_bucket(bucket) for name, bucket in sorted(buckets.items())}


def _interpretation(population: Mapping[str, Any]) -> dict[str, Any]:
    measures = population["measures"]
    boundary = population["boundary_summary"]

    full_reach_lb = measures["full_reach"]["bootstrap"]["lower"]
    generic_reach_lb = measures["generic_reach"]["bootstrap"]["lower"]
    specific_after_generic_lb = measures["specific_after_generic"]["bootstrap"]["lower"]
    signed_s_lb = measures["signed_s_secant"]["bootstrap"]["lower"]
    matched_random_lb = measures["matched_random_contrast"]["bootstrap"]["lower"]
    ts_interaction_lb = measures["ts_interaction"]["bootstrap"]["lower"]

    conditional_structured = (
        specific_after_generic_lb > 0.0 and signed_s_lb > 0.0 and matched_random_lb > 0.0
    )
    base_point_dependence = conditional_structured and ts_interaction_lb > 0.0

    full_reach_present = full_reach_lb > 0.0
    if not full_reach_present:
        supported = "no_detectable_raw_margin_signal"
    elif conditional_structured and base_point_dependence:
        supported = "structured_S_response_with_base_point_dependence"
    elif conditional_structured:
        supported = "structured_S_response_without_base_point_dependence"
    elif generic_reach_lb > 0.0:
        supported = "generic_transport_only"
    else:
        supported = "inconclusive_raw_margin_signature"

    crossing_rate = boundary["negative_crossings"]["full_h"]["fraction"]
    specific_origin_lower = measures["specific_origin"]["bootstrap"]["lower"]
    specific_origin_upper = measures["specific_origin"]["bootstrap"]["upper"]
    if specific_origin_lower > 0.0:
        specific_origin_assessment = "detectably_positive"
    elif specific_origin_upper < 0.0:
        specific_origin_assessment = "detectably_harmful"
    else:
        specific_origin_assessment = "not_detectably_positive"

    return {
        "population": "knowledge_correct",
        "supported_object": supported,
        "conditional_structured_S_response": bool(conditional_structured),
        "base_point_dependence": bool(base_point_dependence),
        "specific_origin_not_standalone": bool(specific_origin_lower <= 0.0),
        "specific_origin_assessment": specific_origin_assessment,
        "generic_transport_moved_margin": bool(generic_reach_lb > 0.0),
        "full_reach_present": bool(full_reach_present),
        "boundary_crossing_fraction": float(crossing_rate),
        "boundary_strength_assessment": "descriptive_only_no_frozen_numeric_threshold",
        "full_reach_confidence_interval": {
            "lower": float(full_reach_lb),
            "upper": float(measures["full_reach"]["bootstrap"]["upper"]),
        },
        "conditional_structured_bounds": {
            "specific_after_generic_lower": float(specific_after_generic_lb),
            "signed_s_secant_lower": float(signed_s_lb),
            "matched_random_contrast_lower": float(matched_random_lb),
            "ts_interaction_lower": float(ts_interaction_lb),
        },
    }


def build_relational_pre_status_causal_response_report(
    runner_jsonl: Path,
    source_manifest: Path,
    *,
    expected_runner_jsonl_sha256: str | None = DEFAULT_RUNNER_JSONL_SHA256,
    expected_source_manifest_sha256: str | None = DEFAULT_SOURCE_MANIFEST_SHA256,
    expected_registered_causal_report_sha256: str | None = DEFAULT_REGISTERED_CAUSAL_REPORT_SHA256,
    expected_registered_causal_report_internal_sha256: str
    | None = DEFAULT_REGISTERED_CAUSAL_REPORT_INTERNAL_SHA256,
    expected_runner_row_count: int | None = EXPECTED_RUNNER_ROW_COUNT,
    registered_causal_report_path: Path | None = DEFAULT_REGISTERED_CAUSAL_REPORT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    argv: Sequence[str] = (),
    extra_source_paths: Sequence[Path] = (),
    allow_unfrozen_test_inputs: bool = False,
) -> dict[str, Any]:
    if not allow_unfrozen_test_inputs:
        frozen_contract = {
            "runner_jsonl": (runner_jsonl.resolve(), DEFAULT_RUNNER_JSONL.resolve()),
            "source_manifest": (
                source_manifest.resolve(),
                DEFAULT_SOURCE_MANIFEST.resolve(),
            ),
            "registered_causal_report_path": (
                registered_causal_report_path.resolve()
                if registered_causal_report_path is not None
                else None,
                DEFAULT_REGISTERED_CAUSAL_REPORT.resolve(),
            ),
            "runner_jsonl_sha256": (
                expected_runner_jsonl_sha256,
                DEFAULT_RUNNER_JSONL_SHA256,
            ),
            "source_manifest_sha256": (
                expected_source_manifest_sha256,
                DEFAULT_SOURCE_MANIFEST_SHA256,
            ),
            "registered_causal_report_sha256": (
                expected_registered_causal_report_sha256,
                DEFAULT_REGISTERED_CAUSAL_REPORT_SHA256,
            ),
            "registered_causal_report_internal_sha256": (
                expected_registered_causal_report_internal_sha256,
                DEFAULT_REGISTERED_CAUSAL_REPORT_INTERNAL_SHA256,
            ),
            "runner_row_count": (
                expected_runner_row_count,
                EXPECTED_RUNNER_ROW_COUNT,
            ),
            "bootstrap_seed": (bootstrap_seed, BOOTSTRAP_SEED),
            "bootstrap_resamples": (bootstrap_resamples, BOOTSTRAP_RESAMPLES),
        }
        changed = [
            name for name, (actual, expected) in frozen_contract.items() if actual != expected
        ]
        if changed:
            raise RelationalPreStatusCausalResponseError(
                "frozen diagnostic contract changed: " + ", ".join(changed)
            )

    if expected_runner_jsonl_sha256 is not None:
        if (
            _sha256(file_sha256(runner_jsonl), "runner_jsonl sha256")
            != expected_runner_jsonl_sha256
        ):
            raise RelationalPreStatusCausalResponseError(
                "runner jsonl hash is not the frozen value"
            )
    if expected_source_manifest_sha256 is not None:
        if (
            _sha256(file_sha256(source_manifest), "source manifest sha256")
            != expected_source_manifest_sha256
        ):
            raise RelationalPreStatusCausalResponseError("manifest hash is not the frozen value")

    if bootstrap_seed < 0:
        raise RelationalPreStatusCausalResponseError("bootstrap_seed must be nonnegative")
    if bootstrap_resamples <= 0:
        raise RelationalPreStatusCausalResponseError("bootstrap_resamples must be positive")

    validated = validate_causal_runner_artifacts(
        runner_jsonl,
        source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_runner_row_count=expected_runner_row_count,
    )

    root_rows, event_seed_counts = _collapse_rows_by_root(validated["rows"])

    populations = {
        "knowledge_correct": [row for row in root_rows if row["knowledge_correct"]],
        "all_roots": root_rows,
        "knowledge_error": [row for row in root_rows if not row["knowledge_correct"]],
    }

    def build_population(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        measures: dict[str, Any] = {
            name: _measure_summary(
                rows,
                name,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            for name in MEASURE_NAMES
        }
        return {
            "root_count": len(rows),
            "event_seed_count": int(sum(int(row["event_seed_count"]) for row in rows)),
            "measures": measures,
            "boundary_summary": _boundary_summary(rows),
            "scenario_count": len({str(row["scenario_id"]) for row in rows}),
            "family_fold_summary": _group_summary(rows, "family_fold"),
            "turn_summary": _group_summary(rows, "turn_index"),
            "margin_samples": {
                "m_noop": _scalar_summary(_number(row["m_noop"], "m_noop") for row in rows),
                "m_generic_t": _scalar_summary(
                    _number(row["m_generic_t"], "m_generic_t") for row in rows
                ),
                "m_full_h": _scalar_summary(_number(row["m_full_h"], "m_full_h") for row in rows),
            },
            "m_noop_quantiles": _quantiles([row["m_noop"] for row in rows]),
            "m_generic_t_quantiles": _quantiles([row["m_generic_t"] for row in rows]),
            "m_full_h_quantiles": _quantiles([row["m_full_h"] for row in rows]),
        }

    population_reports = {name: build_population(rows) for name, rows in populations.items()}
    population_reports["knowledge_correct"]["interpretation"] = _interpretation(
        population_reports["knowledge_correct"]
    )

    inputs = dict(validated["inputs"])
    # The diagnostic-protocol document is privately retained; the frozen
    # SHA-256 pin is recorded archivally instead of re-hashing a file on disk.
    inputs["diagnostic_protocol"] = {
        "sha256": DIAGNOSTIC_PROTOCOL_DOC_SHA256,
        "expected_sha256": DIAGNOSTIC_PROTOCOL_DOC_SHA256,
        "source": "privately_retained_registration_document",
    }

    registered_causal_report: Mapping[str, Any] | None = None
    if registered_causal_report_path is not None:
        registered_causal_report_path = registered_causal_report_path.resolve()
        physical_sha = file_sha256(registered_causal_report_path)
        if (
            expected_registered_causal_report_sha256 is not None
            and _sha256(physical_sha, "registered causal report SHA-256")
            != expected_registered_causal_report_sha256
        ):
            raise RelationalPreStatusCausalResponseError(
                "registered causal report physical SHA-256 does not match frozen value"
            )
        try:
            registered_causal_report = json.loads(
                registered_causal_report_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RelationalPreStatusCausalResponseError(
                "registered causal report is unreadable"
            ) from error
        validate_relational_pre_status_causal_report(registered_causal_report)
        internal_sha = _sha256(
            registered_causal_report.get("report_sha256"),
            "registered causal report internal SHA-256",
        )
        if (
            expected_registered_causal_report_internal_sha256 is not None
            and internal_sha != expected_registered_causal_report_internal_sha256
        ):
            raise RelationalPreStatusCausalResponseError(
                "registered causal report internal SHA-256 does not match frozen value"
            )
        registered_inputs = _mapping(
            registered_causal_report.get("inputs"), "registered causal report inputs"
        )
        if _mapping(registered_inputs.get("runner_jsonl"), "registered runner input").get(
            "sha256"
        ) != file_sha256(runner_jsonl) or _mapping(
            registered_inputs.get("source_manifest"),
            "registered source-manifest input",
        ).get("sha256") != file_sha256(source_manifest):
            raise RelationalPreStatusCausalResponseError(
                "registered causal report does not bind the diagnostic runner inputs"
            )
        inputs["registered_causal_report"] = {
            "path": str(registered_causal_report_path),
            "sha256": physical_sha,
            "report_sha256": internal_sha,
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "success",
        "argv": list(argv),
        "provenance": git_provenance(
            [
                Path(__file__).resolve(),
                *map(Path, extra_source_paths),
            ]
        ),
        "inputs": inputs,
        "assumptions": {
            "analysis_mode": "pre-status raw margin finite-response diagnostic",
            "bootstrap": {
                "seed": bootstrap_seed,
                "resamples": bootstrap_resamples,
                "confidence": 0.95,
                "method": "paired scenario-cluster bootstrap percentile",
            },
            "measurement_scheme": {
                "raw_logit_margin": "l_true - l_false where true is PASS/FAIL status",
                "margin_measures": {
                    "full_reach": "m(H) - m(0)",
                    "generic_reach": "m(T) - m(0)",
                    "specific_origin": "m(S) - m(0)",
                    "specific_after_generic": "m(H) - m(T)",
                    "signed_s_secant": "(m(H) - m(T-S)) / 2",
                    "even_curvature": "m(H) - 2m(T) + m(T-S)",
                    "ts_interaction": "m(H) - m(T) - m(S) + m(0)",
                    "matched_random_contrast": "m(H) - m(T+R)",
                    "local_full_minus_global": "m(H) - m(G)",
                },
                "weighting": "root-balanced after seed deduplication",
            },
            "population_contracts": (
                "knowledge_correct, all_roots, knowledge_error (knowledge_correct is root-level)"
            ),
        },
        "coverage": {
            "root_count": len(root_rows),
            "runner_row_count": int(validated["row_count"]),
            "manifest_expected_row_count": int(validated["manifest"]["expected_row_count"]),
            "manifest_completed_row_count": int(validated["manifest"]["completed_rows"]),
            "scenario_count": len({row["scenario_id"] for row in root_rows}),
            "family_count": len({row["family"] for row in root_rows}),
            "family_fold_count": len({row["family_fold"] for row in root_rows}),
            "turn_count": len({row["turn_index"] for row in root_rows}),
            "event_seed_count_mean": float(np.mean(list(event_seed_counts.values())))
            if event_seed_counts
            else 0.0,
            "event_seed_count_min": int(min(event_seed_counts.values()))
            if event_seed_counts
            else 0,
            "event_seed_count_max": int(max(event_seed_counts.values()))
            if event_seed_counts
            else 0,
            "input_hashes": validated["inputs"]["input_hashes"],
        },
        "populations": population_reports,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def validate_relational_pre_status_causal_response_report(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RelationalPreStatusCausalResponseError("schema_version must be 1")
    if payload.get("kind") != REPORT_KIND:
        raise RelationalPreStatusCausalResponseError("kind is invalid")
    if payload.get("status") != "success":
        raise RelationalPreStatusCausalResponseError("status is not success")
    expected = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    if _string(payload.get("report_sha256"), "report_sha256") != expected:
        raise RelationalPreStatusCausalResponseError("report_sha256 is invalid")

    populations = _mapping(payload.get("populations"), "populations")
    if set(populations) != {"knowledge_correct", "all_roots", "knowledge_error"}:
        raise RelationalPreStatusCausalResponseError("population set is invalid")
    for name in ("knowledge_correct", "all_roots", "knowledge_error"):
        population = populations.get(name)
        if not isinstance(population, Mapping):
            raise RelationalPreStatusCausalResponseError(f"population {name} is missing or invalid")
        measures = population.get("measures")
        if not isinstance(measures, Mapping):
            raise RelationalPreStatusCausalResponseError(f"population {name} is missing measures")
        for measure in MEASURE_NAMES:
            if measure not in measures:
                raise RelationalPreStatusCausalResponseError(
                    f"population {name} is missing measure {measure}"
                )
    if int(populations["knowledge_correct"].get("root_count", -1)) + int(
        populations["knowledge_error"].get("root_count", -1)
    ) != int(populations["all_roots"].get("root_count", -1)):
        raise RelationalPreStatusCausalResponseError("population root counts do not reconcile")

    inputs = _mapping(payload.get("inputs"), "inputs")
    diagnostic_protocol = _mapping(inputs.get("diagnostic_protocol"), "inputs.diagnostic_protocol")
    if diagnostic_protocol.get("sha256") != DIAGNOSTIC_PROTOCOL_DOC_SHA256:
        raise RelationalPreStatusCausalResponseError("diagnostic protocol binding is invalid")


def render_relational_pre_status_causal_response_report(report: Mapping[str, Any]) -> str:
    validate_relational_pre_status_causal_response_report(report)
    interpretation = report["populations"]["knowledge_correct"]["interpretation"]

    kc_population = report["populations"]["knowledge_correct"]
    kc_measures = kc_population["measures"]
    kc_boundary = kc_population["boundary_summary"]
    lines = [
        "# Pre-status causal response diagnostic (v1)",
        "",
        f"Status: {report['status']}",
        "",
        "## Interpretation",
        f"- Supported object: `{interpretation['supported_object']}`",
        f"- Conditional structured-S response: `{interpretation['conditional_structured_S_response']}`",
        f"- Base-point dependence: `{interpretation['base_point_dependence']}`",
        f"- Full reach present: `{interpretation['full_reach_present']}`",
        f"- Boundary crossing from negative margin (`m(0)<0`) under full intervention: `{interpretation['boundary_crossing_fraction']:.4f}`",
        "",
        "## Population summary",
        "| population | roots | full_reach mean | full_reach 95% CI | generic_reach mean | specific_after_generic mean | specific-origin mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, population in sorted(report["populations"].items()):
        full_reach = population["measures"]["full_reach"]
        lines.append(
            f"| {name} | {population['root_count']} | {full_reach['mean']:.4f} | "
            f"[{full_reach['bootstrap']['lower']:.4f}, {full_reach['bootstrap']['upper']:.4f}] | "
            f"{population['measures']['generic_reach']['mean']:.4f} | "
            f"{population['measures']['specific_after_generic']['mean']:.4f} | "
            f"{population['measures']['specific_origin']['mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Boundary reachability (knowledge-correct roots)",
            f"- Negative-m0 roots: `{kc_boundary['negative_count']}` / `{kc_boundary['count']}`",
            f"- Full intervention crossing fraction: `{kc_boundary['negative_crossings']['full_h']['fraction']:.4f}`",
            f"- Generic transport crossing fraction: `{kc_boundary['negative_crossings']['generic_t']['fraction']:.4f}`",
            f"- Fixed-global crossing fraction: `{kc_boundary['negative_crossings']['fixed_global_h']['fraction']:.4f}`",
            f"- Full intervention increase fraction: `{kc_boundary['intervention_increases']['full_h']['fraction']:.4f}`",
            f"- Generic intervention increase fraction: `{kc_boundary['intervention_increases']['generic_t']['fraction']:.4f}`",
            f"- Fixed-global intervention increase fraction: `{kc_boundary['intervention_increases']['fixed_global_h']['fraction']:.4f}`",
            "",
            "Registered sampled-action result is unchanged: the full field did not reliably beat the controls behaviorally.",
        ]
    )
    lines.extend(
        [
            "",
            "## Knowledge-correct measure diagnostics (all 9 measures)",
            "| measure | count | mean | median | positive fraction | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for measure_name in MEASURE_NAMES:
        measure = kc_measures[measure_name]
        lines.append(
            f"| {measure_name} | {measure['count']} | "
            f"{measure['mean']:.4f} | "
            f"{measure['median']:.4f} | "
            f"{measure['positive_fraction']:.4f} | "
            f"[{measure['bootstrap']['lower']:.4f}, {measure['bootstrap']['upper']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Conditional / signed / random / interaction intervals (95% CI, knowledge-correct)",
            "| statistic | lower | upper |",
            "| --- | ---: | ---: |",
            f"| specific_after_generic (conditional lift) | {kc_measures['specific_after_generic']['bootstrap']['lower']:.4f} | {kc_measures['specific_after_generic']['bootstrap']['upper']:.4f} |",
            f"| signed_s_secant (structured-sign contrast) | {kc_measures['signed_s_secant']['bootstrap']['lower']:.4f} | {kc_measures['signed_s_secant']['bootstrap']['upper']:.4f} |",
            f"| matched_random_contrast (random-control contrast) | {kc_measures['matched_random_contrast']['bootstrap']['lower']:.4f} | {kc_measures['matched_random_contrast']['bootstrap']['upper']:.4f} |",
            f"| ts_interaction (interaction term) | {kc_measures['ts_interaction']['bootstrap']['lower']:.4f} | {kc_measures['ts_interaction']['bootstrap']['upper']:.4f} |",
            "",
            "## Raw margin anchor quantiles (knowledge-correct roots)",
            "| anchor | mean | q10 | q25 | q50 | q75 | q90 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| m_noop | {kc_population['margin_samples']['m_noop']['mean']:.4f} | "
            f"{kc_population['m_noop_quantiles']['q10']:.4f} | "
            f"{kc_population['m_noop_quantiles']['q25']:.4f} | "
            f"{kc_population['m_noop_quantiles']['q50']:.4f} | "
            f"{kc_population['m_noop_quantiles']['q75']:.4f} | "
            f"{kc_population['m_noop_quantiles']['q90']:.4f} |",
            f"| m_generic_t | {kc_population['margin_samples']['m_generic_t']['mean']:.4f} | "
            f"{kc_population['m_generic_t_quantiles']['q10']:.4f} | "
            f"{kc_population['m_generic_t_quantiles']['q25']:.4f} | "
            f"{kc_population['m_generic_t_quantiles']['q50']:.4f} | "
            f"{kc_population['m_generic_t_quantiles']['q75']:.4f} | "
            f"{kc_population['m_generic_t_quantiles']['q90']:.4f} |",
            f"| m_full_h | {kc_population['margin_samples']['m_full_h']['mean']:.4f} | "
            f"{kc_population['m_full_h_quantiles']['q10']:.4f} | "
            f"{kc_population['m_full_h_quantiles']['q25']:.4f} | "
            f"{kc_population['m_full_h_quantiles']['q50']:.4f} | "
            f"{kc_population['m_full_h_quantiles']['q75']:.4f} | "
            f"{kc_population['m_full_h_quantiles']['q90']:.4f} |",
            "",
            "## Boundary and intervention crossing counts (knowledge-correct)",
            "| transition | negative crossing count | negative crossing fraction | increase count | increase fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| full_h | {kc_boundary['negative_crossings']['full_h']['count']} | {kc_boundary['negative_crossings']['full_h']['fraction']:.4f} | {kc_boundary['intervention_increases']['full_h']['count']} | {kc_boundary['intervention_increases']['full_h']['fraction']:.4f} |",
            f"| generic_t | {kc_boundary['negative_crossings']['generic_t']['count']} | {kc_boundary['negative_crossings']['generic_t']['fraction']:.4f} | {kc_boundary['intervention_increases']['generic_t']['count']} | {kc_boundary['intervention_increases']['generic_t']['fraction']:.4f} |",
            f"| fixed_global_h | {kc_boundary['negative_crossings']['fixed_global_h']['count']} | {kc_boundary['negative_crossings']['fixed_global_h']['fraction']:.4f} | {kc_boundary['intervention_increases']['fixed_global_h']['count']} | {kc_boundary['intervention_increases']['fixed_global_h']['fraction']:.4f} |",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DIAGNOSTIC_PROTOCOL_DOC_SHA256",
    "DEFAULT_INPUT_RUN_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REGISTERED_CAUSAL_REPORT",
    "DEFAULT_REGISTERED_CAUSAL_REPORT_INTERNAL_SHA256",
    "DEFAULT_REGISTERED_CAUSAL_REPORT_SHA256",
    "DEFAULT_RUNNER_JSONL",
    "DEFAULT_RUNNER_JSONL_SHA256",
    "DEFAULT_SOURCE_MANIFEST",
    "DEFAULT_SOURCE_MANIFEST_SHA256",
    "EXPECTED_RUNNER_ROW_COUNT",
    "MEASURE_NAMES",
    "REPORT_JSON_FILENAME",
    "REPORT_KIND",
    "REPORT_MARKDOWN_FILENAME",
    "RelationalPreStatusCausalResponseError",
    "build_relational_pre_status_causal_response_report",
    "render_relational_pre_status_causal_response_report",
    "validate_relational_pre_status_causal_response_report",
]
