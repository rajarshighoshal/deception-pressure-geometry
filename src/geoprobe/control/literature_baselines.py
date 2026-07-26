"""Standard activation-steering baselines over an action-response field."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from geoprobe.control.action_response import (
    TARGET_STATUSES,
    action_key,
    best_key_for_target,
    choose_key,
    grouped_by_conversation,
    summarize_choices,
    target_from_route,
)


@dataclass(frozen=True)
class SteeringBaselineFamily:
    name: str
    methods: tuple[str, ...]
    literature_role: str
    exactness: str
    note: str


DEFAULT_BASELINE_FAMILIES: tuple[SteeringBaselineFamily, ...] = (
    SteeringBaselineFamily(
        name="caa_route_mean",
        methods=("bidir_linear",),
        literature_role="CAA-style route-conditioned mean-difference residual steering",
        exactness="direct_residual_stream_baseline",
        note="Route-specific contrastive residual direction; train selects layer/alpha per target.",
    ),
    SteeringBaselineFamily(
        name="caa_global_mean",
        methods=("global_mean",),
        literature_role="CAA-style global mean-difference residual steering",
        exactness="direct_residual_stream_baseline",
        note="Single global contrastive direction signed by target; train selects layer/alpha per target.",
    ),
    SteeringBaselineFamily(
        name="repe_probe_residual",
        methods=("global_probe",),
        literature_role="RepE/probe-style residual intervention",
        exactness="residual_stream_proxy",
        note="Logistic-probe residual direction signed by target; train selects layer/alpha per target.",
    ),
    SteeringBaselineFamily(
        name="standard_steering_train_best",
        methods=("bidir_linear", "global_mean", "global_probe"),
        literature_role="train-selected standard activation-steering floor",
        exactness="direct_residual_stream_baseline",
        note="Train-fold best among CAA-like route/global means and probe/RepE-style residual steering.",
    ),
)


def families_by_name(
    names: Iterable[str] | None = None,
    *,
    available: tuple[SteeringBaselineFamily, ...] = DEFAULT_BASELINE_FAMILIES,
) -> list[SteeringBaselineFamily]:
    by_name = {family.name: family for family in available}
    if names is None:
        return list(available)
    out = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"unknown literature baseline family {name!r}; expected one of {sorted(by_name)}")
        out.append(by_name[name])
    return out


def filter_candidate_methods(rows: list[dict], methods: Iterable[str]) -> list[dict]:
    method_set = set(methods)
    return [
        row for row in rows
        if str(row.get("method")) in method_set or str(row.get("method")) == "abstain"
    ]


def evaluate_train_selected_method_family(
    rows: list[dict],
    *,
    folds: list[list[str]],
    family: SteeringBaselineFamily,
    objective: str = "reward",
) -> dict:
    grouped = grouped_by_conversation(rows)
    choices: list[dict] = []
    fold_out: dict[str, dict] = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [
            row for row in filter_candidate_methods(rows, family.methods)
            if str(row["family"]) not in heldout_set
        ]
        best_by_target = {
            target: best_key_for_target(train, target, objective=objective)
            for target in TARGET_STATUSES
        }
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) not in heldout_set:
                continue
            target = target_from_route(str(candidates[0].get("route_action")))
            key = best_by_target.get(target)
            if key is not None and str(key[0]) not in set(family.methods):
                key = None
            fold_choices.append(choose_key(candidates, key))
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": list(heldout),
            "best_by_target": {
                target: list(key) if key is not None else None
                for target, key in best_by_target.items()
            },
            "summary": summarize_choices(fold_choices),
        }
    return {
        "family": family.name,
        "literature_role": family.literature_role,
        "exactness": family.exactness,
        "methods": list(family.methods),
        "note": family.note,
        "summary": summarize_choices(choices),
        "folds": fold_out,
        "choices": choices,
    }


def evaluate_literature_steering_baselines(
    rows: list[dict],
    *,
    folds: list[list[str]],
    families: Iterable[SteeringBaselineFamily] = DEFAULT_BASELINE_FAMILIES,
    objective: str = "reward",
) -> dict[str, dict]:
    return {
        family.name: evaluate_train_selected_method_family(
            rows,
            folds=folds,
            family=family,
            objective=objective,
        )
        for family in families
    }


def exact_iti_status() -> dict:
    return {
        "implementable_from_current_artifact": False,
        "reason": (
            "The powered150 activation/action-response artifacts store residual-stream states and residual "
            "steering vectors, not attention-head activations or head-specific interventions."
        ),
        "current_proxy": "repe_probe_residual",
        "required_for_exact_iti": [
            "attention-head activation capture or hook support",
            "head-level direction fitting",
            "head-level intervention path in the generation/scoring runner",
        ],
    }


def compact_policy_choices(policy: dict) -> list[dict]:
    out = []
    for row in policy.get("choices", []):
        out.append({
            "conversation_id": row.get("conversation_id"),
            "scenario_id": row.get("scenario_id"),
            "family": row.get("family"),
            "status_class": row.get("status_class"),
            "route_action": row.get("route_action"),
            "method": row.get("method"),
            "target_status": row.get("target_status"),
            "layer": row.get("layer"),
            "alpha": row.get("alpha"),
            "action_key": list(action_key(row)) if str(row.get("method")) != "abstain" else None,
            "fixes_error": row.get("fixes_error"),
            "harms_honest": row.get("harms_honest"),
            "reward": row.get("reward"),
        })
    return out
