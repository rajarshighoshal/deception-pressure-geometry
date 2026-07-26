from __future__ import annotations

from geoprobe.control.literature_baselines import (
    SteeringBaselineFamily,
    evaluate_train_selected_method_family,
    exact_iti_status,
    filter_candidate_methods,
)


def row(
    cid: str,
    family: str,
    status_class: str,
    route: str,
    method: str,
    target: str | None,
    reward: float,
) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": cid.split(":")[0],
        "family": family,
        "status_class": status_class,
        "route_action": route,
        "method": method,
        "target_status": target,
        "layer": None if method == "abstain" else 16,
        "alpha": 0.0 if method == "abstain" else 24.0,
        "reward": reward,
        "fixes_error": status_class.startswith("false_") and reward > 0,
        "harms_honest": status_class.startswith("honest_") and reward < 0,
    }


def candidate_rows(cid: str, family: str, status_class: str, route: str, target: str) -> list[dict]:
    return [
        row(cid, family, status_class, route, "abstain", None, 0.0),
        row(cid, family, status_class, route, "global_mean", target, 1.0),
        row(cid, family, status_class, route, "global_probe", target, -1.0),
        row(cid, family, status_class, route, "bidir_linear", target, 0.25),
    ]


def test_train_selected_family_uses_only_allowed_methods() -> None:
    rows = []
    rows.extend(candidate_rows("a:false", "famA", "false_FAIL", "steer_to_PASS", "PASS"))
    rows.extend(candidate_rows("b:honest", "famB", "honest_FAIL", "steer_to_PASS", "PASS"))
    family = SteeringBaselineFamily(
        name="caa_global_mean",
        methods=("global_mean",),
        literature_role="CAA-style",
        exactness="direct",
        note="test",
    )
    policy = evaluate_train_selected_method_family(rows, folds=[["famA"], ["famB"]], family=family)
    methods = {choice["method"] for choice in policy["choices"]}
    assert methods == {"global_mean"}
    assert policy["summary"]["n"] == 2


def test_filter_candidate_methods_keeps_abstain() -> None:
    rows = candidate_rows("a:false", "famA", "false_FAIL", "steer_to_PASS", "PASS")
    kept = filter_candidate_methods(rows, {"global_probe"})
    assert {row["method"] for row in kept} == {"abstain", "global_probe"}


def test_exact_iti_status_is_explicitly_not_claimed() -> None:
    status = exact_iti_status()
    assert status["implementable_from_current_artifact"] is False
    assert status["current_proxy"] == "repe_probe_residual"
    assert "attention-head" in status["reason"]
