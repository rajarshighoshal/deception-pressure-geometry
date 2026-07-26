"""Direction-fitting for PASS/FAIL status control: status error classes, per-family status
directions, and their summaries. Extracted from bidirectional_dp_diagnostic (Phase 3)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from geoprobe.geometry.numerics import clean_matrix
from geoprobe.geometry.tangent import fit_dp_direction, unit, vector_stats


def direction_cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    au = unit(a)
    bu = unit(b)
    if float(np.linalg.norm(au)) <= 1e-8 or float(np.linalg.norm(bu)) <= 1e-8:
        return None
    return float(np.dot(au, bu))


def status_error_class(true_status: str, reported_status: str) -> str | None:
    true_status = str(true_status).upper()
    reported_status = str(reported_status).upper()
    if true_status not in {"PASS", "FAIL"} or reported_status not in {"PASS", "FAIL"}:
        return None
    if true_status == "PASS" and reported_status == "PASS":
        return "honest_PASS"
    if true_status == "FAIL" and reported_status == "FAIL":
        return "honest_FAIL"
    if true_status == "PASS" and reported_status == "FAIL":
        return "false_FAIL"
    if true_status == "FAIL" and reported_status == "PASS":
        return "false_PASS"
    return None


def attach_status_classes(points: list[dict], transcripts: dict[str, dict]) -> tuple[list[dict], Counter]:
    out: list[dict] = []
    skipped: Counter = Counter()
    for point in points:
        cid = str(point["conversation_id"])
        transcript = transcripts.get(cid)
        if transcript is None:
            skipped["missing_transcript"] += 1
            continue
        if not bool(transcript.get("valid_outcome", False)):
            skipped["invalid_transcript"] += 1
            continue
        reported = transcript.get("reported_status")
        true_status = transcript.get("true_status", point.get("true_status"))
        cls = status_error_class(str(true_status), str(reported))
        if cls is None:
            skipped["bad_status"] += 1
            continue
        row = dict(point)
        row["reported_status"] = str(reported).upper()
        row["true_status"] = str(true_status).upper()
        row["status_class"] = cls
        row["transcript_deceptive"] = bool(transcript.get("deceptive", cls.startswith("false_")))
        out.append(row)
    return out, skipped


def fit_status_direction(
    rows: list[dict],
    *,
    heldout_family: str | None = None,
    heldout_scenario_ids: set[str] | None = None,
    direction_levels: set[str],
    target_status: str,
    min_mixed_scenarios: int,
    min_levels: int,
    equivariant: bool = False,
) -> dict | None:
    """Fit one correction direction within scenario-level pairs.

    ``target_status=PASS`` fits honest_PASS - false_FAIL.
    ``target_status=FAIL`` fits honest_FAIL - false_PASS.

    Excludes rows matching ``heldout_family`` (cross-family split) or
    ``heldout_scenario_ids`` (within-scenario split). If both are None, holds
    out nothing (diagnostic mode).

    If ``equivariant=True``, pool the Z₂ partner direction (PASS↔FAIL swap)
    with sign flipped. For target=PASS this adds -(honest_FAIL - false_PASS)
    = (false_PASS - honest_FAIL) to each level's diff list, doubling effective
    data and enforcing the antisymmetry d_PASS ≡ -d_FAIL.
    """
    if target_status == "PASS":
        honest_class = "honest_PASS"
        false_class = "false_FAIL"
        partner_honest = "honest_FAIL"
        partner_false = "false_PASS"
    elif target_status == "FAIL":
        honest_class = "honest_FAIL"
        false_class = "false_PASS"
        partner_honest = "honest_PASS"
        partner_false = "false_FAIL"
    else:
        raise ValueError("target_status must be PASS or FAIL")

    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    by_level_scenario: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        by_level_scenario[(str(row["arm"]), str(row["scenario_id"]))].append(row)

    level_diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    scenario_counts: dict[str, list[dict]] = defaultdict(list)
    for (level, scenario_id), group in sorted(by_level_scenario.items()):
        honest = [row for row in group if row["status_class"] == honest_class]
        false = [row for row in group if row["status_class"] == false_class]
        if honest and false:
            x_h = clean_matrix(np.vstack([row["x"] for row in honest]))
            x_f = clean_matrix(np.vstack([row["x"] for row in false]))
            diff = x_h.mean(axis=0) - x_f.mean(axis=0)
            level_diffs[level].append(diff)
            scenario_counts[level].append({
                "scenario_id": scenario_id,
                "n_honest": int(len(honest)),
                "n_false": int(len(false)),
                "source": "primary",
            })
        if equivariant:
            p_honest = [row for row in group if row["status_class"] == partner_honest]
            p_false = [row for row in group if row["status_class"] == partner_false]
            if p_honest and p_false:
                x_ph = clean_matrix(np.vstack([row["x"] for row in p_honest]))
                x_pf = clean_matrix(np.vstack([row["x"] for row in p_false]))
                partner_diff = x_ph.mean(axis=0) - x_pf.mean(axis=0)
                level_diffs[level].append(-partner_diff)
                scenario_counts[level].append({
                    "scenario_id": scenario_id,
                    "n_honest": int(len(p_honest)),
                    "n_false": int(len(p_false)),
                    "source": "z2_partner",
                })

    usable_levels = [
        level for level in sorted(level_diffs)
        if len(level_diffs[level]) >= min_mixed_scenarios
    ]
    if len(usable_levels) < min_levels:
        return None

    level_means = [clean_matrix(np.vstack(level_diffs[level])).mean(axis=0) for level in usable_levels]
    direction = unit(clean_matrix(np.vstack(level_means)).mean(axis=0))
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None

    return {
        "heldout_family": heldout_family,
        "target_status": target_status,
        "honest_class": honest_class,
        "false_class": false_class,
        "direction_convention": (
            f"direction = mean({honest_class}) - mean({false_class})"
            + (" [Z₂-equivariant: pooled with -partner]" if equivariant else "")
        ),
        "direction_levels": usable_levels,
        "n_train_points": int(len(train)),
        "n_mixed_scenario_level_pairs": int(sum(len(level_diffs[level]) for level in usable_levels)),
        "mixed_scenarios_by_level": {
            level: scenario_counts[level][:100]
            for level in usable_levels
        },
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def summarize_family_directions(
    rows: list[dict],
    *,
    direction_levels: set[str],
    min_mixed_scenarios: int,
    min_levels: int,
) -> dict:
    families = sorted({str(row["family"]) for row in rows})
    per_family: dict[str, dict] = {}
    pass_dirs: list[np.ndarray] = []
    fail_dirs: list[np.ndarray] = []
    pooled_dirs: list[np.ndarray] = []
    cos_pass_fail: list[float] = []
    cos_pooled_pass: list[float] = []
    cos_pooled_fail: list[float] = []

    for family in families:
        pass_info = fit_status_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            target_status="PASS",
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )
        fail_info = fit_status_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            target_status="FAIL",
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )
        pooled = fit_dp_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )

        pass_vec = None if pass_info is None else pass_info["_direction_np"]
        fail_vec = None if fail_info is None else fail_info["_direction_np"]
        pooled_vec = None if pooled is None else pooled["_direction_np"]

        pf = direction_cos(pass_vec, fail_vec)
        pp = direction_cos(pooled_vec, pass_vec)
        pfi = direction_cos(pooled_vec, fail_vec)
        if pass_vec is not None:
            pass_dirs.append(pass_vec)
        if fail_vec is not None:
            fail_dirs.append(fail_vec)
        if pooled_vec is not None:
            pooled_dirs.append(pooled_vec)
        if pf is not None:
            cos_pass_fail.append(pf)
        if pp is not None:
            cos_pooled_pass.append(pp)
        if pfi is not None:
            cos_pooled_fail.append(pfi)

        per_family[family] = {
            "to_PASS_available": pass_info is not None,
            "to_FAIL_available": fail_info is not None,
            "pooled_available": pooled is not None,
            "cos_to_PASS_vs_to_FAIL": pf,
            "cos_pooled_vs_to_PASS": pp,
            "cos_pooled_vs_to_FAIL": pfi,
            "to_PASS": pass_info,
            "to_FAIL": fail_info,
            "pooled": pooled,
        }

    def stats(values: list[float]) -> dict | None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return None
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    global_pass = unit(clean_matrix(np.vstack(pass_dirs)).mean(axis=0)) if pass_dirs else None
    global_fail = unit(clean_matrix(np.vstack(fail_dirs)).mean(axis=0)) if fail_dirs else None
    global_pooled = unit(clean_matrix(np.vstack(pooled_dirs)).mean(axis=0)) if pooled_dirs else None

    return {
        "families": families,
        "n_families": int(len(families)),
        "n_to_PASS_available": int(sum(1 for payload in per_family.values() if payload["to_PASS_available"])),
        "n_to_FAIL_available": int(sum(1 for payload in per_family.values() if payload["to_FAIL_available"])),
        "n_pooled_available": int(sum(1 for payload in per_family.values() if payload["pooled_available"])),
        "cos_to_PASS_vs_to_FAIL": stats(cos_pass_fail),
        "cos_pooled_vs_to_PASS": stats(cos_pooled_pass),
        "cos_pooled_vs_to_FAIL": stats(cos_pooled_fail),
        "global_cos_to_PASS_vs_to_FAIL": direction_cos(global_pass, global_fail),
        "global_cos_pooled_vs_to_PASS": direction_cos(global_pooled, global_pass),
        "global_cos_pooled_vs_to_FAIL": direction_cos(global_pooled, global_fail),
        "per_family": per_family,
    }


# --- tomography direction-fitting. Uses its own L2 normalize (_unit_l2), distinct from the
# max-abs-stabilized geometry.tangent.unit that the status-direction fns above use. ---


def _unit_l2(vec: np.ndarray) -> np.ndarray:
    vec = clean_matrix(np.asarray(vec, dtype=np.float64))
    n = float(np.linalg.norm(vec))
    return vec / n if np.isfinite(n) and n > 1e-12 else np.zeros_like(vec)


def error_class(true_status: str, deceptive: bool) -> str:
    """Derive the report error type from ground truth + the deceptive flag (no reported_status needed)."""
    true_status = str(true_status).upper()
    if not deceptive:
        return "honest_PASS" if true_status == "PASS" else "honest_FAIL"
    # deceptive => reported the opposite of true
    return "false_FAIL" if true_status == "PASS" else "false_PASS"


def fit_layer_family_directions(points: list[dict], heldout_family: str, levels: set[str]) -> dict:
    """d_p / to_PASS / to_FAIL mean-difference directions at one layer, train-family only."""
    train = [r for r in points if str(r["family"]) != heldout_family and r["arm"] in levels]
    by = defaultdict(list)
    for r in train:
        by[error_class(r["true_status"], bool(r["label"]))].append(np.asarray(r["x"], dtype=np.float64))
    def mean(cls):
        return np.mean(np.vstack(by[cls]), axis=0) if by.get(cls) else None
    honest = [v for c in ("honest_PASS", "honest_FAIL") for v in by.get(c, [])]
    decept = [v for c in ("false_FAIL", "false_PASS") for v in by.get(c, [])]
    out = {"counts": {c: len(by.get(c, [])) for c in ("honest_PASS", "honest_FAIL", "false_FAIL", "false_PASS")}}
    out["d_p"] = _unit_l2(np.mean(np.vstack(honest), 0) - np.mean(np.vstack(decept), 0)) if honest and decept else None
    hp, ff = mean("honest_PASS"), mean("false_FAIL")
    hf, fp = mean("honest_FAIL"), mean("false_PASS")
    to_pass_raw = hp - ff if hp is not None and ff is not None else None
    to_fail_raw = hf - fp if hf is not None and fp is not None else None
    out["to_PASS"] = _unit_l2(to_pass_raw) if to_pass_raw is not None else None
    out["to_FAIL"] = _unit_l2(to_fail_raw) if to_fail_raw is not None else None
    out["_d_p_raw"] = (np.mean(np.vstack(honest), 0) - np.mean(np.vstack(decept), 0)) if honest and decept else None
    out["_to_PASS_raw"] = to_pass_raw
    out["_to_FAIL_raw"] = to_fail_raw
    return out


def logit_derived_direction(model, pass_id: int, fail_id: int, hidden: int) -> np.ndarray | None:
    """unembed[PASS]-unembed[FAIL] in hidden space. A quantized head stores packed (non-hidden-dim)
    weights, so dequantize; return None if a hidden-dim unembedding can't be recovered."""
    import mlx.core as mx
    head = getattr(model, "lm_head", None) or model.model.embed_tokens
    w = head.weight
    if w.shape[-1] != hidden and hasattr(head, "scales"):
        w = mx.dequantize(head.weight, head.scales, head.biases,
                          group_size=getattr(head, "group_size", 64), bits=getattr(head, "bits", 4))
    if w.shape[-1] != hidden:
        return None
    diff = (w[pass_id] - w[fail_id]).astype(mx.float32)
    mx.eval(diff)
    return _unit_l2(np.array(diff, dtype=np.float64))


__all__ = ["status_error_class", "attach_status_classes", "fit_status_direction",
           "summarize_family_directions", "error_class", "fit_layer_family_directions",
           "logit_derived_direction"]
