"""Rung-4 holonomy + curvature measurement on the v2 substrate (registered instrument).

Registered in the privately retained results ledger of the program (stage-2 rung registration,
2026-07-23) BEFORE any rung-4 number was computed. This CLI loads the persisted
per-connection SO(3)
transports (no recomputation), enumerates plaquettes, measures gauge-invariant
loop angles, runs the three registered null models, evaluates the adequacy gate
and three-way verdict, and writes the citable artifact. Neutral vocabulary only.

Frozen constants are echoed in the report and must match the registration doc.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from geoprobe.io import write_json  # noqa: E402
from geoprobe.eval.relational_gauge_controller_artifact import (  # noqa: E402
    load_fold_gauge_controller_artifact,
)
from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (  # noqa: E402
    FOLDS,
    load_relational_pre_status_rooted_graph_artifacts,
)
from geoprobe.geometry.relational_gauge_atlas import RelationalGaugeAtlas  # noqa: E402
from geoprobe.geometry.relational_holonomy_curvature import (  # noqa: E402
    NEAR_PI_EXCLUSION_RAD,
    atlas_transport_map,
    build_chart_overlap_graph,
    edge_residual_angles,
    edge_shuffle_null,
    enumerate_quads,
    enumerate_triangles,
    haar_so_null,
    holonomy_group_diagnostics,
    loop_angles_over_transports,
    plaquette_curvature,
    residual_matched_null,
    spanning_tree_gauge_fix,
)
from geoprobe.provenance import git_provenance  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (must match the privately retained registration ledger exactly)
# ---------------------------------------------------------------------------
SUBSTRATE_VIEW = "intervention_masked_action_free"
QUAD_CAP = 20_000
QUAD_SEED = 20_260_723
NULL_SEEDS = tuple(range(1, 101))  # 100 seeded replicates, seeds 1..100
THETA_MIN = 0.1  # rad; adequacy gate threshold
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_723
SVD_RETAIN_RATIO = 0.1
NEAR_PI_RAD = NEAR_PI_EXCLUSION_RAD  # 0.05 rad; loops near pi excluded from logs

VERDICT_CURVATURE = "curvature-found"
VERDICT_FLAT = "flat-under-adequate-instrument"
VERDICT_NOT_FOUND = "not-found-under-this-instrument"


def _center_node_triple(atlas: RelationalGaugeAtlas, cycle: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(str(atlas.get_chart(c).center_node_id) for c in cycle))


def _mean_chart_stress(atlas: RelationalGaugeAtlas, cycle: tuple[str, ...]) -> float:
    return float(np.mean([atlas.get_chart(c).stress for c in cycle]))


def _null_median_angles(
    transports: dict[tuple[str, str], np.ndarray],
    loops: list[tuple[str, ...]],
    null_factory,
    residual_angles: dict[tuple[str, str], float] | None,
    seeds: tuple[int, ...],
) -> np.ndarray:
    """Median plaquette angle per null seed (vectorized loop-angle computation)."""
    medians = np.empty(len(seeds), dtype=np.float64)
    for idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        if residual_angles is not None:
            null_map = null_factory(transports, residual_angles, rng)
        else:
            null_map = null_factory(transports, rng)
        angles = loop_angles_over_transports(null_map, loops)
        medians[idx] = float(np.median(angles))
    return medians


def _bootstrap_corr_ci(
    x: np.ndarray, y: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float]:
    """Pearson r and bootstrap CI (percentile method)."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.integers(0, n, size=n)
        sx, sy = x[idx], y[idx]
        if np.std(sx) > 0 and np.std(sy) > 0:
            samples[d] = np.corrcoef(sx, sy)[0, 1]
        else:
            samples[d] = 0.0
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return r, float(lo), float(hi)


def _measure_fold(
    fold: str,
    atlas: RelationalGaugeAtlas,
) -> dict:
    rng_quad = np.random.default_rng(QUAD_SEED)
    adjacency = build_chart_overlap_graph(atlas.connections)
    n_charts = len(adjacency)
    n_edges = sum(len(v) for v in adjacency.values()) // 2

    triangles = enumerate_triangles(adjacency)
    quads = enumerate_quads(adjacency, cap=QUAD_CAP, rng=rng_quad)
    tri_loops = [tuple(t) for t in triangles]
    quad_loops = [tuple(q) for q in quads]
    all_loops = tri_loops + quad_loops

    # --- measured plaquette records (full fields for jsonl + covariates) ---
    records = []
    for loop in all_loops:
        rec = plaquette_curvature(atlas, loop)
        records.append(
            {
                "fold": fold,
                "loop_type": "triangle" if len(loop) == 3 else "quad",
                "cycle": list(rec.cycle),
                "center_node_triple": list(_center_node_triple(atlas, rec.cycle)),
                "angle": rec.angle,
                "frobenius_defect": rec.frobenius_defect,
                "spectral_defect": rec.spectral_defect,
                "loop_residual_sum": rec.loop_residual_sum,
                "min_overlap": rec.min_overlap,
                "mean_chart_stress": _mean_chart_stress(atlas, rec.cycle),
            }
        )
    measured_angles = np.array([r["angle"] for r in records])
    measured_median = float(np.median(measured_angles))

    # --- null models (vectorized; only need median per seed) ---
    transports = atlas_transport_map(atlas)
    res_angles = edge_residual_angles(atlas)

    n1_medians = _null_median_angles(transports, all_loops, haar_so_null, None, NULL_SEEDS)
    n2_medians = _null_median_angles(
        transports, all_loops, edge_shuffle_null, None, NULL_SEEDS
    )
    n3_medians = _null_median_angles(
        transports, all_loops, residual_matched_null, res_angles, NULL_SEEDS
    )

    n3_p95 = float(np.percentile(n3_medians, 95))
    adequate = n3_p95 < THETA_MIN
    n3_median = float(np.median(n3_medians))
    n3_lo, n3_hi = np.percentile(n3_medians, [2.5, 97.5])

    # --- holonomy-group diagnostics (BFS gauge fix at highest-degree chart) ---
    highest_deg = max(adjacency, key=lambda c: len(adjacency[c]))
    fix = spanning_tree_gauge_fix(atlas, highest_deg)
    generators = list(fix.generators.values())
    closure_eps = max(2.0 * n3_median, 1e-6)
    group_diag = holonomy_group_diagnostics(
        generators,
        closure_epsilon=closure_eps,
        rng=np.random.default_rng(QUAD_SEED),
        svd_retain_ratio=SVD_RETAIN_RATIO,
    )

    # --- covariate correlations (reported, never a gate) ---
    loop_res = np.array([r["loop_residual_sum"] for r in records])
    stress = np.array([r["mean_chart_stress"] for r in records])
    corr_res = float(np.corrcoef(measured_angles, loop_res)[0, 1]) if np.std(loop_res) > 0 else 0.0
    corr_stress = float(np.corrcoef(measured_angles, stress)[0, 1]) if np.std(stress) > 0 else 0.0

    return {
        "fold": fold,
        "n_charts": n_charts,
        "n_edges": n_edges,
        "n_triangles": len(triangles),
        "n_quads": len(quads),
        "measured": {
            "median_angle": measured_median,
            "mean_angle": float(np.mean(measured_angles)),
            "n_loops": len(records),
        },
        "nulls": {
            "N1_haar": {
                "median": float(np.median(n1_medians)),
                "p2_5": float(np.percentile(n1_medians, 2.5)),
                "p97_5": float(np.percentile(n1_medians, 97.5)),
            },
            "N2_edge_shuffle": {
                "median": float(np.median(n2_medians)),
                "p2_5": float(np.percentile(n2_medians, 2.5)),
                "p97_5": float(np.percentile(n2_medians, 97.5)),
            },
            "N3_residual_matched": {
                "median": n3_median,
                "p2_5": float(n3_lo),
                "p97_5": float(n3_hi),
                "p95": n3_p95,
            },
        },
        "adequacy_gate": {
            "n3_p95_median_angle": n3_p95,
            "theta_min": THETA_MIN,
            "adequate": adequate,
        },
        "holonomy_group": {
            "base_chart": highest_deg,
            "n_generators": len(generators),
            "span_dimension": group_diag.span_dimension,
            "singular_values": list(group_diag.singular_values),
            "n_excluded_near_pi": group_diag.n_excluded_near_pi,
            "closure_fraction": group_diag.closure_fraction,
            "closure_epsilon": closure_eps,
            "bootstrap_modal_fraction": group_diag.bootstrap_modal_fraction,
        },
        "covariate_correlation": {
            "theta_vs_loop_residual": corr_res,
            "theta_vs_mean_stress": corr_stress,
        },
        "_plaquette_records": records,
    }


def _cross_fold_correlation(per_fold: list[dict]) -> dict:
    """Shared-plaquette angle correlation across folds (matched center-node triples)."""
    fold_maps: list[dict[tuple[str, ...], float]] = []
    for fd in per_fold:
        mapping: dict[tuple[str, ...], float] = {}
        for rec in fd["_plaquette_records"]:
            key = tuple(rec["center_node_triple"])
            mapping[key] = rec["angle"]
        fold_maps.append(mapping)

    # pool all pairwise (fold_i, fold_j) angle values for shared plaquettes
    xs: list[float] = []
    ys: list[float] = []
    for i in range(len(fold_maps)):
        for j in range(i + 1, len(fold_maps)):
            shared = set(fold_maps[i]) & set(fold_maps[j])
            for key in shared:
                xs.append(fold_maps[i][key])
                ys.append(fold_maps[j][key])
    if len(xs) < 3:
        return {
            "n_shared_plaquettes": len(xs),
            "correlation": float("nan"),
            "bootstrap_ci": [float("nan"), float("nan")],
            "ci_excludes_zero": False,
        }
    r, lo, hi = _bootstrap_corr_ci(
        np.array(xs), np.array(ys), BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
    )
    return {
        "n_shared_plaquettes": len(xs),
        "correlation": r,
        "bootstrap_ci": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def _three_way_verdict(per_fold: list[dict], cross_fold: dict) -> tuple[str, str]:
    all_adequate = all(fd["adequacy_gate"]["adequate"] for fd in per_fold)
    n_adequate = sum(fd["adequacy_gate"]["adequate"] for fd in per_fold)

    # Hard gate (registered): if the instrument is inadequate, the verdict is
    # always not-found-under-this-instrument — full stop. No curvature or flat
    # verdict is evaluated when the noise floor cannot resolve theta_min.
    if not all_adequate:
        return VERDICT_NOT_FOUND, (
            f"adequacy gate FAILED ({n_adequate}/5 folds adequate; "
            f"N3 p95 must be < {THETA_MIN} rad) — instrument inadequate, noise floor unresolved; "
            f"no curvature or flat verdict is evaluated under an inadequate instrument"
        )

    exceedance_count = 0
    within_envelope_count = 0
    for fd in per_fold:
        measured = fd["measured"]["median_angle"]
        n3 = fd["nulls"]["N3_residual_matched"]
        if measured >= 2.0 * n3["median"]:
            exceedance_count += 1
        if n3["p2_5"] <= measured <= n3["p97_5"]:
            within_envelope_count += 1

    corr_ci_excludes_zero = cross_fold["ci_excludes_zero"]
    corr_positive = cross_fold["correlation"] > 0 and corr_ci_excludes_zero

    if exceedance_count >= 4 and corr_positive:
        return VERDICT_CURVATURE, (
            f"measured median >= 2x N3 in {exceedance_count}/5 folds; "
            f"cross-fold correlation {cross_fold['correlation']:.4f} "
            f"CI [{cross_fold['bootstrap_ci'][0]:.4f}, {cross_fold['bootstrap_ci'][1]:.4f}] excludes 0"
        )
    if within_envelope_count >= 4:
        return VERDICT_FLAT, (
            f"measured median within N3 envelope in {within_envelope_count}/5 folds; "
            f"adequacy gate passed ({n_adequate}/5 folds adequate)"
        )
    return VERDICT_NOT_FOUND, (
        f"exceedance {exceedance_count}/5, within-envelope {within_envelope_count}/5, "
        f"correlation CI excludes 0: {corr_ci_excludes_zero} — neither curvature nor flat criterion met"
    )


def _write_plaquette_jsonl(per_fold: list[dict], out_dir: Path) -> None:
    for fd in per_fold:
        path = out_dir / f"plaquettes_{fd['fold']}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for rec in fd["_plaquette_records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _render_markdown(report: dict) -> str:
    lines = [
        "# Rung-4 holonomy + curvature measurement",
        "",
        f"**Verdict: `{report['verdict']}`**",
        "",
        f"Reasoning: {report['verdict_reasoning']}",
        "",
        "## Frozen constants (echoed from registration)",
        "",
        f"- substrate view: `{SUBSTRATE_VIEW}`",
        f"- quad cap: {QUAD_CAP}, seed: {QUAD_SEED}",
        f"- null seeds: {NULL_SEEDS[0]}..{NULL_SEEDS[-1]} ({len(NULL_SEEDS)} replicates)",
        f"- adequacy gate: N3 p95 < {THETA_MIN} rad",
        f"- near-pi exclusion: {NEAR_PI_RAD} rad",
        f"- SVD retain ratio: {SVD_RETAIN_RATIO}",
        f"- bootstrap: {BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}",
        "",
        "## Per-fold summary",
        "",
        "| Fold | Charts | Triangles | Quads | Measured median (rad) | N3 median (rad) | 2xN3? | Adequate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for fd in report["per_fold"]:
        measured = fd["measured"]["median_angle"]
        n3 = fd["nulls"]["N3_residual_matched"]["median"]
        lines.append(
            f"| {fd['fold']} | {fd['n_charts']} | {fd['n_triangles']} | {fd['n_quads']} | "
            f"{measured:.6f} | {n3:.6f} | {'yes' if measured >= 2*n3 else 'no'} | "
            f"{'yes' if fd['adequacy_gate']['adequate'] else 'NO'} |"
        )
    lines += [
        "",
        "## Cross-fold shared-plaquette correlation",
        "",
        f"- shared plaquette pairs: {report['cross_fold']['n_shared_plaquettes']}",
        f"- correlation: {report['cross_fold']['correlation']:.4f}",
        f"- bootstrap CI: [{report['cross_fold']['bootstrap_ci'][0]:.4f}, {report['cross_fold']['bootstrap_ci'][1]:.4f}]",
        f"- CI excludes 0: {report['cross_fold']['ci_excludes_zero']}",
        "",
        "## Holonomy-group diagnostics (neutral vocabulary)",
        "",
        "| Fold | Span dim | Generators | Excluded near-pi | Closure frac |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for fd in report["per_fold"]:
        hg = fd["holonomy_group"]
        lines.append(
            f"| {fd['fold']} | {hg['span_dimension']} | {hg['n_generators']} | "
            f"{hg['n_excluded_near_pi']} | {hg['closure_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Covariate correlations (reported, not gated)",
        "",
        "| Fold | theta vs loop-residual | theta vs mean-stress |",
        "| --- | ---: | ---: |",
    ]
    for fd in report["per_fold"]:
        cv = fd["covariate_correlation"]
        lines.append(
            f"| {fd['fold']} | {cv['theta_vs_loop_residual']:.4f} | {cv['theta_vs_mean_stress']:.4f} |"
        )
    lines += ["", "Per-fold plaquette details: `plaquettes_<fold>.jsonl.gz`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--substrate-root",
        default="results/relational_geometry/gauge_controller_substrate_v2_20260723",
    )
    parser.add_argument(
        "--rooted-graph-root",
        default="results/relational_geometry/pre_status_rooted_graphs_v1_20260721",
    )
    parser.add_argument(
        "--out-dir",
        default="results/relational_geometry/rung4_holonomy_curvature_v1_20260724",
    )
    args = parser.parse_args(argv)

    substrate_root = Path(args.substrate_root).resolve()
    graph_root = Path(args.rooted_graph_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_relational_pre_status_rooted_graph_artifacts(graph_root)
    per_fold: list[dict] = []
    for fold in FOLDS:
        graph = graphs.fold_graph(SUBSTRATE_VIEW, fold)
        bundle = load_fold_gauge_controller_artifact(substrate_root / fold, graph)
        print(f"  {fold}: {len(bundle.atlas.connections)} directed connections")
        fd = _measure_fold(fold, bundle.atlas)
        per_fold.append(fd)

    cross_fold = _cross_fold_correlation(per_fold)
    verdict, reasoning = _three_way_verdict(per_fold, cross_fold)

    per_fold_clean = []
    for fd in per_fold:
        clean = {k: v for k, v in fd.items() if not k.startswith("_")}
        per_fold_clean.append(clean)

    report = {
        "schema_version": 1,
        "kind": "rung4_holonomy_curvature",
        "frozen_constants": {
            "substrate_view": SUBSTRATE_VIEW,
            "quad_cap": QUAD_CAP,
            "quad_seed": QUAD_SEED,
            "null_seeds": list(NULL_SEEDS),
            "theta_min": THETA_MIN,
            "near_pi_exclusion_rad": NEAR_PI_RAD,
            "svd_retain_ratio": SVD_RETAIN_RATIO,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "per_fold": per_fold_clean,
        "cross_fold": cross_fold,
        "verdict": verdict,
        "verdict_reasoning": reasoning,
        "provenance": git_provenance([args.substrate_root, args.rooted_graph_root]),
    }
    write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_plaquette_jsonl(per_fold, out_dir)

    print(f"\nVERDICT: {verdict}")
    print(f"  {reasoning}")
    print(f"  artifacts: {out_dir}/report.json, report.md, plaquettes_*.jsonl.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
