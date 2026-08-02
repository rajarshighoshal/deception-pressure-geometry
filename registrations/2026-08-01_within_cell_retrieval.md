# Within-cell retrieval — activation selection inside the design cell

**Status:** unrun, registered before execution
**Character:** post_evidence_descriptive_diagnostic
**Tier:** retrospective_synthesis

## What this tests

The registered simple-address diagnostic left one question unresolved: whether activation
similarity carries predictive information beyond the design cell. This diagnostic asks it
directly. Within a query root's design cell (turn index, intervention history, pressure
exposure, true status), does selecting the single activation-nearest training exemplar beat
the cell's own mean? If yes, activation similarity orders displacement targets even after
the design is fixed. If no, the cell average is not improved by activation-based selection
at the observed support.

## Frozen inputs (the run must abort on any mismatch)

The five pins of the simple-address registration, unchanged:

| Input | Physical path | SHA-256 |
|---|---|---|
| Rooted-star bank manifest | `results/relational_geometry/pre_status_rooted_stars_v1_20260721/manifest.json` | `ba123cd24d5f17a1796d594652ae53d787aba2700f47a9d2ab57b1cced9ee3ee` |
| Frozen orbit roster | `results/relational_geometry/partial_connection_v1_20260715/frozen_orbit_roster.json` | `750911245de5dfe5285e08c54ff367ea99ab412edd1afbee3aeca2288a8814dd` |
| Outcome report | `results/relational_geometry/post_commitment_growth_outcomes_v1_20260716/outcome_report.json` | `3f7d607131000fd072c42e9234ec3adfedc093e53d8521a895387e580268d572` |
| Rooted-graph manifest | `results/relational_geometry/pre_status_rooted_graphs_v1_20260721/manifest.json` | `79b9c9a386df6e0653cb96dd01337a52b6fd23a956c0d515f821e67a2ece35cd` |
| Sealed honestward report | `results/relational_geometry/pre_status_honestward_field_sealed_v1_20260721/report.json` | `af2681460f01e37f4fc76cfa2c55739f2dc258527448585a2e09ff615166a9cb` (internal `report_sha256` `011461cba07320198b98bec77f918a22c0901ffe0e3d7c1410a5f379a5e5093a`) |

Plus the frozen simple-address report itself:

| Input | Physical path | SHA-256 |
|---|---|---|
| Simple-address report | `results/relational_geometry/simple_address_baselines_v1_20260801/report.json` | `e0d053fb096fee2b49e07d58c71045f6df71438c4554c86a30bd54b3e8853fca` (internal `report_sha256` `44d87fff9f861e5fefbbe89ab545544e0a3c05fc3535578fcfada18151ced35d`) |

## Population and shared machinery (identical to the simple-address run)

Supervision rebuilt with `build_relational_pre_status_supervision` from the pinned bank,
roster, and outcome report (expected 1,680 outcome events; 780 forward roster edges);
root-balanced honestward observations; five outer family folds; both views with primary
`intervention_masked_action_free`; per-layer scales from training root-balanced
observations; scoring by the existing layer-scaled cosine and normalized squared error via
the same metric-row construction.

## Fidelity gates (both must pass before any new number is read)

1. **Sealed honestward gate.** `evaluate_pre_status_honestward_fields` re-run unchanged
   with graph variant `joint` must reproduce the sealed report's per-model aggregate
   subtree within 1e-9 for both views, with matching row counts (the same gate the
   simple-address run passed).
2. **Baseline replication gate.** The recomputed `raw_nn`, `raw_k8`, `design_cell_mean`,
   and `global_mean_recomputed` aggregates (cosine, normalized squared error, defined and
   total counts, per-fold cosines) must match the pinned simple-address report's `models`
   block within 1e-9 for both views, and the design-cell fallback counts must match
   exactly. On any mismatch the run aborts with no new-model output.

## New model (frozen; no alternatives, no tuning, single execution)

`within_cell_nn` — for each view and fold, with training equal to the root-balanced
train-fold observations: the target of the training root nearest to the query in Euclidean
distance on the raw flattened float64 mean root residual, restricted to training roots
sharing the query root's design cell. Defined only when the training fold contains at
least two roots in that cell (with exactly one, the nearest exemplar coincides with the
cell mean and the contrast is degenerate). No calibration. The per-query support (train
cell size) is recorded; the support distribution and coverage are registered outputs.

## Registered outputs and comparison rule

Per view: `within_cell_nn` aggregate cosine, normalized squared error, defined rate,
per-fold values, and the within-cell support distribution (per-fold minimum, median,
maximum, and the count of queries at each support level). Paired per-root pooled-cosine
differences with scenario-cluster 95% percentile intervals (2,000 draws, seed 20260802)
for exactly these contrasts:

- `within_cell_nn - design_cell_mean` (on roots where `within_cell_nn` is defined)
- `raw_k8 - design_cell_mean` (all covered roots)
- `within_cell_nn - raw_nn` (on roots where `within_cell_nn` is defined)
- `within_cell_nn - global_mean_recomputed` (same subset)

**Frozen reading rule.** Activation similarity is described as carrying information beyond
the design cell only if the `within_cell_nn - design_cell_mean` interval excludes zero in
`within_cell_nn`'s favor on the primary view. An interval containing zero is reported as
unresolved at the observed support, with the support distribution stated alongside. An
interval excluding zero in the cell mean's favor is reported as within-cell selection
underperforming the cell average. The `raw_k8 - design_cell_mean` contrast licenses only a
marginal statement about outcome-blind retrieval versus the truth-aware cell. All verdicts
are descriptive either way.

## What this is NOT

- **Not a new controller** — no intervention is applied.
- **Not a confirmation** — the bank, the sealed evaluation, and the simple-address report
  all predate this registration; the verdict is descriptive and reopens no settled claim.
- **Not a universality claim** — one bank, one model, one construction.

## Execution note (2026-08-02, appended after the run; the registration above is preserved verbatim)

1. Registered at commit `0755555`; executed once with no analytic change after
   registration. Both fidelity gates passed: the sealed honestward subtree reproduced
   within 1e-9, and the recomputed baseline aggregates matched the pinned simple-address
   report within 1e-9 with exact fallback counts.
2. Output: `results/relational_geometry/within_cell_retrieval_v1_20260801/report.json`,
   file SHA-256 `06e179053c2f88e1ca4355171b1aa883a1ecc45c290bcdc99b1b9b3f363b7373`,
   internal `report_sha256`
   `0d258a6df6a7a07c82703e3409c7c07eea2df694b7ebdfd2a7336dfac2400ee8`. Both views
   produce identical values, as in the pinned simple-address report.
3. Registered results (primary view): `within_cell_nn` aggregate cosine 0.9471 over
   195/200 covered roots (support >= 2; per-fold support median 12--19, maximum 23--24;
   five roots sit in cells absent from their training folds).
   Contrasts: `within_cell_nn - design_cell_mean` $-0.0022$ $[-0.0127, +0.0068]$ ---
   under the frozen reading rule, unresolved at the observed support;
   `raw_k8 - design_cell_mean` $+0.0103$ $[-0.0049, +0.0261]$ --- marginal tie;
   `within_cell_nn - global_mean` $+0.4474$ $[+0.4065, +0.4846]$;
   `within_cell_nn - raw_nn` identically $0.0000$ on all 195 pairs.
4. Mechanical verification of the identically-zero registered contrast (a determinism
   check, not a new estimand): for every one of the 195 covered queries, the
   unrestricted nearest training root by raw activation distance lies in the query's own
   design cell (195/195 in-cell, 0 out-of-cell). Excluding the five uncovered roots
   raises the nearest-exemplar mean from 0.9304 to 0.9471, so unrestricted retrieval's
   failures concentrate exactly on the roots whose design cells are unseen in training.
5. Deviation disclosure (2026-08-02, appended): the gate as first executed compared the
   registered per-model aggregates but omitted the registered exact fallback-count
   comparison. The counts match (verified post hoc for both views), the producer was
   amended to enforce the comparison, and a re-execution to a scratch directory with the
   amended producer reproduced every registered output (models, comparisons, support,
   fallbacks, fidelity gate) identically. The frozen report of note 2 is unchanged.
