# Simple-address baselines — raw-activation kNN and design-cell mean

**Status:** unrun, registered before execution
**Character:** post_evidence_descriptive_diagnostic
**Tier:** retrospective_synthesis

## What this tests

The honestward reconstruction result shows that the frozen typed-graph address retrieves
held-out-family displacement exemplars at mean cosine 0.9289 (local) and 0.9149 (nearest),
versus 0.4839 for the global train mean. This diagnostic asks whether two strictly simpler
addresses reach comparable reconstruction on the identical population, folds, targets, and
metrics:

1. **Raw-activation nearest exemplar / kNN** — neighbourhoods defined by plain Euclidean
   distance on the raw flattened four-layer root residual, with no typed metadata, no
   attention structure, and no graph.
2. **Design-cell mean** — a metadata-only model: the mean of train-fold displacement targets
   within the query root's design cell (turn index, intervention history, pressure exposure,
   true status), with no activations at all.

This is a descriptive structural diagnostic. It does not reopen any settled claim, is not a
controller evaluation, and makes no universality claim.

## Frozen inputs (the run must abort on any mismatch)

| Input | Physical path | SHA-256 |
|---|---|---|
| Rooted-star bank manifest | `results/relational_geometry/pre_status_rooted_stars_v1_20260721/manifest.json` | `ba123cd24d5f17a1796d594652ae53d787aba2700f47a9d2ab57b1cced9ee3ee` |
| Frozen orbit roster | `results/relational_geometry/partial_connection_v1_20260715/frozen_orbit_roster.json` | `750911245de5dfe5285e08c54ff367ea99ab412edd1afbee3aeca2288a8814dd` |
| Outcome report | `results/relational_geometry/post_commitment_growth_outcomes_v1_20260716/outcome_report.json` | `3f7d607131000fd072c42e9234ec3adfedc093e53d8521a895387e580268d572` |
| Rooted-graph manifest | `results/relational_geometry/pre_status_rooted_graphs_v1_20260721/manifest.json` | `79b9c9a386df6e0653cb96dd01337a52b6fd23a956c0d515f821e67a2ece35cd` |
| Sealed honestward report | `results/relational_geometry/pre_status_honestward_field_sealed_v1_20260721/report.json` | `af2681460f01e37f4fc76cfa2c55739f2dc258527448585a2e09ff615166a9cb` (internal `report_sha256` `011461cba07320198b98bec77f918a22c0901ffe0e3d7c1410a5f379a5e5093a`) |

## Population and shared machinery (identical to the sealed run)

- Supervision rebuilt with `build_relational_pre_status_supervision` from the pinned bank,
  roster, and outcome report (expected 1,680 outcome events; 780 forward roster edges).
- Honestward observations root-balanced with the existing `_root_balanced_observations`;
  242 crossings over 200 deceptive source roots; five outer family folds (`outer_1` ...
  `outer_5`); both views, primary `intervention_masked_action_free`, secondary
  `action_free_full_context`.
- Per-layer scales from `_layer_scales` on training root-balanced observations; scoring by
  the existing layer-scaled cosine and normalized squared error via the same metric-row
  construction as the sealed evaluation.

## Fidelity gate (must pass before any new number is read)

`evaluate_pre_status_honestward_fields` is re-run unchanged with graph variant `joint` on the
regenerated supervision and the frozen rooted graphs. For BOTH views, every frozen model's
aggregate cosine, normalized squared error, and defined rate must match the sealed report's
`views` block within 1e-9, and the regenerated per-row count must equal the sealed row
count. On any mismatch the run aborts with no new-model output.

## New models (frozen; no alternatives, no tuning, single execution)

For each view and fold, with training equal to the root-balanced train-fold observations and
their targets as stored values:

1. `raw_nn` — the target of the single training root nearest to the query in Euclidean
   distance on the raw flattened float64 mean root residual (4 x 4096, flattened). No
   calibration. Defined for every query (coverage 200/200).
2. `raw_k8` — the unweighted mean of the eight nearest training roots' targets under the
   same distance, followed by the same per-layer nonnegative leave-one-root-out calibration
   used by the frozen local estimator (one scalar per layer, computed on training roots with
   their own leave-one-out raw-k8 predictions). Defined for every query.
3. `design_cell_mean` — the mean of train-fold targets whose deceptive roots share the query
   root's design cell (turn index, intervention history, pressure exposure, true status),
   read from the sealed outcome records attached to each root's events (cell consistency
   across a root's events is asserted). If the training fold contains no root in the cell,
   the model falls back to the global train mean and the fallback is counted and reported.

Distance metric, k = 8, the calibration rule, and the cell definition are fixed by this
registration; no other metric, k, weighting, or cell definition may be evaluated.

## Registered outputs and comparison rule

For each view: per-model aggregate cosine, normalized squared error, defined rate, and
per-fold values; and paired per-root cosine differences with scenario-cluster 95% percentile
intervals (2,000 draws, seed 20260801) for exactly these contrasts, pairing the new models
against the sealed report's per-row values:

- `graph_local - raw_nn`, `graph_local - raw_k8`, `graph_nearest - raw_nn`
- `graph_local - design_cell_mean`, `design_cell_mean - global_mean`, `raw_nn - global_mean`

**Frozen reading rule:** a typed-graph advantage over a simple address is claimed only where
the paired scenario-cluster 95% interval excludes zero. Symmetrically, a simple address is
described as matching the graph only where the interval includes zero. The verdict is
descriptive either way.

## What this is NOT

- **Not a new controller** — no intervention is applied.
- **Not a confirmation** — the bank and the sealed evaluation predate this registration; the
  verdict is descriptive and does not re-open or revise any settled claim.
- **Not a universality claim** — one bank, one model, one construction.

## Execution note (2026-08-01, appended after the run; the registration above is preserved verbatim)

1. The producer as first committed (`6a5b60c`) contained two mechanical defects that
   prevented any run: the fidelity comparison assumed scalar per-model aggregates where the
   sealed report stores nested summaries, and the provenance helper was indexed as a tuple
   where it returns a mapping. Both were fixed in commit `edfa61c` before the first
   successful execution. No frozen analytic choice — distance metric, k, calibration rule,
   cell definition, comparison set, bootstrap seed or draw count — was changed.
2. The fidelity gate as executed is stricter than registered: it compares the full sealed
   per-model aggregate subtree (including per-family means and medians) within 1e-9, not
   only the three registered scalars, and additionally checks recomputed per-root
   global-mean values against the sealed rows.
3. Status update: executed once after `edfa61c`. Output:
   `results/relational_geometry/simple_address_baselines_v1_20260801/report.json`,
   internal `report_sha256`
   `e0d053fb096fee2b49e07d58c71045f6df71438c4554c86a30bd54b3e8853fca` is the physical file
   hash; a deterministic rerun reproduced identical values. The design-cell baseline's cell
   includes `true_status` and is described in reporting as the truth-aware design-cell mean.
