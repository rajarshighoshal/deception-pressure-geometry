# Endpoint prototype diagnostic — B = −I constrained follow-up

**Status:** unrun, registered before execution
**Character:** post_evidence_descriptive_diagnostic
**Tier:** retrospective_synthesis

## What this tests

After the free additive transport result (family-macro cosine 0.8915), does a
constrained model that enforces B = −I — i.e., predicts the PCA-projected
endpoint e = c + z from compositional action features q alone, then recovers
ĉ = ê − z — retain the free model's predictive cosine within 0.01?

This is a descriptive structural diagnostic, not a controller evaluation and
not a universality claim.

## Previous report binding (frozen)

| Field | Value |
|---|---|
| Physical path | `results/relational_geometry/additive_compositional_transport_v1_20260730/report.json` |
| Physical SHA-256 | `7c83ae7d6bf0d8bf0261e2b471c6aade0d8c0f801d5a8b38a90cd7c12de8c762` |
| Internal SHA-256 | `8c42763a246a239d3349f5d8b177c64a37f85c834768479419c8296bfe1e284b` |
| Free additive family-macro cosine | `0.8914569246017339` |

The run must abort if the previous report's physical hash, internal hash, or
free additive cosine value has changed from these pinned values.

## Population

Identical to the additive transport follow-up:

- **View:** `intervention_masked_action_free`
- **Layers:** 12, 16, 19, 20
- **Roster:** all 1,560 directed roster edges
- **Folds:** five outer family folds (`outer_1`…`outer_5`)
- **Grouped observations:** 1,165 (same averaging within source_root_id × turn × programs × desired_status)

Uses the same `load_grouped_observations` loader which verifies bank/roster/outcome manifest
hashes internally.

## Constrained model (B = −I)

Per fold:

1. Fit PCA (rank 32, `random_state=0`) on training deltas D_train → `pca.components_` U ∈ ℝ³²ˣ¹⁶³⁸⁴
2. Project: c_train = pca.transform(D_train) ∈ ℝ^{N_train × 32}
3. Compute μG as the mean of per-unique-root averaged training source residuals
4. z = (source_root − μG) @ Uᵀ ∈ ℝ^{N × 32}
5. Construct compositional action features q (15-dim, same as parent report)
6. Standardize q (mean 0, std 1) using training-fold statistics; z is **not** standardized
7. Target: endpoint e_train = c_train + z_train
8. Fit `Ridge(alpha=1.0, fit_intercept=True, random_state=0)` from standardized q → e_train
9. Predict ê_test from standardized q_test
10. Recover ĉ_test = ê_test − z_test (B = −I enforcement)
11. Reconstruct δ̂ = pca.inverse_transform(ĉ_test)
12. Compute raw cosine between actual and predicted deltas
13. Aggregate with the same `_aggregate_cosines` as the parent report

No z standardization, no free B matrix, no hyperparameter tuning.

## Decision rule

**PASS** iff `free_additive_family_macro − constrained_endpoint_family_macro ≤ 0.01`.
Otherwise **FAIL**.

Report per-fold constrained cosine, gap versus the previous report's per-fold
additive value, and counts/provenance.

## What this is NOT

- **Not a new controller** — no counterfactual intervention is applied.
- **Not a universality claim** — the B = −I constraint is a single structural
  hypothesis tested on one bank; it does not assert anything about arbitrary
  models or geometries.
- **Not a confirmation** — the bank predates this registration; the verdict is
  descriptive.
