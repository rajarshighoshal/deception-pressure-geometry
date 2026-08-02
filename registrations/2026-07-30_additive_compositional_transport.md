# Additive compositional transport — post-evidence registered follow-up

**Status:** unrun, registered before execution
**Character:** post_hoc_registered_follow_up
**Tier:** retrospective_synthesis

## What this tests

Does the PCA-projected source root residual (z) add predictive cosine leverage
over a purely compositional action encoding (q) when reconstructing directed
root-residual deltas at the pre-status anchor?

## Artifact bindings

These hashes are pinned: the run must abort if they differ from the physical
files at execution time.

| Artifact | Path (relative to repo root) | SHA-256 |
|---|---|---|
| Roster | `results/relational_geometry/partial_connection_v1_20260715/frozen_orbit_roster.json` | `750911245de5dfe5285e08c54ff367ea99ab412edd1afbee3aeca2288a8814dd` |
| Bank manifest | `results/relational_geometry/pre_status_rooted_stars_v1_20260721/manifest.json` | `ba123cd24d5f17a1796d594652ae53d787aba2700f47a9d2ab57b1cced9ee3ee` |
| Outcome manifest | `results/relational_geometry/pre_status_outcome_shards_v1_20260721/manifest.json` | `25c4ea0879e61a38aa357a2b368e547affdb729df8ea8ca95cca81ec88fb815d` |

## Population

- **View:** `intervention_masked_action_free` (VIEWS[1])
- **Layers:** 12, 16, 19, 20
- **Roster:** all 1,560 directed roster edges (primary + secondary_control, forward + reverse)
- **Folds:** five outer family folds (`outer_1`…`outer_5`), held-out evaluation
- **Transition unit:** one directed roster edge (source x → destination y)
- **Root residual:** `load_rooted_star_root_residuals(index, reference)` → float32 `[4, 4096]`
- **Delta:** `mean_root_residual(target) − mean_root_residual(source)`, flattened to vector of length L = 4 × 4096 = 16384
- **Source root identity:** endpoint `prefix_state_sha256` (verified one identity per multi-member endpoint)
- **Multi-member endpoints:** load all physical member root residuals, verify shared prefix/geometry identity, average
- **Grouping:** observations are grouped (averaged) within (`source_root_id`, `turn`, `source_program`, `target_program`, `desired_status`) before PCA/fitting/scoring

## Compositional programs and primitive channels

Six programs with three primitive channel dimensions: PASS-pressure, FAIL-pressure, caveat-suppression.

| Program | Source primitives | Target primitives |
|---|---|---|
| NN | N = (0,0,0) | N = (0,0,0) |
| AN | A = (1,0,0) or (0,1,0) per desired status | N = (0,0,0) |
| AA | A | A |
| D2N | D2 = 2×A | N |
| AB | A | B = (0,0,1) |
| BA | B = (0,0,1) | A |

A is one unit in the desired-status channel (PASS→(1,0,0), FAIL→(0,1,0)).
D2 is 2×A. Desired status is loaded explicitly from outcome shards
(`scored_events[].desired_status`), matched by `field_event_id`.

## q-feature construction

For each edge, construct q as a concatenated vector:

1. **Turn one-hot** (3 features): indicator for turn 1, 2, or 3
2. **Visible source-slot pressure primitives** (3 × n_visible_slots features):
   per visible slot, the 3-channel primitive vector for the source program's
   slot action at that slot index
3. **Visible signed target-minus-source slot changes** (3 × n_visible_slots features):
   (target slot primitives − source slot primitives), per visible slot

Visible slots: turn 1 → slot 0 only; turns 2,3 → slots 0 and 1.
Max features = 3 + 2×3 + 2×3 = 15.

## PCA / coefficient construction

Per training fold (on grouped observations):

- Collect training deltas into matrix D_train of shape [N_train, 16384]
- Fit PCA (rank 32) on D_train → `pca.components_` shape [32, 16384]
- c_train = pca.transform(D_train): coefficients shape [N_train, 32]
- μ_G = mean of unique training source root residuals (after grouping, averaged per unique source_root_id)
- z = (source_root − μ_G) @ pca.components_.T: shape [N_train, 32]

## Models

Two fixed Ridge regressions fitted per fold to **coefficients** c ∈ ℝ³²:

**Action-only:** predict c from standardized q
**Additive:** predict c from standardized [q, z]

Both use `sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)`.
Features are standardized (mean 0, std 1) using training-fold statistics.
Predicted delta is reconstructed: δ̂ = pca.inverse_transform(ĉ).

No CV, no bootstrap, no feature interactions, no alternate rank/alpha.

## Evaluation

For each held-out grouped observation:
- Predicted δ̂ via coefficient prediction + inverse_transform
- Cosine similarity: cos(δ_actual, δ̂)
- Aggregate: mean duplicates within (transition, source_root_id) → mean across transitions per source root → mean across roots per family → equal-weight macro mean across families

## Decision rule

PASS iff additive-action family-macro cosine delta ≥ 0.01. Otherwise FAIL.

Report: per-fold family-macro cosines and delta, raw observation count, grouped
observation count, unique source roots and families, and input artifact hashes.

## What this is NOT

- Not a causal controller evaluation — no counterfactual intervention is applied
- Not a geometry-specific claim — the PCA + Ridge pipeline is a generic linear tool
- Not a confirmation — the bank predates this registration; verdict is descriptive
  and does not re-open any previously settled claim
