# Experiment design

This document describes the public experimental object behind the eight claims in
[`docs/results_registry.yaml`](results_registry.yaml). It is a methods reference, not a lab
notebook: operational history, abandoned protocols, provider-specific execution details, and
large activation banks are deliberately excluded. Every public claim points to one compact
receipt in [`paper_artifacts/`](../paper_artifacts/), and the receipt records the SHA-256 identities
of its private source artifacts.

## Scope and chronology

All behavioral generations use `meta-llama/Llama-3.1-8B-Instruct`, but the experiments differ in
task, representation, intervention, and information budget. Results from those settings are never
pooled. The study makes no cross-model, deployment, universal-deception, Lie-algebra, or
continuous-flow claim.

Chronology is attached to each claim rather than asserted for the project as a whole:

| Claim | Public role | Chronology |
|---|---|---|
| C5 | prospective natural-prose control | registered before the 96-row held-out-family generation |
| C10 | post-commitment readout plus linear comparator | post-hoc registered; the bank predates the comparator registration |
| C1, C2, C9, C11, C12, C13 | audits and syntheses over completed banks | retrospective synthesis |

We use three outcome states for completed tests: **supported**, **refuted**, and
**not-found-under-this-instrument**. The third is reserved for an instrument that fails its own
resolution or adequacy condition; it is not a softened null result. Unregistered exploratory
quantities may explain a result but cannot replace its declared endpoint.

### 2026-07-27 registration: C10 truth-aware nuisance re-score

Registered before any truth-aware numbers were computed; implemented by
`experiments/rescore_c10_truth_aware_nuisance.py` over the sealed C10 event bank.

The C10 exact nuisance prior is truth-blind: its cell key is (sampled status token, turn,
intervention history, pressure exposure), while the deceptive label is defined relative to
scenario truth. The C11 comparators, by contrast, key on true and desired status. The two
"nuisance prior" baselines are therefore not the same instrument, and the post- versus
pre-commitment contrast in C10/C11 mixes a timing difference with a comparator-strength
difference. This re-score adjudicates the post-commitment residual under a matched truth-aware
comparator.

- **Population and folds**: the same 1,283 pressure-exposed honest/deceptive events and the
  same 5 outer family folds as C10; training pools are fold-safe.
- **Comparator**: an exact nuisance prior keyed on (sampled status token, turn, intervention
  history, pressure exposure, true status, desired status), with coarse fallback (sampled
  token, turn, pressure, true status, desired status) and a family-balanced base-rate final
  fallback, replicating the family-balanced Jeffreys-0.5 semantics of the original prior.
- **Fidelity gate**: recomputation with the original truth-blind key must reproduce the
  published 0.0205183 / 0.0948840 / 0.0743657 within 1e-6 before any new numbers are read.
- **Decision rule**: the graph's residual over a truth-aware nuisance prior is *retained* if
  the family-macro Brier gain is positive and its 10,000-replicate family-cluster bootstrap
  95% CI excludes zero; secondary support requires the nuisance-preserving permutation excess
  over the null mean to be positive at one-sided p < 0.05. Verdict language: **found** /
  **refuted-under-adequate-instrument** / **not-found-under-this-instrument**.

**Outcome (2026-07-27, `post_commitment_truth_aware_nuisance_rescore_v1_20260727`)**: the
fidelity gate passed (published numbers reproduced bit-exact). The truth-aware prior reaches
family-macro Brier 0.02748 against the truth-blind prior's 0.09488, so most of the graph's
raw gain is explained by truth-aware design metadata. The graph's gain over the truth-aware
prior is +0.00696 (16/20 families positive; 1,281 exact-cell hits, 2 coarse fallbacks), with
bootstrap fraction positive 0.9678, so the 95% CI does not exclude zero and the primary
criterion is not met. The nuisance-preserving permutation still rejects (observed excess
+0.0309 over the null mean, one-sided p=1e-4), so the graph carries some real within-cell
signal, but not a family-robust gain. The registered linear probe beats the truth-aware prior
by +0.02598. Verdict: **refuted-under-adequate-instrument**.

Instrument limits logged with the verdict: (i) the truth-aware key contains the label's
defining arguments (sampled token, true status), so the comparator is a near-oracle for every
model — the probe's margin over it (+0.02598) is likewise compressed, and the re-score
adjudicates utility over the design cell, not the existence of information beyond metadata;
(ii) the family bootstrap has only 20 clusters — the sign test on 16/20 positive families
gives p≈0.006 and the bootstrap one-sided tail is ≈0.032, so the result sits at the
instrument's power limit and cannot distinguish "negligible" from "small but real".


## Experimental units and banks

### Natural conversational pressure (C9)

The natural-pressure setting uses free assistant prose in controlled workplace scenarios with
machine-specified ground truth. Pressure accumulates over multiple user turns without placing a
PASS/FAIL label in the assistant's mouth. Three banks are kept separate:

- a 96-conversation scripted bank;
- a 128-conversation adaptive bank;
- a 96-conversation dissociation bank designed to reduce correlation between current and
  accumulated pressure.

The main arms are `smooth`, `late-compressed`, `step`, and `benign`. Smooth pressure spreads the
argument across turns. Late-compressed pressure uses approximately the same argument inventory but
delivers it late. Step pressure introduces stakes abruptly. Benign conversations establish the
unpressured behavior. The registered schedule contrast is smooth minus late-compressed; smooth
minus step is exploratory and is not used to adjudicate C9.

Assistant outcomes and pressure intensity use separate blinded LLM channels. The outcome channel
sees the transcript and scenario truth; the pressure channel sees user turns only. Three judges
are called independently. Categorical outcomes require a strict majority, ordinal scores use the
median of at least two parseable calls, and the planned coverage floor is 0.95. These labels have
not been validated against a human-labelled agreement sample, and the compact public artifact does
not fully bind the auxiliary judge-model revisions. This limits the behavioral claim.

The hazard analysis registers one discrete-time form:

\[
\operatorname{logit} h_t = \delta + \alpha I_t + \gamma C_{t-1},
\]

where \(I_t\) is the judged intensity of the immediately preceding user turn and \(C_{t-1}\) is
the accumulated earlier intensity. The comparison is leave-one-family-out log-loss against the
same model without \(C_{t-1}\). The geometry analysis (P3) reads the last non-special content token
of each assistant turn, fits a commitment axis on the scripted bank, and evaluates the registered
approach-drift statistic on the adaptive bank. A held-out axis AUC of 0.70 is its adequacy floor.

### Sequential structured-action bank (C10, C11, C13)

The sequential bank contains 600 rows, 60 scenarios, 20 families, and four decision turns. It
contains 2,400 status records and 1,680 unique status-draw events after shared-prefix events are
deduplicated. It passed action recognition at 2,400/2,400, but the frozen baseline-knowledge gate
missed by one scenario: 56/60 correct versus a required 57/60. The bank is therefore retained as a
development/exploratory bank; no analysis of it is called confirmatory.

Each turn samples exactly two unrestricted full-vocabulary tokens in sequence:

1. append the exact bytes `Status:` (no trailing whitespace) and sample one token;
2. append that sampled token to the causal transcript;
3. append the exact bytes `\nCaveat:` (no trailing whitespace) and sample one token;
4. append an end-of-turn token.

The status token is the primary operational outcome. Raw token IDs map as follows:

| Field | Token ID | Exact decoded token | Action |
|---|---:|---|---|
| status | 51935 | ` PASS` | PASS |
| status | 34207 | ` FAIL` | FAIL |
| status | 82504 | ` SKIP` | SKIP |
| caveat | 51493 | ` INCLUDE` | INCLUDE |
| caveat | 70500 | ` OMIT` | OMIT |
| caveat | 43969 | ` NONE` | NONE |
| caveat | 82504 | ` SKIP` | SKIP |

Every other sampled token maps to `NO_ACTION`; neither `SKIP` nor `NO_ACTION` is dropped. Caveat is
a secondary outcome conditional on the already sampled status. It is not treated as an independent
pre-status decision.

A status report is **deceptive** when it contradicts scenario truth in a scenario where the shared
unpressured baseline showed correct knowledge. A wrong report without that baseline evidence is
kept as `WRONG_WITHOUT_BASELINE_KNOWLEDGE`, not silently reclassified as deception. This separation
supports the lie-versus-error contrast in C10.

### Offline action-response field (C1, C2)

The powered150 action-response field contains saved candidate responses for 600 deceptive and 600
honest conversations over 20 families. It permits matched evaluation of many interventions without
new generation, but that convenience creates a strict interpretation boundary: an action selector
evaluated on this field is not automatically a prospective controller.

C1 compares three information budgets. The field is oracle-routed at generation, so on the 600
deceptive rows the candidate menu already exposes only the corrective target; every policy is
therefore equally route-constrained there, and the route is also available to the learned ranker
as an input feature. The budgets differ in whether the *corrective target* is handed to the policy
or must be selected:

| Policy | Held-out candidate outcomes at application | Corrective target supplied | Route available | Result |
|---|---|---|---|---|
| learned route-feature ranker | no | no; selects target and action itself | yes, as a feature | 170/600 fixes, 11/600 honest harms |
| Chart-Neural Gate | no | no explicit mask; argmax over the route-constrained menu | yes, via the menu | 599/600 fixes, 1/600 harm |
| route-matched fixed L16, alpha 96 | no | yes; conceived retrospectively | yes | 600/600 fixes, 0 harm |

The learned route-feature ranker receives the route as a feature yet selects a target inconsistent
with scenario truth on 233/1,200 rows, all on honest rows where both targets remain available. The
near-ceiling policies do not establish route inference: they receive the correct route, and so does
the ranker. Because the ranker also differs in action selection and architecture, the observed gap
reflects corrective-coordinate selection and reward estimation, not route access or a geometric
actuation primitive. The historical 539/600 floor is retained only as a mis-specified comparison:
its training selection pooled counter-target candidates. C2 separately compares generated behavior
under a learned dense dose and fixed high-dose policies.

### Prospective natural-prose control (C5)

C5 is a 96-row fresh-generation test over four held-out families: cybersecurity, electrical grid,
structural welding, and water treatment. It compares no intervention, the registered native gate,
a more frequently firing early-window variant, and a family-matched linear comparator. The primary
endpoint is machine-scored change in correct operational status, with family-clustered uncertainty.
Judge-rated honesty, caveat quality, and coherence are secondary because hedging, refusal, or broad
perturbation can move them without fixing the machine-scored action.

The tested controller intervenes in the layer-16 residual stream. It is not an online controller
that attaches novel live typed token, residual, and attention states at layers 12, 16, 19, and 20
and updates throughout fresh generation. Building and prospectively evaluating that richer system
exceeded this study's time and compute budget. Its outcome is unknown; its absence cannot rescue
C5.

### Pilot steering decomposition (C12)

The primary pilot contains 160 rows: 80 deceptive and 80 honest. Correction counts such as 48/80
use the deceptive denominator; honest preservation is reported separately. The six-arm comparison
holds gate, dose, and first-decision-token scope explicit so tangent alignment is compared with a
matched random direction and gated global directions, not only with no intervention. A separate
32-row off-tangent follow-up contains 16 deceptive pairs and is an explicitly retrospective
recalculation; it is not merged into the primary denominator. Status and strict endpoints differ
(48 versus 40 tangent fixes), and every primary arm has 2 honest status harms but 10 honest strict
harms. Behavioral generation in this pilot used a 4-bit derivative, while its activation capture
used a separate higher-precision representation; exact conversion fingerprints were not retained.

## Activation measurements and causal anchors

The sequential structured capture stores every-token BF16 residuals and full 32-head causal
attention at layers 12, 16, 19, and 20. Attention is stored losslessly over the causal lower
triangle. Token role, turn, span, and position metadata preserve the typed point-cloud and graph
structure. The provider-neutral acquisition path is
[`experiments/capture_relational_geometry.py`](../experiments/capture_relational_geometry.py); it
hash-binds the structured rows, protocol, model weights, model config, and tokenizer before writing
checksummed per-row tensors.

Two anchors must not be conflated:

- the **status anchor** is immediately before the sampled status token; neither sampled action is
  visible;
- the **caveat anchor** is immediately before the caveat token, after the status token has entered
  the causal state.

C10 uses the growth edge from status anchor to caveat anchor, so it is deliberately
**post-commitment**. C11 uses action-free pre-status views. C9 uses final content-token residuals,
primarily at layer 16. C5 intervenes only in the layer-16 residual stream. Thus the existence of a
four-layer residual-attention capture does not mean every reported model consumed the complete
capture.

## Claim-specific estimands and comparators

| Claim | Primary estimand | Comparator or null | Uncertainty / unit |
|---|---|---|---|
| C9 | deceptive-commitment rate and smooth-minus-late-compressed contrast; hazard log-loss | late-compressed schedule; alpha-only hazard; P3 smooth-versus-step drift | conversation-level Wilson/Newcombe for arm effects; family-held-out hazard tests |
| C10 | family-macro Brier for honest versus deceptive post-status events | exact sampled-status × turn × intervention-history × pressure prior; nuisance-preserving permutations; same-bank residual probe | 20 held-out families; event draws remain clustered |
| C11 | pre-status AUROC or log-loss gain | design-cell or nuisance-matched models | three instruments retain separate populations |
| C1 | deceptive fixes and honest harms in the saved field | learned context-only, oracle-target CNG, and retrospective oracle-target fixed policy | four held-out-family folds; scenario/family bootstrap |
| C2 | generated structured-action fixes, harms, and strict reward | best fixed high dose | deceptive and honest rows reported separately |
| C5 | change in machine-scored correct status | no intervention and family-matched linear controller | 96 rows, four held-out families |
| C12 | deceptive correction with honest preservation | matched random, gated global, ungated linear, off-tangent follow-up | 160-row primary; separate 32-row follow-up |
| C13 | deceptive-probability change under gauge replay; transport-specific logit remainder | no intervention, sign flip, matched random tangent, generic transport, residual-matched flat holonomy null | common RNG; scenario-cluster bootstrap |

C10's relational model uses a fixed local joint residual-attention graph with eight neighbors and
compares it with an exact nuisance prior. Its same-bank linear-probe comparator uses the same outer
family folds and metric. C11's populations are not interchangeable: the spectral slice contains
106 honest/deceptive events, the connection-path test contains 52, and the masked risk analysis
contains 1,680 events (1,568 honest/deceptive). Numbers from those instruments are never presented
under one denominator.

C13 replays four arms from 402 roots and 656 random streams, yielding 2,624 rows; only 21 roots are
active under the frozen support policy, and all stopped or undefined roots remain in the pooled
denominator. It is a one-step four-layer structured-action controller: a live root is authenticated
against a sealed source-bank query and intervention is applied at an exact frozen prefix. It does
not attach novel states or update a policy across natural-prose generation. Its holonomy analysis
is a separate measurement instrument. Because a
residual-matched flat null exceeds the frozen 0.1-radian resolution threshold in all five folds,
neither curvature nor flatness is adjudicated.

## Reproducibility boundary

The public repository ships compact receipts, their producers, focused tests, exact scientific
protocols, and a manifest binding each claim to its receipt, producer, source hashes, model
identity, and stated registration tier. Large activation banks and generated conversations are
excluded. Where a legacy experiment did not retain immutable checkpoint or source-commit
identity, the manifest records the gap rather than reconstructing it after the fact.

This artifact supports checking receipt integrity, registry/README consistency, frozen protocol
hashes, and figure regeneration. Because the hashed source reports and private banks are not
published, a clean clone cannot independently recompute every receipt number from those sources
or reproduce historical GPU generation. The most important missing experiment is a fresh,
held-out-family natural-prose evaluation of an online state-dependent controller that attaches and
updates the full four-layer token-residual-attention state throughout generation, with
fixed-linear, matched-random, sign-flipped, and shuffled-field controls. Its outcome is unknown.
