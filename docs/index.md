---
layout: article
title: The geometry worked—until I tried to make it universal
deck: "Pressure traces a flow. Local geometric selection works inside the task. The controller stops being universal when the geometry is assumed too early."
byline: "Rajarshi Ghoshal"
assistance_note: >-
  The author wrote the post and owns the research design, methodology, claims, and prose.
  LLMs assisted with experiment code, plots and tables, background lookup, typo correction,
  and limited sentence clarification.
---

I started with a clean story. Deception occupies a structured region of activation
space, that structure can be read, and reading it lets you steer the model away
from lying without pushing it off the language manifold. Two months later I have
a different story: a map of where the geometric hypothesis held and where it
crossed its own information budget.

Three questions turned out to need separate answers:

1. Can you correct deception with local geometric steering, or is one global
   linear direction just as good?
2. Does conversational pressure look like a mathematical object — a flow, a field,
   something geometric — or is it just a scalar that accumulates?
3. If you learn this structure from one task, does it become a universal controller
   for the model, or does it stop at the boundary of the task that produced it?

The answers: **locally and conditionally, yes; descriptively, yes; and no, not yet.**

I use "deceptive commitment" as shorthand for a pressure-induced false operational
report. The structured-action bank supports a stronger knowledge-conditioned form
when its shared unpressured branch demonstrates the correct answer; that baseline
gate passed for 56/60 scenarios, missing its frozen bar by one. The natural-pressure
banks do not have the same branch-level knowledge check. C9 therefore establishes
false operational commitment, not the knowledge state that preceded it. If you
prefer "sycophantic capitulation" or "caving under pressure," the numbers don't
change.

The evidence did not all arrive under one chronology. C5 is the only prospective
public claim; C10 was registered after its evaluation bank existed; C1, C2, C9,
C11, C12, and C13 are retrospective syntheses over completed banks. I'm keeping
those tiers explicit rather than calling the whole program preregistered.

## Act I: local geometric control, within its information budget

I built a large structured-action field: 1,200 rows of machine-checkable
operational-status slots where the model samples one unrestricted token, with
prose kept separate. This is an artificial instrument, not spontaneous
disclosure. It was useful because a hedge cannot silently be reclassified as a
commitment. The field is oracle-routed at generation time: on the 600 deceptive
rows, the menu exposes only the corrective-target candidate, so every policy is
equally route-constrained on deceptive rows.

The Chart-Neural Gate was built as a candidate-response-blind structural-geometry
selector. It scores saved candidates without inspecting their post-action margins,
outcomes, or rewards. The hierarchy is stark: 170 fixes from a held-out-family
route-feature ridge that receives the oracle route and candidate target, 539 from
the historical route floor, then 581, 584, and 598 as richer structural selectors
narrow the gap. CNG's unconstrained argmax reaches 599/600 fixes with 1/600 honest
harm. This is a selection result, not evidence that applying a curved direction is
intrinsically superior.

Two results sharpen what CNG is and isn't doing. A locality proxy comparing
fixes from a local flow against a locality-removed global flow over saved
candidates shows 275/600 versus 52/600, supporting chart-local but not
global structure. The route diagnostic is a separate L20 linear gate across 12 held-out fresh
families making 1,680 PASS/FAIL/abstain routing predictions, same
data-generating process. This is routing-only — not an honest-only evaluation,
not a route-family choice, not OOD/generation/control, and not evidence the
atlas is parochial. It suggests C1's oracle route might be replaceable within
the benchmark. A separately conceived
retrospective hard-route-matched fixed bidirectional-linear L16 policy at
alpha96 reaches 600/600 with zero harm, tying CNG. The fixed coordinate
explicitly maps route to target; CNG selects over a structural hierarchy. Both
hit the same ceiling, and the fixed policy doesn't need a geometry.

Immediate boundaries: this is a BF16 saved-candidate field with oracle
true-status routing, not fresh generation. All policies receive the true target
route. The field exposes only corrective-target candidates on deceptive rows. The
ridge selects 233 counter-target candidates on honest rows without inspecting
held-out candidate outcomes. CNG's hierarchy is held-out-family, pre-response,
and same-distribution only — it does not touch OOD data, fresh prose, or a
generated action. The fixed linear reconstruction was identified retrospectively
from the completed field. CNG supports local conditional correction inside an
oracle-routed structured-action sandbox. It does not establish a
geometry-specific or prospective end-to-end controller.

![Structured-action control and information budgets](figures/structured_action_control_audit.png)

## Act II: pressure traces a flow

My first prompt was blunt — "give me a false report" — and it mostly produced
hedging. A crashed early run came back around 10% deceptive commitment and I
almost wrote that the model just doesn't commit to lies in free prose.

Then I thought about how humans actually do this. It ramps, and it ramps in a
specific order: curiosity first, then disappointment, then stakes, then sunk
cost, then "everyone already agrees with me," and only at the end the direct
question. I built four arms: smooth pressure rising across several turns,
benign/zero pressure, a step arm with one late abrupt demand, and late-compressed
roughly matching smooth but crammed into the last two turns.

I ran 96 scripted conversations and 128 with an adaptive pressurer. Three
separate calls to one judge model/family labeled commitment while a separate
channel scored pressure intensity.
The labels have not been human-validated, and the pressurer and judges share
the same model family.

Smooth pressure produced deceptive commitment in 26/32 scripted and 45/48
adaptive conversations. The registered smooth-minus-late-compressed contrasts
were **+0.31 CI [+0.04, +0.55]** and **+0.22 CI [+0.05, +0.40]**. The
assigned-arm contrast was found in both banks, though scripted-arm
realization was mixed.

But what I expected next — path dependence, accumulation into a basin — did not
show up under the registered tests. Current pressure intensity was positively associated with commitment hazard in
both banks; the one registered cumulative-history term added no held-out log-loss
value. (The held-out result applies only to the history increment; current-intensity
association is in-sample.) A commitment axis could separate landing states
at held-out AUC 0.920, but the registered smooth-versus-step approach-drift
contrast was **−0.020 CI [−0.145, +0.108]**. The axis reads the landing but
does not resolve a distinct route into it under that statistic.

Here's the descriptive result I keep coming back to. Across 1,800 anchor
pseudo-orbits spanning seven assigned pressure levels, the turn-3 pre-response
states deepen monotonically with pressure intensity. Two independent depth
measures give median Spearman 0.714, and consecutive-level displacement fields
are nearly identical across scenario families: median cross-family field cosine
0.907. Pressure mounts along a shared field direction rather than accumulating as
an unstructured scalar. The orbits are pseudo-orbits with offset RNG seeds; I
never tested reversibility or closure, so it's a monotone one-parameter flow, not
a group action. It is in-sample on the same pressure banks used for the
registered comparisons, and pressure schedules, scenario text, and assistant
replies all co-vary across orbit positions. No group, semigroup, causal, or
universal transfer claim is implied. But the shape is coherent under this descriptive instrument.

The arm intervals are conversation-level Wilson/Newcombe, not family-clustered.
The dissociation hazard analysis uses 84/96 conversations after registered
exclusions, and its intensity/history correlation remains +0.332. I'm treating
the pressure profiles as measured, not as a perfectly realized intervention.

![Pressure behavior and commitment hazard](figures/pressure_behavior_and_hazard.png)

## The information-budget bridge

The post-action readout sets the ceiling. On 1,283 post-commitment events, a
relational residual-attention graph reaches family-macro Brier 0.0205 versus
0.0949 for an exact nuisance prior over sampled status, turn, intervention
history, and pressure. The nuisance-preserving permutation null reproduces mean
gain 0.04347, leaving an observed excess of 0.03089 (p=0.0001). Relational
geometry is not necessary for this readout: a same-bank raw-residual linear probe
reaches Brier 0.00150 and beats the graph in all 20 families.

Then I re-registered the test with a truth-aware cell. The original nuisance
prior is blind to scenario truth — it conditions on the sampled token, turn,
history, and pressure, but the label is defined against scenario truth. A
truth-aware prior (adding true and desired status) reaches Brier 0.0275, and the
graph's gain over it shrank to +0.00696, positive in 16/20 families but not
reliably above zero under the registered bootstrap criterion. Verdict:
refuted-under-adequate-instrument. The linear probe still beats the truth-aware
prior by +0.026. The honest summary: post-action activations contain within-cell
signal, but the graph's family-robust advantage over the truth-aware prior is not
established, and a linear probe reads the retained signal better.

Before action, the registered geometry lost. A one-state spectral field reached
AUROC 0.4168, a connection-response field 0.3725, and a masked geometry-only
risk model had log-loss gain −0.0218 over its nuisance prior (C11, refuted).
But the same instrument reads differently on discrimination: the geometry-only
model reaches AUROC 0.9442, the nuisance-only model 0.9446, and the two
combined reach 0.9537 (sealed prior 0.9489). These are receipt-bound
descriptive AUROC orderings, not a registered endpoint, and do not carry
compact-receipt-bound scenario-cluster intervals that exclude zero. The geometry adds a small ordering signal that the
registered calibrated-log-loss endpoint misses. Twenty token-identical
pre-action prefixes produced different outcomes — that's an empirical stochastic
floor for exact deterministic prediction, not a theorem that early warning is
impossible.

![Post-action decodability and pre-action warning](figures/decodability_timing_gap.png)

## Act III: universality became the boundary

This is where the clean story became conditional. If the structured-action field
learned a representation and a selector, should that representation survive
outside the task that produced it? The tests gave one directional fresh-family
positive, one clear response-aware tie, and weak cross-construction transfer.

**Fresh equivariant atlas.** A new 4-bit pilot within the same construction
tests fresh-family generalization, not cross-construction transfer. The
equivariant atlas selector achieves 71 corrected rows versus 64 for a
baseline, with confidence intervals that touch zero. A response-aware variant
that sees measured candidate response margins and outcomes ties plain margin
argmax at 79. It is
explicitly underpowered — the 4-bit precision and small bank size don't support
strong conclusions — and the directional gap does not establish a clear
separation.

**Representation transfer.** A receipt-bound Apollo comparison tests whether the
saved-field source representation carries usable geometry into a different
construction. The transferred detector reports AUROC 0.629 versus 0.816 for
native, 0.514 versus 0.782 for another contrast, and 0.571 versus 0.865. The
source and target banks share no construction architecture, and matched
target-native fits show model classes are capable in-domain; the gap is
consistent with task/domain mismatch but does not isolate its cause.

**Free-prose sign readout and the prospective controller failure.** After the
registered C5 behavioral endpoint, I checked whether a free-prose sign channel
could read the corrective target at all. The no-Z2 sign readout reaches 0.98579
versus 0.61559 for a forced-global-Z2 baseline on the same held-out-family rows.
This is a post-hoc readout from the controller receipt;
it cannot rescue the behavioral failure of C5. It is compatible with chart-local
structure but does not prove local gauge invariance or selective intervention.
The registered result stands: the native layer-16 geometric policy changed
deceptive status by 0.0000 (its gate fired on only 2/48 deceptive rows), a
frequently firing version reached −0.0833, and the family-matched linear
comparator reached +0.3333 (C5, refuted).

**Pilot steering and gauge control.** The 160-row 4-bit pilot shows tangent
steering fixing 48/80 deceptive reports, but matched random-gated (37/80) and
global-gated directions (40/80 each) recover much of it (C12). The gauge
controller authenticated roots against sealed source-bank queries and found only
21/402 roots with an active supported step; gauge-geodesic minus no-intervention
was 0.0000 CI [0, 0] (C13). The holonomy instrument cleared its adequacy gate
in 0/5 folds, so curvature was never adjudicated.

Two further analyses rejected specific priors. Learned dense dose did not beat
fixed high dose; both fixed 591/600 structured actions, while dense dose added
five honest harms (C2). The broad linear policy changes that would be needed to
match the geometric signal are large enough that they would alter honest-row
behavior — the fixed linear coordinate reaches the same ceiling as CNG in the
structured-action field, but its information budget is narrower (oracle-route
mapped directly to target), and it was identified retrospectively.

![Prospective natural-prose control result](figures/natural_prose_control_failure.png)

![Gauge replay contrasts and proposal-support counts](figures/gauge_control_null.png)

## The geometry organizes locally; I pre-imposed structure on it too early

I didn't build a universal deception controller. I found recoverable geometric
structure that is useful **when the information budget is matched to the task**,
and a set of universality attempts that weaken when the scope changes.

The structured-action field shows a clear hierarchy: stronger structural
features narrow the selection gap. A separate L20 linear gate makes 1,680
routing predictions on held-out fresh families drawn from the same process,
suggesting the oracle route might be learnable within this benchmark. The fresh
equivariant atlas is directionally above its route floor but underpowered; the
representation transfer weakens across constructions (0.629 vs 0.816); and the
free-prose readout that looks promising (0.986) cannot rescue the failed
prospective controller. The geometry **organizes** the activation space it was
built from — it does not **decide** the next task's outcome.

The cleanest conclusion I can defend: pressure on this model traces a monotone
one-parameter flow with a shared field direction. Local chart-conditioned
selection works inside the structured task that produced the field. Post-action
readout is strong but overwhelmingly captured by a linear probe. Pre-imposing
global or fixed structure — a forced global Z2 decomposition, a fixed tangent
actuator, or construction-independent transfer — is what prevented universality, not an
absence of geometric organisation. The geometry was doing something. I asked it
to do something else before understanding what "something" was.

What remains unbuilt is the obvious next experiment: an end-to-end learned route
with a dynamic multi-layer controller that attaches novel live typed state at
intermediate layers and updates whether to intervene, where, and at what dose as
natural-prose generation unfolds — fit on training families only, evaluated
against no intervention, fixed residual steering, matched-random, sign-flipped,
and shuffled-field controls. That system exceeded this study's time and compute
budget. I am not claiming it would work. The local structural evidence does not
disappear because the universal controller failed; it tells us what the next
controller would have to learn rather than assume.

---

[Frozen scientific artifact](https://github.com/rajarshighoshal/deception-pressure-geometry/tree/2746821b2b7bdeb3a17a438f1570b2907450d9ae),
[manuscript source](https://github.com/rajarshighoshal/deception-pressure-geometry/tree/2746821b2b7bdeb3a17a438f1570b2907450d9ae/paper),
[results registry](https://github.com/rajarshighoshal/deception-pressure-geometry/blob/2746821b2b7bdeb3a17a438f1570b2907450d9ae/docs/results_registry.yaml), and
[evidence-receipt manifest](https://github.com/rajarshighoshal/deception-pressure-geometry/blob/2746821b2b7bdeb3a17a438f1570b2907450d9ae/paper_artifacts/manifest.json).

If you think I got something wrong, I'd like to hear it.

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
