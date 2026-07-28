---
layout: article
title: "Deception under pressure: reading and steering the geometry"
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

## The ledger

Eight registered claims: four supported, three refuted, one not adjudicated by
its instrument. Every verdict below is bounded — the boundaries are in the
[results registry](https://github.com/rajarshighoshal/deception-pressure-geometry/blob/2746821b2b7bdeb3a17a438f1570b2907450d9ae/docs/results_registry.yaml)
and unpacked in the sections that follow.

| Claim | Verdict | Finding | Key number |
|---|---|---|---|
| **C1** | Supported | Oracle-routed structured-action correction works — but a fixed linear policy ties it | 599/600 fixed, 1/600 harm; fixed linear 600/600 |
| **C9** | Supported | Smooth pressure raises deceptive commitment above a matched-argument control | +0.31 CI [+0.04, +0.55] |
| **C10** | Supported | Post-commitment states decode beyond nuisance — but a linear probe wins | Brier 0.0205 graph vs **0.00150** linear |
| **C12** | Supported | Tangent steering leads on points; matched random/global directions recover most of it | 48/80 vs 37–40/80 |
| **C2** | Refuted | Learned dense dose does not beat fixed high dose | both 591/600; dense adds 5 honest harms |
| **C5** | Refuted | The prospective natural-prose geometric controller fails; the linear comparator doesn't | 0.0000 vs **+0.3333** |
| **C11** | Refuted | Pre-commitment geometric warning fields lose to matched baselines | AUROC 0.4168 / 0.3725 |
| **C13** | Not found under instrument | Gauge-geodesic control is behaviourally null; holonomy unresolvable | 0.0000 CI [0, 0]; adequacy 0/5 folds |

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
outcomes, or rewards. The hierarchy in the figure below is the point: the more
structure a selector is given, the closer it climbs to the ceiling — a weak ranker
that sees the route only as a feature sits at the bottom, and richer structural
selectors close the gap almost entirely. This is a selection result, not evidence
that applying a curved direction is intrinsically superior.

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

*Structural selectors climb from 170 to 599 of 600 as they get richer — but a
retrospectively identified fixed linear policy reaches the same ceiling. Geometry
selects well here; it isn't necessary here.*
{: .figure-caption}

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

Smooth pressure produced deceptive commitment in the large majority of
conversations in both banks. The comparison that carries the weight is smooth
against late-compressed: the same arguments, matched within 25% on word count,
differing only in when they arrive. The registered effect is positive in both
banks, though scripted-arm realization was mixed. Schedule shape matters
independently of argument dose.

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

*Schedule shape matters beyond argument dose: smooth escalation beats the same
arguments compressed late. Commitment tracks current intensity — the cumulative
history term adds no held-out value.*
{: .figure-caption}

## The information-budget bridge

The post-action readout sets the ceiling. On 1,283 post-commitment events, a
relational residual-attention graph beats an exact nuisance prior keyed on
sampled status, turn, intervention history, and pressure, and it survives a
nuisance-preserving permutation null (excess 0.03089, p=0.0001). But the geometry
is not *necessary* for that readout: a same-bank raw-residual linear probe beats
the graph in all 20 families.

Then I re-registered the test with a truth-aware cell, because the original prior
is blind to scenario truth — it conditions on the sampled token, turn, history,
and pressure, while the label itself is defined against scenario truth. Against
a prior that sees true and desired status, the graph's advantage shrank to
+0.00696: positive in 16 of 20 families, not reliably above zero under the
registered bootstrap criterion. Verdict: refuted-under-adequate-instrument. The
honest summary is that post-action activations carry within-cell signal, the
graph's family-robust advantage over a truth-aware prior is not established, and
a linear probe reads whatever is retained better than the geometry does.

Before action, the registered geometry lost outright: both field instruments
landed below chance and the masked geometry-only risk model lost to its nuisance
prior (C11, refuted). The same artifact reads differently on pure discrimination
— geometry-only 0.9442, nuisance-only 0.9446, the two combined 0.9537 — but
those are receipt-bound descriptive orderings, not a registered endpoint, and
they carry no cluster intervals excluding zero. The geometry adds a small
ordering signal that the registered calibrated-log-loss endpoint misses. Twenty
token-identical pre-action prefixes produced different outcomes, which is an
empirical stochastic floor for exact deterministic prediction, not a theorem that
early warning is impossible.

![Post-action decodability and pre-action warning](figures/decodability_timing_gap.png)

*The timing gap, and the core asymmetry of this study: after the model commits,
the state is near-perfectly readable; before it commits, the tested geometric
fields sit at or below chance.*
{: .figure-caption}

## Act III: universality became the boundary

This is where the clean story became conditional. If the structured-action field
learned a representation and a selector, should that representation survive
outside the task that produced it? Seven tests ask versions of that question, and
they weaken in one direction: every time I removed information the geometry had
been fitted on — the oracle route, the construction, the saved candidate field —
the advantage fell back toward its baseline.

| Test | What it asked | Result |
|---|---|---|
| **Fresh equivariant atlas** (4-bit) | Does the selector generalise to fresh families *inside* the same construction? | 71 vs 64 route floor — directional, CI touches zero; a response-aware variant ties margin argmax at 79 |
| **Representation transfer** (Apollo) | Does the source representation carry into a *different* construction? | AUROC 0.629 / 0.514 / 0.571 against native 0.816 / 0.782 / 0.865 |
| **Free-prose sign readout** | Can a sign channel read the corrective target at all? | 0.986 vs 0.616 for a forced-global-Z2 baseline — post-hoc only |
| **Prospective controller** (C5) | Does it steer fresh prose *without* the oracle route? | 0.0000, gate firing on 2/48 deceptive rows; linear comparator +0.3333 |
| **Pilot steering** (C12) | Is tangent geometry the active ingredient? | 48/80, against 37/80 matched-random and 40/80 global-gated |
| **Gauge control** (C13) | Does gauge-geodesic transport steer? | 21/402 supported roots, 0.0000 CI [0, 0]; holonomy adequacy 0/5 folds |
| **Learned dose** (C2) | Does a learned dense dose beat a fixed high dose? | Tie at 591/600; the learned dose adds five honest harms |

The atlas pilot is 4-bit and underpowered, so a directional gap establishes
nothing by itself. The Apollo source and target banks share no construction
architecture, and matched target-native fits show the model classes are capable
in-domain — the gap is consistent with task mismatch but does not isolate its
cause. The gauge replay authenticated its live roots against sealed source-bank
queries, and the holonomy failure means curvature was never adjudicated either
way.

The free-prose readout is the one that stings. A sign channel reads the
corrective target almost perfectly on exactly the rows where the controller moved
nothing. Reading and steering came apart in the same experiment: the information
is present in the state, and the intervention I built on top of it still did not
change what the model did. Note too that the fixed linear coordinate reaches
CNG's ceiling in the structured-action field on a *narrower* budget — the oracle
route mapped straight to the target — and was identified retrospectively.

![Prospective natural-prose control result](figures/natural_prose_control_failure.png)

*Take away the oracle route and move to fresh prose, and the geometric policy
changes deceptive status by exactly nothing — while the family-matched linear
comparator moves it by +0.3333.*
{: .figure-caption}

![Gauge replay contrasts and proposal-support counts](figures/gauge_control_null.png)

*The gauge controller proposes a supported step on only 21 of 402 roots, and its
effect is null with zero action flips. The holonomy instrument fails adequacy in
all five folds, so curvature is unevaluable — not absent.*
{: .figure-caption}

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

**Other work.** *Think Less, Code Better: Probing When Chain-of-Thought Hurts and
How to Route Around It* — [ACL 2026 Student Research Workshop](https://aclanthology.org/2026.acl-srw.13/).
*Parallel k-Clique Counting via Hierarchical Breadth-First Search* — to appear,
SC26. Other code and projects: [github.com/rajarshighoshal](https://github.com/rajarshighoshal).

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
