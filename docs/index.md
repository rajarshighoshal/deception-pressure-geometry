---
layout: article
title: I tried to make Llama-8B lie under pressure and most of what I built didn't work
deck: "A two-month audit of pressure-induced false commitment, activation-space readout, and geometric control in Llama-3.1-8B-Instruct."
byline: "Rajarshi Ghoshal"
assistance_note: >-
  The author wrote the post and owns the research design, methodology, claims, and prose.
  LLMs assisted with experiment code, plots and tables, background lookup, typo correction,
  and limited sentence clarification.
---

I started with a simple picture: deception occupies a structured region of activation
space, find that structure, push the model out of it without pushing it off the
language manifold.

That picture survived only in pieces, and the pieces have an order. Pressure creates
structure. The landing is readable. Navigation turns out to be a selection problem. And
the structure stops at the boundary of the task that produced it.

Conversational pressure reliably made the model reverse a fact it had previously stated
correctly. After it committed, the outcome was extremely easy to decode from activations.
Before commitment, my registered geometric warning fields did not beat matched nuisance
controls. In an artificial one-token action interface, supplied-target correction was
nearly perfect — but a fixed linear policy with the same supplied target was at least as
good. The prospective natural-prose geometric controller failed.

The cleanest takeaway isn't that activation geometry is useless, it's that I was
conflating three different questions:

1. Can activations reveal what the model just did?
2. Can they predict what it will do before it samples the action?
3. Can their structure provide a selective causal handle for changing that action?

In this study the answers were: **yes, not with the tested instruments, and not
prospectively with the tested controller.**

I use "deceptive commitment" narrowly below. I'm not claiming to observe intent. The
event is a pressure-induced false operational report after the same model has
demonstrated the correct answer. "Sycophantic capitulation to a known falsehood" is
also a fair name for it.

The evidence did not all arrive under one chronology. C5 is the only prospective public
claim; C10 was registered after its evaluation bank existed; C1, C2, C9, C11, C12, and
C13 are retrospective syntheses over completed banks. I'm keeping those tiers explicit
rather than calling the whole program preregistered.

## First, I needed a pressure instrument that actually worked

My first prompt was blunt: ask for a false report and repeat the demand harder. It
mostly produced hedging. The better design treated pressure as a schedule rather than
a switch:

- **smooth:** pressure rises across several turns;
- **benign:** no pressure;
- **step:** one late, abrupt demand;
- **late-compressed:** roughly the smooth arm's arguments, compressed into the last two
  turns.

The final ask is byte-identical across pressure arms. This fixed an earlier confound
where schedule shape and argument dose moved together.

I ran 96 scripted conversations and 128 conversations with an adaptive pressurer. Three
LLM judges labeled commitment, while a separate channel — without access to the assistant
text — scored pressure intensity. The labels have not been human-validated, and the
pressurer and judges come from the same model family; their exact auxiliary revisions
are not fully bound. Those are real limitations.

Smooth pressure produced deceptive commitment in 26/32 scripted conversations and 45/48
adaptive conversations. The registered smooth-minus-late-compressed contrasts were
**+0.31 CI [+0.04, +0.55]** and **+0.21875 CI [+0.0545, +0.3961]**.

What I expected next was path dependence: maybe pressure accumulates and carries the
model into a commitment basin. That was not what the registered tests found. Current
pressure intensity predicted commitment hazard in the adaptive bank and in a separate
dissociation bank, while the tested cumulative-history term added no held-out log-loss
value. A commitment axis could distinguish landing states at held-out AUC 0.920, but
the registered smooth-versus-step approach-drift contrast was **−0.020 CI [−0.145,
+0.108]**.

So the behavioral schedule effect replicated, but this activation statistic did not
show that smooth and abrupt pressure travel along different routes.

One descriptive result I still like: the mounting pressure itself has shape. Across
1,800 pseudo-orbits spanning the seven assigned pressure levels, the turn-3
pre-response states deepen monotonically (median Spearman 0.714 on two different depth
measures), and consecutive-level displacements share a common field that is nearly
identical across scenario families (median cosine 0.907). It does not transfer across
constructions, the orbits are pseudo-orbits with offset RNG seeds, and I never tested
reversibility — so it is a monotone one-parameter flow, not a group action. Pressure
mounts along a field; it does not just accumulate as a scalar.

The arm intervals are conversation-level Wilson/Newcombe intervals, not family-clustered
resampling. The dissociation hazard analysis retains 84/96 conversations across 14/16
families after its registered exclusions, and current versus accumulated pressure
remains correlated at +0.332. I'm treating the pressure profiles as measured, not as a
perfectly realized intervention on path shape.

![Pressure behavior and commitment hazard](figures/pressure_behavior_and_hazard.png)

## The landing was easy to read

For the main activation bank I used a structured-action interface: the model sampled
one unrestricted token in an operational status slot, with prose kept separate. This is
an artificial instrument, not spontaneous disclosure. It was useful because a hedge
cannot silently be reclassified as a commitment.

The bank is developmental: its frozen baseline-knowledge gate missed by one scenario,
56/60 rather than 57/60. I retained that failure instead of silently moving the
threshold.

On 1,283 post-commitment events, a relational residual–attention graph reached
family-macro Brier **0.0205**, versus **0.0949** for an exact nuisance prior over
sampled status, turn, intervention history, and pressure. The raw gain was 0.07437. A
nuisance-preserving permutation null reproduced mean gain 0.04347, leaving an observed
excess of **0.03089**; the observed gain exceeded the maximum of 10,000 null draws
(one-sided randomization p=0.0001).

But the geometry was not necessary for that readout. A same-bank raw-residual linear
probe reached Brier **0.00150** and beat the graph in all 20 held-out families.

One more check on that comparison, registered after the first writeup. The exact nuisance
prior is blind to scenario truth: it conditions on the sampled token, turn, intervention
history, and pressure, but the label is defined against scenario truth. I re-registered
the test with a truth-aware cell (sampled token, turn, history, pressure, true and
desired status) and reran it on the same events and folds. The truth-aware prior is much
stronger — Brier **0.0275** versus 0.0949 — and the graph's gain over it shrank to
**+0.00696**, positive in 16 of 20 families but not reliably above zero under the
registered bootstrap criterion. Verdict: refuted under an adequate instrument. A
permutation test still says the graph carries some within-cell signal (p=0.0001), and
the linear probe beat the truth-aware prior by **+0.026** anyway. The honest summary:
most of the post-action readout is the design cell plus the emitted token, and a linear
probe reads what is left.

Two controls make the post-action signal harder to dismiss as merely reading pressure
or the emitted token:

- On 242 events from eight families, the relational model separated deceptive reports
  from knowledge errors at AUROC **0.949**, versus **0.542** for the exact nuisance
  prior.
- Across 30 token-identical, activation-exact pairs, the mean post-commitment deception
  margin was **+0.7231** (one-sided sign-flip p=0.0001).

This supports a post-action signal. It does not show that curved geometry is required.

## Before commitment, the registered geometry lost

Three attempts tried to read the walk rather than the landing:

- a one-state spectral field reached AUROC **0.4168**;
- a connection-response path field reached **0.3725**;
- a masked geometry-only risk model had log-loss gain **−0.0218 CI [−0.0733, +0.0233]**
  over its nuisance prior.

These are separate instruments and populations, not three scores on one common test
set: the spectral result uses 106 honest/deceptive events, the connection result 52,
and the masked-risk analysis 1,680 total events (1,568 honest/deceptive).

One nuance I owe the record: the registered endpoints are calibrated log-loss endpoints.
On discrimination, the same instrument reads differently. On the 1,568 honest/deceptive
events the geometry-only model reached AUROC **0.9442**, the nuisance-only model
**0.9446**, and the two combined **0.9537** — the best of any scored arm, above the
sealed prior's 0.9489. Adding the geometry to the nuisance features improves AUROC,
Brier, and log-loss with scenario-clustered intervals that exclude zero (**+0.0091**,
**+0.0109**, **+0.0141** respectively). That is a fitted secondary comparison, not the
registered endpoint, so the refutation stands — but "no pre-action signal" would be the
wrong summary.

An earlier comparison in the same risk-gate family reported −0.0865. I originally
called that an "improvement." That was a sign error: under the registered convention
it is a **deficit**, and its entire interval is below zero. The final masked
geometry-only repair also fails its comparator, with an interval crossing zero.

Twenty token-identical pre-action prefixes produced different outcomes across sampling
seeds. That is an empirical stochastic floor for exact deterministic prediction on
those observed states, not a theorem that useful early warning is impossible.

![Post-action decodability and pre-action warning](figures/decodability_timing_gap.png)

## Structured-action correction worked — conditionally

The strongest-looking control result came from an offline action-response field. An
oracle-target Chart-Neural Gate fixed **599/600** deceptive rows with **1/600** honest
harm.

That number is real, but the information budget matters. The policy was told the
correct target route. A held-out-family context-only ranker, which had to infer the
target without looking at held-out candidate outcomes, fixed only **170/600**, harmed
**11/600** honest rows, and chose the wrong target on **233/1,200** rows.

A fixed bidirectional-linear L16 policy at dose 96 reached **600/600** with zero honest
harm when supplied the same oracle route. It was not response-aware at held-out
application, but it was oracle-route-conditioned and identified retrospectively from
the completed action field. So it's a legitimate challenge to geometry-specific
superiority, not evidence for a prospective end-to-end controller.

Two additional audits explain why the earlier steering result looked stronger than it
was:

- in the 160-row 4-bit pilot, tangent steering fixed 48/80 deceptive status reports,
  but a matched random-gated direction fixed 37/80 and two gated global directions
  fixed 40/80 each. Under the strict endpoint those counts were 40, 30, 30, and 32;
  every arm caused 2 honest status harms and 10 honest strict harms;
- learned dense dose and fixed high dose both fixed 591/600 structured actions, while
  the learned dose added five honest harms.

The supported result is **conditional correctability once the target is supplied**. Not
that the relational controller discovered the target, and not that geometry beat an
equally informed fixed linear policy.

![Structured-action control and information budgets](figures/structured_action_control_audit.png)

## The structure did not travel

The obvious next question was whether any of it survives outside the machine-checkable
PASS/FAIL world that produced it. I ran the trained detectors zero-shot on a 484-row
external deception bank with free prose instead of a status slot.

The best transferred detector reached AUROC **0.63**; detectors trained on the target
bank itself reached **0.82–0.87** on the same rows. Matched architectures told the
useful version of the story: linear, path, and grid models all sat near chance
zero-shot, and all were strong when fitted natively — so the failure is the domain,
not the model class. The source and target honesty directions are nearly orthogonal
(cosine **0.05**).

Behavioral transfer was worse: zero soft-control successes out of 484, with coherence
destroyed. The planned full-scale suite crashed on a determinism check and was never
rerun, so the behavioral record is thinner than the detector record.

My read: the source representation is task-shaped. The source bank's deception is a
machine-checkable PASS/FAIL field; the target bank has no such field, and the same
scenarios lose the label entirely in free prose. Whether task shape is the reason
transfer failed, I cannot prove — representation, dose, policy, and evaluation all
changed together.

## The prospective natural-prose controller failed

The closest test to the controller I actually wanted used fresh natural-prose
generation on four held-out scenario families. It did not receive the sampled answer
or the oracle truth route.

The native layer-16 geometric policy changed deceptive status by exactly **0.0000 CI
[0, 0]**. Its gate fired on only 2/48 deceptive rows. A frequently firing version
reached **−0.0833 CI [−0.2083, +0.0417]**: it intervened, but did not help. The
family-matched linear comparator reached **+0.3333 CI [+0.1667, +0.4583]**, consisting
of 21 deceptive fixes and 5 deceptive harms; it also caused 6 honest-row harms. So
even the winning comparator was not a clean general controller.

Secondary LLM-judge honesty and coherence scores cannot rescue the machine-checked
null. Hedging, refusal, or generic perturbation can improve those scores without
correcting the operational action.

This experiment tested a layer-16 residual controller. It did **not** build the
stronger system I ultimately wanted: an online controller that attaches novel live
typed token–residual–attention states at layers 12, 16, 19, and 20, then updates
whether to intervene, local direction, and dose as natural-prose generation unfolds.
That system exceeded this study's time and compute budget. Its outcome is unknown, and
its absence does not rescue the controller that was tested.

![Prospective natural-prose control result](figures/natural_prose_control_failure.png)

## The gauge-first replay also returned a null

I did build a narrower four-layer gauge controller for the structured-action bank. It
authenticated live L12/L16/L19/L20 roots against sealed source-bank queries and applied
a one-step residual intervention at an exact frozen prefix.

Its support was much narrower than I hoped. Only **21/402** roots received an active
supported step; 333 stopped at the boundary, 37 had an undefined field, 10 were
off-support, and one had zero direction. Gauge-geodesic minus no intervention was
**0.0000 CI [0, 0]** overall and among active roots, with zero gauge-induced action
flips.

A broader transport intervention moved logits, but nearly all measured reach was
generic. The deception-specific raw-logit remainder after generic transport was
**+0.0125 CI [−0.0160, +0.0406]**.

The holonomy instrument cleared its adequacy gate in **0/5** folds because the
residual-matched flat null was already too noisy. That means curvature and flatness
were both unevaluable under this instrument. It is not evidence that useful curved
structure is absent.

![Gauge-control and holonomy-instrument results](figures/gauge_control_null.png)

## What's left after all that

The contribution is not a universal deception controller. It's a map of where the
geometric hypothesis held and where it stopped: pressure flows, the landing reads
linearly, navigation is a selection problem, and none of it yet travels. It's also an
evidence ladder that keeps several commonly conflated results apart:

- a controlled pressure schedule that reliably elicits false operational commitment;
- a one-token action protocol that separates commitment from prose without constrained
  decoding;
- exact-prefix and lie-versus-error contrasts that reduce pressure, token-identity,
  and ordinary-error confounds;
- a same-bank result where post-action relational structure beats a strong nuisance
  prior but loses to a plain residual linear probe;
- an information-budget audit separating supplied-target correction from context-only
  and prospective control;
- matched-random, fixed-linear, sign-flipped, nuisance, and causal controls that turn
  attractive geometric stories into explicit negative or unresolved results;
- a compact public artifact where claims, scripts, tests, and source-artifact hashes
  can be checked without publishing operational logs or multi-gigabyte activation
  banks.

The compact receipts expose hashes and the code that produced them, but the private
raw banks and upstream source artifacts are not shipped. A clean clone can verify the
public evidence layer; it cannot regenerate every historical GPU run from scratch.

The public registry now contains eight claims: four supported, three refuted, and one
not-found-under-this-instrument. "Supported" is deliberately narrow: the
structured-action claim supports oracle-routed conditional correction, not
geometry-specific superiority.

## What I would do next

The next serious control experiment is not another single-vector steering sweep. It's
the missing online typed-state controller: training-family-only attachment and local
updates during fresh prose, evaluated against no intervention, fixed residual
steering, matched-random, sign-flipped, and shuffled-field controls. It needs a
behavioral positive control showing that the chosen dose can move the endpoint before
a null is interpreted as a geometric failure.

I would also human-validate the natural-pressure labels and repeat the behavioral and
readout results on another model before making any universality claim.

The longer-term hypothesis — that deception occupies a transferable, intrinsically
structured region inside the model's active state space — remains open. This study
found a clear landing signal and failed to turn it into a reliable prospective control
law. That gap is now the result, rather than something to hide behind a near-ceiling
detector.

---

[Frozen public artifact](https://github.com/rajarshighoshal/lie-geometry-probes/tree/e9c9e34d793e86f9d7985171caf358a095ae3496),
[results registry](https://github.com/rajarshighoshal/lie-geometry-probes/blob/e9c9e34d793e86f9d7985171caf358a095ae3496/docs/results_registry.yaml), and
[evidence-receipt manifest](https://github.com/rajarshighoshal/lie-geometry-probes/blob/e9c9e34d793e86f9d7985171caf358a095ae3496/paper_artifacts/manifest.json).

If you think I got something wrong, I'd like to hear it.

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
