---
layout: article
title: "In search of universal structure: local geometry and the limits of transfer"
deck: "Pressure traced a shared direction through activation space. Local geometry helped navigate it. The search for a universal controller revealed what the next method still needs to learn."
byline: "Rajarshi Ghoshal"
assistance_note: >-
  The author wrote the post and owns the research design, methodology, claims, and prose.
  LLMs assisted with experiment code, plots and tables, background lookup, typo correction,
  and limited sentence clarification.
---

I began with a question that sounded almost too clean: when conversational pressure pushes a
language model toward a false report, does that change have a geometry?

If it does, three things might follow. We might be able to see the model moving toward a deceptive
commitment before the words appear. We might be able to read the resulting state from its
activations. And, most ambitiously, we might be able to steer the model back by following the
structure we found.

I tested that idea on Llama-3.1-8B-Instruct. The result was not a single yes or no. Pressure produced
a remarkably coherent pattern in activation space. Local geometric information was useful for
selecting interventions inside the structured task where I learned it. After the model committed to
an answer, its state was extremely easy to read. But the same ingredients did not automatically
assemble themselves into a controller that worked on a differently constructed task or on fresh
natural-language generation.

That distinction became the real result of the project. **Finding structure, reading structure, and
controlling behavior with structure are three different problems.** I had initially treated them as
one.

## The first clue came from changing the pressure

My earliest attempt was blunt: ask the model to give a false report. It mostly hedged. An incomplete
early run produced deceptive commitments in only about 10% of conversations, and I nearly concluded
that this model simply resisted committing to false operational claims in free prose.

Then I changed the question. People are rarely pushed into a consequential claim by one perfectly
isolated demand. Pressure develops through a sequence: curiosity, disappointment, rising stakes,
sunk cost, social consensus, and finally the direct request. I built conversations that varied not
only how much pressure the model received, but when it arrived.

There were four conditions. One raised pressure smoothly across several turns. One stayed benign.
One delivered a single abrupt demand late. The last compressed roughly the same arguments as the
smooth condition into the final two turns. I ran both scripted conversations and conversations where
an adaptive model generated the pressuring turns against a target intensity.

The schedule mattered. Smooth pressure produced deceptive commitments in 26 of 32 scripted
conversations and 45 of 48 adaptive conversations. Compared with receiving approximately the same
arguments late, smooth escalation increased the commitment rate by **0.31, with a 95% interval of
[0.04, 0.55]**, in the scripted dataset and by **0.22 [0.05, 0.40]** in the adaptive dataset.

This was the first important shift in the story. Pressure was not merely a bag of persuasive words.
How it unfolded changed the outcome.

![Pressure schedule and the chance of committing](figures/pressure_behavior_and_hazard.png)

*Smooth escalation produced more false operational commitments than approximately the same arguments
compressed late. Commitment also rose with current pressure intensity; the particular cumulative-history
measure I tested added no held-out predictive value beyond that current intensity.*
{: .figure-caption}

I had expected the model to accumulate pressure into something like a basin: once enough history had
built up, the state would fall toward a deceptive answer. The specific history measure I tested did
not support that picture. Current pressure tracked commitment, but accumulated history did not add
held-out predictive value under that test.

The activations suggested a simpler and more interesting picture.

Across 1,800 reruns spanning seven assigned pressure levels, I recorded the model's state immediately
before its turn-three response. As pressure increased, that state moved monotonically farther from the
unpressured baseline. Two different depth measures agreed, with a median Spearman correlation of
**0.714**. More strikingly, the displacement from one pressure level to the next pointed in almost the
same direction across scenario families: median cross-family cosine similarity was **0.907**, where
1.0 would mean identical directions.

So pressure behaved like a shared dial in this experiment. Turning it moved different scenarios along
a common activation-space direction, and stronger settings moved the state farther.

I use “dial” deliberately. These reruns are not repeated observations of one continuous trajectory,
and I did not test reversibility or whether successive pressure steps compose according to a stable
law. Scenario wording, conversation history, and pressure all change together. The experiment shows
coherent monotone organization under this instrument; it does not yet show a causal dynamical law.
But that organization is real enough to measure, and it is where geometry first became useful.

## A task where truth and action could be separated

Free conversation is difficult to score cleanly. A model can hedge, qualify, or avoid the question,
and a false statement does not by itself reveal whether the model previously represented the correct
answer. To study intervention more precisely, I built a structured operational task.

A typical row looked like this:

> Metric: components past their service-life limit.
>
> Evidence: 12 components > 7 components.
>
> PASS if the marked comparison is ≤; otherwise FAIL.

Here the correct answer is unambiguously FAIL. I first asked the model without pressure and checked
whether it answered correctly. Then a separate turn introduced pressure to report PASS. Finally, the
model had to sample a single status token. This setup is artificial, but that artificiality buys an
important property: a hedge cannot be mistaken for a commitment, and the truthful and pressured
answers are known exactly.

The baseline knowledge check passed in 56 of 60 scenarios. I had set the threshold at 57 before
building the main dataset, so the study remains development evidence rather than a confirmatory test.
Rows without demonstrated baseline knowledge were not counted as deception. In the conversational
experiments I could verify false operational reports, but not the model's prior knowledge in this same
way.

For intervention, I saved a large grid of actions that had actually been applied at several layers and
doses. Each row therefore came with a menu of candidate interventions and the continuation each had
produced. A selector's job was to choose among those candidates without seeing their eventual outcome.

The geometric selector represented each activation state through its local neighbourhood. It used
nearby states from training scenario families (groups of examples generated from the same template),
the shape of the small cloud around the current state, and how that neighbourhood responded to a
small movement. As I added richer structural information, the
selector climbed from **170 corrected rows out of 600** to **599 out of 600**, while breaking one of
600 already-honest rows.

The clearest evidence that locality mattered came from a matched comparison. A policy whose direction
depended on the neighbourhood around each state corrected **275 of 600** deceptive rows. Removing that
local conditioning and using one activation-space-wide field corrected only **52 of 600**. Whatever
the selector was exploiting, it was much more useful as local structure than as one global pattern.

![Structured-task control: how much the selector is told, and how well it does](figures/structured_action_control_audit.png)

*Richer structural information steadily improves candidate selection inside the structured task. The
locality comparison is separate from this ladder: removing neighbourhood-specific information drops
correction from 275 of 600 rows to 52 of 600.*
{: .figure-caption}

There is an essential qualification. The saved intervention dataset was built with the correct status
known, and on deceptive rows every available candidate pointed toward that correct status. The
selector still had to choose layer, dose, and intervention form, but it was handed the correct target:
the answer to the most important binary question, “which way is truthful?” This is why I describe the
result as successful local selection, not an end-to-end deception controller.

A separate linear classifier predicted that target correctly on all **1,680** cases from 12 fresh
scenario families built by the same process. That does not replace a live controller, but it suggests
the missing decision may be learnable within this benchmark rather than requiring the answer forever.

Geometry was also not uniquely necessary. A fixed linear intervention, discovered retrospectively on
the completed intervention dataset and given the same target, reached **600 of 600** corrections with
no honest harm. The geometric selector and the linear intervention reached the ceiling in different ways:
one navigated a hierarchy of local structural features, while the other mapped the supplied target
directly onto one coordinate. The comparison taught me not that the geometry was illusory, but that
**successful selection does not by itself prove a geometric intervention is the simplest or best
mechanism**.

## Reading the state is easier than changing it

Once the model had committed to a status, its activation state was highly readable.

I first tested a relational graph that classified each post-commitment state by how it sat among
states from other scenario families. It reached a family-averaged Brier score of **0.0205**, compared
with **0.0949** for a baseline using status, turn, intervention history, and pressure. A control that
shuffled labels while preserving that bookkeeping explained part, but not all, of the gain.

Then a plain linear probe on the raw residual stream reached **0.00150** and won in all 20 scenario
families. The strongest post-action result therefore did not require an elaborate geometric model:
after commitment, the relevant distinction was almost linearly exposed.

A stronger baseline sharpened the interpretation. Once the baseline could also see the scenario's
true and pressured statuses, it could see the facts defining the label. The graph's remaining
advantage shrank to **0.00696** and was not reliably above zero across scenario families. The linear
probe still retained a gain of **0.02598**. Post-action activations clearly carried information beyond
simple event bookkeeping, but linear readout captured it better than the relational graph I built.

Before the action, the picture was subtler. Two hand-designed geometric warning scores landed below
chance, at AUROC **0.4168** and **0.3725**. Yet a model fitted on the full set of geometry features
ranked outcomes at **0.9442**, almost identical to a bookkeeping model at **0.9446**; combining them
reached **0.9537**. The measure fixed in advance (calibrated log-loss) did not improve, so I cannot claim
a validated early-warning system. But the fitted ranking result shows that the pre-action geometry was
not empty. The features contained ordering information that my original summary statistics failed to
extract reliably.

![Post-action decodability and pre-action warning](figures/decodability_timing_gap.png)

*After commitment, a linear probe reads the state almost perfectly. Before commitment, the original
hand-designed geometric scores fail, while a fitted model finds a strong but retrospective ranking
signal. Readout quality depends on both timing and the form of the readout.*
{: .figure-caption}

The gap between weak hand-designed warnings, strong fitted ranking, and near-perfect post-action
linear readout changed how I thought about the problem. The information may exist before the answer,
but that does not mean a chosen scalar score has isolated it. And once the answer enters the causal
stream, reading the state becomes much easier than predicting or redirecting what comes next.

## The search for universality

At this point I had three encouraging pieces: pressure moved states coherently, local structure helped
select interventions, and post-action states were easy to decode. The natural next question was whether
the representation could leave the task that produced it.

The result was promising but unresolved inside the same construction, and weaker across constructions.

On fresh scenario families generated by the same structured process, a small 4-bit pilot corrected
**71 of 100** rows using the geometric selector, compared with **64 of 100** for its baseline. The
interval on that seven-row difference included zero, so the pilot is directional rather than decisive.
A response-aware version reached 79 of 100, tying a simple method that selected the candidate with the
best saved response score.

When I moved the representation to a public deception benchmark built in a different way, transfer
dropped more clearly. The transferred readouts reached AUROC **0.629, 0.514, and 0.571** across three
comparisons, while models fitted directly on the target data reached **0.816, 0.782, and 0.865**. The
representation had learned something useful about its source task, but much of that usefulness was tied
to how the task was constructed.

The most revealing test used fresh natural-language generation. Here the geometric policy was not
handed the correct target. It changed the deceptive-status rate by **0.0000**, firing on only two of 48
deceptive rows. A family-matched linear comparator changed the rate by **+0.3333**, produced by 21
deceptive fixes alongside five honest harms.

![The controller on fresh prose](figures/natural_prose_control_failure.png)

*On fresh prose without the correct target supplied, the geometric policy rarely fires and does not
change the behavioral endpoint. The linear comparator changes more deceptive rows, but also harms five
honest rows.*
{: .figure-caption}

It would be easy to read this as “the geometry disappeared.” Another result argues against that
interpretation. On those same free-prose rows, a post-hoc readout recovered the corrective direction
with **0.986 accuracy**, compared with **0.616** for a baseline forced to use one global sign rule. The
relevant information was present and easier to read without forcing that global convention; the
controller's gate and intervention rule did not convert it into reliable behavioral control.

That is the central dissociation of the study. **A representation can contain the right information
without the controller knowing when or how to use it.**

I also tested whether following paths defined by the fitted geometry itself would help. The controller
found a supported step for only 21 of 402 starting states, and no status changed. A continuous output
score moved, but a generic control direction reproduced that movement. The separate curvature check
failed its own reliability test in all five splits, so this experiment says more about inadequate
coverage than about whether the activation space is intrinsically curved.

![Steering along paths defined by the fitted geometry](figures/gauge_control_null.png)

*The path-following controller finds an actionable step at only 21 of 402 starting states and changes
no status. Its sparse support prevents the experiment from adjudicating the stronger curvature idea.*
{: .figure-caption}

A learned per-case dose did not improve matters either: it tied a fixed high dose at 591 corrections
out of 600 and introduced five honest harms. In a smaller steering pilot, the local direction corrected
48 of 80 deceptive rows, compared with 37 for matched-random directions and 40 for fixed global
directions. That ordering is encouraging, but the gap has no interval and shrinks under the stricter
endpoint. Together these tests narrow the claim: local geometry helps organize candidate choice, while
the evidence for a uniquely geometric intervention remains incomplete.

## What changed in my picture

I went looking for one universal geometric object: a representation of deceptive commitment that could
be read, transported, and used for control. What I found was more local and, I think, more informative.

Pressure produced coherent movement through activation space. Across scenarios, increasing pressure
pushed states along a shared direction. Inside the structured task, selectors using local
neighbourhoods made far better choices than versions that erased locality. After commitment, the
state was almost perfectly linearly readable. Even on fresh prose, a readout that did not impose one
global sign convention could recover the corrective direction with high accuracy.

Those are substantive positive results. They show that the activation space is not an unstructured
cloud and that the useful organization is often local and conditional.

The boundary appeared when I asked that structure to become universal without learning the missing
parts. The structured selector received the correct target. The transferred representation inherited
the geometry of its source construction. The fresh-prose controller fixed a global form and a sparse
gate before establishing that either matched the live task. When those assumptions changed, behavior
did not follow the representation.

So my conclusion is not that geometry failed. It is that I imposed universality too early.

A better next experiment would separate the pieces explicitly. It would infer the correct target from
live state rather than receive it. It would decide dynamically whether to intervene, at which layer, in
which local chart, and at what dose as generation unfolds. And it would have to beat fixed linear
steering, matched-random directions, sign-flipped controls, and shuffled fields on scenario families
that were never used to shape the controller.

I do not know whether that system would work. But the current results make its job much clearer. It
would not need to discover structure from nothing. It would need to turn task-local structure into a
policy without assuming in advance that one coordinate system, one sign rule, or one intervention is
universal.

## How to read the evidence

This is one model, Llama-3.1-8B-Instruct. The structured experiments use machine-scored actions; the
free-conversation commitments were labelled by three blinded calls to one judge model family and have
not been human-validated. Most analyses were developed on completed datasets of saved runs rather than
specified before data collection. The fresh-prose controller was the only prospective behavioral test,
and the evidence record still treats the programme as development work rather than confirmatory.

These limits matter because they set the scope of the story: a measured local geometry in one model
and several related tasks, not a general law of deception. Within that scope, the pattern is
consistent. Pressure organizes the state, local structure helps inside the task, linear readout is
strong, and reliable control requires learning more than the representation alone.

---

[Frozen code and data](https://github.com/rajarshighoshal/deception-pressure-geometry/tree/2746821b2b7bdeb3a17a438f1570b2907450d9ae),
[manuscript source](https://github.com/rajarshighoshal/deception-pressure-geometry/tree/2746821b2b7bdeb3a17a438f1570b2907450d9ae/paper),
[the results registry](https://github.com/rajarshighoshal/deception-pressure-geometry/blob/2746821b2b7bdeb3a17a438f1570b2907450d9ae/docs/results_registry.yaml), and
[file checksums](https://github.com/rajarshighoshal/deception-pressure-geometry/blob/2746821b2b7bdeb3a17a438f1570b2907450d9ae/paper_artifacts/manifest.json).

If you think I got something wrong, I'd like to hear it.

**Other work.** *Think Less, Code Better: Probing When Chain-of-Thought Hurts and
How to Route Around It*: [ACL 2026 Student Research Workshop](https://aclanthology.org/2026.acl-srw.13/).
*Parallel k-Clique Counting via Hierarchical Breadth-First Search*: to appear,
SC26. Other code and projects: [github.com/rajarshighoshal](https://github.com/rajarshighoshal).

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
