---
layout: article
title: The state before the model lies is an address
deck: "What happened after the first study: a new question, a positive result, and the controls that kept it honest."
byline: "Rajarshi Ghoshal"
assistance_note: >-
  The author owns the research design, methodology, claims, and prose. LLMs assisted
  with experiment code, plots and tables, background lookup, typo correction, and
  drafting support.
---

A few weeks ago I finished the first study and its summary was blunt: most of what I
built didn't work. Linear probes won the reading problem. Structured selection won
one control problem. Almost everything else lost to its own controls.

That study closed one question and left a sharper one behind. The old question was
"does this activation state look deceptive?" — a single state, a label. The new one:

**Can I predict the move itself — the displacement between the state that is about to
lie and the state that stays honest — before the model commits?**

Not classify a state. Predict a transition.

## The object: a displacement, not a label

The bank is the same structured deception task as before. Sixty machine-checkable
scenarios, twenty families. The model first demonstrates the correct answer with no
pressure. Then, under graded pressure, it samples a one-token status report — honest
or deceptive. I capture the activation state at the anchor right before that token.

For each scenario I now have pairs. Picture two identical conversations up to the
pressure point: in one the model caves, in one it holds. The pre-commitment state of
the first, and the pre-commitment state of the second. The **displacement** is just
the arrow between those two states. The question becomes: given a new deceptive state
from a scenario family the estimator has never seen, can I reconstruct what the arrow
to honest looks like there — before ever seeing the honest branch?

This is a different object from a probe. A probe says what a state *is*. This asks
where the state *would move*, and from what information.

## The headline: yes, and by a wide margin

I retrieve the displacement by neighborhood: find training scenarios whose source
states are near the query state, average their measured displacements, reconstruct.
Held-out families only.

| estimator | held-out reconstruction (cosine) | coverage |
|---|---:|---:|
| raw-activation nearest 8 neighbors | **0.9326** | 200/200 |
| raw-activation single nearest | **0.9304** | 200/200 |
| typed-graph neighborhood | **0.9289** | 193/200 |
| truth-aware design-cell mean | **0.9223** | 200/200 |
| typed-graph single nearest | **0.9149** | 193/200 |
| one global mean direction | 0.4839 | 200/200 |
| reusing other roots' targets (shuffle) | 0.4228 | 193/200 |

![Reconstruction chart](figures/representation_reconstruction.png)

Local retrieval beats the global direction by **+0.4252 [+0.3799, +0.4638]** — the
largest gap I have measured in this program. A sign-flipped control lands at −0.93,
so the direction is real, not a cosine artifact.

## The twist: it is lookup, not machinery

I built the first estimator on a typed relational graph — residual streams, attention
rows, typed metadata. Then I ran the registered simple-address diagnostic to check
whether any of that machinery mattered.

It did not. Plain Euclidean nearest neighbors on the raw activations do as well or
better (0.9304–0.9326). Averaging over a neighborhood adds only **+0.0140** over one
retrieved exemplar — one lookup nearly saturates it. Even a mean over the
truth-aware design cell (turn, intervention history, pressure, true status) reaches
0.9223.

So the property is not the graph. The property is that **displacement is smooth over
local source-state similarity** — and largely over the design cell itself.

How largely? I asked the sharpest version: restricted to the query's own design
cell, does the activation-nearest exemplar beat the cell's mean? No (−0.0022, CI
crosses zero). And the restriction never even binds: on all 195 covered queries, the
unrestricted nearest neighbor already lies inside the query's design cell. At this
support, the address's resolution *is* the cell. I would rather report that than
have a reviewer find it.

## Most of the motion is generic

Is any of this specific to deception? I compared against two controls:

- an estimator built from all same-truth transitions, with no deception orientation:
  **0.9060**;
- a nuisance shuffle that keeps the design-cell target distribution but destroys the
  exact source–target pairing: **0.9175**.

Both nearly match the honestward field (0.9289). What remains is small:

| margin | cosine | normalized error |
|---|---|---|
| honestward over generic motion | +0.0135 [+0.0034, +0.0237] | +0.0144 [−0.0091, +0.0362] (crosses zero) |
| exact pairing over the shuffle | **+0.0114 [+0.0064, +0.0166]** | **+0.0198 [+0.0097, +0.0293]** |

Most of what the address retrieves is generic, design-conditioned transition
organization. The deception-specific residue is real — it is positive on both
metrics — and small. Same pattern as the first study, finer instrument.

## The vocabulary is small, and partially a linear rule

Two more results make the object concrete.

**Compactness.** The retrieved displacements live in a rank-32 subspace: projecting
the prediction into it preserves reconstruction almost exactly (0.8760 versus 0.8787
across 849 roots). A 256-landmark compression of the address book misses its frozen
target, so the source side is not as compressible — that part is open.

![Structure chart](figures/representation_structure.png)

**Factorization.** Can I predict the displacement from symbolic inputs? Requested
action alone reaches 0.6808. Adding the source state lifts it to **0.8915**,
positive in all five folds. An endpoint diagnostic shows 66.3% of that lift is plain
algebra (subtract the source); the remaining +0.0710 needs a learned coupling.

![Factorization chart](figures/representation_factorization.png)

So the displacement vocabulary is partially a linear function of *what was asked*
and *where the model was*. That is the result with the most obvious next use: a
predictor for the move, built from the request and the state.

## The boundary, on one table

Everything above is offline reconstruction. The control question is separate, and
the first study's results draw the fence:

| capability | instrument | result | verdict |
|---|---|---|---|
| supplied-target actuation | fixed-dose steering, target given | 591/600 fixes, 0 honest harms | works given the target |
| target-free prose control | prospective L16 controller | net change 0.0000; forced firing −0.083 | refuted |
| pre-commitment warning | three instruments, truth-aware comparators | AUROCs 0.42 / 0.37; geometry −0.022 | utility bound, not absence |
| post-commitment readout | raw-residual probe | Brier 0.0015, wins 20/20 families | reading nearly linear |
| cross-construction transfer | zero-shot readout | 0.51–0.63 vs native 0.78–0.87 | weak; cause unidentified |
| causal injection of the retrieved displacement | — | not run | open |

Actuation works when the target is supplied and fails the one time the controller
had to find the target itself. Inferring the target is the unsolved step.

## What I think this means

The first study said: reading is linear, control is selection. This one adds a third
clause: **the transition is addressable.** The state before the model lies sits next
to the states that already told it, and their moves transfer — across held-out
families, with the simplest possible retrieval.

It also keeps the first study's discipline honest: the gain is retrieval, not
machinery; the specificity is small; the design cell explains most of it. The open
question the whole program keeps circling — does activation similarity carry
anything beyond the design cell — is still open, now with a sharper instrument
pointed at it.

The next experiment is the one that decides what this becomes. Take the retrieved
displacement, **inject it under matched controls**, and ask whether prediction
becomes control. If it works, the address is a handle. If it fails the way the
prose gate failed, that is the second cleanest boundary I have. Either way the
answer is a result.

---

That is where this stands. The state before the model lies is an address into a
small, mostly generic, partially linear vocabulary of moves — and the test that
decides whether it becomes a handle is fully specified and waiting to be run.

The first study's artifact — registry, receipts, code — is public at
[github.com/rajarshighoshal/lie-geometry-probes](https://github.com/rajarshighoshal/lie-geometry-probes).

If you think I got something wrong, I'd like to hear it.

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
