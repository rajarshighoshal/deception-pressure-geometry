---
layout: article
title: The state before the model lies is an address
deck: "A simple nearest-neighbor lookup reconstructs held-out activation displacements at 0.93 cosine. The controls reveal what that address contains—and the causal test that remains."
byline: "Rajarshi Ghoshal"
published: "July 2026"
updated: "August 2, 2026"
og_image: "figures/addressability_social_card.png"
og_image_alt: "Local activation retrieval reconstructs held-out displacement at 0.93 cosine, compared with 0.48 for one global direction."
assistance_note: >-
  The author wrote the post and owns the research design, methodology, claims, and prose. LLMs assisted
  with experiment code, plots and tables, background lookup, typo correction, and
  drafting support.
---

The first study settled an important baseline: after commitment, the model's answer is easy
to read with a linear probe. It also exposed the harder problem. A supplied target can be
actuated, but the prospective target-free controller failed to find one reliably.

That left a sharper question than classification:

**Can I reconstruct the matched activation difference between a state that is about to
report falsely and a state that reports correctly—before the model samples the status token?**

Not classify a state. Reconstruct the matched displacement.

<section class="summary-panel" aria-labelledby="one-minute-title">
  <p class="eyebrow">In one minute</p>
  <h2 id="one-minute-title">The result, the control, and the boundary</h2>
  <div class="summary-grid">
    <article class="summary-card summary-card--blue">
      <p class="summary-kicker">Strong reconstruction</p>
      <p class="summary-number">0.9326</p>
      <p>Raw-activation retrieval reconstructs held-out displacement, versus 0.4839 for one global direction.</p>
    </article>
    <article class="summary-card summary-card--amber">
      <p class="summary-kicker">Simple address</p>
      <p class="summary-number">k = 1</p>
      <p>One nearby training exemplar nearly saturates the result. The relational graph is not required.</p>
    </article>
    <article class="summary-card summary-card--ink">
      <p class="summary-kicker">Control remains open</p>
      <p class="summary-number">Not run</p>
      <p>The reconstructed displacement has not been injected. Accurate prediction is not yet causal control.</p>
    </article>
  </div>
</section>

## The object: a displacement, not a label

Here **lie** is shorthand for a false operational status after the model has demonstrated
the correct answer. It does not assert human-like intent or a hidden goal.

The bank contains sixty machine-checkable scenarios across twenty families. At each decision
turn, the model sees the literal prefix `Status:` and samples one unrestricted token. I record
its internal numerical state—the **activation**—at four layers immediately before that token.

For the same scenario and machine-checked truth, I match one knowledge-correct completion that
later reports the wrong status with one that reports the correct status. Their prefixes need
not be text-identical. The **displacement** is the observed activation difference
`G(honest) − G(deceptive)` between those matched pre-status states.

This is different from a probe. A probe asks what information can be read from one state. This
experiment asks whether an observed difference between matched states can be reconstructed from
the source state. Whether applying that difference changes behavior is a separate causal test.

<div class="concept-figure" role="img" aria-labelledby="concept-title concept-description">
  <div class="concept-heading">
    <p class="eyebrow">How the offline test works</p>
    <h3 id="concept-title">Use a source state as an address into stored displacements</h3>
    <p id="concept-description">Matched branches define honestward displacements in training families. For a held-out source state, nearest training states retrieve stored displacements that are averaged into a prediction. No vector is injected.</p>
  </div>
  <div class="concept-flow">
    <div class="concept-step concept-step--source">
      <span class="step-number">1</span>
      <strong>Measure matched branches</strong>
      <span>Same scenario and truth; one false report, one correct report.</span>
    </div>
    <span class="flow-arrow" aria-hidden="true">→</span>
    <div class="concept-step">
      <span class="step-number">2</span>
      <strong>Store the difference</strong>
      <span>Each training source state points to its measured honestward displacement.</span>
    </div>
    <span class="flow-arrow" aria-hidden="true">→</span>
    <div class="concept-step concept-step--query">
      <span class="step-number">3</span>
      <strong>Look up a held-out state</strong>
      <span>Find nearby training states without using the held-out outcome label in the key.</span>
    </div>
    <span class="flow-arrow" aria-hidden="true">→</span>
    <div class="concept-step concept-step--target">
      <span class="step-number">4</span>
      <strong>Predict its displacement</strong>
      <span>Average retrieved values and compare with the realized matched difference.</span>
    </div>
  </div>
  <p class="concept-boundary"><strong>Offline reconstruction only:</strong> the predicted displacement is scored against a realized target; it is not injected into the model in this study.</p>
</div>

<div class="term-strip" aria-label="Key terms">
  <div><strong>Activation</strong><span>The model's internal numerical state at a token.</span></div>
  <div><strong>Cosine</strong><span>Directional agreement: 1 is aligned, 0 unrelated, −1 opposite.</span></div>
  <div><strong>Held-out family</strong><span>An entire scenario category excluded while fitting the estimator.</span></div>
</div>

## The headline: yes for offline reconstruction

I retrieve a displacement by neighborhood: find training source states near a held-out query,
then reuse their measured displacements. Entire scenario families are held out during fitting.

<div class="table-wrap table-wrap--results">
<table class="responsive-table">
  <caption>Held-out-family displacement reconstruction. Higher cosine is better.</caption>
  <thead><tr><th scope="col">Estimator</th><th scope="col">Reconstruction cosine</th><th scope="col">Coverage</th></tr></thead>
  <tbody>
    <tr><th scope="row">Raw activation, nearest 8</th><td data-label="Cosine"><strong>0.9326</strong></td><td data-label="Coverage">200/200</td></tr>
    <tr><th scope="row">Raw activation, single nearest</th><td data-label="Cosine"><strong>0.9304</strong></td><td data-label="Coverage">200/200</td></tr>
    <tr><th scope="row">Typed-graph neighborhood</th><td data-label="Cosine"><strong>0.9289</strong></td><td data-label="Coverage">193/200</td></tr>
    <tr><th scope="row">Truth-aware design-cell mean</th><td data-label="Cosine"><strong>0.9223</strong></td><td data-label="Coverage">200/200</td></tr>
    <tr><th scope="row">Typed-graph single nearest</th><td data-label="Cosine"><strong>0.9149</strong></td><td data-label="Coverage">193/200</td></tr>
    <tr><th scope="row">One global mean direction</th><td data-label="Cosine">0.4839</td><td data-label="Coverage">200/200</td></tr>
    <tr><th scope="row">Cyclic reuse of other targets</th><td data-label="Cosine">0.4228</td><td data-label="Coverage">193/200</td></tr>
  </tbody>
</table>
</div>

<figure class="evidence-figure">
  <a class="figure-link" href="figures/representation_reconstruction.png" aria-label="Open the full-resolution reconstruction figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_reconstruction_mobile.png">
      <img src="figures/representation_reconstruction.png" alt="Comparison of held-out displacement reconstruction. Raw k-nearest-neighbor, raw nearest, typed-graph local, and the truth-aware cell mean cluster between 0.922 and 0.933 cosine; the global mean reaches 0.484 and the cyclic shuffle 0.423.">
    </picture>
  </a>
  <figcaption><strong>Question:</strong> Does a local address reconstruct a held-out displacement? <strong>Result:</strong> local methods cluster near 0.93 cosine, while one global direction reaches 0.48. <strong>Boundary:</strong> this is held-out-family reconstruction, not intervention.</figcaption>
</figure>

The typed-graph local estimator beats its global direction by
**+0.4252 [+0.3799, +0.4638]**, the largest gap measured in this program. Its
sign-flipped prediction lands near −0.93, an orientation sanity check showing that the
reconstruction is consistently directed rather than benefiting from unsigned similarity.

<aside class="evidence-note" aria-labelledby="evidence-status-title">
  <p class="eyebrow">Evidence status</p>
  <h3 id="evidence-status-title">Strong descriptive result, bounded instrument</h3>
  <p>One Llama-3.1-8B-Instruct model; sixty scenarios and twenty families; a developmental bank whose frozen baseline-knowledge gate reached 56/60 against a required 57/60. The retrieval key is outcome-blind, but its stored displacement values are supervised. Reconstruction, specificity, and compression are retrospective unregistered descriptive analyses; the simple-address, within-cell, and factorization diagnostics are post-evidence registered descriptive follow-ups. None is confirmatory.</p>
</aside>

## The twist: the address is coarse and simple

I built the first estimator on a typed relational graph using residual streams, attention rows,
and typed metadata. The registered simple-address diagnostic then asked whether that machinery
was necessary.

It was not. Euclidean nearest neighbors on raw activations do as well or better
(0.9304–0.9326). Within the typed-graph estimator, averaging its neighborhood adds only
**+0.0140** over its graph-nearest exemplar (0.9289 versus 0.9149). Retrieval, not graph
structure or elaborate aggregation, carries almost all of the gain.

A mean over the design cell—turn, intervention history, pressure exposure, and true status—also
reaches 0.9223. This is a **privileged truth-aware explanatory baseline**, not a deployable
controller. It locates what the address recovers.

Restricted to its own design cell, the activation-nearest exemplar does not beat the cell mean
(−0.0022, interval crossing zero). On all 195 covered queries, the unrestricted nearest
neighbor already falls inside the query's cell. The result is therefore **coarse
addressability**: activation similarity recovers the design-cell partition. At the available
support, it does not detectably order targets within that cell.

## Most of the motion is generic

Two controls ask how specifically this organization belongs to deceptive-to-honest transitions:

- an unoriented estimator built from all same-truth transitions reaches **0.9060**;
- a nuisance shuffle that preserves the design-cell target distribution while destroying exact
  source–target pairing reaches **0.9175**.

<div class="table-wrap table-wrap--compact">
<table class="responsive-table">
  <caption>Specificity margins over generic and design-matched controls.</caption>
  <thead><tr><th scope="col">Comparison</th><th scope="col">Cosine margin</th><th scope="col">Normalized-error margin</th></tr></thead>
  <tbody>
    <tr><th scope="row">Honestward over generic motion</th><td data-label="Cosine">+0.0135 [+0.0034, +0.0237]</td><td data-label="Normalized error">+0.0144 [−0.0091, +0.0362] <span class="verdict verdict--mixed">metric-dependent</span></td></tr>
    <tr><th scope="row">Exact pairing over nuisance shuffle</th><td data-label="Cosine"><strong>+0.0114 [+0.0064, +0.0166]</strong></td><td data-label="Normalized error"><strong>+0.0198 [+0.0097, +0.0293]</strong> <span class="verdict verdict--positive">both positive</span></td></tr>
  </tbody>
</table>
</div>

Most of what the address retrieves is generic, design-conditioned transition organization.
The honestward advantage over generic motion is small and metric-dependent: positive in cosine,
with the normalized-error interval crossing zero. The exact-pairing advantage over the nuisance
shuffle is also small, but positive on both reported metrics.

## The vocabulary is compact and partially factorized

**Output compactness.** A rank-32 projection preserves the full estimator almost exactly
(0.8760 versus 0.8787 across 849 roots). A tested 256-landmark compression of the source-side
address book misses its frozen target. That scheme was insufficient; other source-side
representations remain open.

<figure class="evidence-figure">
  <a class="figure-link" href="figures/representation_structure.png" aria-label="Open the full-resolution specificity and compression figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_structure_mobile.png">
      <img src="figures/representation_structure.png" alt="Specificity margins are small: honestward over generic motion is positive in cosine but crosses zero in normalized error, while exact pairing beats the nuisance shuffle on both metrics. A rank-32 projection retains nearly all reconstruction performance.">
    </picture>
  </a>
  <figcaption><strong>Question:</strong> What part of the displacement is specific, and how compact is its output vocabulary? <strong>Result:</strong> specificity margins are small, while a rank-32 projection preserves the estimator. <strong>Boundary:</strong> the tested landmark scheme does not establish general source-side incompressibility.</figcaption>
</figure>

**Destination-conditioned factorization.** The symbolic descriptor receives the requested
transition, desired status, and destination program. With that destination supplied, action
alone reaches 0.6808; adding the source state lifts reconstruction to **0.8915**, positive in
all five held-out-family folds. Endpoint subtraction explains 66.3% of the observed lift; a
learned source coupling supplies the remaining +0.0710.

<figure class="evidence-figure">
  <a class="figure-link" href="figures/representation_factorization.png" aria-label="Open the full-resolution factorization figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_factorization_mobile.png">
      <img src="figures/representation_factorization.png" alt="Family-macro reconstruction cosine rises from 0.681 for action alone to 0.821 after endpoint subtraction and 0.892 with learned source coupling; five held-out-family points show the same ordering.">
    </picture>
  </a>
  <figcaption><strong>Question:</strong> Can a compact rule construct the displacement once a destination is given? <strong>Result:</strong> action plus source reaches 0.8915 cosine. <strong>Boundary:</strong> the model receives the desired status and destination program; it does not infer the target.</figcaption>
</figure>

The displacement vocabulary is therefore partially a linear function of what destination was
requested and where the model was. This is a compact target constructor, not a target-free
predictor.

## From reconstruction to control

<div class="evidence-ladder" aria-label="Evidence ladder from reconstruction to autonomous control">
  <div class="ladder-step ladder-step--strong"><span>1</span><strong>Reconstruct</strong><small>Strong descriptive result<br>0.9326 cosine</small></div>
  <div class="ladder-connector" aria-hidden="true">→</div>
  <div class="ladder-step ladder-step--conditional"><span>2</span><strong>Factorize</strong><small>Works with desired destination supplied</small></div>
  <div class="ladder-connector" aria-hidden="true">→</div>
  <div class="ladder-step ladder-step--open"><span>3</span><strong>Inject</strong><small>Retrieved displacement<br>not yet tested</small></div>
  <div class="ladder-connector" aria-hidden="true">→</div>
  <div class="ladder-step ladder-step--failed"><span>4</span><strong>Control</strong><small>Prospective target-free controller failed</small></div>
</div>

The earlier experiments draw the boundary:

<div class="table-wrap table-wrap--boundary">
<table class="responsive-table">
  <caption>Different capabilities require different information and cannot be ranked as one leaderboard.</caption>
  <thead><tr><th scope="col">Capability</th><th scope="col">Instrument</th><th scope="col">Result</th><th scope="col">Verdict</th></tr></thead>
  <tbody>
    <tr><th scope="row">Supplied-target actuation</th><td data-label="Instrument">Fixed-dose steering; target given</td><td data-label="Result">591/600 fixes, 0 honest harms</td><td data-label="Verdict"><span class="verdict verdict--positive">Works given target</span></td></tr>
    <tr><th scope="row">Target-free prose control</th><td data-label="Instrument">Prospective L16 controller</td><td data-label="Result">Net 0.0000; forced firing −0.083</td><td data-label="Verdict"><span class="verdict verdict--failed">Refuted</span></td></tr>
    <tr><th scope="row">Pre-commitment warning</th><td data-label="Instrument">Three instruments; truth-aware comparators</td><td data-label="Result">AUROCs 0.42 / 0.37; geometry −0.022</td><td data-label="Verdict"><span class="verdict verdict--mixed">Utility bound</span></td></tr>
    <tr><th scope="row">Post-commitment readout</th><td data-label="Instrument">Raw-residual probe</td><td data-label="Result">Brier 0.0015; wins 20/20 families</td><td data-label="Verdict"><span class="verdict verdict--positive">Nearly linear</span></td></tr>
    <tr><th scope="row">Cross-construction transfer</th><td data-label="Instrument">Zero-shot readout</td><td data-label="Result">0.51–0.63 versus native 0.78–0.87</td><td data-label="Verdict"><span class="verdict verdict--mixed">Weak; cause unknown</span></td></tr>
    <tr><th scope="row">Causal displacement injection</th><td data-label="Instrument">Matched controls</td><td data-label="Result">Not run</td><td data-label="Verdict"><span class="verdict verdict--open">Open</span></td></tr>
  </tbody>
</table>
</div>

Actuation works when the corrective target is supplied and fails in the one prospective test
where the controller had to find a target itself. Inferring the destination remains the
unsolved step. The reconstructed displacement has not yet crossed the separate prediction-to-
intervention boundary.

## What I think this means

The first study said: reading is linear, control is selection. This one adds a third clause:
**the transition is coarsely addressable.** Nearby source activations retrieve displacement
examples that transfer across held-out families with the simplest possible lookup.

The controls make the claim more useful, not weaker. The gain is retrieval rather than graph
machinery. The design cell explains most of the address. The deception-specific residue is
small. A rank-32 output space and a destination-conditioned source-plus-action model make the
next causal test concrete.

Take a retrieved or predicted displacement, inject it under matched random, sign-flipped,
global-direction, and no-intervention controls, then re-decode behavior. If it works, the
address becomes a handle. If it fails, reconstruction and intervention separate cleanly.
Either outcome resolves the next question.

---

The pre-status activation is a coarse address into a compact, mostly generic displacement
vocabulary. The decisive next experiment is whether that address becomes a causal handle.

The registry, compact evidence receipts, figure producers, tests, and scientific code are
public at [github.com/rajarshighoshal/deception-pressure-geometry](https://github.com/rajarshighoshal/deception-pressure-geometry).

If you think I got something wrong, I'd like to hear it.

*GPU compute was supported by a BlueDot Impact Rapid Grant.*
