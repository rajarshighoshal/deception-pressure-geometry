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

<section class="lead-dashboard" aria-labelledby="one-minute-title">
  <div class="lead-finding">
    <p class="eyebrow">The central result</p>
    <div class="lead-score-line">
      <p class="lead-score"><span>0.93</span><small>cosine</small></p>
      <p class="lead-versus">versus <strong>0.48</strong> for one global direction</p>
    </div>
    <h2 id="one-minute-title">A nearby state is a better address than one global vector</h2>
    <p>On held-out scenario families, raw-activation retrieval reconstructs the matched displacement at 0.9326 cosine. The result survives the obvious global and shuffled controls.</p>
  </div>
  <div class="lead-insights">
    <article class="insight-tile insight-tile--retrieval">
      <p class="insight-index">01</p>
      <h3>One example is almost enough</h3>
      <p>A single raw nearest neighbor reaches 0.9304. Elaborate graph machinery is not carrying the result.</p>
    </article>
    <article class="insight-tile insight-tile--boundary">
      <p class="insight-index">02</p>
      <h3>This is not yet control</h3>
      <p>The displacement is reconstructed offline. It has not been injected, and the destination is not inferred autonomously.</p>
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

<section class="method-map" aria-labelledby="concept-title">
  <header class="visual-header visual-header--method">
    <div><p class="visual-kicker">Measurement design</p><h3 id="concept-title">From matched branches to a held-out prediction</h3></div>
    <p>The key is a source activation. The stored value is a supervised displacement.</p>
  </header>
  <div class="method-map-grid" role="img" aria-label="A shared scenario branches into false and correct reports, defining an observed honestward displacement. Training pairs form an address book. A held-out source retrieves nearby entries to predict its displacement.">
    <div class="method-stage method-stage--pair">
      <p class="stage-label">Observed pair</p>
      <div class="shared-root">Same scenario<br><strong>same machine truth</strong></div>
      <div class="branch-lines" aria-hidden="true"></div>
      <div class="branch-nodes">
        <span class="state-node state-node--false">false report</span>
        <span class="state-node state-node--honest">correct report</span>
      </div>
      <div class="delta-token"><span>observed displacement</span><strong>Δ = G<sub>H</sub> − G<sub>D</sub></strong></div>
    </div>
    <div class="method-arrow" aria-hidden="true"><span>store</span>→</div>
    <div class="method-stage method-stage--book">
      <p class="stage-label">Training address book</p>
      <div class="address-entry"><i></i><span>source state</span><b>→ Δ₁</b></div>
      <div class="address-entry"><i></i><span>source state</span><b>→ Δ₂</b></div>
      <div class="address-entry"><i></i><span>source state</span><b>→ Δ₃</b></div>
      <p class="stage-note">Outcome-blind keys.<br>Supervised stored values.</p>
    </div>
    <div class="method-arrow" aria-hidden="true"><span>retrieve</span>→</div>
    <div class="method-stage method-stage--query">
      <p class="stage-label">Held-out family</p>
      <div class="query-orbit"><span class="query-node">query</span><i></i><i></i><i></i></div>
      <div class="prediction-token"><span>predicted displacement</span><strong>Δ̂(query)</strong></div>
      <p class="stage-note">Score against the realized matched difference.</p>
    </div>
  </div>
  <footer class="visual-boundary"><strong>Offline reconstruction only.</strong> <strong>What this tests:</strong> whether the displacement is reconstructible before the status token. <strong>What it does not test:</strong> whether injecting Δ̂ changes behavior.</footer>
</section>

<div class="term-strip" aria-label="Key terms">
  <div><strong>Activation</strong><span>The model's internal numerical state at a token.</span></div>
  <div><strong>Cosine</strong><span>Directional agreement: 1 is aligned, 0 unrelated, −1 opposite.</span></div>
  <div><strong>Held-out family</strong><span>An entire scenario category excluded while fitting the estimator.</span></div>
</div>

## The headline: yes for offline reconstruction

I retrieve a displacement by neighborhood: find training source states near a held-out query,
then reuse their measured displacements. Entire scenario families are held out during fitting.

<section class="visual-card visual-card--ranking" aria-labelledby="ranking-title">
  <header class="visual-header">
    <div><p class="visual-kicker">Finding 1 · Reconstruction</p><h3 id="ranking-title">Local addresses form a separate performance tier</h3></div>
    <p>Exact values and coverage are shown on every row; bar length encodes cosine.</p>
  </header>
  <div class="method-ranking" role="img" aria-label="Raw activation nearest 8 scores 0.9326, raw single nearest 0.9304, typed graph neighborhood 0.9289, truth-aware design cell mean 0.9223, typed graph nearest 0.9149, global mean 0.4839, and cyclic shuffle 0.4228.">
    <div class="ranking-group-label"><span>Local and design-conditioned addresses</span><small>0.91–0.93</small></div>
    <div class="ranking-row ranking-row--retrieval"><span>Raw activation · nearest 8</span><i><b style="--score:93.26%"></b></i><strong>0.9326</strong><small>200/200</small></div>
    <div class="ranking-row ranking-row--retrieval"><span>Raw activation · single nearest</span><i><b style="--score:93.04%"></b></i><strong>0.9304</strong><small>200/200</small></div>
    <div class="ranking-row ranking-row--graph"><span>Typed-graph neighborhood</span><i><b style="--score:92.89%"></b></i><strong>0.9289</strong><small>193/200</small></div>
    <div class="ranking-row ranking-row--cell"><span>Truth-aware design-cell mean</span><i><b style="--score:92.23%"></b></i><strong>0.9223</strong><small>200/200</small></div>
    <div class="ranking-row ranking-row--graph"><span>Typed-graph single nearest</span><i><b style="--score:91.49%"></b></i><strong>0.9149</strong><small>193/200</small></div>
    <div class="ranking-group-label ranking-group-label--baseline"><span>Global and shuffled baselines</span><small>below 0.49</small></div>
    <div class="ranking-row ranking-row--baseline"><span>One global mean direction</span><i><b style="--score:48.39%"></b></i><strong>0.4839</strong><small>200/200</small></div>
    <div class="ranking-row ranking-row--null"><span>Cyclic reuse of other targets</span><i><b style="--score:42.28%"></b></i><strong>0.4228</strong><small>193/200</small></div>
  </div>
  <footer class="visual-caption"><span><b>Read</b> Locality—not graph complexity—is the visible separation.</span><span><b>Coverage</b> Counts are defined roots out of 200.</span></footer>
  <details class="technical-ledger">
    <summary>Open exact estimator ledger</summary>
    <table class="responsive-table">
      <caption class="sr-only">Held-out-family displacement reconstruction. Higher cosine is better.</caption>
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
  </details>
</section>

<figure class="evidence-figure">
  <header class="visual-header"><div><p class="visual-kicker">Comparator audit</p><h3>Where the reconstruction gain comes from</h3></div><p>Paired differences with scenario-cluster 95% intervals.</p></header>
  <a class="figure-link" href="figures/representation_reconstruction_blog.png" aria-label="Open the full-resolution reconstruction figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_reconstruction_mobile_blog.png">
      <img src="figures/representation_reconstruction_blog.png" alt="Comparison of held-out displacement reconstruction. Raw k-nearest-neighbor, raw nearest, typed-graph local, and the truth-aware cell mean cluster between 0.922 and 0.933 cosine; the global mean reaches 0.484 and the cyclic shuffle 0.423.">
    </picture>
  </a>
  <figcaption class="visual-caption"><span><b>Result</b> Local methods cluster near 0.93; the global direction reaches 0.48.</span><span><b>Boundary</b> Held-out-family reconstruction, not intervention.</span></figcaption>
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

<section class="visual-card visual-card--specificity" aria-labelledby="specificity-title">
  <header class="visual-header">
    <div><p class="visual-kicker">Finding 2 · Specificity</p><h3 id="specificity-title">Most of the motion is generic; the remaining margins are small</h3></div>
    <p>Aggregate reconstruction gives orientation. Paired intervals decide the comparisons.</p>
  </header>
  <div class="specificity-dashboard">
    <div class="aggregate-stack">
      <p class="mini-title">Aggregate reconstruction</p>
      <div class="aggregate-row aggregate-row--generic"><span>Generic transitions</span><i style="--score:90.60%"></i><strong>0.9060</strong></div>
      <div class="aggregate-row aggregate-row--shuffle"><span>Design-cell shuffle</span><i style="--score:91.75%"></i><strong>0.9175</strong></div>
      <div class="aggregate-row aggregate-row--honest"><span>Honestward field</span><i style="--score:92.89%"></i><strong>0.9289</strong></div>
      <p class="aggregate-note">These aggregate means are not arithmetic substitutes for the paired margins.</p>
    </div>
    <div class="margin-cards">
      <article class="margin-card margin-card--mixed">
        <p>Honestward over generic motion</p>
        <strong>+0.0135</strong><span>cosine · CI above zero</span>
        <strong>+0.0144</strong><span>normalized error · CI crosses zero</span>
        <em>metric-dependent</em>
      </article>
      <article class="margin-card margin-card--positive">
        <p>Exact pairing over cell shuffle</p>
        <strong>+0.0114</strong><span>cosine · CI above zero</span>
        <strong>+0.0198</strong><span>normalized error · CI above zero</span>
        <em>positive on both metrics</em>
      </article>
    </div>
  </div>
  <footer class="visual-caption"><span><b>Interpretation</b> Design-conditioned transition structure explains most of the result.</span><span><b>Residue</b> Small, and only the exact-pairing margin is stable across both metrics.</span></footer>
  <details class="technical-ledger">
    <summary>Open paired intervals</summary>
    <table class="responsive-table">
      <caption class="sr-only">Specificity margins over generic and design-matched controls.</caption>
      <thead><tr><th scope="col">Comparison</th><th scope="col">Cosine margin</th><th scope="col">Normalized-error margin</th></tr></thead>
      <tbody>
        <tr><th scope="row">Honestward over generic motion</th><td data-label="Cosine">+0.0135 [+0.0034, +0.0237]</td><td data-label="Normalized error">+0.0144 [−0.0091, +0.0362] <span class="verdict verdict--mixed">metric-dependent</span></td></tr>
        <tr><th scope="row">Exact pairing over nuisance shuffle</th><td data-label="Cosine"><strong>+0.0114 [+0.0064, +0.0166]</strong></td><td data-label="Normalized error"><strong>+0.0198 [+0.0097, +0.0293]</strong> <span class="verdict verdict--positive">both positive</span></td></tr>
      </tbody>
    </table>
  </details>
</section>

Most of what the address retrieves is generic, design-conditioned transition organization.
The honestward advantage over generic motion is small and metric-dependent: positive in cosine,
with the normalized-error interval crossing zero. The exact-pairing advantage over the nuisance
shuffle is also small, but positive on both reported metrics.

## The vocabulary is compact and partially factorized

<section class="visual-card visual-card--structure" aria-labelledby="structure-title">
  <header class="visual-header">
    <div><p class="visual-kicker">Finding 3 · Representation</p><h3 id="structure-title">A compact output vocabulary, with a destination-conditioned rule</h3></div>
    <p>Compression describes the outputs. Factorization describes how to construct them.</p>
  </header>
  <div class="structure-dashboard">
    <div class="rank-module">
      <p class="module-label">Output compression</p>
      <div class="rank-number"><strong>32</strong><span>dimensions</span></div>
      <div class="variance-meter"><i></i></div>
      <p><strong>96.75–96.92%</strong> of training variance retained in every fold.</p>
      <div class="performance-pair"><span>Full <b>0.8787</b></span><span>Rank-32 <b>0.8760</b></span></div>
    </div>
    <div class="factor-module">
      <p class="module-label">Destination-conditioned factorization</p>
      <div class="factor-step"><span>Action only</span><i style="--score:68.08%"></i><strong>0.6808</strong></div>
      <div class="factor-arrow">+ endpoint subtraction</div>
      <div class="factor-step factor-step--middle"><span>Constrained source</span><i style="--score:82.05%"></i><strong>0.8205</strong></div>
      <div class="factor-arrow">+ learned source coupling</div>
      <div class="factor-step factor-step--final"><span>Free source + action</span><i style="--score:89.15%"></i><strong>0.8915</strong></div>
      <p class="factor-note">Desired status and destination program are supplied.</p>
    </div>
  </div>
  <footer class="visual-caption"><span><b>Compactness</b> Output prediction survives projection to rank 32.</span><span><b>Use</b> The rule constructs a target; it does not choose one.</span></footer>
</section>

**Output compactness.** A rank-32 projection preserves the full estimator almost exactly
(0.8760 versus 0.8787 across 849 roots). A tested 256-landmark compression of the source-side
address book misses its frozen target. That scheme was insufficient; other source-side
representations remain open.

<figure class="evidence-figure">
  <header class="visual-header"><div><p class="visual-kicker">Specificity + compression audit</p><h3>Small directional residue, compact output space</h3></div><p>Paired intervals and the frozen compression frontier.</p></header>
  <a class="figure-link" href="figures/representation_structure_blog.png" aria-label="Open the full-resolution specificity and compression figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_structure_mobile_blog.png">
      <img src="figures/representation_structure_blog.png" alt="Specificity margins are small: honestward over generic motion is positive in cosine but crosses zero in normalized error, while exact pairing beats the nuisance shuffle on both metrics. A rank-32 projection retains nearly all reconstruction performance.">
    </picture>
  </a>
  <figcaption class="visual-caption"><span><b>Result</b> Rank-32 preserves the estimator; specificity margins remain small.</span><span><b>Boundary</b> One failed landmark scheme does not establish general source-side incompressibility.</span></figcaption>
</figure>

**Destination-conditioned factorization.** The symbolic descriptor receives the requested
transition, desired status, and destination program. With that destination supplied, action
alone reaches 0.6808; adding the source state lifts reconstruction to **0.8915**, positive in
all five held-out-family folds. Endpoint subtraction explains 66.3% of the observed lift; a
learned source coupling supplies the remaining +0.0710.

<figure class="evidence-figure">
  <header class="visual-header"><div><p class="visual-kicker">Factorization audit</p><h3>Source coordinates add beyond endpoint subtraction</h3></div><p>Five held-out-family folds; fold consistency, not a confidence interval.</p></header>
  <a class="figure-link" href="figures/representation_factorization_blog.png" aria-label="Open the full-resolution factorization figure">
    <picture>
      <source media="(max-width: 700px)" srcset="figures/representation_factorization_mobile_blog.png">
      <img src="figures/representation_factorization_blog.png" alt="Family-macro reconstruction cosine rises from 0.681 for action alone to 0.821 after endpoint subtraction and 0.892 with learned source coupling; five held-out-family points show the same ordering.">
    </picture>
  </a>
  <figcaption class="visual-caption"><span><b>Result</b> Action plus source reaches 0.8915 cosine.</span><span><b>Boundary</b> Desired status and destination are inputs; this is not target inference.</span></figcaption>
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

<section class="visual-card visual-card--boundary-table" aria-labelledby="boundary-table-title">
  <header class="visual-header"><div><p class="visual-kicker">Capability ledger</p><h3 id="boundary-table-title">The information budget changes at every rung</h3></div><p>These are different estimands, not one leaderboard.</p></header>
<table class="responsive-table">
  <caption class="sr-only">Different capabilities require different information and cannot be ranked as one leaderboard.</caption>
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
  <footer class="visual-caption"><span><b>Reading</b> Post-commitment state is nearly linear.</span><span><b>Control</b> Supplied targets work; autonomous target inference remains open.</span></footer>
</section>

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
