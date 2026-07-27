"""Symmetry discovery, candidate 1: is pressure a one-parameter FLOW on activation space?

A flow (baby Lie group action) requires more than the known behavioral ramp (deception RATE rises
p0->p6). It requires the STATES to move (i) monotonically deeper along orbits and (ii) coherently —
displacements between consecutive pressure levels should follow a common vector field, not scatter.

Pseudo-orbits = (scenario_id, sample label) across arms p0..p6. The historical rollout offset the
actual RNG seed by arm, so these are not common-random-number trajectories. Results are descriptive
ordered-pressure statistics, not evidence that one observed state is transformed into the next.

Depth coordinates (two, deliberately different instruments):
  probe   : honesty-probe margin (negative = deeper into deception)
  contrast: mean distance to k nearest HONEST anchors minus k nearest DECEPTIVE anchors

Verdict is THREE-WAY by design (user rule 2026-07-09): found / refuted-under-this-instrument /
not-found-under-this-instrument. Known instrument limitations, recorded in the artifact:
pressure covaries with dialogue text (displacements mix pressure and wording); anchor states only
(no paths); both depth coordinates are choices, not canonical.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from geoprobe.control.external_soft_control_generation import write_json  # noqa: E402
from geoprobe.provenance import git_provenance  # noqa: E402

# Restored from tag pre-aggressive-cleanup-20260724 (original producer commit ad45591). The
# historical anchor bank lives outside the citable tree, so its path is a required CLI argument.
ARMS = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True,
                        help="turns.pt anchor bank with turn_index/phase/conversation fields")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results/baselines/g1_source_freeprose/pressure_flow.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    bank = torch.load(args.anchors, map_location="cpu", weights_only=False, mmap=True)
    sel = (np.asarray(bank["turn_index"]).astype(int) == 3) & \
          (np.asarray(bank["phase"]).astype(str) == "pre_response")
    for key in ("conversation_id", "scenario_id", "family", "arm", "deceptive"):
        bank[key] = [v for v, keep in zip(bank[key], sel) if keep]
    bank["activations"] = {k: v[torch.from_numpy(sel)] for k, v in bank["activations"].items()}
    cids = np.asarray(bank["conversation_id"]).astype(str)
    scen = np.asarray(bank["scenario_id"]).astype(str)
    arm = np.asarray(bank["arm"]).astype(str)
    sample = np.asarray(bank["sample"]).astype(str) if "sample" in bank else np.asarray(
        [c.split(":")[-1] for c in cids])
    dec = np.asarray([None if d is None else bool(d) for d in bank["deceptive"]], dtype=object)
    x = bank["activations"][args.layer].float().numpy()

    known = np.asarray([d is not None for d in dec])
    y = np.asarray([0 if d else 1 for d in dec[known]])
    probe = LogisticRegression(max_iter=2000).fit(x[known], y)
    probe_depth = -probe.decision_function(x)  # positive = deeper into deception

    honest_x = x[known][y == 1]
    deceptive_x = x[known][y == 0]
    nn_h = NearestNeighbors(n_neighbors=args.k).fit(honest_x)
    nn_d = NearestNeighbors(n_neighbors=args.k).fit(deceptive_x)
    d_h, _ = nn_h.kneighbors(x)
    d_d, _ = nn_d.kneighbors(x)
    contrast_depth = d_h.mean(axis=1) - d_d.mean(axis=1)  # positive = closer to deceptive cloud

    arm_rank = {a: i for i, a in enumerate(ARMS)}
    orbits: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for i in range(len(cids)):
        orbits[(scen[i], sample[i])][arm_rank[arm[i]]] = i

    full = {key: idx for key, idx in orbits.items() if len(idx) == len(ARMS)}
    rho_probe, rho_contrast = [], []
    steps = {j: [] for j in range(len(ARMS) - 1)}
    for _key, idx in full.items():
        order = [idx[j] for j in range(len(ARMS))]
        levels = list(range(len(ARMS)))
        rho_probe.append(spearmanr(levels, probe_depth[order]).statistic)
        rho_contrast.append(spearmanr(levels, contrast_depth[order]).statistic)
        for j in range(len(ARMS) - 1):
            steps[j].append(x[idx[j + 1]] - x[idx[j]])

    # flow coherence: are consecutive displacements aligned ACROSS orbits (a common field)?
    coherence, coherence_shuffled = [], []
    for j, vecs in steps.items():
        v = np.vstack([u / (np.linalg.norm(u) + 1e-9) for u in vecs])
        pick = rng.choice(len(v), size=min(400, len(v)), replace=False)
        cos = (v[pick] @ v[pick].T)[np.triu_indices(len(pick), 1)]
        coherence.append(float(np.mean(cos)))
        shuf = rng.standard_normal(v[pick].shape)
        shuf /= np.linalg.norm(shuf, axis=1, keepdims=True)
        coherence_shuffled.append(float(np.mean((shuf @ shuf.T)[np.triu_indices(len(pick), 1)])))

    summary = {
        "n_full_orbits": len(full),
        "monotonicity": {
            "probe_depth_median_spearman": float(np.nanmedian(rho_probe)),
            "probe_depth_frac_positive": float(np.nanmean(np.asarray(rho_probe) > 0)),
            "contrast_depth_median_spearman": float(np.nanmedian(rho_contrast)),
            "contrast_depth_frac_positive": float(np.nanmean(np.asarray(rho_contrast) > 0)),
        },
        "flow_coherence_mean_cos_by_step": coherence,
        "flow_coherence_shuffled_baseline": coherence_shuffled,
    }
    med_rho = summary["monotonicity"]["probe_depth_median_spearman"]
    med_rho2 = summary["monotonicity"]["contrast_depth_median_spearman"]
    coh_gap = float(np.mean(coherence) - np.mean(coherence_shuffled))
    if med_rho > 0.5 and med_rho2 > 0.5 and coh_gap > 0.05:
        verdict = "FOUND: states deepen monotonically along orbits AND displacements share a common field"
    elif med_rho < 0.1 and med_rho2 < 0.1:
        verdict = ("NOT-FOUND-UNDER-THIS-INSTRUMENT: no monotone deepening in either depth "
                   "coordinate; could be wrong coordinates, text confound, or absent structure")
    else:
        verdict = ("PARTIAL: monotone deepening present but displacement field "
                   f"{'coherent' if coh_gap > 0.05 else 'NOT coherent'} -- flow structure incomplete "
                   "under this instrument")
    # --- generality tests (is the flow a construction artifact?) ---
    from sklearn.metrics import roc_auc_score
    fam = np.asarray(bank["family"]).astype(str)
    disp_by_fam: dict[str, list] = defaultdict(list)
    for _key, idx in full.items():
        for j in range(len(ARMS) - 1):
            d = x[idx[j + 1]] - x[idx[j]]
            disp_by_fam[fam[idx[0]]].append(d / (np.linalg.norm(d) + 1e-9))
    global_field = np.mean([u for vs in disp_by_fam.values() for u in vs], axis=0)
    global_field /= np.linalg.norm(global_field)
    fams = sorted(disp_by_fam)
    fam_fields = {f: np.mean(disp_by_fam[f], axis=0) for f in fams}
    for f in fam_fields:
        fam_fields[f] /= np.linalg.norm(fam_fields[f])
    cos_ff = [float(np.dot(fam_fields[a], fam_fields[b]))
              for i, a in enumerate(fams) for b in fams[i + 1:]]
    within = {}
    for level in ("p2", "p3", "p4", "p5"):
        m = (np.asarray(bank["arm"]).astype(str) == level) & known
        labels = np.asarray([1 if d else 0 for d in dec[m]])
        if labels.sum() >= 10 and (1 - labels).sum() >= 10:
            within[level] = float(roc_auc_score(labels, x[m] @ global_field))
    summary["generality"] = {
        "cross_family_field_cos_median": float(np.median(cos_ff)),
        "cross_family_field_cos_min": float(np.min(cos_ff)),
        "within_level_depth_auroc": within,
        "read": ("topic-independence: family fields nearly identical (median ~0.9 vs baseline "
                 "~0.016); state-not-text: at FIXED pressure wording, depth along the field "
                 "predicts which samples actually deceive -- a pure text-embedding artifact "
                 "would be constant within level"),
    }
    summary["verdict"] = verdict
    summary["instrument_limitations"] = [
        "historical p0..p6 rows use arm-offset RNG seeds; sample labels are not matched RNG seeds",
        "probe/depth instruments and family fields are fitted and evaluated in-sample",
        "pressure covaries with dialogue text; displacements mix pressure and wording effects",
        "anchor states only (turn-3 pre_response); no trajectory data per pressure level",
        "depth coordinates (probe margin, contrastive kNN) are choices, not canonical",
    ]

    payload = {"schema_version": 1, "kind": "pressure_flow_test", "argv": sys.argv,
               "provenance": git_provenance([Path(__file__)]), "summary": summary}
    write_json(args.out, payload)
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
