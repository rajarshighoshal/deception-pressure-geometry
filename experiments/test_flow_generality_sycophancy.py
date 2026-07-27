"""Generality ladder for the pressure flow: different CONSTRUCTION (sycophancy pushback) + 70B.

Rung A (same model, Llama-3.1-8B, L16): pressure realized as escalating USER PUSHBACK across turns
(synthetic_pressure_v2), not synthetic incentive text (powered150).
  A1 field transfer : mean turn-displacement direction vs the powered150 pressure generator
                      (same ambient space; canonical-frame variant via z-affine pool aligner)
  A2 state-not-text : per-conversation depth slope along the powered150 generator separates
                      sycophantic_flip from steadfast conversations (same pushback text, different
                      internal trajectory)

Rung B (different model, Llama-3.3-70B, functional identity per the car/truck rule): does the SAME
PHENOMENON exist natively — monotone deepening for flips vs steadfast under a 70B-native stance
probe, and a coherent displacement field? (No cross-model direction comparison — dims differ.)

Trajectory classes derived from per-turn stances (labels.jsonl): flip = starts non-accepting,
later accepts persistently; steadfast = never accepts. Verdicts are three-way per the 2026-07-09 rule.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from geoprobe.control.covariant_transfer import ZFrameAligner  # noqa: E402
from geoprobe.control.external_soft_control_generation import write_json  # noqa: E402
from geoprobe.provenance import git_provenance  # noqa: E402

# Restored from tag pre-aggressive-cleanup-20260724. The historical powered bank lives outside
# the citable tree, so its path is a required CLI argument.
SYC_8B = REPO_ROOT / "results/activations/synthetic_pressure_v2_llama8b/turns.pt"
SYC_70B = REPO_ROOT / "results/activations/synthetic_pressure_v2_llama33_70b_limit200/turns.pt"
LABELS = REPO_ROOT / "data/raw/synthetic_pressure/labels.jsonl"


def load_bank(path: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sel = np.asarray(b["phase"]).astype(str) == "pre_response"
    cids = np.asarray(b["conversation_id"]).astype(str)[sel]
    turns = np.asarray(b["turn_index"]).astype(int)[sel]
    x = b["activations"][layer].float().numpy()[sel]
    return cids, turns, x, sel


def stance_classes() -> dict[str, str]:
    per_conv: dict[str, dict[int, str]] = defaultdict(dict)
    for line in LABELS.open():
        row = json.loads(line)
        per_conv[str(row["conversation_id"])][int(row["turn_index"])] = str(row["stance"])
    out = {}
    for cid, stances in per_conv.items():
        seq = [stances[t] for t in sorted(stances)]
        accepts = [s == "accepts" for s in seq]
        if not any(accepts):
            out[cid] = "steadfast"
        elif not accepts[0] and any(accepts) and all(a for a in accepts[next(
                i for i, a in enumerate(accepts) if a):]):
            out[cid] = "flip"
        else:
            out[cid] = "other"
    return out


def powered_generator(path: Path, layer: int) -> np.ndarray:
    b = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sel = (np.asarray(b["turn_index"]).astype(int) == 3) & \
          (np.asarray(b["phase"]).astype(str) == "pre_response")
    cids = np.asarray(b["conversation_id"]).astype(str)[sel]
    scen = np.asarray(b["scenario_id"]).astype(str)[sel]
    arm = np.asarray(b["arm"]).astype(str)[sel]
    x = b["activations"][layer].float().numpy()[sel]
    rank = {a: i for i, a in enumerate(["p0", "p1", "p2", "p3", "p4", "p5", "p6"])}
    samp = np.asarray([c.split(":")[-1] for c in cids])
    orb: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for i in range(len(scen)):
        orb[(scen[i], samp[i])][rank[arm[i]]] = i
    disp = []
    for idx in orb.values():
        if len(idx) < 7:
            continue
        for j in range(6):
            d = x[idx[j + 1]] - x[idx[j]]
            disp.append(d / (np.linalg.norm(d) + 1e-9))
    field = np.mean(disp, axis=0)
    return field / np.linalg.norm(field), x


def per_conv_displacements(cids: np.ndarray, turns: np.ndarray, x: np.ndarray):
    by_conv: dict[str, dict[int, int]] = defaultdict(dict)
    for i, (c, t) in enumerate(zip(cids, turns)):
        by_conv[c][t] = i
    disp, slopes_idx = [], {}
    for c, tmap in by_conv.items():
        ts = sorted(tmap)
        if len(ts) < 3:
            continue
        vecs = []
        for a, bb in zip(ts[:-1], ts[1:]):
            d = x[tmap[bb]] - x[tmap[a]]
            vecs.append(d / (np.linalg.norm(d) + 1e-9))
        disp.extend(vecs)
        slopes_idx[c] = [tmap[t] for t in ts]
    return np.vstack(disp), slopes_idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--powered", type=Path, required=True,
                        help="turns.pt powered bank with arm/scenario/activation fields")
    parser.add_argument("--layer-8b", type=int, default=16)
    parser.add_argument("--layer-70b", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results/baselines/g1_source_freeprose/flow_generality_sycophancy.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    classes = stance_classes()

    # ---- Rung A: 8B ----
    gen_field, powered_x = powered_generator(args.powered, args.layer_8b)
    cids8, turns8, x8, _ = load_bank(SYC_8B, args.layer_8b)
    disp8, slopes8 = per_conv_displacements(cids8, turns8, x8)
    syc_field = np.mean(disp8, axis=0)
    syc_field /= np.linalg.norm(syc_field)
    a1_direct = float(np.dot(syc_field, gen_field))
    aligner = ZFrameAligner().fit(powered_x, x8)
    x8_canon = aligner(x8)
    disp8c, _ = per_conv_displacements(cids8, turns8, x8_canon)
    syc_field_c = np.mean(disp8c, axis=0)
    syc_field_c /= np.linalg.norm(syc_field_c)
    a1_canonical = float(np.dot(syc_field_c, gen_field))
    rand_base = float(np.std([np.dot(u / np.linalg.norm(u), gen_field)
                              for u in rng.standard_normal((200, len(gen_field)))]))

    depth8 = x8 @ gen_field
    slopes, labels = [], []
    for c, idxs in slopes8.items():
        cls = classes.get(c)
        if cls not in ("flip", "steadfast"):
            continue
        slopes.append(float(np.polyfit(range(len(idxs)), depth8[idxs], 1)[0]))
        labels.append(1 if cls == "flip" else 0)
    labels_arr = np.asarray(labels)
    a2_auroc = float(roc_auc_score(labels_arr, np.asarray(slopes))) if 0 < labels_arr.mean() < 1 else float("nan")

    # own-instrument analog of rung B on the 8B: stance probe fit on 8B states
    stance_pairs = {}
    for line in LABELS.open():
        row = json.loads(line)
        stance_pairs[(str(row["conversation_id"]), int(row["turn_index"]))] = row["stance"]
    keep8, y8 = [], []
    for i, (c, tt) in enumerate(zip(cids8, turns8)):
        s = stance_pairs.get((c, tt))
        if s in ("accepts", "rejects"):
            keep8.append(i)
            y8.append(1 if s == "accepts" else 0)
    probe8 = LogisticRegression(max_iter=2000).fit(x8[np.asarray(keep8)], np.asarray(y8))
    depth8_own = probe8.decision_function(x8)
    sl8o, lb8o = [], []
    for c, idxs in slopes8.items():
        cls = classes.get(c)
        if cls not in ("flip", "steadfast"):
            continue
        sl8o.append(float(np.polyfit(range(len(idxs)), depth8_own[idxs], 1)[0]))
        lb8o.append(1 if cls == "flip" else 0)
    lb8o_arr = np.asarray(lb8o)
    a3_own_auroc = float(roc_auc_score(lb8o_arr, np.asarray(sl8o))) if 0 < lb8o_arr.mean() < 1 else float("nan")

    # ---- Rung B: 70B functional replication ----
    cids70, turns70, x70, _ = load_bank(SYC_70B, args.layer_70b)
    stance_by = {}
    for line in LABELS.open():
        row = json.loads(line)
        stance_by[(str(row["conversation_id"]), int(row["turn_index"]))] = row["stance"]
    y70, keep = [], []
    for i, (c, t) in enumerate(zip(cids70, turns70)):
        s = stance_by.get((c, t))
        if s in ("accepts", "rejects"):
            keep.append(i)
            y70.append(1 if s == "accepts" else 0)
    keep_arr = np.asarray(keep)
    probe70 = LogisticRegression(max_iter=2000).fit(x70[keep_arr], np.asarray(y70))
    depth70 = probe70.decision_function(x70)  # positive = accepting/caving side
    disp70, slopes70 = per_conv_displacements(cids70, turns70, x70)
    v = disp70[rng.choice(len(disp70), min(400, len(disp70)), replace=False)]
    coh70 = float(np.mean((v @ v.T)[np.triu_indices(len(v), 1)]))
    sl70, lb70 = [], []
    for c, idxs in slopes70.items():
        cls = classes.get(c)
        if cls not in ("flip", "steadfast"):
            continue
        sl70.append(float(np.polyfit(range(len(idxs)), depth70[idxs], 1)[0]))
        lb70.append(1 if cls == "flip" else 0)
    lb70_arr = np.asarray(lb70)
    b_auroc = float(roc_auc_score(lb70_arr, np.asarray(sl70))) if len(lb70_arr) and 0 < lb70_arr.mean() < 1 else float("nan")

    # per-conversation MONOTONICITY (the powered150-style test) under own instruments
    from scipy.stats import spearmanr
    def monotonicity(slopes_map, depth, cls_want):
        rhos = []
        for c, idxs in slopes_map.items():
            if classes.get(c) != cls_want or len(idxs) < 4:
                continue
            rhos.append(spearmanr(range(len(idxs)), depth[idxs]).statistic)
        return {"median_spearman": float(np.nanmedian(rhos)) if rhos else float("nan"),
                "frac_positive": float(np.nanmean(np.asarray(rhos) > 0)) if rhos else float("nan"),
                "n": len(rhos)}
    mono = {
        "8b_flip": monotonicity(slopes8, depth8_own, "flip"),
        "8b_steadfast": monotonicity(slopes8, depth8_own, "steadfast"),
        "70b_flip": monotonicity(slopes70, depth70, "flip"),
        "70b_steadfast": monotonicity(slopes70, depth70, "steadfast"),
    }

    # DISCRIMINATING TEST: decompose each turn-displacement into along-depth vs orthogonal.
    # "Buried common flow" predicts: along-depth components consistently positive for cavers
    # (sign-consistent small shared part) while orthogonal parts dominate norms and scatter.
    def decompose(slopes_map, x_states, depth_axis, cls_want):
        along_signs, along_fracs = [], []
        for c, idxs in slopes_map.items():
            if classes.get(c) != cls_want:
                continue
            for a, bb in zip(idxs[:-1], idxs[1:]):
                d = x_states[bb] - x_states[a]
                along = float(np.dot(d, depth_axis))
                along_signs.append(along > 0)
                along_fracs.append(abs(along) / (np.linalg.norm(d) + 1e-9))
        return {"frac_steps_descending": float(np.mean(along_signs)),
                "median_along_fraction_of_step": float(np.median(along_fracs)),
                "n_steps": len(along_signs)}
    probe8_axis = probe8.coef_[0] / np.linalg.norm(probe8.coef_[0])
    probe70_axis = probe70.coef_[0] / np.linalg.norm(probe70.coef_[0])
    decomposition = {
        "8b_flip": decompose(slopes8, x8, probe8_axis, "flip"),
        "8b_steadfast": decompose(slopes8, x8, probe8_axis, "steadfast"),
        "70b_flip": decompose(slopes70, x70, probe70_axis, "flip"),
        "70b_steadfast": decompose(slopes70, x70, probe70_axis, "steadfast"),
    }

    summary = {
        "monotonicity_own_instrument": mono,
        "displacement_decomposition": decomposition,
        "rung_A_8b": {
            "field_cos_direct": a1_direct,
            "field_cos_canonical_frame": a1_canonical,
            "random_baseline_std": rand_base,
            "flip_vs_steadfast_depth_slope_auroc_transferred_field": a2_auroc,
            "flip_vs_steadfast_depth_slope_auroc_own_probe": a3_own_auroc,
            "n_flip": int(labels_arr.sum()), "n_steadfast": int((1 - labels_arr).sum()),
        },
        "rung_B_70b_functional": {
            "displacement_coherence_mean_cos": coh70,
            "flip_vs_steadfast_depth_slope_auroc": b_auroc,
            "n_flip": int(lb70_arr.sum()) if len(lb70_arr) else 0,
            "n_steadfast": int((1 - lb70_arr).sum()) if len(lb70_arr) else 0,
        },
        "instrument_limitations": [
            "sycophancy turns advance BOTH pushback pressure and dialogue length; displacement mixes them",
            "trajectory classes derived from stance sequences by rule, not the original taxonomy file",
            "70B rung uses a stance probe (own-model instrument); no cross-model direction comparison is possible",
        ],
    }
    payload = {"schema_version": 1, "kind": "flow_generality_sycophancy", "argv": sys.argv,
               "provenance": git_provenance([Path(__file__)]), "summary": summary}
    write_json(args.out, payload)
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
