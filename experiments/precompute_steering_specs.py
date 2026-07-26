"""Precompute a steering-spec bank from activations (CPU only, no model/GPU).

Phase 1 of the CPU/GPU decouple. Builds every steering vector + geometry metadata
the action-response sweep needs -- directions (Z2 PASS/FAIL), tangent / off-tangent
projections, point-cloud context -- streaming per (layer, family) so a 24GB box can
do it, then writes a spec bank with full identity/provenance. The GPU scorer
(resident_decision_token_pool_runner --steering-spec-bank) then loads this and does
ONLY the cached PASS/FAIL margin forwards, so the A100 is never idle while numpy
does tangent kNN.

The vectors are computed with the SAME leaf primitives the inline runner uses
(fit_status_direction, fit_global_*, fit_tangent_cloud, project onto local tangent,
off_tangent_direction, signed_global_direction, unit), so a spec equals what
action_vector would have produced -- guarded by tests/test_precompute_steering_specs.py.
Leave-one-family-out is enforced: every fit/tangent/context for a row excludes that
row's family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.bidirectional_dp_diagnostic import (  # noqa: E402
    attach_status_classes,
    file_sha256,
    fit_status_direction,
)
from experiments.control_graded_dp_decision_token import (  # noqa: E402
    BIDIR_METHODS,
    GLOBAL_METHODS,
    fit_global_mean_direction,
    fit_global_probe_direction,
    fit_random_direction,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    fit_tangent_cloud,
    load_activation_bank,
    load_activation_points_from_bank,
    local_tangent_bases,
    off_tangent_direction,
    point_index,
    project_direction_onto_tangent_basis,
    to_jsonable,
    unit,
    vector_stats,
)
from experiments.decision_token_action_response import (  # noqa: E402
    PointcloudContextIndex,
    public_direction_info,
    signed_global_direction,
)
from experiments.resident_decision_token_bank_worker import (  # noqa: E402
    load_eval_rows,
    usable_unit_vector,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402

SPEC_SCHEMA_VERSION = 1


def shard_stem(layer: int, family: str) -> str:
    """Collision-proof shard filename stem.

    Sanitizing the family for the filename can map distinct families ('a b' and
    'a_b') to the same name, which would overwrite one family's shard. Append a
    hash of the RAW family so the stem is unique per (layer, family) even when the
    sanitized prefix collides. The manifest still records the exact family.
    """
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(family))[:80]
    digest = hashlib.sha1(str(family).encode("utf-8")).hexdigest()[:10]
    return f"layer{layer}_family{safe}_{digest}"


def targets_for_row(status_class: str, candidate_targets: list[str], unidirectional: bool) -> list[str]:
    """Z2 target rule: false-FAIL only gets PASS, false-PASS only gets FAIL."""
    if not unidirectional:
        return list(candidate_targets)
    if status_class == "false_FAIL":
        return ["PASS"]
    if status_class == "false_PASS":
        return ["FAIL"]
    return list(candidate_targets)


def compute_vector(
    method: str,
    target: str,
    *,
    status_dir: dict,
    global_dir: dict,
    tangent_info: dict | None,
    basis,
    base_info: dict,
) -> tuple[np.ndarray | None, dict | None, dict | None]:
    """Mirror of the runner's action_vector, reusing a precomputed tangent basis.

    Returns (raw_vector, direction_info, projection_info). The basis is the local
    tangent basis for this (row, layer); for tangent methods the per-target raw
    direction is projected onto it (the dedup -- one basis, both methods/targets).
    """
    if method in BIDIR_METHODS:
        direction_info = status_dir.get(target)
        if direction_info is None:
            return None, None, {"reason": "missing_status_direction"}
        raw = np.asarray(direction_info["_direction_np"], dtype=np.float64)
        if method == "bidir_linear":
            return unit(raw), direction_info, None
        if tangent_info is None:
            return None, direction_info, {"reason": "missing_tangent_cloud"}
        if basis is None:
            # same failure project_to_local_tangent would have returned
            return None, direction_info, base_info
        tangent_vec, projection = project_direction_onto_tangent_basis(raw, tangent_info, basis, base_info)
        if tangent_vec is None:
            return None, direction_info, projection
        if method == "bidir_tangent":
            return unit(tangent_vec.detach().float().cpu().numpy()), direction_info, projection
        off = off_tangent_direction(raw, tangent_vec)
        if off is None:
            return None, direction_info, {"reason": "missing_off_tangent", "projection": projection}
        return unit(off.detach().float().cpu().numpy()), direction_info, projection
    direction_info = global_dir.get(method)
    if direction_info is None:
        return None, None, {"reason": "missing_global_direction"}
    return signed_global_direction(direction_info, target), direction_info, None


def spec_identity(args: argparse.Namespace, *, transcript_hashes: dict, activation_hash: str, config_hash: str) -> dict:
    """Vector-affecting identity. The GPU scorer must match this exactly before it
    will score a spec bank -- this is what prevents a bounded bank from being scored
    against an unbounded run, or a different cloud/levels/methods."""
    return {
        "spec_schema_version": SPEC_SCHEMA_VERSION,
        "transcripts_sha256": transcript_hashes,
        "activations_sha256": activation_hash,
        "config_sha256": config_hash,
        "layers": [int(x) for x in args.layers],
        "methods": list(args.methods),
        "candidate_targets": list(args.candidate_targets),
        "direction_turn": int(args.direction_turn),
        "direction_phase": args.direction_phase,
        "query_turn": int(args.query_turn),
        "query_phase": args.query_phase,
        "eval_levels": sorted(args.eval_levels),
        "direction_levels": sorted(args.direction_levels),
        "tangent_levels": sorted(args.tangent_levels),
        "tangent_turns": sorted(int(t) for t in args.tangent_turns),
        "tangent_phases": sorted(args.tangent_phases),
        "min_mixed_scenarios": int(args.min_mixed_scenarios),
        "min_direction_levels": int(args.min_direction_levels),
        "tangent_neighbors": int(args.tangent_neighbors),
        "tangent_dim": int(args.tangent_dim),
        "context_neighbors": int(args.context_neighbors),
        "equivariant_directions": bool(args.equivariant_directions),
        "unidirectional_targets": bool(args.unidirectional_targets),
        "seed": int(args.seed),
        "limit": (int(args.limit) if args.limit is not None else None),
        "limit_per_status_class": (
            int(args.limit_per_status_class) if args.limit_per_status_class is not None else None
        ),
        "limit_strategy": args.limit_strategy,
        "scenario_include": args.scenario_include,
        "scenario_exclude": args.scenario_exclude,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "lofo_rule": "leave_one_family_out",
    }


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out)
    specs_dir = out_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    transcript_paths = [Path(p) for p in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    transcript_hashes = {str(p): file_sha256(p) for p in transcript_paths}
    activation_hash = file_sha256(activation_path)
    config_hash = file_sha256(config_path)
    identity = spec_identity(
        args, transcript_hashes=transcript_hashes, activation_hash=activation_hash, config_hash=config_hash
    )

    layers = [int(x) for x in args.layers]
    candidate_targets = list(args.candidate_targets)
    methods = list(args.methods)
    dlevels = set(args.direction_levels)

    # eval-row selection (canonical): runs its own activation load then frees it,
    # so it never coexists in RAM with the resident bank loaded below.
    print("[precompute] selecting eval rows ...", flush=True)
    rows, transcript_by_cid, _meta = load_eval_rows(
        transcript_paths=transcript_paths,
        activation_path=activation_path,
        layer=layers[0],
        query_turn=args.query_turn,
        query_phase=args.query_phase,
        eval_levels=set(args.eval_levels),
        limit=args.limit,
        limit_per_status_class=args.limit_per_status_class,
        limit_strategy=args.limit_strategy,
        seed=args.seed,
        scenario_include=args.scenario_include,
        scenario_exclude=args.scenario_exclude,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    rows_by_family: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_family.setdefault(str(row["family"]), []).append(row)
    families = sorted(rows_by_family)
    eval_cids = sorted(str(row["conversation_id"]) for row in rows)
    if len(set(eval_cids)) != len(eval_cids):
        raise ValueError("eval rows contain duplicate conversation_id values; spec-bank scoring expects unique cids")
    print(f"[precompute] {len(rows)} eval rows across {len(families)} families", flush=True)

    print("[precompute] loading activation bank ...", flush=True)
    bank = load_activation_bank(activation_path)

    shards: list[dict] = []
    total_specs = 0
    started = time.perf_counter()
    for layer in layers:
        direction_raw, _ = load_activation_points_from_bank(
            bank, layer=layer, turns={args.direction_turn}, phases={args.direction_phase}
        )
        direction_pts, _ = attach_status_classes(direction_raw, transcript_by_cid)
        query_raw, _ = load_activation_points_from_bank(
            bank, layer=layer, turns={args.query_turn}, phases={args.query_phase}
        )
        query_idx = point_index(query_raw)
        region_pts, _ = attach_status_classes(query_raw, transcript_by_cid)
        tangent_pts, _ = load_activation_points_from_bank(
            bank, layer=layer,
            turns={int(t) for t in args.tangent_turns},
            phases=set(args.tangent_phases),
            levels=set(args.tangent_levels),
        )

        for family in families:
            fam_rows = rows_by_family[family]
            status_dir = {
                t: fit_status_direction(
                    direction_pts, heldout_family=family, direction_levels=dlevels,
                    target_status=t, min_mixed_scenarios=args.min_mixed_scenarios,
                    min_levels=args.min_direction_levels, equivariant=args.equivariant_directions,
                )
                for t in candidate_targets
            }
            global_dir: dict = {}
            for method in methods:
                if method not in GLOBAL_METHODS:
                    continue
                if method == "global_mean":
                    global_dir[method] = fit_global_mean_direction(direction_pts, heldout_family=family, direction_levels=dlevels)
                elif method == "global_probe":
                    global_dir[method] = fit_global_probe_direction(direction_pts, heldout_family=family, direction_levels=dlevels)
                elif method == "random_global":
                    global_dir[method] = fit_random_direction(direction_pts, heldout_family=family, direction_levels=dlevels, layer=layer, seed=args.seed)
            tangent_info = fit_tangent_cloud(tangent_pts, heldout_family=family)

            # batched geometry for this family's rows that have a query activation
            present = [(str(r["conversation_id"]), r) for r in fam_rows if str(r["conversation_id"]) in query_idx]
            basis_by_cid: dict[str, tuple] = {}
            context_by_cid: dict[str, dict] = {}
            if present:
                qxs = np.vstack([query_idx[cid]["x"] for cid, _ in present])
                if tangent_info is not None:
                    bases = local_tangent_bases(
                        tangent_info, qxs, tangent_neighbors=args.tangent_neighbors,
                        tangent_dim=args.tangent_dim,
                    )
                else:
                    bases = [(None, {"reason": "missing_tangent_cloud"})] * len(present)
                ctx_index = PointcloudContextIndex(region_pts, heldout_family=family, k=args.context_neighbors)
                ctx = ctx_index.query_batch(qxs)
                for (cid, _), b, c in zip(present, bases, ctx):
                    basis_by_cid[cid] = b
                    context_by_cid[cid] = c

            vectors: list[np.ndarray] = []
            meta: list[dict] = []
            for cid, row in present:
                basis, base_info = basis_by_cid[cid]
                context = context_by_cid[cid]
                targets = targets_for_row(str(row.get("status_class", "")), candidate_targets, args.unidirectional_targets)
                for method in methods:
                    for target in targets:
                        vec, direction_info, projection = compute_vector(
                            method, target, status_dir=status_dir, global_dir=global_dir,
                            tangent_info=tangent_info, basis=basis, base_info=base_info,
                        )
                        clean = usable_unit_vector(vec)
                        if clean is None:
                            continue
                        neigh = None
                        if isinstance(projection, dict) and basis is not None and "_neighbor_idx" in (base_info or {}):
                            neigh = {
                                "neighbor_idx": [int(i) for i in base_info["_neighbor_idx"]],
                                "neighbor_distances": [float(d) for d in base_info["_neighbor_distances"]],
                            }
                        vectors.append(clean.astype(np.float32))
                        meta.append({
                            "conversation_id": cid,
                            "scenario_id": row.get("scenario_id"),
                            "family": family,
                            "status_class": row.get("status_class"),
                            "true_status": row.get("true_status"),
                            "layer": int(layer),
                            "method": method,
                            "target_status": target,
                            "direction_info": public_direction_info(direction_info),
                            "direction_stats": vector_stats(clean),
                            "projection": to_jsonable(projection) if isinstance(projection, dict) else None,
                            "tangent_neighbors_debug": neigh,
                            "context": context,
                        })
            if not vectors:
                continue
            stem = shard_stem(layer, family)
            np.savez_compressed(specs_dir / f"{stem}.npz", vectors=np.vstack(vectors))
            with (specs_dir / f"{stem}.jsonl").open("w") as fh:
                for m in meta:
                    fh.write(json.dumps(to_jsonable(m), sort_keys=True) + "\n")
            shards.append({"layer": int(layer), "family": family, "stem": stem, "n_specs": len(vectors)})
            total_specs += len(vectors)

        # free this layer's points before the next layer
        del direction_pts, direction_raw, query_raw, query_idx, region_pts, tangent_pts
        print(f"[precompute] layer {layer} done; specs={total_specs}; peak_rss={peak_rss_gb():.1f}GB", flush=True)
    del bank

    manifest = {
        "spec_schema_version": SPEC_SCHEMA_VERSION,
        "kind": "steering_spec_bank",
        "identity": identity,
        "provenance": git_provenance([Path(__file__), config_path, activation_path, *transcript_paths]),
        "argv": sys.argv,
        "n_eval_rows": len(rows),
        "n_eval_cids": len(eval_cids),
        "eval_conversation_ids": eval_cids,
        "n_families": len(families),
        "families": families,
        "n_specs": total_specs,
        "shards": shards,
        "duration_sec": round(time.perf_counter() - started, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True))
    print(f"[precompute] DONE: {total_specs} specs, {len(shards)} shards, "
          f"{manifest['duration_sec']}s, peak_rss={manifest['peak_rss_gb']}GB -> {out_dir}", flush=True)
    return manifest


def add_spec_identity_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the data + vector-affecting args that define a spec bank's identity.

    Shared by the precompute and the scorer so the scorer can recompute the SAME
    spec_identity and refuse a bank built with different geometry. Every arg here
    changes which steering vectors exist or their values.
    """
    p.add_argument("--transcripts", nargs="+", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--layers", default="8,12,16,19,20,24,28,32")
    p.add_argument("--methods", default="bidir_linear,bidir_tangent,bidir_off_tangent,global_mean,global_probe,random_global")
    p.add_argument("--candidate-targets", default="PASS,FAIL")
    p.add_argument("--direction-turn", type=int, default=2)
    p.add_argument("--direction-phase", default="pre_response")
    p.add_argument("--query-turn", type=int, default=3)
    p.add_argument("--query-phase", default="pre_response")
    p.add_argument("--eval-levels", default="p3,p4,p5")
    p.add_argument("--direction-levels", default="p3,p4,p5,p6")
    p.add_argument("--tangent-levels", default="p0,p1,p2,p3,p4,p5,p6")
    p.add_argument("--tangent-turns", default="0,1,2,3")
    p.add_argument("--tangent-phases", default="pre_response,post_response")
    p.add_argument("--min-mixed-scenarios", type=int, default=2)
    p.add_argument("--min-direction-levels", type=int, default=2)
    p.add_argument("--tangent-neighbors", type=int, default=16)
    p.add_argument("--tangent-dim", type=int, default=4)
    p.add_argument("--context-neighbors", type=int, default=32)
    p.add_argument("--unidirectional-targets", action="store_true")
    p.add_argument("--equivariant-directions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--limit-per-status-class", type=int, default=None)
    p.add_argument("--limit-strategy", choices=["first", "shuffle", "family_round_robin"], default="family_round_robin")
    p.add_argument("--scenario-include", default=None)
    p.add_argument("--scenario-exclude", default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260620)
    return p


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_spec_identity_args(p)
    p.add_argument("--out", required=True)
    return p


def _normalize(args: argparse.Namespace) -> argparse.Namespace:
    args.layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    args.methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    args.candidate_targets = [t.strip() for t in str(args.candidate_targets).split(",") if t.strip()]
    args.eval_levels = [x.strip() for x in str(args.eval_levels).split(",") if x.strip()]
    args.direction_levels = [x.strip() for x in str(args.direction_levels).split(",") if x.strip()]
    args.tangent_levels = [x.strip() for x in str(args.tangent_levels).split(",") if x.strip()]
    args.tangent_turns = [x.strip() for x in str(args.tangent_turns).split(",") if x.strip()]
    args.tangent_phases = [x.strip() for x in str(args.tangent_phases).split(",") if x.strip()]
    return args


def main() -> None:
    args = _normalize(make_parser().parse_args())
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, num_shards)")
    build(args)


if __name__ == "__main__":
    main()
