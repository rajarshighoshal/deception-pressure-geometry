from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.eval import relational_gauge_training_supervision as supervision
from geoprobe.geometry.relational_pre_status_rooted_graph import FOLDS
from geoprobe.io import file_sha256


def _node(event_id: str, fold: str, view: str) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=f"{view}:{event_id}",
        family=f"family:{fold}",
        family_fold=fold,
        turn_index=1,
        prefix_state_sha256=(fold[-1] * 64),
    )


def test_fold_training_supervision_never_opens_heldout_shard(
    tmp_path: Path, monkeypatch: object
) -> None:
    source_sha = "a" * 64
    rows = {}
    nodes = []
    event_to_node_ids = {}
    for fold in FOLDS:
        event_id = f"event:{fold}"
        rows[fold] = {
            "field_event_id": event_id,
            "family": f"family:{fold}",
            "family_fold": fold,
            "turn_index": 1,
            "prefix_state_sha256": fold[-1] * 64,
            "intervention_history": ["A"],
            "pressure_exposed": True,
            "true_status": "PASS",
            "desired_status": "FAIL",
            "outcome_class": "DECEPTIVE",
        }
        view_nodes = {}
        for view in VIEWS:
            node = _node(event_id, fold, view)
            nodes.append(node)
            view_nodes[view] = node.node_id
        event_to_node_ids[event_id] = MappingProxyType(view_nodes)

    manifest = {
        "folds": list(FOLDS),
        "source_report": {"file_sha256": source_sha},
        "shards": {
            fold: {
                "file_sha256": str(index + 1) * 64,
                "content_sha256": chr(ord("a") + index) * 64,
            }
            for index, fold in enumerate(FOLDS)
        },
    }
    root = tmp_path / "shards"
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = file_sha256(manifest_path)
    quotient = SimpleNamespace(
        nodes=tuple(nodes),
        event_to_node_ids=MappingProxyType(event_to_node_ids),
    )
    monkeypatch.setattr(
        supervision, "build_label_free_prefix_state_quotient", lambda _: quotient
    )
    opened = []

    def load(_root: Path, fold: str, **_: object) -> SimpleNamespace:
        if fold == "outer_1":
            raise AssertionError("held-out outcome shard was opened")
        opened.append(fold)
        return SimpleNamespace(
            scored_events=(MappingProxyType(rows[fold]),),
            shard_file_sha256=manifest["shards"][fold]["file_sha256"],
            content_sha256=manifest["shards"][fold]["content_sha256"],
            shard_sha256="f" * 64,
        )

    monkeypatch.setattr(supervision, "load_relational_pre_status_outcome_shard", load)
    result = supervision.build_fold_gauge_training_supervision(
        SimpleNamespace(),
        held_out_family_fold="outer_1",
        outcome_shard_root=root,
        expected_outcome_shard_manifest_file_sha256=manifest_sha,
        expected_source_report_file_sha256=source_sha,
    )
    assert opened == ["outer_2", "outer_3", "outer_4", "outer_5"]
    assert len(result.opened_training_shards) == 4
    assert all(
        len(result.risk_events_by_view[view]) == 4 for view in VIEWS
    )
    assert all(
        row.family_fold != "outer_1"
        for view in VIEWS
        for row in result.risk_events_by_view[view]
    )
