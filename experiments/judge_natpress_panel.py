"""Blinded three-judge Sonnet panel over natpress rollouts (build / judge / aggregate).

Pipeline:
  build     rows.jsonl -> tasks.jsonl (one blinded task per conversation-channel)
  judge     tasks.jsonl -> {out-dir}/raw/{blind_item_id}.j{judge_index}.json
            (every raw reply is written verbatim BEFORE parsing; resumable;
             one repair-retry per unparseable reply)
  aggregate raw replies -> judged_rows.jsonl + panel_manifest.json + disagreements.jsonl
            (coverage gate: nonzero exit if <0.95 of tasks have >=2 parseable judgements)

Backends: ``mock`` (deterministic, network-free; for tests/dev) and ``judge-cli``
(external CLI backend for a model of your choice).
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # noqa: E402

from geoprobe.io import file_sha256  # noqa: E402

COVERAGE_MIN = 0.95
# Instrument registry: every module must export the full judge surface (build_judge_tasks,
# parse_judge_response, render_task_prompt, mock_judge_reply, aggregate_panel,
# build_panel_manifest, INSTRUMENT_VERSION). natpress_agentic re-exports the shared machinery.
INSTRUMENTS = {
    "natpress": "geoprobe.eval.natpress_judge_instrument",
    "natpress_agentic": "geoprobe.eval.natpress_agentic_judge_instrument",
}


def _load_instrument(name: str):
    import importlib

    if name not in INSTRUMENTS:
        raise ValueError(f"unknown instrument: {name!r} (choices: {sorted(INSTRUMENTS)})")
    return importlib.import_module(INSTRUMENTS[name])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: line is not an object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _call_judge_cli(
    prompt: str,
    model: str,
    timeout: int,
    command: str | None,
) -> str:
    if command is None:
        raise ValueError("--judge-cli is required for the judge-cli backend")
    cli = shlex.split(command)
    if len(cli) != 1:
        raise ValueError(f"--judge-cli must be a single executable name/path, got {command!r}")
    out = subprocess.run(
        [cli[0], "-p", prompt, "--model", model],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if out.returncode != 0:
        raise RuntimeError(f"{cli[0]} -p failed: {out.stderr[:300]}")
    return out.stdout.strip()


def _judge_call(
    task: dict[str, Any],
    judge_index: int,
    *,
    inst: Any,
    backend: str,
    model: str,
    command: str | None,
    timeout: int,
    repair: bool,
) -> str:
    if backend == "mock":
        return inst.mock_judge_reply(task, judge_index)
    if backend == "judge-cli":
        prompt = inst.render_task_prompt(task)
        if repair:
            prompt = (
                prompt
                + "\n\nYour previous reply was NOT valid JSON. Return ONLY the single "
                "JSON object described above, with no prose or code fences."
            )
        return _call_judge_cli(prompt, model, timeout, command=command)
    raise ValueError(f"unknown backend: {backend!r}")


def _raw_path(raw_dir: Path, blind_item_id: str, judge_index: int) -> Path:
    return raw_dir / f"{blind_item_id}.j{judge_index}.json"


def _write_raw(
    path: Path,
    task: dict[str, Any],
    judge_index: int,
    *,
    judge_model: str,
    backend: str,
    reply: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "blind_item_id": task["blind_item_id"],
                "conversation_id": task["conversation_id"],
                "channel": task["channel"],
                "judge_index": judge_index,
                "judge_model": judge_model,
                "backend": backend,
                "raw": reply,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _existing_parseable(path: Path, channel: str, inst: Any) -> bool:
    if not path.exists():
        return False
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        inst.parse_judge_response(str(wrapper["raw"]), channel)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def cmd_build(args: argparse.Namespace) -> int:
    inst = _load_instrument(args.instrument)
    rows_path = Path(args.rows)
    out_path = Path(args.out)
    if out_path.resolve() == rows_path.resolve():
        raise ValueError("--out must not overwrite --rows")
    rows = _read_jsonl(rows_path)
    rows_sha256 = file_sha256(rows_path)
    tasks = inst.build_judge_tasks(rows, rows_sha256)
    _write_jsonl(out_path, tasks)
    print(
        f"built {len(tasks)} tasks from {len(rows)} conversations "
        f"(instrument={args.instrument}, rows_sha256={rows_sha256[:12]}) -> {out_path}"
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    inst = _load_instrument(args.instrument)
    tasks = _read_jsonl(Path(args.tasks))
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    judged = skipped = repaired = failed = 0
    for task in tasks:
        channel = str(task["channel"])
        for judge_index in range(args.judges):
            path = _raw_path(raw_dir, str(task["blind_item_id"]), judge_index)
            if _existing_parseable(path, channel, inst):
                skipped += 1
                continue
            # A hung/failed backend call must never kill the worker: log, count,
            # and move on -- the resume loop re-attempts missing raw files.
            try:
                reply = _judge_call(
                    task,
                    judge_index,
                    inst=inst,
                    backend=args.backend,
                    model=args.judge_model,
                    command=args.judge_cli,
                    timeout=args.timeout,
                    repair=False,
                )
            except Exception as exc:  # noqa: BLE001 - includes subprocess.TimeoutExpired
                print(f"judge call error ({task['blind_item_id']} j{judge_index}): {exc}")
                failed += 1
                continue
            _write_raw(
                path, task, judge_index, judge_model=args.judge_model,
                backend=args.backend, reply=reply,
            )
            ok = _parses(reply, channel, inst)
            if not ok:
                try:
                    reply = _judge_call(
                        task,
                        judge_index,
                        inst=inst,
                        backend=args.backend,
                        model=args.judge_model,
                        command=args.judge_cli,
                        timeout=args.timeout,
                        repair=True,
                    )
                    _write_raw(
                        path, task, judge_index, judge_model=args.judge_model,
                        backend=args.backend, reply=reply,
                    )
                    repaired += 1
                    ok = _parses(reply, channel, inst)
                except Exception as exc:  # noqa: BLE001
                    print(f"repair call error ({task['blind_item_id']} j{judge_index}): {exc}")
                    ok = False
            judged += 1
            if not ok:
                failed += 1
    print(
        f"judge: {judged} new calls ({skipped} skipped, {repaired} repair-retries, "
        f"{failed} still unparseable) with backend={args.backend} "
        f"model={args.judge_model} instrument={args.instrument} -> {raw_dir}"
    )
    return 0


def _parses(reply: str, channel: str, inst: Any) -> bool:
    try:
        inst.parse_judge_response(reply, channel)
    except ValueError:
        return False
    return True


def cmd_aggregate(args: argparse.Namespace) -> int:
    inst = _load_instrument(args.instrument)
    tasks_path = Path(args.tasks)
    tasks = _read_jsonl(tasks_path)
    raw_dir = Path(args.raw_dir) / "raw"
    if not tasks:
        raise ValueError("no tasks to aggregate")
    rows_sha256 = str(tasks[0].get("rows_sha256", ""))
    if any(str(t.get("rows_sha256", "")) != rows_sha256 for t in tasks):
        raise ValueError("tasks disagree on rows_sha256; refusing to aggregate")

    assistant_texts: dict[tuple[str, int], str] = {}
    for task in tasks:
        if task["channel"] == "assistant":
            for turn in task["payload"]["transcript"]:
                assistant_texts[(task["conversation_id"], int(turn["turn_index"]))] = str(
                    turn.get("assistant_text", "")
                )

    per_task: list[dict[str, Any]] = []
    parse_counts: dict[str, int] = {}
    for task in tasks:
        blind = str(task["blind_item_id"])
        channel = str(task["channel"])
        count = 0
        for path in sorted(raw_dir.glob(f"{blind}.j*.json")):
            try:
                wrapper = json.loads(path.read_text(encoding="utf-8"))
                parsed = inst.parse_judge_response(str(wrapper["raw"]), channel)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            per_task.append(
                {
                    "blind_item_id": blind,
                    "conversation_id": str(task["conversation_id"]),
                    "channel": channel,
                    "judge_index": wrapper.get("judge_index"),
                    "parsed": parsed,
                }
            )
            count += 1
        parse_counts[blind] = count

    aggregate = inst.aggregate_panel(per_task, assistant_texts=assistant_texts)
    n_tasks = len(tasks)
    tasks_two_plus = sum(1 for t in tasks if parse_counts.get(str(t["blind_item_id"]), 0) >= 2)
    coverage_rate = tasks_two_plus / n_tasks if n_tasks else 0.0
    coverage = {
        "n_tasks": n_tasks,
        "tasks_with_2plus_parseable": tasks_two_plus,
        "coverage_rate": coverage_rate,
        "min_required": COVERAGE_MIN,
        "passed": coverage_rate >= COVERAGE_MIN,
    }

    judged_rows = [aggregate["conversations"][cid] for cid in sorted(aggregate["conversations"])]
    _write_jsonl(Path(args.out_judged), judged_rows)
    _write_jsonl(Path(args.out_disagreements), aggregate["disagreements"])
    manifest = inst.build_panel_manifest(
        rows_sha256=rows_sha256,
        tasks_sha256=file_sha256(tasks_path),
        judge_model=args.judge_model,
        judges_per_task=args.judges,
        aggregate=aggregate,
        coverage=coverage,
        source_paths=[tasks_path],
    )
    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"aggregate: {len(judged_rows)} conversations, coverage "
        f"{tasks_two_plus}/{n_tasks}={coverage_rate:.3f} "
        f"(min {COVERAGE_MIN}), {len(aggregate['disagreements'])} disagreements"
    )
    if not coverage["passed"]:
        print(
            f"ERROR: coverage {coverage_rate:.3f} below required {COVERAGE_MIN}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="rows.jsonl -> blinded tasks.jsonl")
    build.add_argument("--rows", required=True)
    build.add_argument("--out", required=True, help="tasks.jsonl output path")
    build.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="natpress")
    build.set_defaults(func=cmd_build)

    judge = sub.add_parser("judge", help="tasks.jsonl -> raw judge replies")
    judge.add_argument("--tasks", required=True)
    judge.add_argument("--out-dir", required=True, help="run dir; raw files go under raw/")
    judge.add_argument("--backend", choices=["mock", "judge-cli"], default="mock")
    judge.add_argument(
        "--judge-cli",
        help="Executable for external judge CLI backend.",
    )
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judges", type=int, default=3)
    judge.add_argument("--timeout", type=int, default=180)
    judge.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="natpress")
    judge.set_defaults(func=cmd_judge)

    agg = sub.add_parser("aggregate", help="raw replies -> judged rows + manifest")
    agg.add_argument("--tasks", required=True)
    agg.add_argument("--raw-dir", required=True, help="judge --out-dir (contains raw/)")
    agg.add_argument("--out-judged", required=True)
    agg.add_argument("--out-manifest", required=True)
    agg.add_argument("--out-disagreements", required=True)
    agg.add_argument("--judge-model", required=True)
    agg.add_argument("--judges", type=int, default=3)
    agg.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="natpress")
    agg.set_defaults(func=cmd_aggregate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
