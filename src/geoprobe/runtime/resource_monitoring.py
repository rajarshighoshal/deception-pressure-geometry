"""Optional host/GPU telemetry and deterministic family sharding."""
from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path


def _host_ram_snapshot() -> dict[str, int | None]:
    meminfo: dict[str, int] = {}
    proc = Path("/proc/meminfo")
    if proc.exists():
        for line in proc.read_text().splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                meminfo[key] = int(parts[0]) * 1024
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_bytes = max_rss * 1024 if sys.platform.startswith("linux") else max_rss
    return {
        "mem_total_bytes": meminfo.get("MemTotal"),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "process_max_rss_bytes": max_rss_bytes,
    }


def _gpu_snapshot() -> list[dict[str, int | float | str | None]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except Exception:
        return []
    rows: list[dict[str, int | float | str | None]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, utilization, used, total, power = parts
        rows.append(
            {
                "index": int(index),
                "name": name,
                "utilization_gpu_pct": float(utilization),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
                "power_draw_w": None if power in {"[N/A]", "N/A"} else float(power),
            }
        )
    return rows


def resource_snapshot(stage: str, start_time: float) -> dict[str, object]:
    """Return optional telemetry without affecting experiment semantics."""
    return {
        "stage": stage,
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "host_ram": _host_ram_snapshot(),
        "gpus": _gpu_snapshot(),
    }


def shard_rows_by_family(
    rows: list[dict], *, num_shards: int, shard_index: int
) -> list[dict]:
    """Assign whole families deterministically to one of ``num_shards`` shards."""
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if num_shards == 1:
        return list(rows)
    families = sorted({str(row["family"]) for row in rows})
    keep = {
        family for index, family in enumerate(families) if index % num_shards == shard_index
    }
    return [row for row in rows if str(row["family"]) in keep]


__all__ = ["resource_snapshot", "shard_rows_by_family"]
