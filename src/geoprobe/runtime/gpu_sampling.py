from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


def gpu_sample() -> dict | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).splitlines()[0]
    except Exception:
        return None
    parts = [part.strip() for part in out.split(",")]
    if len(parts) != 3:
        return None
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_util_pct": float(parts[0]),
        "memory_used_mib": float(parts[1]),
        "memory_total_mib": float(parts[2]),
    }


def gpu_sampler(path: Path, stop_event: threading.Event, interval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "gpu_util_pct", "memory_used_mib", "memory_total_mib"],
        )
        writer.writeheader()
        while not stop_event.is_set():
            row = gpu_sample()
            if row:
                writer.writerow(row)
                handle.flush()
            stop_event.wait(interval)


def summarize_samples(path: Path) -> dict:
    rows = []
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append({
                        "gpu_util_pct": float(row["gpu_util_pct"]),
                        "memory_used_mib": float(row["memory_used_mib"]),
                        "memory_total_mib": float(row["memory_total_mib"]),
                    })
                except Exception:
                    continue
    utils = [row["gpu_util_pct"] for row in rows]
    mems = [row["memory_used_mib"] for row in rows]
    total = rows[-1]["memory_total_mib"] if rows else None
    return {
        "sample_count": len(rows),
        "gpu_util_pct_max": max(utils) if utils else None,
        "gpu_util_pct_mean": (sum(utils) / len(utils)) if utils else None,
        "memory_used_mib_max": max(mems) if mems else None,
        "memory_used_mib_mean": (sum(mems) / len(mems)) if mems else None,
        "memory_total_mib": total,
        "memory_used_frac_max": (max(mems) / total) if mems and total else None,
    }
