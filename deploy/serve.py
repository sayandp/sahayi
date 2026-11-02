"""Start the API in the deploy image: thread count from the CPUs actually available.

os.cpu_count() reports the host's cores even when a container is limited to 2, which would
oversubscribe the CPU, so the affinity mask and the cgroup quota cap it.
--cpus N pins the process to N cores, for simulating a small Space without a container.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def cgroup_cpus(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """The CPU quota of this container in cores, or None if it has none."""
    try:
        quota, period = (root / "cpu.max").read_text().split()  # cgroup v2
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int((root / "cpu/cpu.cfs_quota_us").read_text())
        period = int((root / "cpu/cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


def effective_cpus(root: Path = Path("/sys/fs/cgroup")) -> int:
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    else:
        import psutil

        count = min(count, len(psutil.Process().cpu_affinity() or [count]))
    quota = cgroup_cpus(root)
    return max(1, min(count, math.ceil(quota))) if quota else max(1, count)


def pin_cpus(n: int) -> None:
    """Simulation only: use every other logical core, so each is a separate physical core."""
    import psutil

    psutil.Process().cpu_affinity(list(range(0, 2 * n, 2)))


def configure_threads() -> int:
    """Set before torch/onnxruntime load. Passive waiting is the busy-wait fix."""
    n = effective_cpus()
    os.environ["SAHAYI_TORCH_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["SAHAYI_ORT_SPIN"] = "0"
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, help="simulation: pin to this many cores")
    args = ap.parse_args()
    if args.cpus:
        pin_cpus(args.cpus)
    print(f"threads: {configure_threads()}", flush=True)

    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        app_dir=str(Path(__file__).resolve().parents[1]),
        host="0.0.0.0",  # noqa: S104 - inside the container
        port=int(os.environ.get("PORT", 7860)),
        proxy_headers=True,  # Spaces sits behind a proxy: rate-limit by the real client IP
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
