#!/usr/bin/env python3
"""
Live memory/CPU monitor for a process tree (parent + joblib workers).

Use this alongside a long-running simulation to observe:
- current total RSS (resident memory) across the target process + all children
- peak total RSS observed since the monitor started

Example
-------
1) Run simulation in terminal A:
   python code/simulations/binary_comparison/Comprehensive_simulation_v4/lambda_comparison.py --cache

2) Find parent PID in terminal B:
   pgrep -af "Comprehensive_simulation_v4/lambda_comparison.py"

3) Monitor in terminal B:
   python scripts/monitor_peak_ram.py <PID> --include-children
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _read_mem_total_bytes() -> int | None:
    """
    Return total system memory in bytes (MemTotal) from /proc/meminfo.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # MemTotal is in kB
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _read_proc_stat(pid: int) -> tuple[int, int] | None:
    """
    Return (utime_ticks, stime_ticks) from /proc/<pid>/stat or None if gone.
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            s = f.read()
    except OSError:
        return None
    # /proc/<pid>/stat has comm in parens which may contain spaces.
    rparen = s.rfind(")")
    if rparen == -1:
        return None
    rest = s[rparen + 2 :].split()
    # utime is field 14, stime field 15 (1-indexed) => in `rest` (0-indexed) they are 11 and 12
    try:
        utime = int(rest[11])
        stime = int(rest[12])
    except (IndexError, ValueError):
        return None
    return utime, stime


def _read_proc_rss_bytes(pid: int) -> int | None:
    """
    Return RSS bytes from /proc/<pid>/statm or None if gone.
    """
    try:
        with open(f"/proc/{pid}/statm", "r", encoding="utf-8") as f:
            parts = f.read().split()
    except OSError:
        return None
    if len(parts) < 2:
        return None
    try:
        rss_pages = int(parts[1])
    except ValueError:
        return None
    return int(rss_pages) * int(os.sysconf("SC_PAGE_SIZE"))


def _build_ppid_index() -> dict[int, list[int]]:
    """
    Return mapping PPid -> [child pids] by scanning /proc.
    """
    idx: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                status = f.read().splitlines()
        except OSError:
            continue
        ppid = None
        for line in status:
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split()[1])
                except Exception:
                    ppid = None
                break
        if ppid is None:
            continue
        idx.setdefault(ppid, []).append(pid)
    return idx


def _collect_tree_pids(root_pid: int, include_children: bool) -> list[int]:
    if not include_children:
        return [root_pid]
    idx = _build_ppid_index()
    out: list[int] = []
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(idx.get(pid, []))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Live peak RSS monitor for a PID (optionally including children).")
    ap.add_argument("pid", type=int, help="PID of the main (parent) process to monitor.")
    ap.add_argument("--interval", type=float, default=10.0, help="Sampling interval in seconds (default: 10.0).")
    ap.add_argument(
        "--include-children",
        action="store_true",
        help="Include all child processes recursively (recommended for joblib).",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional CSV path to append samples (ts,rss_bytes,rss_pct,cpu_percent,n_procs).",
    )
    args = ap.parse_args()

    root_pid = int(args.pid)
    mem_total = _read_mem_total_bytes()
    clk_tck = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    last_ts = time.time()
    last_cpu_ticks: dict[int, int] = {}
    last_total_ticks = 0

    # Prime CPU readings.
    pids0 = _collect_tree_pids(root_pid, include_children=bool(args.include_children))
    for pid in pids0:
        st = _read_proc_stat(pid)
        if st is not None:
            last_cpu_ticks[pid] = st[0] + st[1]
            last_total_ticks += last_cpu_ticks[pid]

    peak_rss = 0
    peak_ts = None

    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        if not args.log.exists():
            args.log.write_text("ts,rss_bytes,rss_pct,cpu_percent,n_procs\n")

    try:
        while True:
            if not os.path.exists(f"/proc/{root_pid}"):
                break

            pids = _collect_tree_pids(root_pid, include_children=bool(args.include_children))
            rss = 0
            total_ticks = 0
            ticks_now: dict[int, int] = {}
            for pid in pids:
                rb = _read_proc_rss_bytes(pid)
                if rb is not None:
                    rss += rb
                st = _read_proc_stat(pid)
                if st is not None:
                    t = st[0] + st[1]
                    ticks_now[pid] = t
                    total_ticks += t

            now = time.time()
            dt = max(1e-6, now - last_ts)
            d_ticks = total_ticks - last_total_ticks
            cpu = (d_ticks / clk_tck) / dt * 100.0
            last_ts = now
            last_total_ticks = total_ticks
            last_cpu_ticks = ticks_now
            n_procs = len(pids)

            if rss > peak_rss:
                peak_rss = rss
                peak_ts = time.time()

            rss_mib = rss / 1024 / 1024
            peak_mib = peak_rss / 1024 / 1024
            rss_pct = (rss / mem_total * 100.0) if mem_total else float("nan")
            peak_pct = (peak_rss / mem_total * 100.0) if mem_total else float("nan")

            line = (
                f"rss={rss_mib:9.1f} MiB ({rss_pct:5.2f}%) | "
                f"peak={peak_mib:9.1f} MiB ({peak_pct:5.2f}%) | "
                f"cpu={cpu:7.1f}% | procs={n_procs:3d}"
            )
            print(line, flush=True)

            if args.log is not None:
                with args.log.open("a") as f:
                    f.write(f"{time.time()},{rss},{rss_pct},{cpu},{n_procs}\n")

            time.sleep(float(args.interval))
    except KeyboardInterrupt:
        pass

    peak_mib = peak_rss / 1024 / 1024
    peak_pct = (peak_rss / mem_total * 100.0) if mem_total else float("nan")
    print(f"\nPEAK RSS: {peak_mib:.1f} MiB ({peak_pct:.2f}%)", flush=True)
    if peak_ts is not None:
        print(f"Peak time: {time.ctime(peak_ts)}", flush=True)


if __name__ == "__main__":
    main()

