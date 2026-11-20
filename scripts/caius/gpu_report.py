"""
GPU monitor for the run (stdlib only; run from the repo root, any python):

    python scripts/caius/gpu_report.py            # running + finished jobs
    python scripts/caius/gpu_report.py --running  # only jobs whose log is still growing

Running jobs: from logs/run_abcfold/<protein>/<depth>_<backend>.log, the GPU
check line (device, torch CUDA) and the latest "[gpu] t=.. util=..% mem=.."
sample. A log with a check but no util > 5 % after 10 min is listed as IDLE.
Finished jobs: from results/*/predictions/*/*/run.json, jobs per GPU type with
median fold minutes and GPU busy fraction, and every job flagged
gpu_idle_warning (busy < 20 % of a fold longer than 5 min).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

CHECK = re.compile(r"^\[gpu\] check (\S+) \(\S+\): (.*)$")
SAMPLE = re.compile(r"^\[gpu\] t=([\d.]+)min util=(\d+)% mem=(\d+)MiB")


def running(logs: Path, done: set[str], fresh_s: float) -> None:
    rows = []
    now = time.time()
    for log in sorted(logs.glob("*/*.log")):
        key = f"{log.parent.name}/{log.stem}"
        if key in done or now - log.stat().st_mtime > fresh_s:
            continue
        dev, last, busy, n = "?", None, 0, 0
        for line in log.read_text(errors="replace").splitlines():
            if m := CHECK.match(line):
                try:
                    c = json.loads(m[2])
                    dev = f"{c.get('device') or 'NO CUDA'} sm{c.get('capability', '')} torch{c.get('torch', '')}"
                except json.JSONDecodeError:
                    dev = "check unparsable"
            elif m := SAMPLE.match(line):
                last = (float(m[1]), int(m[2]), int(m[3]))
                n += 1
                busy += int(m[2]) > 5
        flag = ""
        if last and last[0] > 10 and busy == 0:
            flag = "IDLE"
        elif dev.startswith("NO CUDA"):
            flag = "NO GPU"
        rows.append((key, dev, last, busy, n, flag))
    print(f"== running ({len(rows)} logs written to in the last {fresh_s / 60:.0f} min)")
    for key, dev, last, busy, n, flag in rows:
        s = f"t={last[0]:.0f}min util={last[1]}% mem={last[2] / 1024:.1f}GiB busy {busy}/{n}" if last else "no sample yet"
        print(f"  {flag:6s} {key:45s} {dev:45s} {s}")


def finished(results: Path) -> set[str]:
    runs = [json.loads(f.read_text()) for f in sorted(results.glob("*/predictions/*/*/run.json"))]
    by = defaultdict(list)
    for r in runs:
        by[r.get("gpu", "?")].append(r)
    print(f"== finished: {len(runs)} jobs")
    for gpu, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        mins = [r["fold_s"] / 60 for r in rs if "fold_s" in r]
        busy = [r["gpu_busy_frac"] for r in rs if "gpu_busy_frac" in r]
        print(f"  {gpu:40s} {len(rs):5d} jobs  median {statistics.median(mins) if mins else float('nan'):6.1f} min"
              f"  GPU busy {statistics.median(busy) if busy else float('nan'):.0%}")
    idle = [r for r in runs if r.get("gpu_idle_warning")]
    for r in idle:
        print(f"  IDLE WARNING {r['protein']} {r['depth']} {r['backend']} on {r.get('gpu')} "
              f"({r.get('host')}): busy {r.get('gpu_busy_frac', 0):.0%}, {r['fold_s'] / 60:.0f} min")
    print(f"  {len(idle)} job(s) with gpu_idle_warning")
    return {f"{r['protein']}/{r['depth']}_{r['backend']}" for r in runs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--logs", type=Path, default=Path("logs/run_abcfold"))
    ap.add_argument("--running", action="store_true")
    ap.add_argument("--fresh-min", type=float, default=15)
    a = ap.parse_args()
    done = finished(a.results) if not a.running else {
        f"{f.parents[3].name}/{f.parents[1].name}_{f.parent.name}" for f in a.results.glob("*/predictions/*/*/run.json")}
    running(a.logs, done, a.fresh_min * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
