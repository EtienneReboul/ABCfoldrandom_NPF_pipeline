"""
Step 5: run_stats: GPU time and storage of every run_abcfold job (its run.json),
as one table and a figure: fold minutes per job vs protein length by backend,
the stored MB per model by backend next to ABCfold's raw output size, and the
jobs per GPU type (median fold minutes, GPU busy fraction; jobs flagged
gpu_idle_warning, i.e. probably run on CPU, are counted in red).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import BACKEND_COLORS, BACKENDS, save_fig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    ap.add_argument("--inputs", type=Path, default=Path("data/inputs"), help="<protein>/sequence.fasta (length axis)")
    a = ap.parse_args()

    rows = []
    for f in a.runs:
        r = json.loads(f.read_text())
        r.pop("bytes", None)
        r.pop("pae_modes", None)
        rows.append(r)
    df = pd.DataFrame(rows)
    lengths = {}
    for prot in df["protein"].unique():
        fa = a.inputs / prot / "sequence.fasta"
        lengths[prot] = (sum(len(x.strip()) for x in fa.read_text().splitlines() if not x.startswith(">"))
                         if fa.exists() else float("nan"))
    df["length"] = df["protein"].map(lengths)
    df["fold_min"] = df["fold_s"] / 60
    df["kb_per_model"] = df["bytes_total"] / df["n_models"] / 1e3
    df["raw_kb_per_model"] = df["raw_bytes"] / df["n_models"] / 1e3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 4.8))
    for b in BACKENDS:
        s = df[df["backend"] == b]
        if s.empty:
            continue
        a1.scatter(s["length"], s["fold_min"], s=10, color=BACKEND_COLORS[b], alpha=0.7,
                   label=f"{BACKENDS[b]['label']}: {s['fold_min'].sum() / 60:.0f} GPU-h")
    a1.set_xlabel("protein length (aa)")
    a1.set_ylabel("ABCfold minutes per job (20 seeds x 5 samples)")
    a1.legend(fontsize=8, frameon=False)
    a1.grid(alpha=0.3)
    g = df.groupby("backend")[["kb_per_model", "raw_kb_per_model"]].median().reindex([b for b in BACKENDS if b in set(df["backend"])])
    x = range(len(g))
    a2.bar([i - 0.2 for i in x], g["raw_kb_per_model"], 0.4, color="#bbbbbb", label="ABCfold output (deleted)")
    a2.bar([i + 0.2 for i in x], g["kb_per_model"], 0.4, color=[BACKEND_COLORS[b] for b in g.index], label="model store")
    a2.set_yscale("log")
    a2.set_xticks(list(x), [BACKENDS[b]["label"] for b in g.index])
    a2.set_ylabel("KB per model (median job)")
    a2.legend(fontsize=8, frameon=False)
    a2.grid(axis="y", alpha=0.3)
    gp = df.assign(gpu=df.get("gpu", pd.Series("?", index=df.index)).fillna("?"),
                   idle=df.get("gpu_idle_warning", pd.Series(False, index=df.index)).fillna(False).astype(bool))
    gs = gp.groupby("gpu").agg(n=("fold_min", "size"), minutes=("fold_min", "median"), idle=("idle", "sum"),
                               busy=("gpu_busy_frac", "median") if "gpu_busy_frac" in gp else ("fold_min", "size"))
    gs = gs.sort_values("n")
    a3.barh(range(len(gs)), gs["n"], color="#4c72b0")
    a3.barh(range(len(gs)), gs["idle"], color="#c44e52", label="gpu_idle_warning")
    a3.set_yticks(range(len(gs)), [f"{g} ({m:.0f} min, busy {b:.0%})" if "gpu_busy_frac" in gp else f"{g} ({m:.0f} min)"
                                   for g, m, b in zip(gs.index, gs["minutes"], gs["busy"])], fontsize=7)
    a3.set_xlabel("jobs (label: median fold minutes, GPU busy fraction)")
    a3.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{len(df)} jobs, {df['n_models'].sum()} models: {df['fold_min'].sum() / 60:.0f} GPU-h, "
                 f"store {df['bytes_total'].sum() / 1e9:.1f} GB (ABCfold wrote {df['raw_bytes'].sum() / 1e9:.0f} GB)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, a.fig, a.formats.split(","))
    print(f"[run_stats] {len(df)} jobs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
