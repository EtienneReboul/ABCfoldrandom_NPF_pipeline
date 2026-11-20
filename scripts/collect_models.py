"""
Step 3: collect_models: every model of one protein (all depths x backends) in
one table without the binary columns, plus TM-helix pLDDT, and the
confidence-vs-depth figure of the report.

Output results/<protein>/models.parquet: one row per model (protein, depth,
backend, seed, sample, model_id, the backend's scores, mean_plddt, tm_plddt,
n_res, store = the predictions/<depth>/<backend> dir holding its coordinates /
PAE). Per-residue pLDDT stays in the per-store models.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common import BACKENDS, depth_colors, depth_label, load_topology, save_fig, tm_residues  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=Path, nargs="+", required=True, help="predictions/<depth>/<backend>/models.parquet")
    ap.add_argument("--topology", type=Path, required=True)
    ap.add_argument("--depths", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    topo = load_topology(a.topology)
    tm_idx = tm_residues(topo) - 1
    frames = []
    for f in a.stores:
        cols = [c for c in pq.read_schema(f).names if c not in ("coords", "bfactor")]
        df = pd.read_parquet(f, columns=cols)
        pl = np.stack([np.asarray(v, np.float32) for v in df.pop("plddt")])
        if pl.shape[1] != topo["length"]:
            raise ValueError(f"{f}: {pl.shape[1]} residues, topology says {topo['length']}")
        df["tm_plddt"] = pl[:, tm_idx].mean(1)
        df["store"] = str(f.parent)
        frames.append(df)
    models = pd.concat(frames, ignore_index=True)
    models["depth"] = pd.Categorical(models["depth"], categories=a.depths, ordered=True)
    models = models.sort_values(["depth", "backend", "seed", "sample"]).reset_index(drop=True)
    models["depth"] = models["depth"].astype(str)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    models.to_parquet(a.out, index=False)

    backends = [b for b in BACKENDS if b in set(models["backend"])]
    dcol = depth_colors(a.depths)
    metrics = [("tm_plddt", "TM-helix pLDDT"), ("ptm", "pTM")]
    fig, axes = plt.subplots(len(metrics), len(backends), figsize=(2.6 * len(backends) + 1, 6.2),
                             sharey="row", squeeze=False)
    for j, b in enumerate(backends):
        sub = models[models["backend"] == b]
        for i, (col, lab) in enumerate(metrics):
            ax = axes[i, j]
            data = [sub.loc[sub["depth"] == d, col].dropna().to_numpy() for d in a.depths]
            bp = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
            for patch, d in zip(bp["boxes"], a.depths):
                patch.set_facecolor(dcol[d])
                patch.set_alpha(0.85)
            for k, (d, v) in enumerate(zip(a.depths, data)):
                ax.scatter(np.full(len(v), k + 1) + np.random.default_rng(k).uniform(-0.22, 0.22, len(v)),
                           v, s=2, color="black", alpha=0.25, lw=0)
            ax.set_xticks(range(1, len(a.depths) + 1), [depth_label(d) for d in a.depths], rotation=60, fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{BACKENDS[b]['label']} ({len(sub)})", fontsize=10)
    fig.suptitle(f"{topo['protein']}: confidence vs MSA depth ({len(models)} models)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, a.fig, a.formats.split(","))
    print(f"[collect] {topo['protein']}: {len(models)} models from {len(a.stores)} stores -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
