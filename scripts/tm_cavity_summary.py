"""
Step 6b: tm_cavity_summary: how often the pyKVFinder pocket (step 6) agrees
with the PCA-extremity state (step 5), over every exported model of the
family: confusion matrix (state x pocket call, resolved proteins), agreement
as a function of alpha, and per protein / state agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import save_fig  # noqa: E402

STATES = ["OF", "Occ", "IF"]
CALLS = ["extra-open", "occluded", "intra-open", "both-open", "lateral-only", "no pocket"]
STATE_COLORS = {"OF": "#d62728", "Occ": "#f0a30a", "IF": "#2457c5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, nargs="+", required=True, help="cavity.parquet (tm_states.json next to it)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    parts = []
    for t in a.tables:
        js = json.loads((t.parent / "tm_states.json").read_text())
        parts.append(pd.read_parquet(t).assign(verdict=js["verdict"]))
    d = pd.concat(parts, ignore_index=True)
    res = d[d["verdict"] == "resolved"]

    per = (d.groupby(["protein", "state"])["agree"].mean().unstack().reindex(columns=STATES)
           .join(d.groupby("protein")["verdict"].first())
           .join(d.groupby("protein")["pocket_source"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else "-")))
    per = per.loc[sorted(per.index, key=lambda p: [int(x) if x.isdigit() else x for x in p.split("_")[0][3:].split(".")])]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    per.round(3).to_csv(a.out)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 2.2], height_ratios=[1, 1])
    # confusion matrix (resolved proteins)
    ax = fig.add_subplot(gs[0, 0])
    cm = pd.crosstab(res["state"], res["call"]).reindex(index=STATES, columns=CALLS, fill_value=0)
    frac = cm.div(cm.sum(1).replace(0, 1), axis=0)
    ax.imshow(frac.to_numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(len(STATES)):
        for j in range(len(CALLS)):
            v = frac.iat[i, j]
            ax.text(j, i, f"{v:.0%}\n({cm.iat[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else "black")
    ax.set_xticks(range(len(CALLS)))
    ax.set_xticklabels(CALLS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(STATES)
    ax.set_xlabel("pyKVFinder pocket call")
    ax.set_ylabel("PCA-extremity state")
    n_prot = res["protein"].nunique()
    ax.set_title(f"resolved proteins ({n_prot}): {res['agree'].mean():.0%} agree", fontsize=10)
    # agreement vs alpha
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0.3, 1.0, 8)
    for s in STATES:
        x = res[res["state"] == s]
        b = pd.cut(x["alpha"], bins, include_lowest=True)
        g = x.groupby(b, observed=True)["agree"].agg(["mean", "size"])
        mid = [iv.mid for iv in g.index]
        ax.plot(mid, g["mean"], "-o", color=STATE_COLORS[s], label=s, ms=4)
    ax.set_xlabel("alpha of the model to its state's corner")
    ax.set_ylabel("fraction agreeing with the pocket call")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("agreement vs alpha (resolved proteins)", fontsize=10)
    # unresolved proteins confusion, for contrast
    ax = fig.add_subplot(gs[1, 0:2])
    un = d[d["verdict"] != "resolved"]
    if len(un):
        cmu = pd.crosstab(un["state"], un["call"]).reindex(index=STATES, columns=CALLS, fill_value=0)
        fu = cmu.div(cmu.sum(1).replace(0, 1), axis=0)
        ax.imshow(fu.to_numpy(), cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for i in range(len(STATES)):
            for j in range(len(CALLS)):
                ax.text(j, i, f"{fu.iat[i, j]:.0%} ({cmu.iat[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if fu.iat[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(CALLS)))
        ax.set_xticklabels(CALLS, fontsize=8)
        ax.set_yticks(range(len(STATES)))
        ax.set_yticklabels(STATES)
        ax.set_title(f"unresolved proteins ({un['protein'].nunique()}): {un['agree'].mean():.0%} agree",
                     fontsize=10)
    else:
        ax.axis("off")
    # per protein
    ax = fig.add_subplot(gs[:, 2])
    m = per[STATES].to_numpy(dtype=float)
    ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(3):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(STATES)
    ax.set_yticks(range(len(per)))
    ax.set_yticklabels([p.split("_")[0] + ("" if v == "resolved" else " *") + (" (geom)" if s == "geometry" else "")
                        for p, v, s in zip(per.index, per["verdict"], per["pocket_source"])], fontsize=6)
    ax.set_title("agreement per protein (* unresolved, geom = no CDD site)", fontsize=10)
    fig.suptitle("pyKVFinder substrate pocket vs PCA-extremity state, all exported models", fontsize=12)
    fig.tight_layout()
    save_fig(fig, a.fig, a.formats.split(","))
    print(f"[tm_cavity_summary] resolved: {res['agree'].mean():.1%} agree over {len(res)} models; "
          f"unresolved: {un['agree'].mean() if len(un) else float('nan'):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
