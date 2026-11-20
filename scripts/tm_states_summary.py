"""
Step 5b: tm_states_summary: family overview of the extremity-based states
(step 5): one PC1-PC2 panel per protein (colour = alpha mix of the corners,
title = verdict) and a CSV with each protein's verdict, QC values and
state sizes.
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
from tm_states import STATE_COLORS, STATES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, nargs="+", required=True, help="tm_states.parquet (json next to it)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    n = len(a.tables)
    nc = min(8, n)
    nr = -(-n // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.6, nr * 2.8), squeeze=False)
    cols = np.array([matplotlib.colors.to_rgb(STATE_COLORS[s]) for s in STATES])
    rows = []
    for ax, t in zip(axes.ravel(), a.tables):
        js = json.loads((t.parent / "tm_states.json").read_text())
        d = pd.read_parquet(t)
        d = d[d["fitted"]]
        rgb = np.clip(d[[f"alpha_{s}" for s in STATES]].to_numpy() @ cols, 0, 1)
        ax.scatter(d["PC1"], d["PC2"], s=2, c=rgb, lw=0, alpha=0.6, rasterized=True)
        T = np.array([[js["corners"][s]["PC1"], js["corners"][s]["PC2"]] for s in STATES])
        ax.plot(*np.vstack([T, T[:1]]).T, color="black", lw=0.5, alpha=0.4)
        for s, (x, y) in zip(STATES, T):
            ax.text(x, y, s, fontsize=7, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=STATE_COLORS[s], lw=1))
        lo, hi = np.percentile(d[["PC1", "PC2"]], [0.5, 99.5], axis=0)
        pad = 0.12 * (hi - lo) + 1
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ok = js["verdict"] == "resolved"
        ax.set_title(f"{js['protein'].split('_')[0]}{'' if ok else '  (unresolved)'}", fontsize=8,
                     color="black" if ok else "#b00000")
        ax.set_xticks([])
        ax.set_yticks([])
        if not ok:
            for sp in ax.spines.values():
                sp.set_edgecolor("#b00000")
        rows.append({"protein": js["protein"], "verdict": js["verdict"], "reasons": "; ".join(js["reasons"]),
                     "asym_gap_bundle": round(js["asym_gap_bundle"], 2),
                     "asym_gap_gates": round(js["asym_gap_gates"], 2), "betweenness": round(js["betweenness"], 3),
                     "flatness": round(js["flatness"], 3), "occ_tight": js["occ_tight"],
                     **{f"n_{s}": js["sizes"][s] for s in STATES}})
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    tab = pd.DataFrame(rows)
    nres = int((tab["verdict"] == "resolved").sum())
    fig.suptitle(f"OF / Occ / IF from the TM-Ca PCA extremities: {nres}/{n} proteins resolved "
                 "(colour = alpha mix: OF red, Occ amber, IF blue)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (nr * 2.8)))
    save_fig(fig, a.fig, a.formats.split(","))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    print(f"[tm_states_summary] {nres}/{n} resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
