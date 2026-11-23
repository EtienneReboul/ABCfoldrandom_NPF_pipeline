"""
Step 6: tm_cavity: does the substrate pocket of each exported model (step 5)
open to the outside, to the inside, or not at all? An independent check of
the OF / Occ / IF call made from the PCA geometry.

Per model (structures/pool/*.pdb.gz):
  1. pyKVFinder cavities with ChimeraX's kvfinder settings (probe in 1.4 A,
     removal distance 2.4 A, step 0.6 A, --volume-cutoff 100 A^3 as
     `kvfinder ... volumeCutoff 100`) EXCEPT the outer probe: --probe-out
     8 A, not 4. With 4 A a wide-open vestibule counts as bulk solvent and
     the pocket vanishes (NPF2.10 outward-open models: "no pocket" or a
     lateral remnant at 4 A, extra-open with 70 % of the CDD residues lining
     it at 6 and 8 A; closed pockets give the same calls at 4, 6 and 8 A).
  2. pocket = the cavity lined by the most CDD "chemical substrate binding
     pocket" residues (data/inputs/<protein>/cdd_sites.json, InterPro API);
     without a CDD site: the cavity whose centroid lies nearest the transport
     axis (pocket_source = "geometry")
  3. its openings to the bulk (pyKVFinder.openings), each placed on the
     membrane normal of this model: z = height of the opening's centre above
     the TM mid-plane, in units of the half-membrane H (the inside -> outside
     axis between the centroids of the TM helix-end thirds on each leaflet).
     z > +side_z: extracellular, z < -side_z: intracellular, otherwise
     lateral (leak towards the lipid). Openings < min_area A^2 are ignored.
  4. call: occluded (no opening), extra-open, intra-open, both-open,
     lateral-only, or no pocket (fewer than min_cdd_frac of the CDD residues
     line any cavity: a collapsed or misfolded site).
     agree: OF <-> extra-open, Occ <-> occluded, IF <-> intra-open.

Run on the raw exported models (structures/ -> cavity.parquet) and on their
OpenMM-minimized copies (step 5b, minimized/ -> cavity_minimized.parquet,
--structures): the openings depend on side-chain placement. Hydrogens are
dropped before detection so both sets see the same heavy atoms; a model
whose minimization failed is called "not minimized".

Outputs (results/<protein>/tm_states/): cavity[_minimized].parquet (per
exported model) and a figure: per state, the opening areas of its exported
models (outside up, inside down) ordered by alpha as in the file names.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyKVFinder  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)   # "OXT not in dictionary" per model

STATES = ["OF", "Occ", "IF"]
STATE_COLORS = {"OF": "#d62728", "Occ": "#f0a30a", "IF": "#2457c5"}
EXPECT = {"OF": "extra-open", "Occ": "occluded", "IF": "intra-open"}
STEP, PROBE_IN, REMOVAL = 0.6, 1.4, 2.4


def label(name: str) -> int:
    """pyKVFinder names KAA / OAA ... -> grid labels 2, 3, ..."""
    return (ord(name[1]) - 65) * 26 + (ord(name[2]) - 65) + 2


def membrane_frame(atomic, topo):
    xyz = atomic[:, 4:7].astype(float)
    ca = {int(a[0]): xyz[i] for i, a in enumerate(atomic) if a[3] == "CA"}

    def end(h, side):
        r = np.arange(h["start"], h["end"] + 1)
        n = max(4, len(r) // 3)
        sel = r[:n] if h["start_side"] == side else r[-n:]
        return np.mean([ca[x] for x in sel if x in ca], 0)

    out_c = np.mean([end(h, "outside") for h in topo["helices"]], 0)
    in_c = np.mean([end(h, "inside") for h in topo["helices"]], 0)
    axis = out_c - in_c
    H = float(np.linalg.norm(axis)) / 2
    return (out_c + in_c) / 2, axis / (2 * H), H


def analyse(path: Path, topo, cdd, a, vdw):
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(gzip.open(path, "rt").read())
        tmp = fh.name
    try:
        atomic = pyKVFinder.read_pdb(tmp, vdw)
    finally:
        os.unlink(tmp)
    atomic = atomic[[not str(n).startswith("H") for n in atomic[:, 3]]]   # heavy atoms only
    ctr, nrm, H = membrane_frame(atomic, topo)
    verts = pyKVFinder.get_vertices(atomic, probe_out=a.probe_out, step=STEP)
    origin = verts[0]
    ncav, cav = pyKVFinder.detect(atomic, verts, step=STEP, probe_in=PROBE_IN, probe_out=a.probe_out,
                                  removal_distance=REMOVAL, volume_cutoff=a.volume_cutoff, nthreads=a.threads)
    row = {"n_cavities": int(ncav)}
    if ncav == 0:
        return {**row, "call": "no pocket"}
    _, vol, area = pyKVFinder.spatial(cav, step=STEP, nthreads=a.threads)
    names = sorted(vol)
    if cdd:
        res = pyKVFinder.constitutional(cav, atomic, verts, step=STEP, probe_in=PROBE_IN, ignore_backbone=False,
                                        nthreads=a.threads)
        lined = {k: len({int(x[0]) for x in res.get(k, [])} & cdd) for k in names}
        best = max(names, key=lambda k: (lined[k], vol[k]))
        row.update(pocket_source="CDD", cdd_lined=int(lined[best]), cdd_frac=lined[best] / len(cdd))
    else:
        def radial(k):
            p = origin + np.argwhere(cav == label(k)) * STEP - ctr
            c = p.mean(0)
            return float(np.linalg.norm(c - (c @ nrm) * nrm))
        best = min(names, key=radial)
        row.update(pocket_source="geometry", cdd_lined=np.nan, cdd_frac=np.nan)
    pts = origin + np.argwhere(cav == label(best)) * STEP
    row.update(pocket=best, volume=float(vol[best]), area=float(area[best]),
               pocket_z=float(((pts - ctr) @ nrm).mean() / H), H=H)
    if cdd and row["cdd_frac"] < a.min_cdd_frac:
        return {**row, "call": "no pocket"}
    depths, _, _ = pyKVFinder.depth(cav, step=STEP, nthreads=a.threads)
    _, opg, oparea = pyKVFinder.openings(cav, depths, step=STEP, selection=[best], nthreads=a.threads)
    sides = {"extra": 0.0, "intra": 0.0, "lateral": 0.0}
    ops = []
    for o, ar in oparea.get(best, {}).items():
        if ar < a.min_area:
            continue
        z = float(((origin + np.argwhere(opg == label(o)) * STEP - ctr) @ nrm).mean() / H)
        side = "extra" if z > a.side_z else "intra" if z < -a.side_z else "lateral"
        sides[side] += float(ar)
        ops.append(f"{side}:{ar:.0f}@{z:+.2f}")
    ex, inn, lat = sides["extra"] > 0, sides["intra"] > 0, sides["lateral"] > 0
    call = ("both-open" if ex and inn else "extra-open" if ex else "intra-open" if inn
            else "lateral-only" if lat else "occluded")
    return {**row, **{f"open_{k}": v for k, v in sides.items()}, "openings": " ".join(ops), "call": call}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True, help="tm_states.parquet (structures/ next to it)")
    ap.add_argument("--topology", type=Path, required=True)
    ap.add_argument("--cdd", type=Path, required=True)
    ap.add_argument("--structures", type=Path, default=None, help="default: structures/ next to --states")
    ap.add_argument("--label", default="raw models", help="figure title: which structures")
    ap.add_argument("--volume-cutoff", type=float, default=100.0)
    ap.add_argument("--probe-out", type=float, default=8.0)
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--side-z", type=float, default=0.5)
    ap.add_argument("--min-cdd-frac", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    topo = json.loads(a.topology.read_text())
    protein = topo["protein"]
    feats = [f for f in json.loads(a.cdd.read_text())["features"] if "pocket" in (f["description"] or "")]
    cdd = set(feats[0]["residues"]) if feats else set()
    st = pd.read_parquet(a.states)
    st = st[st["exported"] != ""].reset_index(drop=True)
    sdir = a.structures or a.states.parent / "structures"
    vdw = pyKVFinder.read_vdw()
    rows = []
    for r in st.itertuples(index=False):
        s, link = r.exported.split("/")
        f = sdir / r.exported
        res = analyse(f, topo, cdd, a, vdw) if f.exists() else {"call": "not minimized"}
        rows.append({"protein": protein, "state": s, "rank": int(link[:2]), "alpha": float(getattr(r, f"alpha_{s}")),
                     "depth": r.depth, "backend": r.backend, "model_id": r.model_id, "file": r.exported, **res})
    out = pd.DataFrame(rows)
    for c in ("open_extra", "open_intra", "open_lateral"):
        out[c] = out.get(c, pd.Series(np.nan, index=out.index)).fillna(0.0)
    out["agree"] = out["call"] == out["state"].map(EXPECT)
    out = out.sort_values(["state", "rank"], key=lambda c: c.map(STATES.index) if c.name == "state" else c)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)

    # ── figure: per state, opening areas of its exported models by rank ──────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    top = max(50.0, float(out[["open_extra", "open_intra", "open_lateral"]].to_numpy().max()) * 1.1)
    for ax, s in zip(axes, STATES):
        d = out[out["state"] == s]
        x = d["rank"].to_numpy()
        ax.bar(x, d["open_extra"], color="#d62728", width=0.8, label="extracellular opening")
        ax.bar(x, -d["open_intra"], color="#2457c5", width=0.8, label="intracellular opening")
        ax.bar(x, d["open_lateral"], bottom=d["open_extra"], color="#9a9a9a", width=0.8, label="lateral opening")
        for xi, (c, ok) in enumerate(zip(d["call"], d["agree"])):
            mk = "o" if c == "occluded" else "x" if c == "no pocket" else None
            if mk:
                ax.scatter(x[xi], 0, marker=mk, s=40, color="black", zorder=3)
            if not ok:
                ax.axvspan(x[xi] - 0.5, x[xi] + 0.5, color="#ffd6d6", zorder=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-top, top)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in d["alpha"]], rotation=90, fontsize=7)
        ax.set_xlabel(f"alpha_{s} of the exported models (file ranks 01 ->)")
        n_ok = int(d["agree"].sum())
        ax.set_title(f"{s}: {n_ok}/{len(d)} pockets {EXPECT[s]}", fontsize=10, color=STATE_COLORS[s])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("pocket opening area (A^2): outside up, inside down")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    src = "CDD pocket residues" if cdd else "no CDD site: pocket = cavity nearest the transport axis"
    fig.suptitle(f"{protein}: pyKVFinder substrate pocket of the exported {a.label} ({src}; o = occluded, "
                 f"x = no pocket, pink = disagrees with the PCA state)", fontsize=10)
    fig.tight_layout()
    for fmt in a.formats.split(","):
        fig.savefig(a.fig.with_suffix(f".{fmt}"), dpi=150 if fmt == "png" else None)
    plt.close(fig)
    agr = out.groupby("state")["agree"].mean().reindex(STATES)
    print(f"[tm_cavity] {protein}: agreement " + ", ".join(f"{s} {v:.0%}" for s, v in agr.items())
          + f"; calls {out['call'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
