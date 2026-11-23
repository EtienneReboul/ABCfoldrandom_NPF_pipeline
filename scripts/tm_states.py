"""
Step 5: tm_states: outward-open / occluded / inward-open annotation of the
TM-Ca PCA (step 4) by the EXTREMITIES of the point cloud, not its density.

Why not a GMM / HDBSCAN: in PC1-PC2 the ensemble draws a V (tip up or down)
or a lambda (one arm joins the other before its end, so the occluded end
sticks out). The states sit at the shape's extremities, while the density
sits along the arms, often very unevenly: density-based clustering splits
dense arms and drops the sparse tip.

  1. trim: a fitted model is kept when >= mass_frac of the models (min 15)
     lie within radius_frac x the cloud diameter (a radius that scales with
     the cloud's size, not its density, so a sparse arm survives and only
     small isolated blobs go)
  2. corners: the largest-area triangle on the convex hull of the kept
     models (V: the two arm ends + the tip; lambda: the two ends + the
     overshoot); each corner's "cap" = its m nearest kept models
  3. occluded = the corner lying on (V) or just off (lambda) the path the
     data draw between the other two: min of d(a,b) + d(a,c) - d(b,c), d
     geodesic on a kNN + MST graph of the kept models (Euclidean fails on a
     sharp V)
  4. OF vs IF: the two other caps compared on (extra - intra) opening, a
     relative sign test. Opening = distance between the N-bundle (TM1-6) and
     C-bundle (TM7-12) Ca centroids of the helix halves on that leaflet: no
     gate residue pair to choose. The cavity-pair gates (TM1-7, 2-11, 4-10,
     5-8, helix-end thirds) are a second opinion.
  5. every fitted model: barycentric weights alpha_OF / alpha_Occ / alpha_IF
     to the cap centres (clipped >= 0, sum 1), state = argmax alpha.

QC (tm_states.json "verdict" = resolved / unresolved, with the reasons):
betweenness ratio (best / second-best excess, small = a clear tip),
flatness of the triangle (2 x area / longest side^2: a line has no tip),
OF-IF asymmetry gap, the two opening definitions agreeing, and "occ_tight"
(the occluded cap no more open than the OF cap outside and the IF cap
inside: chosen from geometry alone, so an independent check).

Outputs (results/<protein>/tm_states/): tm_states.parquet (per model: alpha_*,
state, openings, exported), tm_states.json, structures/ = n_export models
per state spread over the alpha range of the state's members whose alpha
for it is >= alpha_floor (01 = the most extreme; fewer if fewer pass; the
models between two corners, partway along an arm, are left out),
superposed on the mean TM frame of step 4: each written once to pool/ as
.pdb.gz (ChimeraX reads gzipped PDB, not gzipped mmCIF), the per-state folders OF/ Occ/ IF/ hold relative symlinks, with
ChimeraX scripts view.cxc (all) and view_<state>.cxc. Plus a 2-panel figure
and an interactive plotly page.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path

import gemmi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from abcstore import Store  # noqa: E402
from common import BACKENDS, load_topology, save_fig  # noqa: E402

STATES = ["OF", "Occ", "IF"]
STATE_NAMES = {"OF": "outward-open", "Occ": "occluded", "IF": "inward-open"}
STATE_COLORS = {"OF": "#d62728", "Occ": "#f0a30a", "IF": "#2457c5"}
GATE_PAIRS = [(1, 7), (2, 11), (4, 10), (5, 8)]


def kabsch(P, Q):
    pm, qm = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qm - R @ pm


# ── geometry of the cloud ──────────────────────────────────────────────────
def corners(X, mass_frac, radius_frac):
    m = max(15, int(mass_frac * len(X)))
    lo, hi = np.percentile(X, [0.5, 99.5], axis=0)
    R = radius_frac * float(np.linalg.norm(hi - lo))
    tree = cKDTree(X)
    keep = np.array([len(x) for x in tree.query_ball_point(X, R)]) >= m
    if keep.sum() < 3 * m:
        keep[:] = True
    kidx = np.flatnonzero(keep)
    K = X[kidx]
    hv = ConvexHull(K).vertices
    best, tri = -1.0, None
    for i, j, k in itertools.combinations(hv, 3):
        u, v = K[j] - K[i], K[k] - K[i]
        a = abs(u[0] * v[1] - u[1] * v[0])
        if a > best:
            best, tri = a, (i, j, k)
    kt = cKDTree(K)
    caps = [kidx[kt.query(K[v], m)[1]] for v in tri]
    centres = np.array([X[c].mean(0) for c in caps])
    sides = [np.linalg.norm(centres[i] - centres[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    flat = float(abs(np.cross(np.r_[centres[1] - centres[0], 0], np.r_[centres[2] - centres[0], 0])[2])
                 / max(sides) ** 2)
    return keep, caps, centres, m, R, flat


def geodesic(K, centres, k=10, nmax=2000, seed=0):
    rng = np.random.default_rng(seed)
    G = K if len(K) <= nmax else K[rng.choice(len(K), nmax, replace=False)]
    D = squareform(pdist(G))
    knn = np.argsort(D, 1)[:, 1:k + 1]
    rows = np.repeat(np.arange(len(G)), knn.shape[1])
    W = np.zeros_like(D)
    W[rows, knn.ravel()] = D[rows, knn.ravel()]
    W = np.maximum(W, minimum_spanning_tree(csr_matrix(D)).toarray())   # always connected
    W = np.maximum(W, W.T)
    src = [int(np.argmin(((G - c) ** 2).sum(1))) for c in centres]
    return dijkstra(csr_matrix(W), indices=src)[:, src]


def barycentric(P, T):
    A = np.vstack([T.T, np.ones(3)])
    lam = np.linalg.solve(A, np.vstack([P.T, np.ones(len(P))])).T
    lam = np.clip(lam, 0, None)
    return lam / lam.sum(1, keepdims=True)


# ── openings ───────────────────────────────────────────────────────────────
def leaflet_sets(topo):
    """Residue numbers per (helix, side): half (bundle) and third (gates) of each helix."""
    half, third = {}, {}
    for h in topo["helices"]:
        r = np.arange(h["start"], h["end"] + 1)
        for side in ("outside", "inside"):
            first = h["start_side"] == side
            nh, nt = max(3, len(r) // 2), max(4, len(r) // 3)
            half[h["index"], side] = r[:nh] if first else r[-nh:]
            third[h["index"], side] = r[:nt] if first else r[-nt:]
    return half, third


def openings(xyz, pos, half, third):
    def cen(res):
        return xyz[[pos[x] for x in res]].mean(0)

    o = {}
    for side, key in (("outside", "extra"), ("inside", "intra")):
        nb = cen(np.concatenate([half[h, side] for h in range(1, 7)]))
        cb = cen(np.concatenate([half[h, side] for h in range(7, 13)]))
        o[f"{key}_bundle"] = float(np.linalg.norm(nb - cb))
        o[f"{key}_gates"] = float(np.mean([np.linalg.norm(cen(third[i, side]) - cen(third[j, side]))
                                           for i, j in GATE_PAIRS]))
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=Path, required=True, help="tm_pca.parquet (model.npz next to it)")
    ap.add_argument("--topology", type=Path, required=True)
    ap.add_argument("--mass-frac", type=float, default=0.005)
    ap.add_argument("--radius-frac", type=float, default=0.05)
    ap.add_argument("--n-export", type=int, default=20)
    ap.add_argument("--alpha-floor", type=float, default=0.8,
                    help="export only members whose alpha for their state is >= this")
    ap.add_argument("--cdd", type=Path, default=None, help="cdd_sites.json: pocket residues shown in the .cxc")
    ap.add_argument("--max-betweenness", type=float, default=0.5)
    ap.add_argument("--min-flatness", type=float, default=0.1)
    ap.add_argument("--min-gap", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    topo = load_topology(a.topology)
    protein = topo["protein"]
    if len(topo["helices"]) < 12:
        raise ValueError(f"{protein}: {len(topo['helices'])} TM helices, bundles need 12")
    npz = np.load(a.pca.parent / "model.npz")
    mean, tm = npz["mean"].astype(float), npz["tm_resids"]
    df = pd.read_parquet(a.pca).reset_index(drop=True)
    fit = df["fitted"].to_numpy()
    X = df.loc[fit, ["PC1", "PC2"]].to_numpy()
    fidx = np.flatnonzero(fit)

    # ── corners, occluded, alpha ──────────────────────────────────────────
    keep, caps, centres, m, R, flat = corners(X, a.mass_frac, a.radius_frac)
    dd = geodesic(X[keep], centres)
    excess = [float(dd[i, j] + dd[i, k] - dd[j, k]) for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1))]
    occ = int(np.argmin(excess))
    s_exc = sorted(excess)
    betw = float(s_exc[0] / s_exc[1]) if s_exc[1] > 0 else 1.0

    # ── openings of every model ───────────────────────────────────────────
    half, third = leaflet_sets(topo)
    stores, sel = {}, {}
    rows = []
    for r in df.itertuples(index=False):
        if r.store not in stores:
            st = stores[r.store] = Store(r.store)
            at = st.atoms
            i = np.flatnonzero(((at["atom_name"] == "CA") & ~at["het"]).to_numpy())
            sel[r.store] = (i, {int(x): k for k, x in enumerate(at["res_seq"].to_numpy()[i])})
        i, pos = sel[r.store]
        rows.append(openings(stores[r.store].coords(r.model_id)[i], pos, half, third))
    op = pd.DataFrame(rows)

    others = [i for i in range(3) if i != occ]
    cap_op = [op.iloc[fidx[c]].median().to_dict() for c in caps]
    asym = {d: [c[f"extra_{d}"] - c[f"intra_{d}"] for c in cap_op] for d in ("bundle", "gates")}
    of = max(others, key=lambda i: asym["bundle"][i])
    if_ = [i for i in others if i != of][0]
    of_gates = max(others, key=lambda i: asym["gates"][i])
    lab = {of: "OF", occ: "Occ", if_: "IF"}
    gap = float(asym["bundle"][of] - asym["bundle"][if_])
    occ_tight = bool(cap_op[occ]["extra_bundle"] <= cap_op[of]["extra_bundle"]
                     and cap_op[occ]["intra_bundle"] <= cap_op[if_]["intra_bundle"])
    reasons = []
    if betw > a.max_betweenness:
        reasons.append(f"no clear tip (betweenness {betw:.2f} > {a.max_betweenness:g})")
    if flat < a.min_flatness:
        reasons.append(f"near-linear cloud (flatness {flat:.2f} < {a.min_flatness:g})")
    if gap < a.min_gap:
        reasons.append(f"OF/IF ends barely differ ({gap:.1f} < {a.min_gap:g} A)")
    if of_gates != of:
        reasons.append("bundle and gate openings disagree on OF/IF")
    if not occ_tight:
        reasons.append("occluded corner not the tightest on both sides")
    verdict = "resolved" if not reasons else "unresolved"

    alpha = np.full((len(df), 3), np.nan)
    alpha[fidx] = barycentric(X, centres)
    order = [next(i for i in range(3) if lab[i] == s) for s in STATES]   # corner index of OF, Occ, IF
    out = df[["protein", "depth", "backend", "seed", "sample", "model_id", "store", "tm_plddt", "fitted",
              "PC1", "PC2"]].copy()
    for s, i in zip(STATES, order):
        out[f"alpha_{s}"] = alpha[:, i]
    state = np.array(["low confidence"] * len(df), dtype=object)
    state[fidx] = [STATES[j] for j in np.argmax(alpha[fidx][:, order], 1)]
    out["state"] = state
    out["cap"] = ""
    for s, i in zip(STATES, order):
        out.loc[fidx[caps[i]], "cap"] = s
    out = pd.concat([out, op], axis=1)

    # ── exported structures: n per state over its alpha range >= floor ────
    sdir = a.out.parent / "structures"
    pool = sdir / "pool"
    for d in [pool, *(sdir / s for s in STATES)]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    for f in sdir.glob("*.cxc"):
        f.unlink()
    out["exported"] = ""
    show = []
    if a.cdd is not None and a.cdd.exists():
        for f in json.loads(a.cdd.read_text())["features"]:
            if "pocket" in (f["description"] or ""):
                show = [f"# {f['cdd']} {f['name']}: {f['description']}",
                        f"show :{','.join(str(x) for x in f['residues'])} atoms", "style stick",
                        "color byhetero"]
    show.append("# pocket cavities as in ChimeraX's KVFinder tool: kvfinder #1-20 volumeCutoff 100")
    tm_set = set(int(x) for x in tm)
    cxc_all = [f"# {protein}: {a.n_export} models per state, alpha >= {a.alpha_floor} (01 = most extreme),"
               " superposed on the mean TM frame.",
               f"# verdict: {verdict}" + (f" ({'; '.join(reasons)})" if reasons else ""), "close"]
    n_open = 0
    export = {}
    for s in STATES:
        mem = np.flatnonzero(out["state"].to_numpy() == s)
        n_mem = len(mem)
        mem = mem[out[f"alpha_{s}"].to_numpy()[mem] >= a.alpha_floor]
        mem = mem[np.argsort(-out[f"alpha_{s}"].to_numpy()[mem])]
        export[s] = {"members": int(n_mem), "above_floor": int(len(mem))}
        pick = mem[np.unique(np.linspace(0, len(mem) - 1, min(a.n_export, len(mem))).round().astype(int))]
        cxc = [f"# {protein} {STATE_NAMES[s]}: {len(pick)}/{len(mem)} models over the alpha_{s} range"
               f" >= {a.alpha_floor} ({n_mem} members)", "close"]
        first = n_open + 1
        for rank, k in enumerate(pick, 1):
            r = out.iloc[k]
            st = stores[r["store"]]
            res, ca = st.ca(r["model_id"])
            mtm = np.isin(res, list(tm_set))
            Rm, t = kabsch(ca[mtm], mean)
            g = st.structure(r["model_id"])
            g[0].transform_pos_and_adp(gemmi.Transform(gemmi.Mat33(Rm.tolist()), gemmi.Vec3(*t.tolist())))
            base = f"{r['depth']}__{r['model_id']}.pdb.gz"
            with gzip.open(pool / base, "wt", compresslevel=6) as fh:
                fh.write(g.make_pdb_string())
            link = f"{rank:02d}_a{r[f'alpha_{s}']:.2f}__{base}"
            os.symlink(Path("..") / "pool" / base, sdir / s / link)
            out.loc[k, "exported"] = f"{s}/{link}"
            cxc.append(f"open {link} name {s}_{rank:02d}")
            cxc_all.append(f"open {s}/{link} name {s}_{rank:02d}")
        n_open += len(pick)
        export[s]["exported"] = len(pick)
        if len(pick) < a.n_export:
            print(f"WARNING {protein} {s}: only {len(pick)} members with alpha >= {a.alpha_floor}")
        col = STATE_COLORS[s]
        cxc += [f"color #1-{len(pick)} {col}", "cartoon", "hide atoms", *show, "view"]
        (sdir / s / f"view_{s}.cxc").write_text("\n".join(cxc) + "\n")
        cxc_all.append(f"color #{first}-{n_open} {col}")
    cxc_all += ["cartoon", "hide atoms", *show, "view"]
    (sdir / "view.cxc").write_text("\n".join(cxc_all) + "\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    summary = {
        "protein": protein, "verdict": verdict, "reasons": reasons,
        "n_fitted": int(fit.sum()), "n_trimmed": int((~keep).sum()), "cap_size": int(m), "radius": R,
        "betweenness": betw, "flatness": flat, "asym_gap_bundle": gap,
        "asym_gap_gates": float(asym["gates"][of] - asym["gates"][if_]), "occ_tight": occ_tight,
        "corners": {lab[i]: {"PC1": float(centres[i][0]), "PC2": float(centres[i][1]), "excess": excess[i],
                             **{k: round(v, 2) for k, v in cap_op[i].items()}} for i in range(3)},
        "sizes": {s: int((out["state"] == s).sum()) for s in STATES},
        "alpha_floor": a.alpha_floor, "export": export,
    }
    (a.out.parent / "tm_states.json").write_text(json.dumps(summary, indent=1))

    # ── static figure ─────────────────────────────────────────────────────
    cols = np.array([matplotlib.colors.to_rgb(STATE_COLORS[s]) for s in STATES])
    rgb = np.clip(np.nan_to_num(out[[f"alpha_{s}" for s in STATES]].to_numpy()) @ cols, 0, 1)
    ex = out["exported"] != ""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    low = ~out["fitted"]
    a1.scatter(out.loc[low, "PC1"], out.loc[low, "PC2"], s=4, color="#d0d0d0", lw=0, alpha=0.5, zorder=0)
    trim = np.zeros(len(out), bool)
    trim[fidx[~keep]] = True
    a1.scatter(out.loc[trim, "PC1"], out.loc[trim, "PC2"], s=5, color="#9a9a9a", lw=0, alpha=0.6, zorder=1,
               label=f"trimmed ({trim.sum()}: < {m} models within {R:.1f})")
    kk = fit & ~trim
    a1.scatter(out.loc[kk, "PC1"], out.loc[kk, "PC2"], s=6, c=rgb[kk], lw=0, alpha=0.7, zorder=2)
    T = np.array([centres[i] for i in order])
    a1.plot(*np.vstack([T, T[:1]]).T, color="black", lw=0.6, alpha=0.4)
    for s, (x, y) in zip(STATES, T):
        a1.text(x, y, s, fontsize=10, weight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STATE_COLORS[s], lw=1.5))
    a1.scatter(out.loc[ex, "PC1"], out.loc[ex, "PC2"], s=40, facecolor="none", edgecolor="black", lw=0.8,
               zorder=4, label=f"exported ({ex.sum()})")
    npz_evr = npz["explained_variance_ratio"]
    a1.set_xlabel(f"PC1 ({npz_evr[0]:.1%})")
    a1.set_ylabel(f"PC2 ({npz_evr[1]:.1%})" if len(npz_evr) > 1 else "PC2")
    a1.set_title("colour = alpha mix of the corners (OF red, Occ amber, IF blue)", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="best")
    kp = out.loc[kk, ["PC1", "PC2"]]
    pad = 0.12 * (kp.max() - kp.min()) + 1
    a1.set_xlim(kp["PC1"].min() - pad["PC1"], kp["PC1"].max() + pad["PC1"])
    a1.set_ylim(kp["PC2"].min() - pad["PC2"], kp["PC2"].max() + pad["PC2"])
    for s in STATES:
        mm = out["state"] == s
        a2.scatter(out.loc[mm, "extra_bundle"], out.loc[mm, "intra_bundle"], s=5, color=STATE_COLORS[s], lw=0,
                   alpha=0.5, label=f"{STATE_NAMES[s]} ({mm.sum()})")
    a2.scatter(out.loc[ex, "extra_bundle"], out.loc[ex, "intra_bundle"], s=40, facecolor="none",
               edgecolor="black", lw=0.8)
    lim = [min(out.loc[fit, "extra_bundle"].min(), out.loc[fit, "intra_bundle"].min()) - 0.5,
           max(out.loc[fit, "extra_bundle"].max(), out.loc[fit, "intra_bundle"].max()) + 0.5]
    a2.plot(lim, lim, color="black", lw=0.6, alpha=0.4)
    a2.set_xlim(*lim)
    a2.set_ylim(*lim)
    a2.set_xlabel("extracellular opening: N-/C-bundle distance (A)")
    a2.set_ylabel("intracellular opening: N-/C-bundle distance (A)")
    a2.set_title("check: bundle openings of the fitted models, by state", fontsize=10)
    a2.legend(fontsize=8, frameon=False, markerscale=2)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.suptitle(f"{protein}: states from the PCA extremities: {verdict.upper()}"
                 + (f"\n{'; '.join(reasons)}" if reasons else
                    f"  (OF-IF gap {gap:.1f} A, betweenness {betw:.2f}, flatness {flat:.2f})"),
                 fontsize=11, color="black" if verdict == "resolved" else "#b00000")
    fig.tight_layout()
    save_fig(fig, a.fig, a.formats.split(","))
    plt.close(fig)

    # ── interactive ───────────────────────────────────────────────────────
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figp = make_subplots(rows=1, cols=2, subplot_titles=("TM-Ca PCA", "bundle openings (A)"))
    hov = out.assign(backend=out["backend"].map(lambda b: BACKENDS[b]["label"]))
    cd = hov[["model_id", "depth", "backend", "alpha_OF", "alpha_Occ", "alpha_IF", "tm_plddt", "exported"]]
    tmpl = ("%{customdata[0]} | %{customdata[1]} | %{customdata[2]}<br>alpha OF %{customdata[3]:.2f}, "
            "Occ %{customdata[4]:.2f}, IF %{customdata[5]:.2f}<br>TM pLDDT %{customdata[6]:.1f}<br>"
            "%{customdata[7]}<extra></extra>")
    for s in [*STATES, "low confidence"]:
        mm = (out["state"] == s).to_numpy()
        col = STATE_COLORS.get(s, "#c8c8c8")
        name = STATE_NAMES.get(s, s)
        for c, (x, y) in enumerate((("PC1", "PC2"), ("extra_bundle", "intra_bundle")), 1):
            figp.add_trace(go.Scattergl(x=out.loc[mm, x], y=out.loc[mm, y], mode="markers", name=name,
                                        legendgroup=s, showlegend=c == 1, customdata=cd[mm].to_numpy(),
                                        hovertemplate=tmpl, marker=dict(size=4, color=col, opacity=0.6)),
                           row=1, col=c)
    for c, (x, y) in enumerate((("PC1", "PC2"), ("extra_bundle", "intra_bundle")), 1):
        figp.add_trace(go.Scattergl(x=out.loc[ex, x], y=out.loc[ex, y], mode="markers", name="exported",
                                    legendgroup="exp", showlegend=c == 1, customdata=cd[ex].to_numpy(),
                                    hovertemplate=tmpl,
                                    marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=1.2, color="black"))),
                       row=1, col=c)
    figp.update_layout(template="plotly_white", height=620,
                       title=f"{protein}: OF / Occ / IF from the PCA extremities ({verdict})")
    figp.write_html(a.html, include_plotlyjs="cdn", auto_open=False)
    print(f"[tm_states] {protein}: {verdict} {reasons}; sizes {summary['sizes']}; gap {gap:.1f}; "
          f"betweenness {betw:.2f}; flatness {flat:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
