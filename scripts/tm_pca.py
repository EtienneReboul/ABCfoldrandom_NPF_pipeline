"""
Step 4: tm_pca: PCA of the TM-helix Ca of every model of one protein (all
depths x backends), the recipe of CFrandom_NPF_pipeline step 8 (itself the
ABCfold "pca_k3" macro-state recipe):

  1. Ca of the 12 DeepTMHMM TM helices, read from the model store
  2. fit set: models with TM-helix pLDDT >= tm_pca.min_tm_plddt (70); shallow
     MSAs give misfolds that would otherwise own PC1-PC2
  3. iterative Procrustes onto the converged mean (<= 5 iterations)
  4. PCA (5 components), GMM k = tm_pca.n_states on PC1-PC2 (PC1-only BIC
     fallback when PC1 > 95 %); states numbered by increasing PC1 mean
  5. the other models are Kabsch-fitted onto the mean and projected
     (pca_state = -1)

Outputs (results/<protein>/tm_pca/): tm_pca.parquet (PC1..PC5, state, fitted,
rmsd_tm per model), model.npz (mean, components: to project other ensembles),
tm_pca.json, structures/ (the model nearest each state mean, superposed on
the mean frame, .cif.gz + ChimeraX view.cxc), plus a 2x2 figure (states,
depth, backend, TM pLDDT) and an interactive plotly scatter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from abcstore import Store  # noqa: E402
from common import BACKEND_COLORS, BACKENDS, depth_colors, depth_label, load_topology, save_fig, tm_residues  # noqa: E402

N_PC = 5
FALLBACK_EVR1 = 0.95
STATE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def kabsch(P, Q):
    pm, qm = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qm - R @ pm


def align_ensemble(frames, n_iter=5, tol=1e-4):
    ref = frames[0].copy()
    for _ in range(n_iter):
        aligned = np.stack([(R @ f.T).T + t for f, (R, t) in zip(frames, [kabsch(f, ref) for f in frames])])
        new = aligned.mean(0)
        shift = float(np.sqrt(((new - ref) ** 2).sum(1).mean()))
        ref = new
        if shift < tol:
            break
    return ref


def fit_states(x2, evr, k, seed=42):
    if evr[0] > FALLBACK_EVR1:
        x = x2[:, :1]
        fits = {kk: GaussianMixture(kk, covariance_type="full", n_init=20, random_state=seed).fit(x)
                for kk in range(1, min(6, len(x) - 1) + 1)}
        k_used = min(fits, key=lambda kk: fits[kk].bic(x))
        return fits[k_used], 1, k_used, True
    k_used = min(k, len(x2) - 1)
    return GaussianMixture(k_used, covariance_type="full", n_init=20, random_state=seed).fit(x2[:, :2]), 2, k_used, False


def ellipse(ax, mean, cov, color, n_std=1.5):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * n_std * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(mean, w, h, angle=ang, facecolor=color, alpha=0.12, edgecolor=color, lw=1.2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--topology", type=Path, required=True)
    ap.add_argument("--depths", nargs="+", required=True)
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--min-tm-plddt", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--formats", default="svg")
    a = ap.parse_args()

    topo = load_topology(a.topology)
    protein = topo["protein"]
    tm = tm_residues(topo)
    df = pd.read_parquet(a.models).reset_index(drop=True)
    stores = {s: Store(s) for s in df["store"].unique()}
    frames = []
    for r in df.itertuples(index=False):
        res, xyz = stores[r.store].ca(r.model_id, tm)
        if len(res) != len(tm):
            raise ValueError(f"{r.store} {r.model_id}: {len(res)}/{len(tm)} TM Ca found")
        frames.append(xyz)
    frames = np.stack(frames)

    n_min = max(10, 3 * a.n_states)
    fitted = (df["tm_plddt"] >= a.min_tm_plddt).to_numpy()
    if fitted.sum() < n_min:
        fitted = df["tm_plddt"].rank(ascending=False, method="first").le(n_min).to_numpy()
        print(f"[tm_pca] WARNING: < {n_min} models with TM pLDDT >= {a.min_tm_plddt}; fitting the {n_min} best")
    mean = align_ensemble(frames[fitted])
    transforms = [kabsch(f, mean) for f in frames]
    aligned = np.stack([(R @ f.T).T + t for f, (R, t) in zip(frames, transforms)])
    X = aligned.reshape(len(aligned), -1)
    pca = PCA(n_components=min(N_PC, int(fitted.sum()), X.shape[1])).fit(X[fitted])
    coords = pca.transform(X)
    evr = pca.explained_variance_ratio_
    gmm, nd, k_used, fallback = fit_states(coords[fitted], evr, a.n_states, a.seed)
    order = np.argsort(gmm.means_[:, 0])
    relabel = {int(o): i for i, o in enumerate(order)}
    state = np.array([relabel[int(s)] for s in gmm.predict(coords[:, :nd])])
    state[~fitted] = -1

    out = df[["protein", "depth", "backend", "seed", "sample", "model_id", "store", "ptm", "ranking_score",
              "mean_plddt", "tm_plddt"]].copy()
    for i in range(coords.shape[1]):
        out[f"PC{i + 1}"] = coords[:, i]
    out["pca_state"], out["fitted"] = state, fitted
    out["rmsd_tm"] = np.sqrt(((aligned - mean) ** 2).sum(2).mean(1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    np.savez_compressed(a.out.parent / "model.npz", mean=mean.astype(np.float32), tm_resids=tm,
                        components=pca.components_.astype(np.float32), pca_mean=pca.mean_.astype(np.float32),
                        explained_variance_ratio=evr, gmm_means=gmm.means_[order], n_states=k_used,
                        fallback_1d=fallback)

    # ── the model nearest each state mean, superposed on the mean TM frame ────
    sdir = a.out.parent / "structures"
    sdir.mkdir(parents=True, exist_ok=True)
    for f in sdir.iterdir():
        f.unlink()
    cxc = [f"# {protein}: TM-Ca PCA states (GMM k={k_used}), model nearest each state mean,",
           "# superposed on the converged mean TM frame. Open with: chimerax view.cxc"]
    reps = []
    for s in range(k_used):
        idx = np.flatnonzero(state == s)
        best = idx[np.argmin(((coords[idx, :nd] - gmm.means_[order[s]]) ** 2).sum(1))]
        r = out.iloc[best]
        R, t = transforms[best]
        st = stores[r["store"]].structure(r["model_id"])
        st[0].transform_pos_and_adp(gemmi.Transform(gemmi.Mat33(R.tolist()), gemmi.Vec3(*t.tolist())))
        name = f"S{s}_{r['depth']}_{r['model_id']}.cif"
        st.make_mmcif_document().write_file(str(sdir / name))
        cxc += [f"open {name}", f"color #{s + 1} {STATE_COLORS[s % len(STATE_COLORS)]}"]
        comp = out.loc[idx]
        reps.append({"state": s, "size": len(idx), "model_id": r["model_id"], "depth": r["depth"], "file": name,
                     "backends": comp["backend"].value_counts().to_dict(),
                     "depths": comp["depth"].value_counts().to_dict()})
    cxc += ["cartoon", "hide atoms", "view", "tile"]
    (sdir / "view.cxc").write_text("\n".join(cxc) + "\n")
    (a.out.parent / "tm_pca.json").write_text(json.dumps(
        {"protein": protein, "n_models": len(out), "n_fitted": int(fitted.sum()), "min_tm_plddt": a.min_tm_plddt,
         "n_tm_ca": len(tm), "explained_variance_ratio": [float(v) for v in evr], "n_states": int(k_used),
         "fallback_1d": bool(fallback), "states": reps}, indent=1))

    # ── static figure ───────────────────────────────────────────────────────
    y = "PC2" if len(evr) > 1 else "PC1"
    xl, yl = f"PC1 ({evr[0]:.1%})", (f"PC2 ({evr[1]:.1%})" if len(evr) > 1 else "PC1")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    (a1, a2), (a3, a4) = axes
    fit, low = out[out["fitted"]], out[~out["fitted"]]
    for ax in axes.ravel():
        ax.scatter(low["PC1"], low[y], s=6, color="#d0d0d0", alpha=0.5, zorder=0, lw=0,
                   label=f"TM pLDDT < {a.min_tm_plddt:g} ({len(low)}, projected)")
    for s in range(k_used):
        m = out["pca_state"] == s
        col = STATE_COLORS[s % len(STATE_COLORS)]
        a1.scatter(out.loc[m, "PC1"], out.loc[m, y], s=7, color=col, alpha=0.7, lw=0, label=f"S{s} ({m.sum()})")
        if not fallback:
            ellipse(a1, gmm.means_[order[s]], gmm.covariances_[order[s]], col)
    rep_ids = {(r["depth"], r["model_id"]) for r in reps}
    rp = out[[(d, m) in rep_ids for d, m in zip(out["depth"], out["model_id"])]]
    a1.scatter(rp["PC1"], rp[y], s=110, facecolor="none", edgecolor="black", lw=1.3, label="nearest to mean")
    a1.set_title(f"GMM states (k={k_used}{', PC1-only fallback' if fallback else ''})", fontsize=10)
    dcol = depth_colors(a.depths)
    for d in a.depths:
        m = fit["depth"] == d
        a2.scatter(fit.loc[m, "PC1"], fit.loc[m, y], s=7, color=dcol[d], alpha=0.7, lw=0,
                   label=f"{depth_label(d)} ({m.sum()})")
    a2.set_title("MSA depth", fontsize=10)
    for b in BACKENDS:
        m = fit["backend"] == b
        if m.any():
            a3.scatter(fit.loc[m, "PC1"], fit.loc[m, y], s=7, color=BACKEND_COLORS[b], alpha=0.7, lw=0,
                       label=f"{BACKENDS[b]['label']} ({m.sum()})")
    a3.set_title("backend", fontsize=10)
    sc = a4.scatter(fit["PC1"], fit[y], s=7, c=fit["tm_plddt"], cmap="viridis", vmin=a.min_tm_plddt, vmax=100, lw=0)
    fig.colorbar(sc, ax=a4, label="TM-helix pLDDT", shrink=0.8)
    a4.set_title("TM-helix pLDDT", fontsize=10)
    px_, py_ = fit["PC1"], fit[y]
    pad_x, pad_y = 0.15 * (px_.max() - px_.min()) + 1, 0.15 * (py_.max() - py_.min()) + 1
    xlim, ylim = (px_.min() - pad_x, px_.max() + pad_x), (py_.min() - pad_y, py_.max() + pad_y)
    for ax in axes.ravel():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if ax is not a4:
            ax.legend(fontsize=7, frameon=False, markerscale=2)
    for ax in (a3, a4):
        ax.set_xlabel(xl)
    for ax in (a1, a3):
        ax.set_ylabel(yl)
    fig.suptitle(f"{protein}: PCA of TM-helix Ca ({len(tm)} atoms; fit on {int(fitted.sum())}/{len(out)} models "
                 f"with TM pLDDT >= {a.min_tm_plddt:g})", fontsize=11)
    fig.tight_layout()
    save_fig(fig, a.fig, a.formats.split(","))
    plt.close(fig)

    import plotly.express as px

    pdf = out.assign(pca_state=out["pca_state"].map(lambda s: "low confidence" if s == -1 else f"S{s}"),
                     depth=out["depth"].map(depth_label), backend=out["backend"].map(lambda b: BACKENDS[b]["label"]))
    figp = px.scatter(pdf, x="PC1", y=y, color="pca_state", symbol="backend",
                      color_discrete_map={**{f"S{s}": STATE_COLORS[s % len(STATE_COLORS)] for s in range(k_used)},
                                          "low confidence": "#c8c8c8"},
                      category_orders={"pca_state": [f"S{s}" for s in range(k_used)] + ["low confidence"]},
                      hover_data={"model_id": True, "depth": True, "tm_plddt": ":.1f", "ptm": ":.2f",
                                  "rmsd_tm": ":.2f", "PC3": ":.2f"},
                      labels={"PC1": xl, y: yl}, template="plotly_white", height=680,
                      title=f"{protein}: TM-Ca PCA (colour = GMM state, symbol = backend)")
    figp.update_traces(marker=dict(size=5))
    figp.update_xaxes(range=list(xlim))
    figp.update_yaxes(range=list(ylim))
    figp.write_html(a.html, include_plotlyjs="cdn", auto_open=False)
    print(f"[tm_pca] {protein}: EVR {', '.join(f'{v:.2f}' for v in evr[:3])}; k={k_used}; "
          f"fitted {int(fitted.sum())}/{len(out)}; sizes {np.bincount(state[fitted]).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
