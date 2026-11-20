"""
export_models: write working mmCIF / PDB files from the model store (no CIF is
kept by the pipeline; see scripts/abcstore.py and README "Getting structure
files back").

    # every Boltz model of NPF2.10 at depth 8:16, as mmCIF
    python scripts/export_models.py --protein NPF2.10_Q944G5 --depth max8_ext16 \
        --backend boltz --out exported/

    # the 10 best-ranked models of each backend at every depth, gzipped PDB
    python scripts/export_models.py --protein NPF2.10_Q944G5 --top 10 --format pdb.gz --out exported/

    # a pandas query on the score columns (ptm, ranking_score, mean_plddt, seed, sample, ...)
    python scripts/export_models.py --protein NPF2.10_Q944G5 --query "ptm > 0.85 and mean_plddt > 80" --out x/

    # named models (model_id as in models.parquet; needs --depth when ambiguous)
    python scripts/export_models.py --protein NPF2.10_Q944G5 --depth full --model-id boltz_s123_m0 --out x/

Files are named <protein>__<depth>__<model_id>.<format>; manifest.csv lists
them with their scores. Coordinates are those of the backend's CIF (to 0.001 A),
B-factors its per-atom pLDDT, and the written mmCIF carries entities and
label_seq_id (gemmi), so ChimeraX, PyMOL, Foldseek, DSSP, MDAnalysis and
gemmi read it as they would ABCfold's own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from abcstore import Store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", nargs="*", default=None)
    ap.add_argument("--backend", nargs="*", default=None)
    ap.add_argument("--model-id", nargs="*", default=None)
    ap.add_argument("--query", default=None, help="pandas query on models.parquet columns")
    ap.add_argument("--top", type=int, default=None, help="best N per (depth, backend)")
    ap.add_argument("--by", default="ranking_score", help="column for --top (descending)")
    ap.add_argument("--format", default="cif", choices=["cif", "cif.gz", "pdb", "pdb.gz"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    root = a.results / a.protein / "predictions"
    dirs = sorted(d for d in root.glob("*/*") if (d / "models.parquet").exists())
    if a.depth:
        dirs = [d for d in dirs if d.parent.name in a.depth]
    if a.backend:
        dirs = [d for d in dirs if d.name in a.backend]
    if not dirs:
        print(f"no store under {root} matches", file=sys.stderr)
        return 1

    rows = []
    for d in dirs:
        st = Store(d)
        m = st.models
        if a.model_id:
            m = m[m["model_id"].isin(a.model_id)]
        if a.query:
            m = m.query(a.query)
        if a.top:
            m = m.sort_values(a.by, ascending=False).head(a.top)
        for r in m.itertuples(index=False):
            f = a.out / f"{a.protein}__{r.depth}__{r.model_id}.{a.format}"
            st.write(r.model_id, f)
            rows.append({"file": f.name, "protein": a.protein, "depth": r.depth, "backend": r.backend,
                         "model_id": r.model_id, "seed": r.seed, "sample": r.sample, "ptm": r.ptm,
                         "ranking_score": r.ranking_score, "mean_plddt": r.mean_plddt})
    if not rows:
        print("no model matches", file=sys.stderr)
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "manifest.csv", index=False)
    print(f"[export] {len(rows)} models -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
