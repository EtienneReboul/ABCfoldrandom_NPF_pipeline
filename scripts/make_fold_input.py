"""
Step 2a: make_fold_input: one ABCfold input (AlphaFold3 dialect) per
protein x MSA depth.

  full           the whole MSA (each backend applies its own internal cap)
  max<m>_ext<e>  the query + m + e sequences drawn at random, without
                 replacement, from the rest of the MSA (all of them if the MSA
                 is shallower). CF-random's max_seq:max_extra_seq are AF2
                 model inputs (cluster centres : extra rows) with no
                 equivalent in Boltz / Chai-1 / OpenFold3 / Protenix / RF3,
                 so the depth is applied to the MSA itself. The same
                 subsample is fed to every backend.

Every draw (the MSA rows, then the modelSeeds) comes from a generator keyed
on (master_seed, protein, depth): reruns reproduce the inputs exactly. No
templates, no paired MSA (one chain). Written as .json.xz (the full-MSA one is
~10 MB plain); run_abcfold.py unpacks it on the compute node.
"""
from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np

from common import FULL, depth_n_seqs, read_a3m, rng_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3m", type=Path, required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, required=True)
    ap.add_argument("--templates", action="store_true", help="not implemented: inputs carry no templates")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.templates:
        raise SystemExit("templates: true is not supported (no template search is run)")

    recs = read_a3m(a.a3m)
    query, rest = recs[0], recs[1:]
    seq = query[1]
    if any(c.islower() or c == "-" for c in seq):
        raise ValueError(f"{a.a3m}: first record is not an ungapped query")
    rng = rng_for(a.master_seed, a.protein, a.depth)
    n = depth_n_seqs(a.depth)
    if a.depth == FULL or n >= len(rest):
        keep = rest
    else:
        idx = np.sort(rng.choice(len(rest), size=n, replace=False))   # MSA order kept (by E-value)
        keep = [rest[i] for i in idx]
    msa = "".join(f">{h}\n{s}\n" for h, s in [query, *keep])
    seeds = draw_seeds(a.master_seed, a.protein, a.depth, a.n_seeds)

    job = {
        "name": f"{a.protein}__{a.depth}",
        "dialect": "alphafold3",
        "version": 1,
        "modelSeeds": seeds,
        "sequences": [{"protein": {"id": "A", "sequence": seq, "unpairedMsa": msa,
                                   "pairedMsa": "", "templates": []}}],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".part")
    with lzma.open(tmp, "wt", preset=6) as fh:
        json.dump(job, fh)
    tmp.replace(a.out)
    print(f"[fold_input] {a.protein} {a.depth}: {len(keep)}/{len(rest)} MSA rows + query, "
          f"{len(seeds)} seeds -> {a.out}")
    return 0


def draw_seeds(master_seed: int, protein: str, depth: str, n: int) -> list[int]:
    """modelSeeds: n distinct values in [1, 2^31), from their own keyed generator."""
    rng = rng_for(master_seed, protein, depth, "modelSeeds")
    seeds: list[int] = []
    while len(seeds) < n:
        s = int(rng.integers(1, 2**31 - 1))
        if s not in seeds:
            seeds.append(s)
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
