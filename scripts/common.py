"""
common: helpers shared by every step script (stdlib + numpy only at import).

Scripts are run as `python scripts/<step>.py` from the repo root, so
`scripts/` is sys.path[0] and `import common` works in every rule env.
"""
from __future__ import annotations

import gzip
import json
import lzma
import zlib
from pathlib import Path

import numpy as np

# ── backends ────────────────────────────────────────────────────────────────
# ABCfold flag letter, and the substring of the output dir ABCfold names after it
# and the micromamba env ABCfold 1.1.6 runs it in (abcfold/data/config.ini);
# min_cc = lowest CUDA compute capability it runs on (CAIUS 2026-11-18: OpenFold3 /
# Protenix "no kernel image" on P100 sm6.0 (Protenix hangs at 0 % GPU), Chai-1 OOM
# on 16 GB, RF3 fmha kernel fails on RTX 6000 sm7.5; Boltz-2 fine on P100 / RTX 6000)
BACKENDS = {
    "boltz":        {"flag": "b", "dir": "boltz",       "label": "Boltz-2",   "env": "abcfold-boltz-py311", "min_cc": 6.0},
    "chai1":        {"flag": "c", "dir": "chai",        "label": "Chai-1",    "env": "abcfold-chai-py311", "min_cc": 8.0},
    "openfold3":    {"flag": "o", "dir": "openfold",    "label": "OpenFold3", "env": "abcfold-openfold-py311", "min_cc": 8.0},
    "protenix":     {"flag": "p", "dir": "protenix",    "label": "Protenix",  "env": "abcfold-protenix-py311", "min_cc": 8.0},
    "rosettafold3": {"flag": "r", "dir": "rosettafold", "label": "RF3",       "env": "abcfold-rosetta-py312", "min_cc": 8.0},
}
BACKEND_COLORS = {"boltz": "#1b9e77", "chai1": "#d95f02", "openfold3": "#7570b3",
                  "protenix": "#e7298a", "rosettafold3": "#66a61e"}

# ── MSA depths ──────────────────────────────────────────────────────────────
FULL = "full"


def depth_name(max_seq: int, max_extra_seq: int) -> str:
    return f"max{max_seq}_ext{max_extra_seq}"


def depth_names(depths) -> list[str]:
    """[full, max64_ext128, ..., max1_ext2]: deepest first, as in every figure."""
    return [FULL] + [depth_name(m, e) for m, e in sorted(depths, key=lambda d: -d[0])]


def depth_label(name: str) -> str:
    if name == FULL:
        return "full MSA"
    m, e = name[3:].split("_ext")
    return f"{m}:{e}"


def depth_n_seqs(name: str) -> int | None:
    """Sequences drawn besides the query (max_seq + max_extra_seq); None = full MSA."""
    if name == FULL:
        return None
    m, e = name[3:].split("_ext")
    return int(m) + int(e)


def rng_for(master_seed: int, *keys: str) -> np.random.Generator:
    """Generator keyed on (master_seed, keys...): reruns reproduce every draw."""
    return np.random.default_rng([int(master_seed), *[zlib.crc32(k.encode()) for k in keys]])


# ── files ──────────────────────────────────────────────────────────────────
def open_text(path: str | Path, mode: str = "rt"):
    """Text handle on a plain, .gz or .xz file."""
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode)
    if p.endswith(".xz"):
        return lzma.open(p, mode)
    return open(p, mode.replace("t", ""))


def read_a3m(path: str | Path) -> list[tuple[str, str]]:
    """[(header, sequence)] of an a3m (plain/.gz/.xz); the query comes first."""
    recs, head, seq = [], None, []
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if head is not None:
                    recs.append((head, "".join(seq)))
                head, seq = line[1:], []
            else:
                seq.append(line.strip())
    if head is not None:
        recs.append((head, "".join(seq)))
    return recs


# ── topology ───────────────────────────────────────────────────────────────
def load_topology(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def tm_residues(topo: dict, helices=None) -> np.ndarray:
    hs = topo["helices"] if helices is None else [topo["helices"][i - 1] for i in helices]
    return np.concatenate([np.arange(h["start"], h["end"] + 1) for h in hs])


# ── figures ────────────────────────────────────────────────────────────────
def depth_colors(names: list[str]) -> dict[str, tuple]:
    import matplotlib

    cmap = matplotlib.colormaps["viridis"]
    n = max(len(names) - 1, 1)
    return {d: cmap(i / n) for i, d in enumerate(names)}


def save_fig(fig, out: str | Path, formats=("svg",)) -> None:
    """Save `out` (its own suffix) plus any extra formats next to it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    for fmt in formats:
        if f".{fmt}" != out.suffix:
            fig.savefig(out.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=200)
