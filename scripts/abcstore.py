"""
abcstore: the compact model store of this pipeline (Parquet only, no CIF kept).

One ABCfold run of one backend on one protein x MSA depth (20 seeds x 5
samples = 100 models) becomes three Parquet files in
results/<protein>/predictions/<depth>/<backend>/:

  models.parquet  one row per model: ids, the backend's own scores, per-residue
                  pLDDT (float16 list, from the CA B-factors) and two binary
                  columns:
                    coords   every atom's x, y, z as int32 thousandths of an A,
                             delta-coded along the atom order, byte-planes
                             shuffled, zstd. Lossless for Boltz / Chai-1 /
                             Protenix (their CIFs carry 3 decimals); OpenFold3
                             and RF3 write float32 digits, rounded here to
                             0.0005 A at most
                    bfactor  every atom's B-factor (= the backend's per-atom
                             pLDDT; RF3 writes it on a 0-1 scale) as uint16
                             at the row's b_scale steps per unit (500 for
                             0-100 values, 50000 for 0-1 values: at most
                             0.001 / 0.00001 off), zstd
  pae.parquet     one row per seed: the PAE matrices of that seed's samples,
                  quantised to uint8 at `step` A (0.25 A: max error 0.125 A,
                  a quarter of the backends' own 0.5 A PAE bins), coded with
                  the 2-D "plane" predictor (x - left - up + upleft, mod 256;
                  inverse = 2-D cumulative sum) and, when smaller, as
                  differences to the seed's first sample (diffusion samples of
                  one seed share the trunk), zstd. The variant is per seed
                  (`mode`).
  atoms.parquet   the atom table shared by every model of this backend
                  (chain, residue, atom name, element, occupancy): the
                  topology needed to rebuild a structure file.

The binary columns are stored uncompressed by Parquet (they are zstd already).
Every encoding is checked against its source before ABCfold's files are
deleted (run_abcfold.py): coordinates to 0.0005 A, B-factors to 0.5 / b_scale,
PAE to within step / 2.

Reading (the only API downstream code should use):

    from abcstore import Store
    st = Store("results/NPF2.10_Q944G5/predictions/full/boltz")
    st.models                      # DataFrame without the binary columns
    st.coords(model_id)            # (n_atoms, 3) float64, A
    st.ca(model_id, resids=None)   # CA xyz (+ residue numbers)
    st.pae(model_id)               # (n_tok, n_tok) float32, A
    st.write(model_id, "x.cif")    # working mmCIF (or .pdb / .cif.gz / .pdb.gz)

and scripts/export_models.py writes any selection of models as CIF/PDB.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STORE_VERSION = 1
COORD_SCALE = 1000      # int32 thousandths of an A
PAE_MODES = ("plane", "plane+ref", "plane+refplane")
BLOB_COLUMNS = ("coords", "bfactor", "blob")


# ═════ codecs ══════════════════════════════════════════════════════════════════
def _codec(level: int = 19):
    import pyarrow as pa

    return pa.Codec("zstd", compression_level=level)


def _z(raw: bytes, level: int = 19) -> bytes:
    return _codec(level).compress(raw, asbytes=True)


def _unz(blob: bytes, size: int) -> bytes:
    return _codec().decompress(blob, decompressed_size=size, asbytes=True)


def encode_coords(xyz: np.ndarray, level: int = 19) -> bytes:
    q = np.rint(np.asarray(xyz, np.float64) * COORD_SCALE).astype(np.int64)
    d = np.diff(q, axis=0, prepend=0)
    if np.abs(d).max(initial=0) >= 2**31:
        raise ValueError("coordinate step does not fit int32")
    planes = np.ascontiguousarray(d.T.astype("<i4")).view(np.uint8).reshape(-1, 4).T   # byte planes
    return _z(np.ascontiguousarray(planes).tobytes(), level)


def decode_coords(blob: bytes, n_atoms: int) -> np.ndarray:
    raw = np.frombuffer(_unz(blob, n_atoms * 12), np.uint8).reshape(4, 3 * n_atoms)
    d = np.ascontiguousarray(raw.T).view("<i4").reshape(3, n_atoms).T
    return np.cumsum(d.astype(np.int64), axis=0) / COORD_SCALE


def b_scale_for(b: np.ndarray) -> int:
    """uint16 steps per unit: pLDDT on 0-100 (500) or 0-1 (50000, RF3)."""
    return 50000 if float(np.max(b, initial=0)) <= 1.0 else 500


def encode_b(b: np.ndarray, scale: int, level: int = 19) -> bytes:
    q = np.rint(np.asarray(b, np.float64) * scale)
    if q.min(initial=0) < 0 or q.max(initial=0) > 65535:
        raise ValueError("B-factor out of uint16 range")
    return _z(q.astype("<u2").tobytes(), level)


def decode_b(blob: bytes, n_atoms: int, scale: int) -> np.ndarray:
    return np.frombuffer(_unz(blob, n_atoms * 2), "<u2").astype(np.float64) / scale


def _plane(q: np.ndarray) -> np.ndarray:
    x = q.astype(np.int16)
    d = x.copy()
    d[1:, :] -= x[:-1, :]
    d[:, 1:] -= d[:, :-1].copy()
    return (d & 0xFF).astype(np.uint8)


def _plane_inv(r: np.ndarray) -> np.ndarray:
    return (np.cumsum(np.cumsum(r.astype(np.int64), 0), 1) & 0xFF).astype(np.uint8)


def quantise_pae(pae: np.ndarray, step: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(pae, np.float64) / step), 0, 255).astype(np.uint8)


def encode_pae_seed(paes: list[np.ndarray], step: float, level: int = 19) -> tuple[bytes, str]:
    """The PAE matrices of one seed's samples -> (zstd blob, mode), smallest of PAE_MODES."""
    qs = [quantise_pae(p, step) for p in paes]
    ref = qs[0]
    dif = [((q.astype(np.int16) - ref) & 0xFF).astype(np.uint8) for q in qs[1:]]
    cand = {
        "plane": np.stack([_plane(q) for q in qs]),
        "plane+ref": np.stack([_plane(ref), *dif]),
        "plane+refplane": np.stack([_plane(ref), *[_plane(d) for d in dif]]),
    }
    if len(qs) == 1:
        cand = {"plane": cand["plane"]}
    blobs = {m: _z(a.tobytes(), level) for m, a in cand.items()}
    mode = min(blobs, key=lambda m: len(blobs[m]))
    return blobs[mode], mode


def decode_pae_seed(blob: bytes, mode: str, n_samples: int, n_tok: int, step: float) -> np.ndarray:
    """-> (n_samples, n_tok, n_tok) float32, A."""
    r = np.frombuffer(_unz(blob, n_samples * n_tok * n_tok), np.uint8).reshape(n_samples, n_tok, n_tok)
    if mode == "plane":
        q = np.stack([_plane_inv(x) for x in r])
    elif mode in ("plane+ref", "plane+refplane"):
        ref = _plane_inv(r[0])
        rest = [(_plane_inv(x) if mode == "plane+refplane" else x) for x in r[1:]]
        q = np.stack([ref, *[((d.astype(np.int16) + ref) & 0xFF).astype(np.uint8) for d in rest]])
    else:
        raise ValueError(f"unknown PAE mode {mode!r}")
    return q.astype(np.float32) * step


# ═════ ABCfold output layout ═══════════════════════════════════════════════════
# Confirmed on the ABCfold NPF run (ABCfold_NPF_pipeline/scripts/abcfold_backends.py)
# and re-checked on this pipeline's pilot. Each backend writes its own subtree
# under the ABCfold output dir; per (seed, sample) one model CIF (OpenFold3 and
# RF3 also a '_fixed' copy: the one kept), a scores file and a PAE source.
@dataclass
class RawModel:
    backend: str
    seed: int
    sample: int
    cif: Path


DIR_KEY = {"boltz": "boltz", "chai1": "chai", "openfold3": "openfold", "protenix": "protenix",
           "rosettafold3": "rosettafold"}


def _strip_model(stem: str) -> str:
    return re.sub(r"_model(_fixed)?$", "", stem)


def _seed_sample(rel: str, stem: str) -> tuple[int | None, int | None]:
    m = re.search(r"seed[-_]?(\d+)_sample[-_]?(\d+)", rel, re.I)
    if m:
        return int(m[1]), int(m[2])
    s = re.search(r"seed[-_]?(\d+)", rel, re.I)
    k = (re.search(r"model_idx_(\d+)", stem) or re.search(r"sample[-_]?(\d+)", stem)
         or re.search(r"_model_(\d+)$", stem))
    return (int(s[1]) if s else None), (int(k[1]) if k else None)


def discover(raw_dir: Path, backend: str, name: str) -> list[RawModel]:
    """One CIF per (seed, sample) of `backend` under an ABCfold output dir."""
    roots = [d for d in Path(raw_dir).iterdir() if d.is_dir() and DIR_KEY[backend] in d.name.lower()]
    found: dict[tuple[int, int], Path] = {}
    for root in roots:
        for cif in sorted(root.rglob("*.cif")):
            if "templates" in cif.parts or cif.stem == f"{name}_model":   # per-run / per-seed best-model copies
                continue
            seed, sample = _seed_sample(str(cif.relative_to(raw_dir)), cif.stem)
            if seed is None or sample is None:
                raise ValueError(f"cannot parse seed/sample from {cif}")
            key = (seed, sample)
            if key not in found or cif.stem.endswith("_fixed"):
                found[key] = cif
    return [RawModel(backend, s, k, c) for (s, k), c in sorted(found.items())]


def _json(p: Path) -> dict:
    return json.loads(p.read_text())


def summary_scores(m: RawModel) -> dict:
    """The backend's own per-model scores, unified to ptm / iptm / ranking_score /
    fraction_disordered / has_clash / backend_plddt (+ the raw dict as JSON)."""
    c, base = m.cif, _strip_model(m.cif.stem)
    if m.backend == "rosettafold3":
        p = c.with_name(f"{base}_summary_confidences.json")
        raw = _json(p)
        u = dict(ptm=raw.get("ptm"), iptm=raw.get("iptm"), ranking_score=raw.get("ranking_score"),
                 fraction_disordered=raw.get("fraction_disordered"), has_clash=raw.get("has_clash"),
                 backend_plddt=None)
    elif m.backend == "boltz":
        p = c.with_name(f"confidence_{c.stem}.json")
        raw = _json(p)
        pl = raw.get("complex_plddt")
        u = dict(ptm=raw.get("ptm"), iptm=raw.get("iptm"), ranking_score=raw.get("confidence_score"),
                 fraction_disordered=None, has_clash=None, backend_plddt=None if pl is None else 100 * pl)
    elif m.backend == "chai1":
        p = c.with_name(c.name.replace("pred.", "scores.", 1)).with_suffix(".npz")
        with np.load(p) as d:
            raw = {k: d[k].tolist() for k in d.files}
        first = lambda v: (v[0] if isinstance(v, list) and v else v)  # noqa: E731
        u = dict(ptm=first(raw.get("ptm")), iptm=first(raw.get("iptm")),
                 ranking_score=first(raw.get("aggregate_score")), fraction_disordered=None,
                 has_clash=first(raw.get("has_inter_chain_clashes")), backend_plddt=None)
    elif m.backend == "openfold3":
        p = c.with_name(f"{base}_confidences_aggregated.json")
        raw = _json(p)
        u = dict(ptm=raw.get("ptm"), iptm=raw.get("iptm"), ranking_score=raw.get("sample_ranking_score"),
                 fraction_disordered=raw.get("disorder"), has_clash=raw.get("has_clash"),
                 backend_plddt=raw.get("avg_plddt"))
    elif m.backend == "protenix":
        mm = re.match(r"(.+)_sample([-_])(\d+)$", c.stem)
        p = c.with_name(f"{mm[1]}_summary_confidence_sample{mm[2]}{mm[3]}.json")
        raw = _json(p)
        u = dict(ptm=raw.get("ptm"), iptm=raw.get("iptm"), ranking_score=raw.get("ranking_score"),
                 fraction_disordered=raw.get("disorder"), has_clash=raw.get("has_clash"),
                 backend_plddt=raw.get("plddt"))
    else:
        raise ValueError(m.backend)
    for k, v in list(u.items()):
        if isinstance(v, list):
            u[k] = v[0] if v else None
        if u[k] is not None:
            u[k] = float(u[k])
    u["scores_json"] = json.dumps(raw)
    return u


def raw_pae(m: RawModel) -> np.ndarray:
    """The model's full PAE matrix (tokens x tokens), A."""
    c, base = m.cif, _strip_model(m.cif.stem)
    if m.backend in ("rosettafold3", "openfold3"):
        return np.asarray(_json(c.with_name(f"{base}_confidences.json"))["pae"], np.float32)
    if m.backend == "boltz":
        p = c.with_name(f"pae_{c.stem}.npz")
        try:
            with np.load(p) as d:
                return np.asarray(d["pae"], np.float32)
        except (ValueError, OSError):           # ABCfold may rewrite it as AF3-style JSON in place
            return np.asarray(_json(p)["pae"], np.float32)
    if m.backend == "protenix":
        mm = re.match(r"(.+)_sample([-_])(\d+)$", c.stem)
        d = _json(c.with_name(f"{mm[1]}_full_data_sample{mm[2]}{mm[3]}.json"))
        return np.asarray(d["pae"] if "pae" in d else d["token_pair_pae"], np.float32)
    if m.backend == "chai1":
        stack = np.load(c.with_name("pae_scores.npy"))
        return np.asarray(stack[m.sample], np.float32)
    raise ValueError(m.backend)


def read_cif_atoms(cif: Path):
    """(atom table rows, xyz (n,3), B (n,)) in file order, first model only."""
    import gemmi

    st = gemmi.read_structure(str(cif), merge_chain_parts=False)
    rows, xyz, b = [], [], []
    for ch in st[0]:
        for res in ch:
            for at in res:
                rows.append((ch.name, res.seqid.num, res.seqid.icode.strip(), res.name, at.name,
                             at.element.name, res.het_flag == "H", at.occ))
                xyz.append(at.pos.tolist())
                b.append(at.b_iso)
    return rows, np.asarray(xyz, np.float64), np.asarray(b, np.float64)


# ═════ packing (run on the compute node, right after ABCfold) ═════════════════
ATOM_COLS = ["chain", "res_seq", "icode", "res_name", "atom_name", "element", "het", "occupancy"]


def pack(raw_dir: Path, backend: str, name: str, out_dir: Path, *, protein: str, depth: str,
         seeds: list[int], n_samples: int, pae_step: float = 0.25, level: int = 19) -> dict:
    """ABCfold output of one backend -> {models,pae,atoms}.parquet in out_dir, verified."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    models = discover(raw_dir, backend, name)
    expect = {(s, k) for s in seeds for k in range(n_samples)}
    got = {(m.seed, m.sample) for m in models}
    if got != expect:
        miss, extra = sorted(expect - got)[:5], sorted(got - expect)[:5]
        # samples may be numbered from 1
        shifted = {(s, k - 1) for s, k in got}
        if shifted == expect:
            models = [RawModel(m.backend, m.seed, m.sample - 1, m.cif) for m in models]
        else:
            raise RuntimeError(f"{backend}: {len(got)} models found, {len(expect)} expected "
                               f"(missing e.g. {miss}, unexpected e.g. {extra})")

    atoms_ref = None
    rows, by_seed, checks = [], {}, {"coord_max_err": 0.0, "b_rel_err": 0.0, "pae_max_err": 0.0}
    for m in models:
        arows, xyz, b = read_cif_atoms(m.cif)
        if atoms_ref is None:
            atoms_ref = arows
        elif [r[:7] for r in arows] != [r[:7] for r in atoms_ref]:
            raise RuntimeError(f"{m.cif}: atom table differs from the first model of {backend}")
        bs = b_scale_for(b)
        cblob, bblob = encode_coords(xyz, level), encode_b(b, bs, level)
        n = len(xyz)
        checks["coord_max_err"] = max(checks["coord_max_err"], float(np.abs(decode_coords(cblob, n) - xyz).max()))
        # in steps: <= 0.5 when right (float32 B values add a hair)
        checks["b_rel_err"] = max(checks["b_rel_err"], float(np.abs(decode_b(bblob, n, bs) - b).max() * bs))
        ca = [i for i, r in enumerate(arows) if r[4] == "CA" and not r[6]]
        plddt = b[ca] * (100.0 if b.max() <= 1.0 else 1.0)
        pae = raw_pae(m)
        by_seed.setdefault(m.seed, {})[m.sample] = pae
        rows.append({
            "protein": protein, "depth": depth, "backend": backend, "seed": m.seed, "sample": m.sample,
            "model_id": f"{backend}_s{m.seed}_m{m.sample}", **summary_scores(m),
            "mean_plddt": float(plddt.mean()), "n_res": len(ca), "n_tok": int(pae.shape[0]),
            "n_atoms": n, "b_scale": bs, "plddt": plddt.astype(np.float16), "coords": cblob, "bfactor": bblob,
            "source": str(m.cif.relative_to(raw_dir)),
        })
    if checks["coord_max_err"] > 0.5 / COORD_SCALE + 1e-6 or checks["b_rel_err"] > 0.5 + 1e-3:
        raise RuntimeError(f"{backend}: coordinate / B-factor round trip off: {checks}")

    pae_rows = []
    for seed, samples in sorted(by_seed.items()):
        order = sorted(samples)
        mats = [samples[k] for k in order]
        if len({x.shape for x in mats}) != 1:
            raise RuntimeError(f"{backend} seed {seed}: PAE shapes differ")
        blob, mode = encode_pae_seed(mats, pae_step, level)
        back = decode_pae_seed(blob, mode, len(mats), mats[0].shape[0], pae_step)
        err = max(float(np.abs(back[i] - np.clip(x, 0, 255 * pae_step)).max()) for i, x in enumerate(mats))
        checks["pae_max_err"] = max(checks["pae_max_err"], err)
        pae_rows.append({"protein": protein, "depth": depth, "backend": backend, "seed": seed,
                         "samples": order, "n_tok": int(mats[0].shape[0]), "step": pae_step,
                         "mode": mode, "blob": blob})
    if checks["pae_max_err"] > pae_step / 2 + 1e-4:
        raise RuntimeError(f"{backend}: PAE round trip off by {checks['pae_max_err']:.4f} A")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {b"abcstore_version": str(STORE_VERSION).encode(), b"pae_step": str(pae_step).encode(),
            b"coord_scale": str(COORD_SCALE).encode()}

    def write(df, path, binary):
        df = df.copy()
        pl = df.pop("plddt") if "plddt" in df else None
        t = pa.Table.from_pandas(df, preserve_index=False)
        if pl is not None:
            t = t.append_column("plddt", pa.array([np.asarray(v, np.float16) for v in pl], pa.list_(pa.float16())))
        t = t.replace_schema_metadata({**(t.schema.metadata or {}), **meta})
        comp = {c: ("NONE" if c in binary else "ZSTD") for c in t.column_names}
        tmp = path.with_name(path.name + ".part")
        pq.write_table(t, tmp, compression=comp)
        tmp.replace(path)

    atoms = pd.DataFrame(atoms_ref, columns=ATOM_COLS)
    write(atoms, out_dir / "atoms.parquet", ())
    write(pd.DataFrame(pae_rows), out_dir / "pae.parquet", ("blob",))
    write(pd.DataFrame(rows).sort_values(["seed", "sample"]), out_dir / "models.parquet", ("coords", "bfactor"))
    sizes = {f: (out_dir / f).stat().st_size for f in ("models.parquet", "pae.parquet", "atoms.parquet")}
    return {"backend": backend, "n_models": len(rows), "n_atoms": len(atoms_ref),
            "pae_modes": pd.Series([r["mode"] for r in pae_rows]).value_counts().to_dict(),
            **checks, "bytes": sizes, "bytes_total": sum(sizes.values())}


# ═════ reading ═══════════════════════════════════════════════════════════════
class Store:
    """Read access to one predictions/<depth>/<backend>/ directory."""

    def __init__(self, path: str | Path):
        import pandas as pd

        self.path = Path(path)
        full = pd.read_parquet(self.path / "models.parquet")
        self._blobs = full[["model_id", "coords", "bfactor", "n_atoms", "b_scale"]].set_index("model_id")
        self.models = full.drop(columns=["coords", "bfactor"])
        self._pae = None
        self._atoms = None

    @property
    def atoms(self):
        import pandas as pd

        if self._atoms is None:
            self._atoms = pd.read_parquet(self.path / "atoms.parquet")
        return self._atoms

    def coords(self, model_id: str) -> np.ndarray:
        r = self._blobs.loc[model_id]
        return decode_coords(r["coords"], int(r["n_atoms"]))

    def bfactor(self, model_id: str) -> np.ndarray:
        r = self._blobs.loc[model_id]
        return decode_b(r["bfactor"], int(r["n_atoms"]), int(r["b_scale"]))

    def ca(self, model_id: str, resids=None) -> tuple[np.ndarray, np.ndarray]:
        a = self.atoms
        sel = (a["atom_name"] == "CA") & ~a["het"]
        if resids is not None:
            sel &= a["res_seq"].isin(set(int(r) for r in resids))
        idx = np.flatnonzero(sel.to_numpy())
        return a["res_seq"].to_numpy()[idx], self.coords(model_id)[idx]

    def pae(self, model_id: str) -> np.ndarray:
        import pandas as pd

        if self._pae is None:
            self._pae = pd.read_parquet(self.path / "pae.parquet").set_index("seed")
        m = self.models.set_index("model_id").loc[model_id]
        r = self._pae.loc[int(m["seed"])]
        samples = list(r["samples"])
        stack = decode_pae_seed(r["blob"], r["mode"], len(samples), int(r["n_tok"]), float(r["step"]))
        return stack[samples.index(int(m["sample"]))]

    def structure(self, model_id: str):
        """gemmi.Structure of one model (entities set up: writes a working mmCIF / PDB)."""
        import gemmi

        a = self.atoms
        xyz, b = self.coords(model_id), self.bfactor(model_id)
        st = gemmi.Structure()
        st.name = model_id
        model = gemmi.Model("1")
        chain, res, key = None, None, None
        for i, r in enumerate(a.itertuples(index=False)):
            if chain is None or chain.name != r.chain:
                if chain is not None:
                    chain.add_residue(res)
                    model.add_chain(chain)
                chain, res, key = gemmi.Chain(r.chain), None, None
            k = (r.res_seq, r.icode, r.res_name)
            if k != key:
                if res is not None:
                    chain.add_residue(res)
                res = gemmi.Residue()
                res.name = r.res_name
                res.seqid = gemmi.SeqId(int(r.res_seq), r.icode or " ")
                res.het_flag = "H" if r.het else "A"
                key = k
            at = gemmi.Atom()
            at.name = r.atom_name
            at.element = gemmi.Element(r.element)
            at.pos = gemmi.Position(*xyz[i])
            at.b_iso = float(b[i])
            at.occ = float(r.occupancy)
            res.add_atom(at)
        if chain is not None:
            chain.add_residue(res)
            model.add_chain(chain)
        st.add_model(model)
        st.setup_entities()
        st.assign_label_seq_id()
        return st

    def write(self, model_id: str, out: str | Path) -> Path:
        """mmCIF (.cif[.gz]) or PDB (.pdb[.gz]) of one model."""
        import gzip

        out = Path(out)
        st = self.structure(model_id)
        fmt = out.name.removesuffix(".gz").rsplit(".", 1)[-1]
        if fmt == "cif":
            text = st.make_mmcif_document().as_string()
        elif fmt == "pdb":
            text = st.make_pdb_string()
        else:
            raise ValueError(f"{out}: .cif, .pdb (optionally .gz) only")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.name.endswith(".gz"):
            with gzip.open(out, "wt", compresslevel=6) as fh:
                fh.write(text)
        else:
            out.write_text(text)
        return out
