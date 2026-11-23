"""
Step 5b: tm_minimize: OpenMM energy minimization of the exported models of
step 5 (structures/pool/*.pdb.gz), ported from NanoPulldown
(src/nanopulldown/structure/minimize.py), so that the pocket check (step 6)
can be run on relaxed side chains as well as on the raw predictions.

Per model, as in NanoPulldown:
  1. PDBFixer: missing heavy atoms, then hydrogens at pH 7
  2. pre-relaxation: vacuum, nonbonded cutoff (cutoff_nm), charges OFF
     (Lennard-Jones + bonded terms push overlaps apart without opposite
     charges collapsing), until a chunk changes the energy by
     < convergence_delta or pre_steps
  3. amber14-all + implicit/gbn2 (NoCutoff) L-BFGS in chunks of
     trace_interval iterations up to max_steps: divergence guard on every
     chunk (NaN, |E| > max_abs_energy), probe check at probe_steps (energy
     turning positive and > divergence_factor x |E_start|), early stop when
     a chunk changes the energy by < convergence_delta, wall-clock budget
     (max_minutes per model)
Platform: the first of --platforms that starts (OpenCL = a GPU vendor's
platform only: Apple / NVIDIA / AMD). Unlike NanoPulldown the CPU platform
may be listed: one NPF monomer in implicit solvent is small.
Differences: no cg2all fallback (a model whose first evaluation is NaN is
recorded as failed), and a failed model does not fail the job (status
column); every model keeps its raw frame (no re-superposition).

Outputs (results/<protein>/tm_states/): minimized/{pool,OF,Occ,IF}/ (same
names as structures/, hydrogens included) with view.cxc, and
minimize.parquet: per model status, energies (raw, after pre-relax, final),
steps, converged, seconds, platform, CA / heavy-atom RMSD to the raw model.
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["OF", "Occ", "IF"]


def opencl_gpu_index(platform) -> int | None:
    from openmm import Context, System, VerletIntegrator

    probe = System()
    probe.addParticle(1.0)
    for i in range(8):
        try:
            ctx = Context(probe, VerletIntegrator(0.001), platform, {"OpenCLPlatformIndex": str(i)})
        except Exception:  # noqa: BLE001
            break
        vendor = platform.getPropertyValue(ctx, "OpenCLPlatformName")
        del ctx
        if any(v in vendor for v in ("NVIDIA", "AMD Accelerated", "Apple")):
            return i
    return None


class Platforms:
    """First platform of the preference list that starts; remembered after."""

    def __init__(self, preference, threads):
        self.preference, self.threads, self.chosen = preference, threads, None

    def simulation(self, topology, system):
        from openmm import LangevinMiddleIntegrator, Platform, unit
        from openmm.app import Simulation

        names = [self.chosen[0]] if self.chosen else self.preference
        for name in names:
            try:
                platform = Platform.getPlatformByName(name)
                props = {}
                if name == "OpenCL":
                    idx = opencl_gpu_index(platform) if not self.chosen else self.chosen[1].get("OpenCLPlatformIndex")
                    if idx is None:
                        raise RuntimeError("no GPU OpenCL platform")
                    props = {"OpenCLPlatformIndex": str(idx)}
                elif name == "CPU":
                    props = {"Threads": str(self.threads)}
                integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
                sim = Simulation(topology, system, integ, platform, props)
            except Exception as e:  # noqa: BLE001
                print(f"[minimize] platform {name} unusable ({str(e).splitlines()[0][:120]})", flush=True)
                continue
            if not self.chosen:
                dev = ""
                if name in ("OpenCL", "CUDA"):
                    dev = " (" + platform.getPropertyValue(sim.context, "DeviceName").strip() + ")"
                print(f"[minimize] platform: {name}{dev}", flush=True)
                self.chosen = (name, props)
            return sim
        raise RuntimeError(f"no OpenMM platform could start (tried {', '.join(names)})")


def minimize(path: Path, out: Path, a, plat: Platforms) -> dict:
    from openmm import LocalEnergyMinimizer, NonbondedForce, unit
    from openmm.app import CutoffNonPeriodic, ForceField, Modeller, PDBFile
    from pdbfixer import PDBFixer

    t0 = time.monotonic()
    with gzip.open(path, "rt") as fh:
        fixer = PDBFixer(pdbfile=io.StringIO(fh.read()))
    fixer.findMissingResidues()
    fixer.missingResidues = {}            # never build termini / gaps the predictor did not model
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    modeller = Modeller(fixer.topology, fixer.positions)
    sim = plat.simulation(modeller.topology, ForceField(*a.forcefield).createSystem(modeller.topology))
    sim.context.setPositions(modeller.positions)

    def energy(s=sim):
        return s.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    e0 = energy()
    pre_done = 0
    if a.pre_steps > 0:
        pre_sys = ForceField("amber14-all.xml").createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                                                             nonbondedCutoff=a.cutoff_nm * unit.nanometer)
        for f in pre_sys.getForces():
            if isinstance(f, NonbondedForce):
                for i in range(f.getNumParticles()):
                    _, sig, eps = f.getParticleParameters(i)
                    f.setParticleParameters(i, 0.0, sig, eps)
                for k in range(f.getNumExceptions()):
                    p1, p2, _, sig, eps = f.getExceptionParameters(k)
                    f.setExceptionParameters(k, p1, p2, 0.0, sig, eps)
        pre = plat.simulation(modeller.topology, pre_sys)
        pre.context.setPositions(modeller.positions)
        prev = energy(pre)
        while pre_done < a.pre_steps:
            n = min(a.trace_interval, a.pre_steps - pre_done)
            LocalEnergyMinimizer.minimize(pre.context, maxIterations=n)
            pre_done += n
            e = energy(pre)
            if e != e:
                raise RuntimeError(f"pre-relaxation diverged (nan) at step {pre_done}")
            if abs(e - prev) < a.convergence_delta:
                break
            prev = e
        sim.context.setPositions(pre.context.getState(getPositions=True).getPositions())
        del pre
    e_start = energy()
    done, probed, prev, converged = 0, False, e_start, False
    budget = 60.0 * a.max_minutes
    while done < a.max_steps:
        n = min(a.trace_interval, a.max_steps - done)
        LocalEnergyMinimizer.minimize(sim.context, maxIterations=n)
        done += n
        e = energy()
        bad = e != e or abs(e) > a.max_abs_energy
        if not probed and done >= a.probe_steps:
            probed = True
            bad = bad or (e_start < 0 < e and abs(e) > a.divergence_factor * abs(e_start))
        if bad:
            raise RuntimeError(f"minimization diverged at step {done} (start {e_start:.1f}, now {e:.1f} kJ/mol)")
        if abs(e - prev) < a.convergence_delta:
            converged = True
            break
        if probed and budget and time.monotonic() - t0 > budget:
            break
        prev = e
    pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    buf = io.StringIO()
    PDBFile.writeFile(modeller.topology, pos, buf, keepIds=True)
    with gzip.open(out, "wt", compresslevel=6) as fh:
        fh.write(buf.getvalue())

    # RMSD to the raw model, same frame (no fit): how far minimization moved it
    raw_xyz = read_heavy(path)
    new_xyz = {k: v for k, v in zip(atom_keys(modeller.topology), pos) if not k[2].startswith("H")}
    common = [k for k in raw_xyz if k in new_xyz]
    d2 = np.array([((raw_xyz[k] - new_xyz[k]) ** 2).sum() for k in common])
    ca = np.array([k[2] == "CA" for k in common])
    return {"status": "ok", "e_raw": e0, "e_prerelaxed": e_start, "e_final": float(e), "pre_steps": pre_done,
            "steps": done, "converged": converged, "seconds": time.monotonic() - t0,
            "rmsd_ca": float(np.sqrt(d2[ca].mean())), "rmsd_heavy": float(np.sqrt(d2.mean()))}


def atom_keys(topology):
    return [(a.residue.chain.id, int(a.residue.id), a.name) for a in topology.atoms()]


def read_heavy(path: Path) -> dict:
    out = {}
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                name = ln[12:16].strip()
                if name.startswith("H") or ln[76:78].strip() == "H":
                    continue
                out[(ln[21], int(ln[22:26]), name)] = np.array([float(ln[30:38]), float(ln[38:46]),
                                                                float(ln[46:54])])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True, help="tm_states.parquet (structures/ next to it)")
    ap.add_argument("--platforms", default="CUDA,OpenCL")
    ap.add_argument("--threads", type=int, default=4, help="CPU platform threads")
    ap.add_argument("--forcefield", nargs="+", default=["amber14-all.xml", "implicit/gbn2.xml"])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--trace-interval", type=int, default=50)
    ap.add_argument("--convergence-delta", type=float, default=1e-3)
    ap.add_argument("--max-abs-energy", type=float, default=1e15)
    ap.add_argument("--divergence-factor", type=float, default=100.0)
    ap.add_argument("--pre-steps", type=int, default=500, help="0 = no pre-relaxation")
    ap.add_argument("--cutoff-nm", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=15.0, help="per model; 0 = no budget")
    ap.add_argument("--limit", type=int, default=0, help="test: first N models only")
    ap.add_argument("--out", type=Path, required=True, help="minimize.parquet")
    a = ap.parse_args()

    st = pd.read_parquet(a.states)
    st = st[st["exported"] != ""].reset_index(drop=True)
    if a.limit:
        st = st.head(a.limit)
    src = a.states.parent / "structures"
    mdir = a.out.parent / "minimized"
    for d in [mdir / "pool", *(mdir / s for s in STATES)]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    for f in mdir.glob("*.cxc"):
        f.unlink()
    plat = Platforms([p.strip() for p in a.platforms.split(",") if p.strip()], a.threads)
    rows = []
    for r in st.itertuples(index=False):
        s, link = r.exported.split("/")
        base = os.path.basename(os.readlink(src / r.exported))
        row = {"protein": r.protein, "state": s, "file": r.exported, "model_id": r.model_id, "depth": r.depth}
        try:
            row.update(minimize(src / "pool" / base, mdir / "pool" / base, a, plat))
            os.symlink(Path("..") / "pool" / base, mdir / s / link)
        except Exception as e:  # noqa: BLE001
            row.update(status="failed", error=str(e).splitlines()[0][:300])
        row["platform"] = plat.chosen[0] if plat.chosen else ""
        rows.append(row)
        print(f"[minimize] {r.exported}: {row['status']}"
              + (f" {row['e_raw']:.3g} -> {row['e_final']:.1f} kJ/mol, {row['steps']} steps, "
                 f"CA {row['rmsd_ca']:.2f} / heavy {row['rmsd_heavy']:.2f} A, {row['seconds']:.0f} s"
                 if row["status"] == "ok" else f" ({row['error']})"), flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)

    # ChimeraX scripts, as structures/ (same colours), only the models that minimized
    cols = {"OF": "#d62728", "Occ": "#f0a30a", "IF": "#2457c5"}
    cxc = [f"# {st['protein'].iloc[0]}: minimized exported models (OpenMM amber14 + GBn2), mean TM frame", "close"]
    n = 0
    for s in STATES:
        names = sorted(p.name for p in (mdir / s).iterdir() if p.is_symlink())
        if not names:
            continue
        cxc += [f"open {s}/{x} name {s}_{x[:2]}" for x in names]
        cxc.append(f"color #{n + 1}-{n + len(names)} {cols[s]}")
        n += len(names)
    side = src / "view.cxc"
    tail = [ln for ln in side.read_text().splitlines() if ln.startswith(("show :", "style", "color byhetero", "# cd"))]
    cxc += ["cartoon", "hide atoms", *tail, "hide H", "view"]
    (mdir / "view.cxc").write_text("\n".join(cxc) + "\n")
    ok = out["status"].eq("ok")
    print(f"[minimize] {ok.sum()}/{len(out)} minimized; median {out.loc[ok, 'seconds'].median():.0f} s/model; "
          f"median CA RMSD {out.loc[ok, 'rmsd_ca'].median():.2f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
