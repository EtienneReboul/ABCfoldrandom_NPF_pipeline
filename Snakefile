"""
ABCfoldrandom_NPF_pipeline: CF-random sampling with the ABCfold backends
=======================================================================
CF-random's recipe (full MSA + randomly subsampled shallow MSAs, Lee et al.
2025) run with the five open-weight ABCfold backends (Boltz-2, Chai-1,
OpenFold3, Protenix, RosettaFold3) instead of AF2, for every Arabidopsis NPF:
53 proteins x 8 MSA depths x 5 backends x 20 seeds x 5 samples.

  1  import_inputs   MSA + DeepTMHMM topology from CFrandom_NPF_pipeline (Mac only)
  2  fold_input      one ABCfold input per protein x depth (subsampled MSA, seeds)
     run_abcfold     1 GPU job per protein x depth x backend; ABCfold's output is
                     packed into the Parquet model store and deleted in the job
  3  collect_models  all models of a protein + confidence vs depth      -> report
  4  tm_pca          TM-helix Ca PCA + GMM states (static + plotly)    -> report
  5  tm_states       OF / Occ / IF from the PCA extremities + 20 models
                    per state for ChimeraX (pdb.gz pool + symlinks)    -> report
  5b tm_minimize     OpenMM minimization of those models (GPU; NanoPulldown's scheme)
  6  tm_cavity       pyKVFinder substrate pocket of those models, raw and
                    minimized: open outside / inside / occluded vs the PCA state -> report
  7  run_stats       timings + storage of every GPU job                -> report
  onsuccess: reports/report.zip

No CIF is kept: scripts/export_models.py writes any model as mmCIF / PDB from
the store (README "Getting structure files back").

Local (Mac):   snakemake --use-conda --cores 4 fold_inputs   (inputs only)
CAIUS (SLURM): scripts/caius/run.sh   (see README)
"""
import json
import sys
from pathlib import Path

REPO = Path(workflow.basedir)
sys.path.insert(0, str(REPO / "scripts"))
from common import BACKENDS, depth_names  # noqa: E402

configfile: str(REPO / "config.yaml")
if (REPO / "config.local.yaml").exists():
    configfile: str(REPO / "config.local.yaml")

report: "report/workflow.rst"

DIRS = config["dirs"]
INP, FIN = Path(DIRS["inputs"]), Path(DIRS["fold_inputs"])
RES, REP, LOG = Path(DIRS["results"]), Path(DIRS["reports"]), Path(DIRS["logs"])
AB = config["abcfold"]
ST = config["store"]
TP = config.get("tm_pca") or {}
TS = config.get("tm_states") or {}
TC = config.get("tm_cavity") or {}
TM = config.get("tm_minimize") or {}
HPC = config.get("hpc") or {}
FMTS = ",".join(config["report"]["figure_formats"])
DEPTHS = depth_names(config["depths"])
DEPTH_ARGS = " ".join(DEPTHS)
BACKS = [b for b in AB["backends"] if b in BACKENDS]
PY = "python scripts"

# ── proteins ────────────────────────────────────────────────────────────────
CFR = (REPO / config["cfrandom_repo"]).resolve()
_plist = INP / "proteins.txt"
_cfr_list = CFR / "data/inputs/proteins.txt"


def _npf_order(ps):
    return sorted(ps, key=lambda p: [int(x) if x.isdigit() else x for x in p.split("_")[0][3:].split(".")])


if _cfr_list.exists():
    ALL_PROTEINS = _npf_order(_cfr_list.read_text().split())
elif _plist.exists():
    ALL_PROTEINS = _npf_order(_plist.read_text().split())
else:
    raise ValueError(f"no protein list: provide {_plist} (with the imported inputs) or {_cfr_list}")
PROTEINS = _npf_order(config.get("proteins") or ALL_PROTEINS)


def _topo(p):
    f = INP / p / "topology.json"
    return json.loads(f.read_text()) if f.exists() else {}


TOPO = {p: _topo(p) for p in PROTEINS}
SUBCAT = {p: "GA importers" if TOPO[p].get("ga_importer") else "other NPFs" for p in PROTEINS}


def labels(wc):
    t = TOPO.get(wc.protein, {})
    return {"protein": wc.protein, "ligand group": t.get("ligand") or "-", "length": str(t.get("length", "?"))}


def store(p, d, b):
    return RES / p / "predictions" / d / b


wildcard_constraints:
    protein="[^/]+",
    depth="full|max\\d+_ext\\d+",
    backend="|".join(BACKENDS),


rule all:
    default_target: True
    input:
        str(INP / "proteins.txt"),
        [str(RES / p / "tm_pca" / "tm_pca.parquet") for p in PROTEINS],
        str(REP / "tm_states" / "tm_states_summary.csv"),
        str(REP / "tm_cavity" / "tm_cavity_summary.csv"),
        str(REP / "tm_cavity" / "tm_cavity_summary_minimized.csv"),
        str(REP / "run_stats" / "run_stats.csv"),


# ═════ 1. inputs (Mac) ═════════════════════════════════════════════════════
# The CF-random inputs (ABCfold apo MSA + DeepTMHMM topology): both runs see
# the same MSA. On CAIUS the outputs arrive by rsync and the rule has no
# inputs, so it is never scheduled there.
rule import_inputs:
    input:
        lambda wc: [str(CFR / "data/inputs" / wc.protein / f) for f in
                    (f"{wc.protein}.a3m.xz", "sequence.fasta", "topology.json")] if CFR.exists() else [],
    output:
        a3m=str(INP / "{protein}" / "{protein}.a3m.xz"),
        fasta=str(INP / "{protein}" / "sequence.fasta"),
        topology=str(INP / "{protein}" / "topology.json"),
    shell:
        "cp -p {input} $(dirname {output.a3m})/"


rule protein_list:
    input:
        [str(INP / p / "topology.json") for p in ALL_PROTEINS],
    output:
        str(INP / "proteins.txt"),
    run:
        Path(output[0]).write_text("\n".join(ALL_PROTEINS) + "\n")


# CDD site residues from the InterPro API (network: Mac). tm_states shows
# them in the ChimeraX scripts, tm_cavity uses them to find the pocket.
rule cdd_sites:
    localrule: True
    input:
        topology=str(INP / "{protein}" / "topology.json"),
        fasta=str(INP / "{protein}" / "sequence.fasta"),
    output:
        str(INP / "{protein}" / "cdd_sites.json"),
    shell:
        "{PY}/fetch_cdd_sites.py --topology {input.topology} --fasta {input.fasta} --out {output}"


rule cdd_inputs:
    input:
        [str(INP / p / "cdd_sites.json") for p in PROTEINS],


rule fold_input:
    input:
        str(INP / "{protein}" / "{protein}.a3m.xz"),
    output:
        str(FIN / "{protein}" / "{depth}.json.xz"),
    log:
        str(LOG / "fold_input" / "{protein}" / "{depth}.log"),
    conda:
        "envs/analysis.yaml"
    params:
        seed=config["master_seed"], n=AB["n_seeds"], tpl="--templates" if AB.get("templates") else "",
    shell:
        """
        {PY}/make_fold_input.py --a3m {input} --protein {wildcards.protein} --depth {wildcards.depth} \
          --master-seed {params.seed} --n-seeds {params.n} {params.tpl} --out {output} > {log} 2>&1
        """


rule fold_inputs:
    input:
        [str(FIN / p / f"{d}.json.xz") for p in PROTEINS for d in DEPTHS],


# ═════ 2. ABCfold (GPU) ════════════════════════════════════════════════════
FIT = HPC.get("time_fit") or {}


def abcfold_runtime(wc, attempt):
    """Walltime (min): load + per-backend minutes at 600 aa x (L/600)^exp x safety (config hpc.time_fit).

    Attempt 1 may land on any GPU (no constraint), pre-Ampere cards included,
    which are several times slower: x first_attempt_factor. Retries run on
    hpc.gpu_constraint_retry cards with the plain estimate.
    """
    L = TOPO.get(wc.protein, {}).get("length", 600)
    per = float((FIT.get("minutes_600aa") or {}).get(wc.backend, 90))
    scale = (int(AB["n_seeds"]) * int(AB["number_of_models"])) / 100
    t = per * scale * (L / 600) ** float(FIT.get("exponent", 2.0)) * float(FIT.get("safety", 2.0))
    if attempt == 1:
        t *= float(FIT.get("first_attempt_factor", 4.0))
    t += float(FIT.get("load_min", 15))
    return int(min(float(HPC.get("runtime_max", 1440)), max(30.0, t)))


def gpu_constraint(wc, attempt):
    """Any GPU first (the run is large: take whatever node is free); after a
    failure (crash on an old card, OOM, timeout) only hpc.gpu_constraint_retry."""
    if attempt > 1 or BACKENDS[wc.backend]["min_cc"] >= 8.0:   # these crash / hang below Ampere
        return HPC.get("gpu_constraint_retry", "")
    return HPC.get("gpu_constraint", "")


GPU_RES = {"slurm_partition": HPC.get("gpu_partition", "gpu"), "mem_mb": int(HPC.get("mem_mb", 64000)),
           "cpus_per_task": int(HPC.get("cpus", 8))}
if HPC.get("gpu_request", "gpus") == "gres":   # CAIUS: sbatch rejects --gpus
    GPU_RES.update(gpu=0, gres="gpu:1")
else:
    GPU_RES.update(gpu=1)
if HPC.get("gpu_constraint") or HPC.get("gpu_constraint_retry"):
    GPU_RES["constraint"] = gpu_constraint
if HPC.get("exclude_nodes"):
    GPU_RES["slurm_extra"] = f"--exclude={HPC['exclude_nodes']}"

# shell prefix, not params: machine plumbing must not mark finished folds outdated
AB_ENV = ""
if AB.get("mamba_root_prefix"):
    AB_ENV += f"MAMBA_ROOT_PREFIX={AB['mamba_root_prefix']} "
if AB.get("chai_downloads_dir"):
    AB_ENV += f"CHAI_DOWNLOADS_DIR={AB['chai_downloads_dir']} "
if AB.get("protenix_root_dir"):
    AB_ENV += f"PROTENIX_ROOT_DIR={AB['protenix_root_dir']} "
if AB.get("conda_lib_first"):   # RHEL 8 libstdc++ too old for the env's numpy (CAIUS)
    AB_ENV += ('LD_LIBRARY_PATH="$(python -c \'import sys; print(sys.prefix)\')/lib'
               '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ')


rule run_abcfold:
    input:
        str(FIN / "{protein}" / "{depth}.json.xz"),
    output:
        models=str(store("{protein}", "{depth}", "{backend}") / "models.parquet"),
        pae=str(store("{protein}", "{depth}", "{backend}") / "pae.parquet"),
        atoms=str(store("{protein}", "{depth}", "{backend}") / "atoms.parquet"),
        run=str(store("{protein}", "{depth}", "{backend}") / "run.json"),
    log:
        str(LOG / "run_abcfold" / "{protein}" / "{depth}_{backend}.log"),
    conda:
        "envs/abcfold.yaml"
    resources:
        runtime=abcfold_runtime,
        **GPU_RES,
    params:
        n=AB["number_of_models"], rec=AB["num_recycles"], step=ST["pae_step"], lvl=ST["zstd_level"],
        scratch=HPC.get("scratch") or "",
    shell:
        """
        {AB_ENV}{PY}/run_abcfold.py --fold-input {input} --backend {wildcards.backend} \
          --protein {wildcards.protein} --depth {wildcards.depth} --out-dir $(dirname {output.models}) \
          --number-of-models {params.n} --num-recycles {params.rec} --pae-step {params.step} \
          --zstd-level {params.lvl} --scratch '{params.scratch}' --abcfold "$(command -v abcfold)" > {log} 2>&1
        """


# ═════ 3-5. analysis (CPU) ═════════════════════════════════════════════════
rule collect_models:
    input:
        stores=[str(store("{protein}", d, b) / "models.parquet") for d in DEPTHS for b in BACKS],
        topology=str(INP / "{protein}" / "topology.json"),
    output:
        table=str(RES / "{protein}" / "models.parquet"),
        fig=report(str(REP / "confidence" / "{protein}.svg"), category="2. Confidence vs MSA depth",
                   subcategory=lambda wc: SUBCAT[wc.protein], labels=labels, caption="report/captions/confidence.rst"),
    log:
        str(LOG / "collect_models" / "{protein}.log"),
    conda:
        "envs/analysis.yaml"
    resources:
        mem_mb=8000,
    shell:
        """
        {PY}/collect_models.py --stores {input.stores} --topology {input.topology} --depths {DEPTH_ARGS} \
          --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1
        """


rule tm_pca:
    input:
        models=str(RES / "{protein}" / "models.parquet"),
        topology=str(INP / "{protein}" / "topology.json"),
    output:
        table=str(RES / "{protein}" / "tm_pca" / "tm_pca.parquet"),
        fig=report(str(REP / "tm_pca" / "{protein}.svg"), category="3. TM-helix Ca PCA",
                   subcategory=lambda wc: SUBCAT[wc.protein], labels=lambda wc: {**labels(wc), "view": "static"},
                   caption="report/captions/tm_pca.rst"),
        html=report(str(REP / "tm_pca" / "{protein}.html"), category="3. TM-helix Ca PCA",
                    subcategory=lambda wc: SUBCAT[wc.protein], labels=lambda wc: {**labels(wc), "view": "interactive"},
                    caption="report/captions/tm_pca.rst"),
    log:
        str(LOG / "tm_pca" / "{protein}.log"),
    conda:
        "envs/analysis.yaml"
    resources:
        mem_mb=16000,
        runtime=120,
    params:
        k=TP.get("n_states", 3), plddt=TP.get("min_tm_plddt", 70),
    shell:
        """
        {PY}/tm_pca.py --models {input.models} --topology {input.topology} --depths {DEPTH_ARGS} \
          --n-states {params.k} --min-tm-plddt {params.plddt} --out {output.table} --fig {output.fig} \
          --html {output.html} --formats {FMTS} > {log} 2>&1
        """


rule tm_states:
    input:
        pca=str(RES / "{protein}" / "tm_pca" / "tm_pca.parquet"),
        topology=str(INP / "{protein}" / "topology.json"),
        cdd=str(INP / "{protein}" / "cdd_sites.json"),
    output:
        table=str(RES / "{protein}" / "tm_states" / "tm_states.parquet"),
        structures=directory(str(RES / "{protein}" / "tm_states" / "structures")),
        fig=report(str(REP / "tm_states" / "{protein}.svg"), category="4. Conformational states (OF / Occ / IF)",
                   subcategory=lambda wc: SUBCAT[wc.protein], labels=lambda wc: {**labels(wc), "view": "static"},
                   caption="report/captions/tm_states.rst"),
        html=report(str(REP / "tm_states" / "{protein}.html"), category="4. Conformational states (OF / Occ / IF)",
                    subcategory=lambda wc: SUBCAT[wc.protein],
                    labels=lambda wc: {**labels(wc), "view": "interactive"},
                    caption="report/captions/tm_states.rst"),
    log:
        str(LOG / "tm_states" / "{protein}.log"),
    conda:
        "envs/analysis.yaml"
    resources:
        mem_mb=8000,
        runtime=60,
    params:
        mass=TS.get("mass_frac", 0.005), rad=TS.get("radius_frac", 0.05), n=TS.get("n_export", 20), floor=TS.get("alpha_floor", 0.8),
        betw=TS.get("max_betweenness", 0.5), flat=TS.get("min_flatness", 0.1), gap=TS.get("min_gap", 5.0),
    shell:
        """
        {PY}/tm_states.py --pca {input.pca} --topology {input.topology} --mass-frac {params.mass} \
          --radius-frac {params.rad} --n-export {params.n} --alpha-floor {params.floor} --max-betweenness {params.betw} \
          --min-flatness {params.flat} --min-gap {params.gap} --cdd {input.cdd} --out {output.table} --fig {output.fig} \
          --html {output.html} --formats {FMTS} > {log} 2>&1
        """


rule tm_states_summary:
    input:
        [str(RES / p / "tm_states" / "tm_states.parquet") for p in PROTEINS],
    output:
        table=str(REP / "tm_states" / "tm_states_summary.csv"),
        fig=report(str(REP / "tm_states" / "family_overview.svg"), category="4. Conformational states (OF / Occ / IF)",
                   subcategory="family overview", labels={"view": "all proteins"},
                   caption="report/captions/tm_states_summary.rst"),
    log:
        str(LOG / "tm_states_summary.log"),
    conda:
        "envs/analysis.yaml"
    shell:
        "{PY}/tm_states_summary.py --tables {input} --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1"


# GPU job (CUDA on SLURM, OpenCL = Apple GPU on the Mac: tm_minimize.platforms).
MIN_RES = {k: v for k, v in GPU_RES.items() if k != "constraint"}
if HPC.get("gpu_constraint_retry"):   # OpenMM's CUDA kernels: a recent card
    MIN_RES["constraint"] = HPC["gpu_constraint_retry"]


rule tm_minimize:
    input:
        states=str(RES / "{protein}" / "tm_states" / "tm_states.parquet"),
        structures=str(RES / "{protein}" / "tm_states" / "structures"),
    output:
        table=str(RES / "{protein}" / "tm_states" / "minimize.parquet"),
        structures=directory(str(RES / "{protein}" / "tm_states" / "minimized")),
    log:
        str(LOG / "tm_minimize" / "{protein}.log"),
    conda:
        "envs/minimize.yaml"
    resources:
        runtime=180,
        **MIN_RES,
    params:
        plat=",".join(TM.get("platforms", ["CUDA", "OpenCL"])), steps=TM.get("max_steps", 2000),
        probe=TM.get("probe_steps", 200), pre=TM.get("pre_relax_steps", 500), budget=TM.get("max_minutes", 15),
    shell:
        """
        {PY}/tm_minimize.py --states {input.states} --platforms {params.plat} --max-steps {params.steps} \
          --probe-steps {params.probe} --pre-steps {params.pre} --max-minutes {params.budget} \
          --out {output.table} > {log} 2>&1
        """


rule tm_cavity:
    input:
        states=str(RES / "{protein}" / "tm_states" / "tm_states.parquet"),
        structures=str(RES / "{protein}" / "tm_states" / "structures"),
        topology=str(INP / "{protein}" / "topology.json"),
        cdd=str(INP / "{protein}" / "cdd_sites.json"),
    output:
        table=str(RES / "{protein}" / "tm_states" / "cavity.parquet"),
        fig=report(str(REP / "tm_cavity" / "{protein}.svg"), category="4. Conformational states (OF / Occ / IF)",
                   subcategory=lambda wc: SUBCAT[wc.protein],
                   labels=lambda wc: {**labels(wc), "view": "pocket (pyKVFinder)"},
                   caption="report/captions/tm_cavity.rst"),
    log:
        str(LOG / "tm_cavity" / "{protein}.log"),
    conda:
        "envs/cavity.yaml"
    threads: 4
    resources:
        mem_mb=8000,
        runtime=60,
    params:
        vol=TC.get("volume_cutoff", 100), po=TC.get("probe_out", 8.0), area=TC.get("min_area", 20),
        z=TC.get("side_z", 0.5),
        cdd=TC.get("min_cdd_frac", 0.4),
    shell:
        """
        KMP_WARNINGS=0 {PY}/tm_cavity.py --states {input.states} --topology {input.topology} --cdd {input.cdd} \
          --volume-cutoff {params.vol} --probe-out {params.po} --min-area {params.area} --side-z {params.z} --min-cdd-frac {params.cdd} \
          --threads {threads} --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1
        """


rule tm_cavity_minimized:
    input:
        states=str(RES / "{protein}" / "tm_states" / "tm_states.parquet"),
        structures=str(RES / "{protein}" / "tm_states" / "minimized"),
        topology=str(INP / "{protein}" / "topology.json"),
        cdd=str(INP / "{protein}" / "cdd_sites.json"),
    output:
        table=str(RES / "{protein}" / "tm_states" / "cavity_minimized.parquet"),
        fig=report(str(REP / "tm_cavity" / "{protein}_minimized.svg"),
                   category="4. Conformational states (OF / Occ / IF)",
                   subcategory=lambda wc: SUBCAT[wc.protein],
                   labels=lambda wc: {**labels(wc), "view": "pocket (pyKVFinder), minimized"},
                   caption="report/captions/tm_cavity.rst"),
    log:
        str(LOG / "tm_cavity" / "{protein}_minimized.log"),
    conda:
        "envs/cavity.yaml"
    threads: 4
    resources:
        mem_mb=8000,
        runtime=60,
    params:
        vol=TC.get("volume_cutoff", 100), po=TC.get("probe_out", 8.0), area=TC.get("min_area", 20),
        z=TC.get("side_z", 0.5),
        cdd=TC.get("min_cdd_frac", 0.4),
    shell:
        """
        KMP_WARNINGS=0 {PY}/tm_cavity.py --states {input.states} --structures {input.structures} \
          --label "OpenMM-minimized models" --topology {input.topology} --cdd {input.cdd} \
          --volume-cutoff {params.vol} --probe-out {params.po} --min-area {params.area} --side-z {params.z} --min-cdd-frac {params.cdd} \
          --threads {threads} --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1
        """


rule tm_cavity_summary:
    input:
        [str(RES / p / "tm_states" / "cavity.parquet") for p in PROTEINS],
    output:
        table=str(REP / "tm_cavity" / "tm_cavity_summary.csv"),
        fig=report(str(REP / "tm_cavity" / "agreement.svg"), category="4. Conformational states (OF / Occ / IF)",
                   subcategory="family overview", labels={"view": "pocket agreement"},
                   caption="report/captions/tm_cavity_summary.rst"),
    log:
        str(LOG / "tm_cavity_summary.log"),
    conda:
        "envs/analysis.yaml"
    shell:
        "{PY}/tm_cavity_summary.py --tables {input} --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1"


rule tm_cavity_summary_minimized:
    input:
        [str(RES / p / "tm_states" / "cavity_minimized.parquet") for p in PROTEINS],
    output:
        table=str(REP / "tm_cavity" / "tm_cavity_summary_minimized.csv"),
        fig=report(str(REP / "tm_cavity" / "agreement_minimized.svg"),
                   category="4. Conformational states (OF / Occ / IF)",
                   subcategory="family overview", labels={"view": "pocket agreement, minimized"},
                   caption="report/captions/tm_cavity_summary.rst"),
    log:
        str(LOG / "tm_cavity_summary_minimized.log"),
    conda:
        "envs/analysis.yaml"
    shell:
        "{PY}/tm_cavity_summary.py --tables {input} --label 'OpenMM-minimized models' --out {output.table} "
        "--fig {output.fig} --formats {FMTS} > {log} 2>&1"


rule run_stats:
    input:
        [str(store(p, d, b) / "run.json") for p in PROTEINS for d in DEPTHS for b in BACKS],
    output:
        table=str(REP / "run_stats" / "run_stats.csv"),
        fig=report(str(REP / "run_stats" / "run_stats.svg"), category="1. Runs: GPU time and storage",
                   caption="report/captions/run_stats.rst"),
    log:
        str(LOG / "run_stats.log"),
    conda:
        "envs/analysis.yaml"
    shell:
        "{PY}/run_stats.py --runs {input} --out {output.table} --fig {output.fig} --formats {FMTS} > {log} 2>&1"


onsuccess:
    # only once the whole run is there: --report fails on any missing report item
    if (REP / "run_stats" / "run_stats.csv").exists() and not workflow.remote_exec:
        shell(f"cd {REPO} && {PY}/report_bundle.py build --out {REP / 'report.zip'} || true")
