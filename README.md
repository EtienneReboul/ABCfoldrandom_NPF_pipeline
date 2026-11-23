# ABCfoldrandom_NPF_pipeline

CF-random's sampling recipe, run with the ABCfold backends instead of AF2, for
every Arabidopsis NPF.

[CF-random](https://github.com/ncbi/CF-random_software) (Lee et al. 2025,
*Nat Commun* 16:5622) looks for alternative conformations by predicting from
the full MSA and from randomly subsampled shallow MSAs.
`../CFrandom_NPF_pipeline` ran it with AF2 (ColabFold) and concluded that one
predictor is not enough. Here the same MSAs and depths go to the five
open-weight backends that [ABCfold](https://github.com/rigdenlab/ABCFold)
drives:

- Boltz-2
- Chai-1
- OpenFold3
- Protenix
- RosettaFold3

AlphaFold3 is left out on purpose because of its weights terms of use.

The scale is 53 NPFs × 8 MSA depths × 5 backends × 20 seeds × 5 samples, which
is **212,000 models**. Kept the way the ABCfold NPF run kept its output (CIFs plus a float16 HDF5 of PAE, contact probabilities and per-atom pLDDT), it would take about 180 GB. It is
stored in about **27 GB** of Parquet instead, and no CIF is kept (see
[Storage](#storage)).

## Pipeline

```text
 Mac (needs ../CFrandom_NPF_pipeline)            CAIUS (SLURM, offline compute nodes)
 ─────────────────────────────────────           ────────────────────────────────────────────────────
 1 import_inputs  MSA + DeepTMHMM topology  ──►  2 run_abcfold    53 x 8 depths x 5 backends = 2,120 GPU jobs
 2 fold_input     424 ABCfold inputs        rsync    (20 seeds x 5 samples each; output packed into
                  (subsampled MSA + seeds)            the Parquet store and deleted inside the job)
   cdd_sites      CDD pocket residues (InterPro API)
                                                 3 collect_models  all models of a protein, confidence vs depth  [report]
                                                 4 tm_pca          TM-helix Ca PCA + GMM states            [report, static + plotly]
                                                 5 tm_states       OF / Occ / IF from the PCA extremities, [report]
                                                                   20 models per state for ChimeraX
                                                 5b tm_minimize    OpenMM minimization of those models (GPU)
                                                 6 tm_cavity       pyKVFinder pocket openings of those      [report]
                                                                   models, raw + minimized, vs the state
                                                 7 run_stats       GPU time + storage of every job          [report]
                                                   onsuccess       reports/report.zip
```

### Sampling

- **MSA and topology:** copied from `../CFrandom_NPF_pipeline/data/inputs`.
  This is the ABCfold apo `unpairedMsa` and the DeepTMHMM helices, so the two
  runs differ only in the predictor.
- **Depths:** the full MSA plus CF-random's seven depths, 1:2, 2:4, … 64:128.
  - In CF-random a depth is AF2's `max_seq:max_extra_seq` (cluster centres :
    extra rows). These backends have no equivalent setting.
  - So depth m:e means the query plus **m + e sequences drawn at random**,
    without replacement, from the full MSA (3 to 192 sequences).
  - The same subsample goes to every backend.
  - At `full`, each backend applies its own internal MSA cap.
- **Seeds:** 20 per protein × depth, shared by all five backends. Each seed
  gives 5 samples (ABCfold's `--number_of_models`).
- **Reproducibility:** the MSA draw and the seeds come from a generator keyed
  on `(master_seed, protein, depth)` (`scripts/make_fold_input.py`), so a
  rerun gives the same inputs.
- **No templates** at any depth, as in CF-random. Templates would pull every
  depth back to the known structure.
- **Recycles:** 10, ABCfold's default (`abcfold.num_recycles`). The ABCfold NPF
  run used 25. CF-random used 3, AF2's default.

## Storage

ABCfold writes 1–2 GB per job:

- a CIF per model
- pretty-printed JSON with the full PAE, contact probabilities and per-atom
  pLDDT
- duplicate `_fixed` CIFs and best-model copies
- backend input caches

`run_abcfold` writes all of this to node-local scratch. In the same job it
converts it into three Parquet files in
`results/<protein>/predictions/<depth>/<backend>/`, checks them, and deletes
the scratch directory. The code is in `scripts/abcstore.py`.

| File | One row per | Content |
| --- | --- | --- |
| `models.parquet` | model | ids, the backend's scores (pTM, ranking score, …, raw dict in `scores_json`), per-residue pLDDT (float16), `coords`, `bfactor` |
| `pae.parquet` | seed | PAE of the seed's 5 samples (`blob`, `mode`, `step`) |
| `atoms.parquet` | atom | the atom table shared by every model of that backend: chain, residue, atom name, element, occupancy |
| `run.json` | job | GPU, timings, sizes, round-trip errors |

The binary columns are encoded as follows:

- **`coords`:** every atom as int32 thousandths of an Å.
  - Encoding: delta-coded along the atom order, then the four byte planes are
    shuffled, then zstd level 19.
  - Lossless for Boltz, Chai-1 and Protenix, whose CIFs carry 3 decimals.
  - OpenFold3 and RF3 write float32 digits, so they are rounded by at most
    0.0005 Å.
- **`bfactor`:** the per-atom pLDDT as uint16 at `b_scale` steps per unit (500
  for 0–100 values, 50,000 for 0–1 values), then zstd. At most 0.001 off.
- **PAE:** uint8 at 0.25 Å per step, so the maximum error is 0.125 Å; the
  backends' own PAE bins are 0.5 Å.
  - Each matrix is coded with the 2-D "plane" predictor: x − left − up +
    upleft, mod 256. Its inverse is a 2-D cumulative sum, so decoding is
    vectorised.
  - Because samples of one seed share the trunk, the store can instead code
    the other samples as differences from the seed's first sample.
  - For each seed, the smaller of the three variants is kept.
- **Parquet compression:** the binary columns are stored without it (they are
  zstd already); the other columns use zstd.

Every conversion is checked against ABCfold's files before anything is
deleted. Coordinates must match to 0.0005 Å, B-factors to half a step, and PAE
to 0.125 Å. ABCfold exits 0 even when a backend produced nothing, so a job
also fails if it doesn't find exactly 20 × 5 models.

Size on the earlier ABCfold run's NPF2.10 output (636 aa, re-encoded):

| | Boltz | Chai-1 | OpenFold3 | Protenix | RF3 |
| --- | --- | --- | --- | --- | --- |
| structure (KB / model) | 30 | 38 | 39 | 38 | 38 |
| PAE (KB / model) | 100 | 101 | 100 | 100 | 44 |
| total vs CIF + float16 HDF5 (~850 KB / model) | ~6x smaller | | | | |

The PAE share decides the total. For comparison, float16 + zstd takes about
690 KB per model, and plain uint8 at 1/8 Å takes about 370 KB.

There is only one file type, Parquet; there's no HDF5. The PAE and coordinate
predictors are plain numpy code, so HDF5's filters (gzip or blosc with
shuffle) would add nothing they don't already do.

### Getting structure files back

No CIF is kept. Any model can be written back as a working mmCIF or PDB (with
entities and `label_seq_id`, via gemmi). ChimeraX, PyMOL, Foldseek, DSSP,
MDAnalysis and gemmi read it the way they read ABCfold's own files.

From the command line, with the analysis env (any env with numpy, pandas,
pyarrow and gemmi works):

```bash
# every Boltz model of NPF2.10 at depth 8:16, as mmCIF
python scripts/export_models.py --protein NPF2.10_Q944G5 --depth max8_ext16 --backend boltz --out exported/

# the 10 best-ranked models of every depth x backend, gzipped PDB
python scripts/export_models.py --protein NPF2.10_Q944G5 --top 10 --format pdb.gz --out exported/

# a pandas query on the score columns
python scripts/export_models.py --protein NPF2.10_Q944G5 --query "ptm > 0.85 and mean_plddt > 80" --out exported/

# named models (model_id = <backend>_s<seed>_m<sample>, as in results/<protein>/models.parquet)
python scripts/export_models.py --protein NPF2.10_Q944G5 --depth full --model-id boltz_s123_m0 chai1_s123_m4 --out exported/
```

Each export writes `<protein>__<depth>__<model_id>.<format>` files plus a
`manifest.csv` listing them with their scores.

From Python (run from the repo root, or put `scripts/` on `sys.path`):

```python
from abcstore import Store
st = Store("results/NPF2.10_Q944G5/predictions/max8_ext16/boltz")
st.models                         # scores, one row per model (no binary columns)
xyz = st.coords("boltz_s123_m0")  # (n_atoms, 3) A, in st.atoms order
res, ca = st.ca("boltz_s123_m0")  # CA only (resids= to restrict)
pae = st.pae("boltz_s123_m0")     # (n_tok, n_tok) A
st.write("boltz_s123_m0", "m.cif")   # or .pdb / .cif.gz / .pdb.gz
st.structure("boltz_s123_m0")        # gemmi.Structure
```

What is not kept:

- ABCfold's contact probabilities and per-atom pLDDT JSON. The B-factors carry
  the per-atom pLDDT.
- Boltz's PDE and distograms.
- The backends' input caches.
- The `_fixed` / best-model duplicate CIFs. When both exist, the `_fixed` CIF
  is the one stored.

## Running

### 1. Inputs (Mac, once)

```bash
conda activate nanopulldown        # controller env: snakemake 9 + datavzrd
snakemake --use-conda --cores 8 protein_list fold_inputs cdd_inputs
/opt/homebrew/bin/rsync -a --exclude .snakemake --exclude results --exclude logs --exclude .git \
  ./ hpc-login.u-strasbg.fr:ABCfoldrandom_NPF_pipeline/
```

Activate the controller env in a clean shell. Don't prepend its `bin/` to
`PATH` by hand.

### 2. CAIUS setup (login node, once)

```bash
ssh hpc-login.u-strasbg.fr
cd ~/ABCfoldrandom_NPF_pipeline
export TMUX_TMPDIR=~/.tmux-sock
tmux new -d -s abcfr_setup 'nice -n 19 scripts/caius/setup.sh 2>&1 | tee -a logs/setup.log'
```

`setup.sh` does the following:

1. Writes `config.local.yaml` from `config.caius.yaml.example`.
2. Builds the two rule envs:
   - `envs/abcfold.yaml`: ABCfold 1.1.6 plus the packing code.
   - `envs/analysis.yaml`.
3. Runs `scripts/caius/prime_backends.sh`. This needs the login node's internet
   access; compute nodes have none.
   - It runs `abcfold --dry_run`, one backend at a time. That builds the five
     backend micromamba envs under `~/micromamba/envs/abcfold-*` (about 40 GB)
     and fetches the OpenFold3 and RF3 checkpoints into `~/.openfold3` and
     `~/.rosettafold3`.
   - **Protenix** compiles a CUDA layer-norm kernel when it is imported, so
     even its `--help` smoke test fails on CAIUS: there is no `nvcc` matching
     its torch (CUDA 12.6) and the system GCC is 8.5.
     `build_protenix_kernel.sh` installs CUDA 12.6 `nvcc`, the CUDA headers
     and GCC 12 from conda-forge into the Protenix env, then builds the kernel
     once into the package. It covers sm_70 to sm_90; a P100 job crashes and
     is retried on an sm_80+ card.
   - It then runs `fetch_weights.py`. ABCfold's dry run only runs `--help` for
     Boltz-2, Chai-1 and Protenix, so the first GPU job would otherwise try to
     download their weights offline. They go into `~/.boltz` (7.5 GB),
     `abcfold.chai_downloads_dir` (7 GB, with ESM2-3B) and
     `abcfold.protenix_root_dir` (2.5 GB).
4. **Protenix v2 checkpoint, by hand, before `setup.sh`.** ABCfold 1.1.6 runs
   `protenix-v2`, but its official URL (a Chinese object store) returns 403.
   Copy `protenix-v2.pt` from Hugging Face, or from the IFB copy the ABCfold
   NPF run used (`~/checkpoint/protenix-v2.pt` there), into
   `<protenix_root_dir>/checkpoint/`. `fetch_weights.py` checks it against
   the published sha256 (`8f931f97…d599`, 1,859,785,497 bytes).

### 3. Pilot, then the full run

```bash
export TMUX_TMPDIR=~/.tmux-sock
# one protein first: 40 GPU jobs (8 depths x 5 backends); refit hpc.time_fit from logs/run_abcfold
tmux new -d -s abcfr 'scripts/caius/run.sh --config proteins="[NPF2.10_Q944G5]" 2>&1 | tee -a logs/caius_controller.log'
# everything
tmux new -d -s abcfr 'scripts/caius/run.sh 2>&1 | tee -a logs/caius_controller.log'
```

- **Walltime per job:** set by `hpc.time_fit`. The per-backend minutes come
  from the IFB L40S ABCfold NPF run (25 recycles) and use a 2x safety factor.
- **Preemption:** `publicgpu` jobs can be preempted. A requeued job reruns its
  backend from scratch.
- **Progress:** `ls results/*/predictions/*/*/run.json | wc -l` counts finished
  jobs; 2,120 means everything is done.
- **After a crash:** run `snakemake --unlock`, then relaunch.

### GPUs: any type, checked in every job

The run is large, so no GPU type is excluded: attempt 1 of a job takes any
`publicgpu` card (`hpc.gpu_constraint: "gpu|gpuh200"`, which covers every
node), pre-Ampere P100 / V100 / RTX 5000-6000 included, with a 4x walltime
(`time_fit.first_attempt_factor`). An old card can still crash a backend (no
bf16, cuEquivariance kernels need sm_80+) or run out of time; the retry then
goes to sm_80+ cards only (`hpc.gpu_constraint_retry`).

Every job checks its GPU and records how much it used it:

- **Before folding:** the backend's own torch (in its micromamba env) must
  see a CUDA device. Otherwise the job fails at once (exit 3) and is retried,
  rather than folding on CPU for hours. The check (device, compute capability,
  torch/CUDA versions) is the `[gpu] check` line of
  `logs/run_abcfold/<protein>/<depth>_<backend>.log`.
- **While folding:** `nvidia-smi` is sampled every 30 s (`[gpu] t=.. util=..%
  mem=..` lines). `run.json` gets `gpu`, `gpu_capability`, `gpu_util_mean`,
  `gpu_busy_frac`, `gpu_mem_max_mib`, and `gpu_idle_warning` (busy in < 20 %
  of the samples of a fold longer than 5 min).
- **Summary:** `~/micromamba/envs/nanopulldown/bin/python scripts/caius/gpu_report.py` lists the running jobs
  with their device and latest utilisation (IDLE / NO GPU flagged), and the
  finished ones per GPU type (median minutes, busy fraction, every idle
  warning). The report's run-stats figure has the same per-GPU panel.

### 4. Back to the Mac

```bash
/opt/homebrew/bin/rsync -a hpc-login.u-strasbg.fr:ABCfoldrandom_NPF_pipeline/{results,reports} ./
```

## Layout

| Path | What |
| --- | --- |
| `Snakefile` | the whole pipeline |
| `config.yaml` | depths, ABCfold sampling, storage, TM-PCA, SLURM defaults |
| `config.caius.yaml.example` | CAIUS overrides, copied to `config.local.yaml` |
| `profiles/caius/` | SLURM profile |
| `scripts/abcstore.py` | the model store: codecs, ABCfold output parsing, `pack()`, `Store` reader + CIF/PDB writer |
| `scripts/run_abcfold.py` | GPU job: ABCfold, then pack, verify and delete |
| `scripts/export_models.py` | CIF / PDB files from the store |
| `scripts/make_fold_input.py` | subsampled MSA + seeds, as ABCfold (AF3-dialect) input |
| `scripts/{collect_models,tm_pca,run_stats}.py` | analysis + report figures |
| `scripts/caius/` | `setup.sh`, `prime_backends.sh`, `fetch_weights.py`, `run.sh` |
| `data/inputs/<protein>/` | `<protein>.a3m.xz`, `sequence.fasta`, `topology.json`, `cdd_sites.json` |
| `data/fold_inputs/<protein>/<depth>.json.xz` | ABCfold inputs (20 seeds each) |
| `results/<protein>/predictions/<depth>/<backend>/` | `models.parquet`, `pae.parquet`, `atoms.parquet`, `run.json` |
| `results/<protein>/models.parquet` | every model of a protein, scores + TM pLDDT, no binary columns |
| `results/<protein>/tm_pca/` | PCA table, `model.npz`, `structures/` (+ `view.cxc`) |
| `results/<protein>/tm_states/` | per-model state + alpha, `cavity[_minimized].parquet` (pyKVFinder pocket), `structures/` and `minimized/` (`{pool,OF,Occ,IF}/` + `view.cxc`), `minimize.parquet` |
