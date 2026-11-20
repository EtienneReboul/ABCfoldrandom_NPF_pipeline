# Working notes for Claude in this repo

Sibling of `../CFrandom_NPF_pipeline` (same layout, same CAIUS setup, same
report machinery). Read its CLAUDE.md for the general rules; the ones that
matter most here:

- Never open a browser or GUI while testing: plotly `PLOTLY_RENDERER=json
  BROWSER=true`, matplotlib Agg. Don't use `qlmanage` to rasterise SVGs (it
  hung on a missing file): save a PNG next to the SVG with `--formats svg,png`.
- `/opt/homebrew/bin/rsync`, not macOS openrsync.
- Long CAIUS work in tmux (`export TMUX_TMPDIR=~/.tmux-sock`).
- Local snakemake: `env -i HOME=$HOME USER=$USER TERM=xterm zsh -lic "cd $PWD &&
  conda activate nanopulldown && snakemake --use-conda ..."`. Pipe through
  `tail`: the `reason:` lines of a big DAG are huge.

## The model store is the only copy of the structures

No CIF is kept (user decision, 2026-11-17): `scripts/abcstore.py` packs each
ABCfold run into `{models,pae,atoms}.parquet` and `run_abcfold.py` deletes the
raw output in the same job. Consequences:

- Any change to an encoding must stay decodable for stores already written:
  bump `STORE_VERSION` and branch on it in the readers, never change a codec
  in place.
- `pack()` verifies every encoding against ABCfold's files before the job
  deletes them; keep that check when touching it. ABCfold exits 0 when a
  backend produced nothing (`--no_visuals` returns before its own check): the
  exact 20 x 5 model count in `pack()` is what fails the job.
- Structure files for any tool: `scripts/export_models.py` /
  `abcstore.Store.write()`. Never re-add CIF output to the pipeline.

## ABCfold facts (1.1.6, checked against its source)

- Backend output layouts and score/PAE file names: `abcstore.discover()`,
  `summary_scores()`, `raw_pae()`. OpenFold3 and RF3 also write `_fixed` CIFs
  (kept) and best-model copies `<name>_model.cif` (skipped). OpenFold3 numbers
  samples from 1 (pack() shifts to 0).
- 1.1.6 runs OpenFold3 once with all seeds (`openfold_results/`); the n^2 seed
  bug patched in ABCfold_NPF_pipeline (`patch_abcfold_openfold3_seed_bug.py`)
  is fixed upstream.
- `--dry_run` builds the backend envs and fetches Boltz / OpenFold3 / RF3
  weights, but only runs `--help` for Chai-1 and Protenix: their weights are
  fetched by `scripts/caius/fetch_weights.py` on the login node, into
  `abcfold.chai_downloads_dir` / `protenix_root_dir` (exported to the GPU jobs).
- ABCfold finds micromamba at `~/.local/bin/micromamba` on CAIUS; backend envs
  live under `MAMBA_ROOT_PREFIX` (`abcfold.mamba_root_prefix`).

## GPU monitoring (user requirement, 2026-11-17)

Any GPU type is acceptable ("as long as it runs without crashing"), but GPU use
must be monitored while jobs run: `~/micromamba/envs/nanopulldown/bin/python
scripts/caius/gpu_report.py` on CAIUS (the login node's `python3` is 3.6 and
fails on it) (running jobs: device + latest util; finished: per-GPU stats + every
`gpu_idle_warning`). Check it whenever looking at the run. Attempt 1 =
`hpc.gpu_constraint` ("gpu|gpuh200": the H200 nodes lack the plain "gpu"
feature), retries = sm_80+ only.

## CAIUS backend gotchas found in the pilot (2026-11-17)

- **micromamba concurrency:** ABCfold calls `micromamba run` for every step;
  with dozens of jobs at once on the shared root they crash on each other's
  `~/.cache/mamba/proc/*.json` ("parse error ... empty input") or env locks
  ("Could not set lock"). `run_abcfold.py` gives each job its own
  `XDG_CACHE_HOME` + `MAMBA_ROOT_PREFIX` (envs/ symlinked to the shared ones)
  and `MAMBA_USE_LOCKFILES=false`. Keep that if touching the job wrapper.
- **Env fixes ABCfold's installers miss** (in `prime_backends.sh`): Chai-1 env
  needs `pyarrow==19.0.1` (MSA -> .aligned.pqt; numpy 1.26 there); RF3 env needs
  `atomworks[ml]==2.2.1` (3.x dropped `AtomSelectionStack`); Protenix needs its
  CUDA kernel prebuilt (`build_protenix_kernel.sh`); Boltz-2 weights and the
  Protenix v2 checkpoint (403 upstream; copy + sha256 check) are fetched by
  `fetch_weights.py`. ABCfold hides backend helper stderr (`quiet=True`): to
  debug, monkeypatch `abcfold.backend_envs.MicromambaEnv.run` to quiet=False.
- **`~/.abcfold_config.ini` race:** every `abcfold` call rewrites it in place
  (`verify_config_file`), so a concurrent job can read it truncated (`KeyError:
  'chai_env'`). `run_abcfold.py` passes each job a private copy via
  `--config-file`; edit the shared file (e.g. `protenix_model`) only between runs.
- `nvidia-smi` sees every GPU of a node: the sampler queries `-i <UUID>` (from
  the backend's torch in the GPU check), not the index.
- RTX PRO 6000 Blackwell node (sm12.0, feature `gpurtx6000`, matches "gpu"):
  the cu124/cu126 torch of Boltz-2 / Chai-1 / Protenix has no kernels for it.
  The GPU check runs a real CUDA op (`kernel_ok`) and exits 3 on it.
- CAIUS H200 nodes (hpc-n977-981) don't confine GPUs: jobs without a GPU
  request use the cards anyway (2026-11-19, another user's MACE runs) and our
  folds OOM mid-run on a 141 GB card. The GPU check fails a job whose card is
  >10 % used at start. hpc-n862 (V100) has torch cuda=False: black-hole node.
- Chai-1 runs one process per seed (~3 min/seed H200, ~5.5 A100): the 30 s
  GPU samples look idle-ish during its per-seed load; check its log progress
  ("Trunk recycles: 100%") before calling it hung.
- Only Boltz-2 runs below Ampere (full run, 2026-11-18): OpenFold3 / Protenix
  "no kernel image" on P100 (Protenix *hangs* at 0 % GPU), Chai-1 OOM on 16 GB,
  RF3 fmha kernel fails on RTX 6000 (sm7.5), Protenix OOM on 24 GB at ~640 aa.
  `common.BACKENDS[*]["min_cc"]` encodes it: the GPU check exits 3 below it, and
  the Snakefile asks the sm_80+ constraint from attempt 1 for those backends.
