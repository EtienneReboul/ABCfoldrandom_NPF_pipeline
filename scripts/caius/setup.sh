#!/usr/bin/env bash
# One-time setup on CAIUS, ON THE LOGIN NODE (niced; compute nodes are offline):
#
#   nice -n 19 scripts/caius/setup.sh
#
# 1. config.local.yaml from config.caius.yaml.example (if absent)
# 2. the rule conda envs (envs/abcfold.yaml, envs/analysis.yaml) into
#    .snakemake/conda/, with the NanoPulldown controller env (snakemake 9 +
#    SLURM plugin)
# 3. scripts/caius/prime_backends.sh: ABCfold's five backend micromamba envs
#    + every weight file (ABCfold's own dry run does not fetch Chai-1's or
#    Protenix's: they would be downloaded by the first GPU job, offline)
# /tmp on hpc-login has <1 GB free, so TMPDIR and caches go under $HOME.
set -euo pipefail
cd "$(dirname "$0")/../.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
CTRL=${CTRL:-$MAMBA_ROOT_PREFIX/envs/nanopulldown}
export PATH=$CTRL/bin:$PATH
export TMPDIR=$HOME/tmp/abcfr_setup PIP_CACHE_DIR=$HOME/tmp/abcfr_setup/pip CONDA_CHANNEL_PRIORITY=strict
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "host $(hostname), controller $(snakemake --version)"

[ -e config.local.yaml ] || { cp config.caius.yaml.example config.local.yaml; echo "wrote config.local.yaml"; }
snakemake --use-conda --conda-create-envs-only --cores 2
scripts/caius/prime_backends.sh
diskquota | tail -2
