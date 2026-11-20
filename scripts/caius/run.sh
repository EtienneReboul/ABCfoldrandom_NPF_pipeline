#!/usr/bin/env bash
# Launch (or resume) the run on CAIUS from the login node, inside tmux:
#
#   export TMUX_TMPDIR=~/.tmux-sock   # hpc-login1's /tmp can be full
#   tmux new -d -s abcfr 'scripts/caius/run.sh 2>&1 | tee -a logs/caius_controller.log'
#
# Everything goes to SLURM via profiles/caius (fold inputs come pre-built in
# data/fold_inputs/ from the Mac). Extra args are passed to snakemake, e.g.
#   scripts/caius/run.sh --config proteins='[NPF2.10_Q944G5]'    # a pilot
# Do NOT start this under CUDA_VISIBLE_DEVICES="" / thread caps: sbatch
# exports the controller's environment to every job.
set -euo pipefail
cd "$(dirname "$0")/../.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
export PATH=${CTRL:-$MAMBA_ROOT_PREFIX/envs/nanopulldown}/bin:$PATH
export CONDA_CHANNEL_PRIORITY=strict
export TMPDIR=$HOME/tmp/abcfr_controller
mkdir -p "$TMPDIR" logs
echo "== $(date): ABCfold-random NPF on SLURM (profiles/caius)"
rc=0
snakemake --profile profiles/caius "$@" || rc=$?
echo "== $(date): snakemake exit $rc"
exit $rc
