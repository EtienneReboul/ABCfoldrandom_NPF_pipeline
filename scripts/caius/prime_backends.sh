#!/usr/bin/env bash
# Build ABCfold's five backend micromamba envs and fetch every weight file, ON
# THE LOGIN NODE (internet; compute nodes are offline). Called by setup.sh;
# safe to rerun (ABCfold skips what is installed, downloads skip what exists).
#
#   1. abcfold --dry_run, one backend at a time: envs abcfold-{boltz,chai,openfold,protenix}-py311
#      + abcfold-rosetta-py312 under $MAMBA_ROOT_PREFIX, Boltz weights
#      (~/.boltz), OpenFold3 (~/.openfold3) and RF3 (~/.rosettafold3)
#      checkpoints, and a --help smoke test of each backend; Protenix needs
#      its CUDA kernel built first (build_protenix_kernel.sh)
#   2. Chai-1 and Protenix weights: ABCfold's dry run only runs --help for
#      those two, so their weights would be fetched by the first GPU job
#      (offline): scripts/caius/fetch_weights.py downloads them here
set -euo pipefail
cd "$(dirname "$0")/../.."
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
export TMPDIR=${TMPDIR:-$HOME/tmp/abcfr_prime}
mkdir -p "$TMPDIR"
cfg() { python3 -c "import yaml,sys; c={}; [c.update(yaml.safe_load(open(f)) or {}) for f in ('config.yaml','config.local.yaml')]; print(c['abcfold'].get('$1') or '')"; }
AB=""
for d in .snakemake/conda/*/; do [ -x "$d/bin/abcfold" ] && AB=${d%/}; done
[ -n "$AB" ] || { echo "no env with abcfold under .snakemake/conda (run setup.sh)"; exit 1; }
export LD_LIBRARY_PATH="$PWD/$AB/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$AB/bin/abcfold" --help >/dev/null
echo "abcfold $("$AB/bin/python" -c 'import importlib.metadata as m; print(m.version("abcfold"))') from $AB"

json=$(ls data/fold_inputs/*/max1_ext2.json.xz | head -1)
xz -dc "$json" > "$TMPDIR/prime.json"
prime() { "$AB/bin/abcfold" "$TMPDIR/prime.json" "$TMPDIR/prime_out" "$1" --dry_run --override; }
prime -b
prime -c
# Protenix: the env is created, then its smoke test fails until the fused
# layer-norm CUDA kernel is built (no nvcc / GCC 8.5 on CAIUS):
prime -p || echo "(protenix smoke test failed as expected before the kernel build)"
scripts/caius/build_protenix_kernel.sh
prime -p
prime -o
prime -r
# Env fixes ABCfold 1.1.6's installers miss (2026-11-17 pilot):
#  - Chai-1: ABCfold converts the MSA to Chai's .aligned.pqt inside the chai
#    env with pandas.to_parquet, but the env has no parquet engine (pyarrow
#    19.0.1 = ABCfold's own pin; the chai env keeps numpy 1.26)
#  - RF3: rc-foundry 0.1.12 accepts any atomworks >= 2.1.1, pip takes 3.1.0,
#    which dropped atomworks.io.utils.selection.AtomSelectionStack (rf3 needs it;
#    2.2.1 is the last release that has it)
M=${MICROMAMBA:-$HOME/.local/bin/micromamba}
"$M" run -n abcfold-chai-py311 python -m pip install -q "pyarrow==19.0.1"   # chai env has numpy 1.26; newer pyarrow needs numpy 2
"$M" run -n abcfold-rosetta-py312 python -m pip install -q "atomworks[ml]==2.2.1"
"$M" run -n abcfold-chai-py311 python -c "import pyarrow; print('chai env pyarrow', pyarrow.__version__)"
"$M" run -n abcfold-rosetta-py312 python -c "from rf3.inference_engines.rf3 import RF3InferenceEngine; print('RF3 engine import OK')"
rm -rf "$TMPDIR/prime_out" "$TMPDIR/prime.json"

CHAI_DOWNLOADS_DIR=$(cfg chai_downloads_dir) PROTENIX_ROOT_DIR=$(cfg protenix_root_dir) \
  "$AB/bin/python" scripts/caius/fetch_weights.py
ls -la ~/.boltz ~/.openfold3 ~/.rosettafold3 2>/dev/null | head -30
