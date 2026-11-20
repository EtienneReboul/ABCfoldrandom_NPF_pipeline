#!/usr/bin/env bash
# Build Protenix's fused layer-norm CUDA extension ONCE, on the login node.
#
# protenix (2.0.0) JIT-compiles `fast_layer_norm_cuda_v2` at import time and
# keeps the .so in its package dir. On CAIUS that build fails everywhere:
# no nvcc (only a cuda/13.0 module; the env's torch is cu126) and GCC 8.5
# (torch needs >= 9) - so even `protenix pred --help` (ABCfold's smoke test)
# dies. Fix: cuda-nvcc + cudart / cuBLAS / cuSPARSE / cuSOLVER / cuRAND / NVTX headers
# (torch's ATen CUDA headers include them all) 12.6 + gxx 12 from conda-forge into the
# protenix env, a CUDA_HOME shim (bin, include, lib -> targets/x86_64-linux,
# the layout torch.utils.cpp_extension expects), then one import builds the
# kernel (sm_70/80/86/89/90; sm_75 Turing runs the sm_70 code; P100 sm_60 has
# none and fails over to the sm_80+ retry). GPU jobs then import the prebuilt .so.
set -euo pipefail
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
M=${MICROMAMBA:-$HOME/.local/bin/micromamba}
ENV=abcfold-protenix-py311
P=$MAMBA_ROOT_PREFIX/envs/$ENV
CUDA_VER=$("$M" run -n $ENV python -c 'import torch; print(torch.version.cuda)')
echo "protenix env torch CUDA $CUDA_VER"
"$M" install -y -n $ENV -c conda-forge "cuda-version=$CUDA_VER" cuda-nvcc cuda-cudart-dev \
  libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev cuda-nvtx-dev cuda-profiler-api "gxx_linux-64=12"
SHIM=$P/cuda_home_shim
mkdir -p "$SHIM"
T=$(ls -d "$P"/targets/*-linux | head -1)
ln -sfn "$P/bin" "$SHIM/bin"
ln -sfn "$T/include" "$SHIM/include"
ln -sfn "$T/lib" "$SHIM/lib"
ln -sfn "$T/lib" "$SHIM/lib64"
export CUDA_HOME=$SHIM
export CC=$P/bin/x86_64-conda-linux-gnu-gcc CXX=$P/bin/x86_64-conda-linux-gnu-g++
export PATH=$P/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-4}
"$M" run -n $ENV python -c 'import protenix.model.layer_norm.layer_norm as m; print("built:", m.fast_layer_norm_cuda_v2.__file__)'
"$M" run -n $ENV protenix pred --help > /dev/null && echo "protenix pred --help OK"
