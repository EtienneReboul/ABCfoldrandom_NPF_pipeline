"""
Fetch the Boltz-2, Chai-1 and Protenix weights ON THE LOGIN NODE (called by
prime_backends.sh). ABCfold's --dry_run only runs `--help` for these three
backends (it does fetch the OpenFold3 and RF3 checkpoints), so their weights
would otherwise be downloaded by the first GPU job, on an offline compute node. Each download runs inside the backend's own
micromamba env with the backend's own download code, into the directories
the GPU jobs are pointed at (config.local.yaml abcfold.chai_downloads_dir /
protenix_root_dir, exported by the Snakefile's run_abcfold rule):

  Boltz-2 (boltz 2.2.1)    ~/.boltz (ABCfold's default cache: boltz.main.download_boltz2)
  Chai-1 (chai_lab 0.6.1)  $CHAI_DOWNLOADS_DIR/models_v2/{feature_embedding,
                           bond_loss_input_proj,token_embedder,trunk,
                           diffusion_module,confidence_head}.pt,
                           conformers_v1.apkl, esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt
  Protenix (2.0.0)         $PROTENIX_ROOT_DIR/checkpoint/protenix-v2.pt and the
                           CCD / PDB-cluster data cache, via the same
                           download_inference_cache() `protenix predict` calls

protenix-v2.pt: the official URL (protenix.tos-cn-beijing.volces.com, a
Chinese object store) answers 403 to everyone since 2026; ABCfold 1.1.6 asks
for that model. Put the checkpoint in place by hand first, from Hugging Face
or the IFB copy the ABCfold NPF run used (~/checkpoint/protenix-v2.pt on IFB,
copied here 2026-11-17); this script then only checks its sha256 (the
published digest, PROTENIX_V2_SHA256) and fetches the small data cache.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROTENIX_V2_SHA256 = "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
PROTENIX_V2_BYTES = 1859785497
MICROMAMBA = shutil.which("micromamba") or str(Path.home() / ".local/bin/micromamba")

BOLTZ = r'''
from pathlib import Path
from boltz.main import download_boltz2
cache = Path("~/.boltz").expanduser()
cache.mkdir(parents=True, exist_ok=True)
download_boltz2(cache)
print("boltz", sorted(p.name for p in cache.iterdir()), flush=True)
'''

CHAI = r'''
from chai_lab.utils.paths import chai1_component, cached_conformers, downloads_path, download_if_not_exists
from chai_lab.data.dataset.embeddings.esm import ESM_URL
for k in ("feature_embedding.pt", "bond_loss_input_proj.pt", "token_embedder.pt", "trunk.pt",
          "diffusion_module.pt", "confidence_head.pt"):
    print("chai", chai1_component(k), flush=True)
print("chai", cached_conformers.get_path(), flush=True)
esm = downloads_path / "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
download_if_not_exists(ESM_URL, esm)
print("chai", esm, flush=True)
'''

PROTENIX = r'''
import runner.batch_inference as bi
from protenix.config.config import parse_configs
from ml_collections.config_dict import ConfigDict
name = "{model}"
bi.inference_configs["model_name"] = name
cfg = parse_configs(configs={{**bi.configs_base, "data": bi.data_configs, **bi.inference_configs}},
                    fill_required_with_null=True)
cfg.update(ConfigDict(bi.model_configs[name]))
cfg.use_template = False
bi.download_inference_cache(cfg)
print("protenix", cfg.load_checkpoint_dir, flush=True)
'''


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(env: str, code: str, extra_env: dict) -> None:
    print(f"== {env}", flush=True)
    subprocess.run([MICROMAMBA, "run", "-n", env, "python", "-c", code],
                   env={**os.environ, **extra_env}, check=True)


def main() -> int:
    chai_dir = os.environ.get("CHAI_DOWNLOADS_DIR")
    prot_dir = os.environ.get("PROTENIX_ROOT_DIR")
    if not chai_dir or not prot_dir:
        sys.exit("set CHAI_DOWNLOADS_DIR and PROTENIX_ROOT_DIR (config.local.yaml abcfold:)")
    Path(chai_dir).mkdir(parents=True, exist_ok=True)
    Path(prot_dir).mkdir(parents=True, exist_ok=True)
    run("abcfold-boltz-py311", BOLTZ, {})
    run("abcfold-chai-py311", CHAI, {"CHAI_DOWNLOADS_DIR": chai_dir})
    ckpt = Path(prot_dir) / "checkpoint/protenix-v2.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} missing: copy protenix-v2.pt there (Hugging Face, or IFB ~/checkpoint/), see docstring")
    if ckpt.stat().st_size != PROTENIX_V2_BYTES or sha256(ckpt) != PROTENIX_V2_SHA256:
        sys.exit(f"{ckpt}: size / sha256 differ from the published protenix-v2 checkpoint")
    print(f"protenix-v2.pt sha256 OK ({PROTENIX_V2_SHA256[:12]}...)", flush=True)
    run("abcfold-protenix-py311", PROTENIX.format(model=os.environ.get("PROTENIX_MODEL", "protenix-v2")),
        {"PROTENIX_ROOT_DIR": prot_dir})
    for d in (chai_dir, prot_dir):
        files = [f for f in Path(d).rglob("*") if f.is_file()]
        print(f"{d}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
