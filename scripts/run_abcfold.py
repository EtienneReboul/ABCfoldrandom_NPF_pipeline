"""
Step 2b: run_abcfold: one backend on one protein x MSA depth (GPU job), then
pack its output into the model store and delete ABCfold's files.

  0. GPU check: the backend's own torch (in its micromamba env) must see a
     CUDA device of at least the backend's min_cc (common.BACKENDS), else the job fails at once (exit 3) and SLURM retries it
     elsewhere, instead of ABCfold silently folding on CPU for hours
  1. unpack data/fold_inputs/<protein>/<depth>.json.xz into a scratch dir
     (node-local $TMPDIR by default: ABCfold's raw output is ~1-2 GB per job,
     mostly pretty-printed PAE/contact JSON, and never reaches the shared FS)
  2. abcfold <json> <scratch>/out -<flag> --number_of_models N --num_recycles R
       --no_server --no_visuals --override
  3. abcstore.pack(): models / pae / atoms .parquet in --out-dir, every
     encoding checked against ABCfold's files (see scripts/abcstore.py)
  4. run.json next to them (timings, GPU, sizes, round-trip errors), then the
     scratch dir is removed (also on failure, unless --keep-raw)

While ABCfold runs, nvidia-smi is sampled every --gpu-sample-s seconds
("[gpu] t=.. util=..% mem=.. MiB" lines in the job log); run.json gets
gpu_util_mean, gpu_busy_frac (samples with util > 5 %), gpu_mem_max_mib and
gpu_idle_warning (busy < 20 % of a fold longer than 5 min: the backend probably
ran on CPU or stalled). scripts/caius/gpu_report.py summarises them.

Private micromamba state: ABCfold calls `micromamba run -n <env>` for every
step (Chai-1 several times per job), and micromamba (1) records each running
process in ~/.cache/mamba/proc/ ($XDG_CACHE_HOME) and parses the others'
files, (2) locks the env it runs in. With dozens of jobs starting at once
(2026-11-17 pilot) one reads another's half-written proc file ("[json.
exception.parse_error.101] ... empty input"), or fails "Could not set lock",
and the backend dies. Each job therefore gets, in its scratch dir: its own
XDG_CACHE_HOME (nothing the backends need lives in ~/.cache: weights are in
~/.boltz, ~/.openfold3, ~/.rosettafold3 and the chai / protenix dirs) and its
own MAMBA_ROOT_PREFIX whose envs/ is a symlink to the shared envs; and
MAMBA_USE_LOCKFILES=false (the jobs only read the envs).

Private config file: every `abcfold` call rewrites ~/.abcfold_config.ini in
place (output/utils.py verify_config_file: open "w" then write), so a job
starting while another rewrites it reads a truncated file ("KeyError:
'chai_env'", 2026-11-17 pilot). Each job copies it into its scratch dir and
passes --config-file (the copy is rewritten by that job only).

GPU index: CAIUS GPU nodes don't hide the other cards from a job, so
nvidia-smi lists every GPU of the node; it is queried with
-i $CUDA_VISIBLE_DEVICES (the job's card) for the check and the samples.

ABCfold exits 0 even when a backend produced nothing (--no_visuals returns
before its own check): the model count check in pack() is what fails the job.
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import abcstore
from common import BACKENDS


TORCH_CHECK = r"""
import json, torch
ok = torch.cuda.is_available()
d = {'cuda': ok, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'device': '', 'capability': ''}
if ok:
    d['device'] = torch.cuda.get_device_name(0)
    d['capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
    try:  # nvidia-smi -i <uuid>: the job's own card whatever the index mapping
        u = str(torch.cuda.get_device_properties(0).uuid)
        d['uuid'] = u if u.startswith('GPU-') else 'GPU-' + u
    except Exception:
        pass
    try:  # a real kernel: torch builds without this card's arch fail here ("no kernel image")
        d['kernel_ok'] = float((torch.arange(8, device='cuda', dtype=torch.float32) * 2).sum().item()) == 56.0
    except Exception as e:
        d['kernel_ok'] = False
        d['kernel_error'] = str(e)[:300]
print(json.dumps(d))
"""


GPU_UUID = ""   # set by gpu_check from the backend's torch


def smi_index() -> list[str]:
    """nvidia-smi -i <the job's GPU>: its UUID once known, else CUDA_VISIBLE_DEVICES, [] when unknown."""
    if GPU_UUID:
        return ["-i", GPU_UUID]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return ["-i", cvd] if cvd else []


def private_mamba_root(work: Path) -> Path:
    """A per-job MAMBA_ROOT_PREFIX whose envs/ points at the shared envs (see module doc)."""
    shared = Path(os.environ.get("MAMBA_ROOT_PREFIX") or Path.home() / "micromamba")
    root = work / "mamba_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "envs").symlink_to(shared / "envs", target_is_directory=True)
    return root


def gpu_check(backend: str, envs: Path) -> dict:
    """nvidia-smi + the backend env's torch.cuda (its python called directly); raises if no usable GPU."""
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}
    try:
        smi = subprocess.run(["nvidia-smi", *smi_index(), "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        info["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"failed: {e}"
    env = BACKENDS[backend]["env"]
    r = subprocess.run([str(envs / env / "bin" / "python"), "-c", TORCH_CHECK],
                       capture_output=True, text=True, timeout=600)
    try:
        info.update(json.loads(r.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        info.update(cuda=False, torch_error=(r.stderr or r.stdout)[-2000:])
    print(f"[gpu] check {backend} ({env}): {json.dumps(info)}", flush=True)
    global GPU_UUID
    GPU_UUID = info.get("uuid", "")
    if not info.get("cuda"):
        raise SystemExit(3)
    if not info.get("kernel_ok"):
        # e.g. RTX PRO 6000 Blackwell (sm12.0) with the cu124 / cu126 torch of Boltz-2, Chai-1, Protenix
        print(f"[gpu] {backend}'s torch {info.get('torch')} has no kernels for {info.get('device')} "
              f"(sm{info.get('capability')}): failing now so the retry lands on another card", flush=True)
        raise SystemExit(3)
    # CAIUS H200 nodes don't confine GPUs: jobs that asked for no GPU (2026-11-19:
    # another user's MACE runs) sit on allocated cards and our fold OOMs mid-run.
    # A card already >10 % full before we start is taken: fail now, retry elsewhere.
    try:
        used, total = (float(x) for x in subprocess.run(
            ["nvidia-smi", *smi_index(), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0].split(","))
        info["mem_used_at_start_mib"] = used
        if used > 0.10 * total:
            print(f"[gpu] {used:.0f} of {total:.0f} MiB already used by other processes on this card: "
                  "failing now so the retry lands on a free one", flush=True)
            raise SystemExit(3)
    except (IndexError, ValueError, subprocess.SubprocessError):
        pass
    cc = tuple(int(x) for x in (info.get("capability") or "0.0").split("."))
    need = BACKENDS[backend]["min_cc"]
    if cc < tuple(int(x) for x in str(need).split(".")):
        print(f"[gpu] {info.get('device')} (sm{info.get('capability')}) is below {backend}'s minimum sm{need}: "
              "failing now so the retry lands on hpc.gpu_constraint_retry cards", flush=True)
        raise SystemExit(3)
    return info


class GpuSampler(threading.Thread):
    """nvidia-smi utilization / memory every `every` s while ABCfold runs."""

    def __init__(self, every: float):
        super().__init__(daemon=True)
        self.every, self.samples, self.stop = every, [], threading.Event()

    def run(self):
        t0 = time.time()
        while not self.stop.wait(self.every):
            try:
                out = subprocess.run(["nvidia-smi", *smi_index(), "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=30).stdout.strip().splitlines()[0]
                util, mem = (float(x) for x in out.split(","))
            except Exception:  # noqa: BLE001
                continue
            self.samples.append((util, mem))
            print(f"[gpu] t={(time.time() - t0) / 60:.1f}min util={util:.0f}% mem={mem:.0f}MiB", flush=True)

    def summary(self, fold_s: float) -> dict:
        if not self.samples:
            return {"gpu_samples": 0, "gpu_idle_warning": fold_s > 300}
        u = [x[0] for x in self.samples]
        busy = sum(x > 5 for x in u) / len(u)
        return {"gpu_samples": len(u), "gpu_util_mean": round(sum(u) / len(u), 1), "gpu_busy_frac": round(busy, 3),
                "gpu_mem_max_mib": max(x[1] for x in self.samples),
                "gpu_idle_warning": bool(fold_s > 300 and busy < 0.2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-input", type=Path, required=True)
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--number-of-models", type=int, default=5)
    ap.add_argument("--num-recycles", type=int, default=10)
    ap.add_argument("--pae-step", type=float, default=0.25)
    ap.add_argument("--zstd-level", type=int, default=19)
    ap.add_argument("--scratch", default="", help="parent of the raw output dir ('' = $TMPDIR)")
    ap.add_argument("--abcfold", default="abcfold")
    ap.add_argument("--keep-raw", action="store_true", help="keep ABCfold's output (debugging)")
    ap.add_argument("--pack-only", type=Path, default=None, help="skip ABCfold, pack this existing raw dir")
    ap.add_argument("--gpu-sample-s", type=float, default=30.0)
    a = ap.parse_args()

    with lzma.open(a.fold_input, "rt") as fh:
        job = json.load(fh)
    name, seeds = job["name"], [int(s) for s in job["modelSeeds"]]
    t0 = time.time()
    info = {"protein": a.protein, "depth": a.depth, "backend": a.backend, "host": socket.gethostname(),
            "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "slurm_restart": os.environ.get("SLURM_RESTART_COUNT", "0"),
            "n_seeds": len(seeds), "number_of_models": a.number_of_models, "num_recycles": a.num_recycles}
    work = None
    try:
        if a.pack_only:
            raw = a.pack_only
        else:
            parent = Path(a.scratch or os.environ.get("TMPDIR") or tempfile.gettempdir())
            parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f"abcfr_{name}_{a.backend}_", dir=parent))
            root = private_mamba_root(work)
            g = gpu_check(a.backend, (root / "envs").resolve())
            info.update(gpu=g.get("device", "?"), gpu_capability=g.get("capability", ""), torch=g.get("torch", ""),
                        torch_cuda=g.get("torch_cuda", ""))
            # inherited by ABCfold and every micromamba call it makes (see module doc)
            os.environ["MAMBA_ROOT_PREFIX"] = str(root)
            os.environ["XDG_CACHE_HOME"] = str(work / "xdg_cache")
            os.environ["MAMBA_USE_LOCKFILES"] = "false"
            inp = work / "input" / f"{name}.json"
            inp.parent.mkdir(parents=True)
            inp.write_text(json.dumps(job))
            cfg = work / "abcfold_config.ini"
            shared_cfg = Path.home() / ".abcfold_config.ini"
            if not shared_cfg.exists():
                raise SystemExit(f"{shared_cfg} missing: run scripts/caius/prime_backends.sh first")
            shutil.copy(shared_cfg, cfg)
            raw = work / "out"
            cmd = [a.abcfold, str(inp), str(raw), f"-{BACKENDS[a.backend]['flag']}", "--config-file", str(cfg),
                   "--number_of_models", str(a.number_of_models), "--num_recycles", str(a.num_recycles),
                   "--no_server", "--no_visuals", "--override"]
            print("[run_abcfold]", " ".join(cmd), flush=True)
            sampler = GpuSampler(a.gpu_sample_s)
            sampler.start()
            try:
                subprocess.run(cmd, check=True)
            finally:
                sampler.stop.set()
            info.update(sampler.summary(time.time() - t0))
            if info.get("gpu_idle_warning"):
                print(f"[gpu] WARNING: GPU busy in {info.get('gpu_busy_frac', 0):.0%} of the samples: "
                      "the backend probably ran on CPU or stalled", flush=True)
        info["fold_s"] = round(time.time() - t0, 1)
        info["raw_bytes"] = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file())

        t1 = time.time()
        stats = abcstore.pack(raw, a.backend, name, a.out_dir, protein=a.protein, depth=a.depth,
                              seeds=seeds, n_samples=a.number_of_models, pae_step=a.pae_step,
                              level=a.zstd_level)
        info.update(stats, pack_s=round(time.time() - t1, 1))
        info["compression_ratio"] = round(info["raw_bytes"] / max(stats["bytes_total"], 1), 1)
        (a.out_dir / "run.json").write_text(json.dumps(info, indent=1, default=str))
        print(f"[run_abcfold] {a.protein} {a.depth} {a.backend}: {stats['n_models']} models, fold "
              f"{info['fold_s'] / 60:.1f} min on {info.get('gpu', '?')} (busy {info.get('gpu_busy_frac', float('nan')):.0%}), raw {info['raw_bytes'] / 1e9:.2f} GB -> "
              f"{stats['bytes_total'] / 1e6:.1f} MB (x{info['compression_ratio']}), "
              f"max err coords {stats['coord_max_err']:.4f} A, PAE {stats['pae_max_err']:.3f} A", flush=True)
    finally:
        if work is not None:
            if a.keep_raw:
                print(f"[run_abcfold] raw output kept: {work}", flush=True)
            else:
                shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
