# Terminal-Bench 2.0 on the L40S box

Official benchmark (`harborframework/terminal-bench-2.0`, 89 tasks, commit `f2e8c75`; byte-identical to
`third_party/terminal-bench` on the DGX), run with [Harbor](https://github.com/harbor-framework/harbor) 0.24.0 and
the reference agent **terminus-2**. Agent and serving settings are the same as `scripts/bench_terminal.sh` /
`scripts/serve_model.sh`, so numbers are comparable with the DGX pipeline; only the sandbox differs.

## Host

| item | value |
|---|---|
| GPU | 1× NVIDIA L40S 48GB (driver 580, CUDA 13.0) |
| CPU / RAM / disk | 224 cores / 1 TB / 147 GB at `/root/gpufree-data` |
| container | k8s pod: no usable Docker, no CAP_NET_ADMIN, masked `/proc`, no `/dev/net/tun`, no `/dev/kvm` |
| network | pypi.org, github.com, huggingface.co, Docker Hub, raw.githubusercontent.com **blocked**; international bandwidth is slow |
| workdir | `/root/gpufree-data/tb2` (scripts, logs, `jobs/`, `results/`) |
| model | `models/Qwen3-8B` rev `b968826d` (sha256 identical to the DGX copy) |
| serving | sglang 0.5.21 (torch 2.13), `configs/chat_templates/qwen3_nothink.jinja`, context 32768, seed 42 |
| agent | terminus-2, `openai/Qwen3-8B`, temperature 0.7, `max_input_tokens=28672`, `max_output_tokens=4096`, 1 attempt, task-default timeouts |

## Sandbox: Harbor `singularity` environment + shim

dockerd starts here but containers fail (`operation not permitted`: a nested PID namespace cannot mount procfs in
this pod). Harbor's built-in **singularity** environment (Apptainer 1.5.4, unprivileged user namespaces) is used
instead, through `bin/singularity`, which harbor finds first on `PATH`:

| problem | fix |
|---|---|
| new PID namespace refused | `--containall --pid` → `--contain --cleanenv --ipc` |
| no PID ns ⇒ container root could signal vLLM/harbor | containers run as unprivileged `tbsandbox` (uid 2000; fakeroot root ↦ 2000) |
| no PID ns ⇒ killed containers leave orphans | one cgroup per trial (`/sys/fs/cgroup/harbor/trial-<pid>`), `cgroup.kill` on exit, `pids.max=8192`; `cg_reaper.sh` for SIGKILLed wrappers |
| `--no-mount bind-paths` empties `/etc/resolv.conf` | bind host resolv.conf read-only |

Harbor itself needs 4 small patches (`harbor_patches.sh`): server wait 60→300 s, local `get-pip.py`, keep the
image `PATH` (upstream prepends `/usr/bin`, which hides `/usr/local/bin/python3` in python:* images), reuse the
image's `python3` instead of apt-installing a second one.

## Network stand-ins (all on 127.0.0.1, containers share the host network)

| port | service | replaces |
|---|---|---|
| 3142 | apt-cacher-ng → tuna (ubuntu, debian, debian-security) | archive.ubuntu.com / deb.debian.org (too slow for the bootstrap) |
| 18080 | `mirror_server.py`: uv release tarballs (built from the PyPI wheels on demand), python-build-standalone (local copies, else 302 → npmmirror), `get-pip.py` (pip 25.0.1 wheel; venv-only sysconfig switch for distutils-less python<3.10) | github.com releases, bootstrap.pypa.io |
| 18081 | `hf_proxy.py` → hf-mirror.com, rewrites `huggingface.co` in `Link`/`Location` | huggingface.co (hf-mirror's pagination links point back to huggingface.co) |
| 18082 | nginx caching proxy → tuna PyPI (`nginx-pypi-cache.conf`) | pypi.org (large wheels such as CUDA torch fetched once) |
| — | `GIT_CONFIG_*` env: `url.https://githubfast.com/.insteadOf=https://github.com/` | github.com git clones / archives |
| — | `docker_image` → `docker.1ms.run/alexgshaw/...` in `datasets/tb2-mirror` (only change to the tasks) | Docker Hub |

## Tasks that interfere with each other

Without per-container network/PID namespaces, containers share host ports and can see (and `pkill`) each other's
processes. `lanes.sh` puts the 17 affected tasks into 3 serial lanes (conflicting tasks always share a lane: :8080
group + qemu pair + install-windows' `pgrep qemu-system`; :22/:8000/:5328 + the chrome pair hit by
filter-js-from-html's `pkill -f chrome`; the :29500 torch.distributed pair + other single-port tasks); `run_split.sh`
runs them next to the other 72 tasks. Tasks needing ports < 1024 (sshd :22, smtp :25, :80) cannot pass here: the
pod's sshd owns :22 and container root is an unprivileged host uid.

## Known differences from the official docker environment

- **No CPU limit**: harbor's singularity backend enforces neither `cpus` nor `memory` (docker does both). Memory is
  approximated by harbor's watchdog (kill at 95 % of the task limit); CPU is unlimited (84 tasks specify `cpus = 1`).
  This can only make tasks easier/faster.
- Shared hostname (`gpufree-container`), host PID/network namespaces, mirror env vars visible to the agent
  (`PIP_INDEX_URL`, `UV_*`, `HF_ENDPOINT`, `GIT_CONFIG_*`), apt proxy config file in `/etc/apt/apt.conf.d/`.
- Identical: task files (only `docker_image` rewritten; 50/50 image manifest digests checked equal to Docker Hub via
  the DGX), harbor verifier code path, task timeouts.

## Run

```bash
ssh -p <port> root@<x86-host>
cd /root/gpufree-data/tb2
bash start_services.sh                    # model :30000 + mirrors/caches + cgroup reaper
bash harbor_patches.sh                    # re-apply after any harbor reinstall
bash prepull.sh                           # docker image -> SIF cache (/root/gpufree-data/sif-cache, 44 GB)

bash run_split.sh oracle oracle 12        # reference solutions = infra ceiling on this host
bash run_split.sh qwen3-8b-base model 8   # terminus-2 + Qwen3-8B (8+3 agents fit the KV cache)
python3 tb2_summary.py --model-jobs jobs/qwen3-8b-base-* --oracle-jobs jobs/oracle-final-* --output results/qwen3-8b-base
python3 tb2_diagnose.py --jobs jobs/qwen3-8b-base-* --output results/qwen3-8b-base/diagnostics.jsonl  # loops, parse errors, context
bash resume_split.sh qwen3-8b-base r2 11  # after an interruption: rerun only unfinished tasks
```

Other checkpoints: `bash serve_sglang.sh checkpoints/<dir> <name>` (tmux session `model`), then
`MODEL_NAME=<name> bash run_split.sh <name> model 12`.
