#!/bin/bash
# Bring up everything Terminal-Bench 2.0 needs on the GPU box (idempotent; each service in its own tmux session).
#   model   : sglang OpenAI-compatible server on 127.0.0.1:30000 (serve_sglang.sh -> scripts/serve_model.sh)
#   mirror  : 127.0.0.1:18080 stand-in for GitHub releases (uv tarballs built from PyPI wheels, python builds -> npmmirror)
#   acng    : apt-cacher-ng on 127.0.0.1:3142 (upstream: tuna ubuntu/debian)
#   hfproxy : 127.0.0.1:18081 -> hf-mirror.com, rewrites huggingface.co pagination links (hf_proxy.py)
#   pypi    : 127.0.0.1:18082 nginx caching proxy -> tuna PyPI (nginx-pypi-cache.conf)
#   reaper  : kills cgroups of trials whose wrapper died (cg_reaper.sh)
T=/root/gpufree-data/tb2
cd "$T"

# cgroup2 is mounted read-only in the pod; per-trial cgroups need it writable.
mount -o remount,rw /sys/fs/cgroup
if [ ! -d /sys/fs/cgroup/init ]; then
  mkdir -p /sys/fs/cgroup/init
  xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || :
fi
grep -q memory /sys/fs/cgroup/cgroup.subtree_control || echo "+memory +pids +cpu" > /sys/fs/cgroup/cgroup.subtree_control
mkdir -p /sys/fs/cgroup/harbor
grep -q memory /sys/fs/cgroup/harbor/cgroup.subtree_control || echo "+memory +pids +cpu" > /sys/fs/cgroup/harbor/cgroup.subtree_control

up() { tmux has-session -t "$1" 2>/dev/null && echo "$1: already running" || { tmux new-session -d -s "$1" "$2"; echo "$1: started"; }; }

up model  "bash serve_sglang.sh 2>&1 | tee model_server.log"
up mirror "python3 $T/mirror_server.py 2>&1 | tee mirror.log"
up acng   "/usr/sbin/apt-cacher-ng -c /etc/apt-cacher-ng ForeGround=1 2>&1 | tee acng.log"
up hfproxy "python3 $T/hf_proxy.py 2>&1 | tee hf_proxy.log"
up reaper "bash cg_reaper.sh 2>&1 | tee reaper.log"
[ -f nginx-pypi.pid ] && kill -0 "$(cat nginx-pypi.pid)" 2>/dev/null && echo "pypi: already running" || { mkdir -p /root/gpufree-data/pypi-cache; nginx -c $T/nginx-pypi-cache.conf && echo "pypi: started"; }

for i in $(seq 1 60); do curl -sf 127.0.0.1:30000/v1/models >/dev/null && break; sleep 5; done
for p in 30000/v1/models 18080/ 18081/api/models/Qwen/Qwen3-8B 18082/simple/pip/ 3142/acng-report.html; do
  printf "%-28s %s\n" "127.0.0.1:$p" "$(curl -s -o /dev/null -w %{http_code} 127.0.0.1:$p)"
done
