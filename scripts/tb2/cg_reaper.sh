#!/bin/bash
# Kill + remove harbor trial cgroups whose wrapper process is gone (e.g. wrapper got SIGKILL).
mkdir -p /sys/fs/cgroup/harbor
grep -q memory /sys/fs/cgroup/harbor/cgroup.subtree_control || echo "+memory +pids +cpu" > /sys/fs/cgroup/harbor/cgroup.subtree_control
while true; do
  for d in /sys/fs/cgroup/harbor/trial-*; do
    [ -d "$d" ] || continue
    pid=${d##*trial-}
    if ! kill -0 "$pid" 2>/dev/null; then
      echo 1 > "$d/cgroup.kill" 2>/dev/null; sleep 1; rmdir "$d" 2>/dev/null && echo "$(date +%T) reaped $d"
    fi
  done
  sleep 20
done
