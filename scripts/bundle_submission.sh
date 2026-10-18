#!/usr/bin/env bash
# Pack the generated artefacts of the DGX checkout for the submission repository:
#   /tmp/submission_artifacts.tar.gz  results/ (per-task files, no harbor trial dirs), results_rep2/, analysis/,
#                                     curves/, logs/ (= logs_submission/: per-run_id logs + resources.csv)
#   /tmp/harbor_jobs.tar.gz           full harbor trial directories (trajectories, recordings, patches, verifier
#                                     output) of every TB2/SWE run, incl. the base TB2 job dirs on the x86 host
set -euo pipefail
cd ~/moe-merge
rm -rf /tmp/sub && mkdir -p /tmp/sub
rsync -a --exclude 'jobs/' --exclude '_smoke/' results results_rep2 analysis curves /tmp/sub/
rsync -a logs_submission/ /tmp/sub/logs/
mkdir -p /tmp/sub/data && for d in swe hle asi terminal; do mkdir -p /tmp/sub/data/$d; cp data/$d/stats.json /tmp/sub/data/$d/ 2>/dev/null || true; done
tar -C /tmp/sub -czf /tmp/submission_artifacts.tar.gz .
tar -czf /tmp/harbor_jobs.tar.gz $(find results results_rep2 -type d \( -path '*/terminal/jobs' -o -path '*/swe/jobs' \) -prune) \
  results/_validation $( [ -d x86/base_tb2_jobs ] && echo x86/base_tb2_jobs )
ls -la /tmp/submission_artifacts.tar.gz /tmp/harbor_jobs.tar.gz
sha256sum /tmp/harbor_jobs.tar.gz
