#!/usr/bin/env bash
# Second Terminal-Bench 2.0 run of base and merged_linear (repeat-run variance) on the x86 multi-LoRA server,
# identical settings to the first run. Starts only after the remaining first-run TB2 jobs finish, so the
# extra load cannot change their timeouts. Hard stop at $DEADLINE; a partial run is scored on the trials
# that completed and reported as such.
set -uo pipefail
cd /root/gpufree-data/projects/swe-eval
source /root/gpufree-data/tb2/env.sh
export PATH="/root/gpufree-data/tb2/bin:/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
export PYTHON=python MODEL_BASE_URL=http://127.0.0.1:30000/v1 TB_CONCURRENCY="${TB_CONCURRENCY:-12}"
DEADLINE="${DEADLINE:-2026-10-18T08:15:00+08:00}"

until [ -f results/expert_terminal/terminal/results.json ] && [ -f results/mopd/terminal/results.json ]; do sleep 60; done
echo "$(date -Is) first-run TB2 jobs done; starting TB2 repeat run"
LEFT=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) ))
for pair in base:base merged_linear:base:merged_linear; do
  M=${pair%%:*}; S=${pair#*:}
  ( RESULTS_DIR=results_rep2/$M SERVED_MODEL=$S timeout "$LEFT" bash scripts/timed_run.sh "terminal_rep2_$M" \
        bash scripts/bench_terminal_x86.sh "$M" > "logs/eval_rep2_${M}_terminal.log" 2>&1
    rc=$?
    pkill -f "results_rep2/$M/terminal/jobs" 2>/dev/null; sleep 20
    [ -f results_rep2/$M/terminal/results.json ] || python -m src.evaluation.harbor_results \
        --job_dir results_rep2/$M/terminal/jobs/tb2 --benchmark terminal_bench_2.0 --metric pass_rate \
        --output results_rep2/$M/terminal >> "logs/eval_rep2_${M}_terminal.log" 2>&1
    echo "$(date -Is) rep2 $M terminal exit $rc" ) &
done
wait
echo "$(date -Is) TB2 repeat run complete"
