#!/usr/bin/env bash
# expert_terminal's TB2 harbor process wrote its job result.json (05:52) but did not exit; kill it and score.
cd /root/gpufree-data/projects/swe-eval
export PATH="/root/gpufree-data/tb2/sglangenv/bin:$PWD/.venv-harness/bin:$PATH"
for p in $(ps -ww -eo pid,args | awk '/harbor run/ && /base:expert_terminal/ && !/awk/ {print $1}'); do kill $p; done
sleep 20
for p in $(ps -ww -eo pid,args | awk '/harbor run/ && /base:expert_terminal/ && !/awk/ {print $1}'); do kill -9 $p; done
python -m src.evaluation.harbor_results --job_dir results/expert_terminal/terminal/jobs/tb2 \
  --benchmark terminal_bench_2.0 --metric pass_rate --output results/expert_terminal/terminal
echo "$(date -Is) expert_terminal terminal scored manually"
