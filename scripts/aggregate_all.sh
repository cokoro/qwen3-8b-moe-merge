#!/usr/bin/env bash
# Final aggregation on the DGX checkout: pull the x86 host's TB2/SWE results + logs, merge them into
# results/<model>/ (and results_rep2/), then regenerate every table, CI, failure list, curve and log index.
#   bash scripts/aggregate_all.sh            # pull + rebuild
#   NO_PULL=1 bash scripts/aggregate_all.sh  # rebuild from what is already in x86/
set -uo pipefail
cd ~/moe-merge
source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
X86="${X86:?set X86=user@x86-host}"; X86_PORT="${X86_PORT:?set X86_PORT}"; KEY="${KEY:-$HOME/.ssh/l40s_transfer}"
SRC=/root/gpufree-data/projects/swe-eval

if [ -z "${NO_PULL:-}" ]; then
  mkdir -p x86
  for d in results logs results_rep2; do
    rsync -a -e "ssh -i $KEY -p $X86_PORT -o BatchMode=yes" "$X86:$SRC/$d" x86/ 2>/dev/null || echo "pull $d: not present / partial"
  done
  # base TB2 run 1 was executed from the host's shared tb2 workspace (same serve_model.sh stack)
  mkdir -p x86/base_tb2_jobs
  rsync -a -e "ssh -i $KEY -p $X86_PORT -o BatchMode=yes" "$X86:/root/gpufree-data/tb2/jobs/qwen3-8b-base-main" \
        "$X86:/root/gpufree-data/tb2/jobs/qwen3-8b-base-r2-main" x86/base_tb2_jobs/ 2>/dev/null || echo "pull base tb2 jobs: partial"
fi
# x86 owns terminal/ and swe/ (and the SWE oracle validation); DGX owns hle/ and asi/
for root in results results_rep2; do
  [ -d "x86/$root" ] || continue
  for md in x86/$root/*/; do
    m=$(basename "$md")
    for b in terminal swe; do
      [ -d "$md/$b" ] && mkdir -p "$root/$m" && rsync -a "$md/$b" "$root/$m/"
    done
  done
done
cp x86/results/experiment_times.csv x86/experiment_times_x86.csv 2>/dev/null || true

python scripts/summarize.py --results results --out_csv results/summary.csv --out_json analysis/uncertainty.json
python scripts/analyze_failures.py --results results --models merged_linear merged_ties mopd --out analysis
python scripts/plot_curves.py --checkpoints checkpoints --out curves
python scripts/collect_logs.py --x86 x86 --out logs_submission

python scripts/render_results.py
echo "$(date -Is) aggregation done"
