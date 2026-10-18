#!/usr/bin/env bash
# Second HLE run (repeat-run variance): same 500 questions, same settings; sampling is
# stochastic (server-side Qwen3 generation_config), so run-to-run spread is measurable.
cd ~/moe-merge; source ~/miniconda3/etc/profile.d/conda.sh; conda activate moe-merge
export MODEL_BASE_URL=http://127.0.0.1:30001/v1 HLE_SKIP_JUDGE=1
model_dir() { case "$1" in base) echo models/Qwen3-8B;; mopd) echo checkpoints/full/expert_mopd;; *) echo checkpoints/$1;; esac; }
for M in base merged_linear mopd; do
  R=results_rep2/$M; [ -f $R/hle/hle_$M.json ] && continue
  MEM_FRACTION=0.4 bash scripts/serve_model.sh "$(model_dir $M)" "$M" 30001 > logs/serve_rep2_$M.log 2>&1 &
  S=$!
  for _ in $(seq 1 240); do curl -sf $MODEL_BASE_URL/models | grep -q "\"$M\"" && break; sleep 5; done
  RESULTS_DIR=$R bash scripts/bench_hle.sh $M > logs/hle_rep2_$M.log 2>&1 && echo "$(date -Is) rep2 $M generated" || echo "$(date -Is) rep2 $M FAILED"
  kill $S; wait $S 2>/dev/null; sleep 15
done
echo "$(date -Is) rep2 done"
