#!/usr/bin/env bash
# Time-boxed evaluation plan on the DGX Spark: HLE + ASI only.
# Terminal-Bench 2.0 and SWE-bench Pro ship prebuilt linux/amd64 images; under qemu on
# this aarch64 host their verifiers crash (oracle solutions score 0), so both run on the
# x86 host (scripts/run_x86_plan.sh). Priority follows exam.md 5.1: base and both merged
# models on every benchmark, each expert on its own benchmark.
set -uo pipefail
mkdir -p logs

until grep -q "after_training done" logs/after_training.log 2>/dev/null; do
  pgrep -f "scripts/after_training.sh" >/dev/null || { echo "after_training.sh not running and not done" >&2; exit 1; }
  sleep 60
done
echo "$(date -Is) training/merge done"

export MEM_FRACTION="${MEM_FRACTION:-0.4}"
PLAN="${PLAN_A:-base:hle,asi merged_ties:hle,asi expert_hle:hle}" PORT=30000 \
  bash scripts/evaluate_all.sh > logs/lane_a.log 2>&1 &
A=$!
sleep 120  # stagger model loading
PLAN="${PLAN_B:-merged_linear:hle,asi expert_asi:asi}" PORT=30001 \
  bash scripts/evaluate_all.sh > logs/lane_b.log 2>&1 &
B=$!
wait "$A" "$B"
echo "$(date -Is) all lanes done"
