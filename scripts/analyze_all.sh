#!/usr/bin/env bash
set -euo pipefail

echo "Running all analyses..."

echo "Collecting results..."
python -m src.evaluation.collect_results --results_dir results --output results/summary.json

echo "Computing confidence intervals..."
python -m src.analysis.bootstrap_ci --results_dir results --output results/confidence_intervals.json

echo "Analyzing failure cases..."
python -m src.analysis.analyze_failures --results_dir results --output results/failure_analysis.json --max_samples 20

echo "Summarizing costs..."
python -m src.analysis.summarize_costs \
    --checkpoints_dir checkpoints \
    --results_dir results \
    --gpu_rate 1.5 \
    --output results/cost_summary.json

echo "Analysis complete!"
echo ""
echo "Results available in:"
echo "  - results/summary.json"
echo "  - results/summary.csv"
echo "  - results/confidence_intervals.json"
echo "  - results/failure_analysis.json"
echo "  - results/cost_summary.json"
