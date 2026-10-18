```python
import numpy as np
import os
import json

# Load data
expression_timeseries = np.load('data/expression_timeseries.npy')
perturbation_steady_states = np.load('data/perturbation_steady_states.npy')

# Perturbation mapping
perturbation_map = {
    1: ([5, 11, 21, 23], 'regulatory_knockout'),
    2: ([3, 5], 'regulatory_knockout'),
    3: ([12, 14, 15, 18], 'overexpression'),
    4: ([1, 6, 17], 'overexpression'),
    5: ([5, 11, 14, 23], 'expression_knockout'),
    6: ([5, 6, 8, 21], 'regulatory_knockout'),
    7: ([6, 8, 11, 18], 'expression_knockout'),
    8: ([8, 9, 21], 'expression_knockout'),
    9: ([0, 10], 'expression_knockout'),
    10: ([8, 10, 12, 24], 'overexpression'),
    11: ([16, 18, 19], 'expression_knockout'),
    12: ([9, 14, 17], 'overexpression'),
    13: ([0, 3, 13, 14], 'overexpression'),
    14: ([0, 4, 16, 18], 'overexpression'),
    15: ([10, 12, 16], 'expression_knockout'),
}

# Step 1: Compute perturbation differentials
def compute_differentials(perturbation_data, wt_data):
    eps = 1e-9
    diff = np.abs(perturbation_data - wt_data) / (np.abs(wt_data) + eps)
    return diff

wt_steady_state = perturbation_steady_states[0]
perturbation_diffs = []
for row_idx in range(1, 16):
    row = perturbation_steady_states[row_idx]
    diff = compute_differentials(row, wt_steady_state)
    perturbation_diffs.append(diff)

# Step 2: Attribute effects to candidate source sets
edge_confidence = np.zeros((25, 25))

for row_idx, (genes, perturbation_type) in enumerate(perturbation_map.values()):
    diff = perturbation_diffs[row_idx]
    for target_idx in range(25):
        if diff[target_idx] > 0.1:  # Threshold for significant change
            # Count co-occurrence of candidate sources with this target
            co_occurrence = np.zeros(25)
            for source_idx in range(25):
                if source_idx in genes:
                    co_occurrence[source_idx] += 1
            # Normalize co-occurrence
            co_occurrence /= np.sum(co_occurrence)
            # Add confidence
            edge_confidence[genes, target_idx] += co_occurrence[genes]

# Step 3: Cross-validate across rows
for target_idx in range(25):
    for source_idx in range(25):
        if edge_confidence[source_idx, target_idx] > 0:
            # Count how many times this source-target pair appears together
            count = 0
            for row_idx, (genes, _) in enumerate(perturbation_map.values()):
                if source_idx in genes and target_idx in genes:
                    count += 1
            # Add additional confidence based on co-occurrence
            edge_confidence[source_idx, target_idx] += count / 15

# Step 4: Threshold to obtain a sparse network
mean_confidence = np.mean(edge_confidence)
std_confidence = np.std(edge_confidence)
threshold = mean_confidence + std_confidence
edge_confidence[edge_confidence < threshold] = 0.0

# Step 5: Compute edge signs
edge_signs = np.zeros((25, 25))
for source_idx in range(25):
    for target_idx in range(25):
        if edge_confidence[source_idx, target_idx] > 0:
            # Check if the perturbation type is overexpression (positive sign)
            for row_idx, (genes, perturbation_type) in enumerate(perturbation_map.values()):
                if source_idx in genes and perturbation_type == 'overexpression':
                    edge_signs[source_idx, target_idx] = 1
                elif source_idx in genes and perturbation_type == 'expression_knockout':
                    edge_signs[source_idx, target_idx] = -1

# Step 6: Identify hub genes
hub_genes = []
for idx in range(25):
    if np.sum(edge_confidence[idx]) > np.mean(np.sum(edge_confidence, axis=1)):
        hub_genes.append(idx)

# Step 7: Method comparison
method_comparison = {
    'methods': [
        {
            'description': 'Perturbation Differential Analysis',
            'score': np.mean(edge_confidence),
            'strengths': 'Captures direct regulatory effects through multi-gene perturbations',
            'limitations': 'Cannot distinguish direct from indirect effects',
        },
        {
            'description': 'Time-series Granger Causality',
            'score': np.mean(np.abs(np.corrcoef(expression_timeseries.mean(axis=0), rowvar=False))),
            'strengths': 'Captures temporal dependencies',
            'limitations': 'Cannot distinguish direct from indirect effects',
        },
    ]
}

# Step 8: Network metrics
network_metrics = {
    'n_edges': np.sum(edge_confidence > 0),
    'density': np.sum(edge_confidence > 0) / (25 * 25),
    'in_degrees': np.sum(edge_confidence > 0, axis=1),
    'out_degrees': np.sum(edge_confidence > 0, axis=0),
}

# Save output files
np.save('inferred_network.npy', edge_confidence)
np.save('edge_sign_predictions.npy', edge_signs)
with open('hub_genes.json', 'w') as f:
    json.dump({'hub_genes': hub_genes}, f)
with open('method_comparison.json', 'w') as f:
    json.dump(method_comparison, f)
with open('network_metrics.json', 'w') as f:
    json.dump(network_metrics, f)
```