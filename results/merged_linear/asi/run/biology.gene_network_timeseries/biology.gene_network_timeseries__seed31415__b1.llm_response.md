```python
import numpy as np
import pandas as pd
from scipy import stats
import json
import os

# Load data
expression_timeseries = np.load('data/expression_timeseries.npy')  # Shape (5, 50, 25)
perturbation_steady_states = np.load('data/perturbation_steady_states.npy')  # Shape (16, 25)

# Step 1: Compute perturbation differentials
wt_expression = perturbation_steady_states[0]
perturbation_diff = np.zeros((15, 25))
for r in range(1, 16):
    perturbed_expression = perturbation_steady_states[r]
    diff = np.abs(perturbed_expression - wt_expression) / (np.abs(wt_expression) + 1e-6)
    perturbation_diff[r-1, :] = diff

# Step 2: Attribute effects to candidate source sets
perturbation_signatures = {
    1: {'genes': [5, 11, 21, 23], 'type': 'regulatory_knockout'},
    2: {'genes': [3, 5], 'type': 'regulatory_knockout'},
    3: {'genes': [12, 14, 15, 18], 'type': 'overexpression'},
    4: {'genes': [1, 6, 17], 'type': 'overexpression'},
    5: {'genes': [5, 11, 14, 23], 'type': 'expression_knockout'},
    6: {'genes': [5, 6, 8, 21], 'type': 'regulatory_knockout'},
    7: {'genes': [6, 8, 11, 18], 'type': 'expression_knockout'},
    8: {'genes': [8, 9, 21], 'type': 'expression_knockout'},
    9: {'genes': [0, 10], 'type': 'expression_knockout'},
    10: {'genes': [8, 10, 12, 24], 'type': 'overexpression'},
    11: {'genes': [16, 18, 19], 'type': 'expression_knockout'},
    12: {'genes': [9, 14, 17], 'type': 'overexpression'},
    13: {'genes': [0, 3, 13, 14], 'type': 'overexpression'},
    14: {'genes': [0, 4, 16, 18], 'type': 'overexpression'},
    15: {'genes': [10, 12, 16], 'type': 'expression_knockout'}
}

edge_confidence = np.zeros((25, 25))
edge_sign = np.zeros((25, 25))

# Step 3: Cross-validate across rows
for r in range(1, 16):
    perturbed_genes = perturbation_signatures[r]['genes']
    perturbation_type = perturbation_signatures[r]['type']
    diff = perturbation_diff[r-1, :]
    for i in range(25):
        if diff[i] > 0.1:  # Threshold for significant effect
            for j in perturbed_genes:
                if j != i:
                    # Handle regulatory knockout (outgoing edges removed)
                    if perturbation_type == 'regulatory_knockout':
                        # Check if there is a known regulatory edge from j to i
                        if edge_confidence[j, i] > 0:
                            edge_confidence[j, i] += 1
                            # Determine sign based on effect direction
                            # Assume activation if effect is positive, inhibition if negative
                            edge_sign[j, i] = 1 if diff[i] > 0 else -1
                    elif perturbation_type == 'overexpression':
                        # Check if there is a known regulatory edge from j to i
                        if edge_confidence[j, i] > 0:
                            edge_confidence[j, i] += 1
                            # Assume activation
                            edge_sign[j, i] = 1
                    elif perturbation_type == 'expression_knockout':
                        # Check if there is a known regulatory edge from j to i
                        if edge_confidence[j, i] > 0:
                            edge_confidence[j, i] += 1
                            # Assume inhibition
                            edge_sign[j, i] = -1

# Step 4: Threshold and finalize network
mean_confidence = np.mean(edge_confidence)
std_confidence = np.std(edge_confidence)
threshold = mean_confidence + std_confidence
inferred_network = np.zeros((25, 25))
inferred_network[edge_confidence > threshold] = edge_confidence[edge_confidence > threshold]
inferred_network[inferred_network > 0] = 1.0

# Step 5: Identify hub genes
hub_gene_indices = np.argsort(np.sum(inferred_network, axis=1))[-2:]

# Step 6: Save outputs
np.save('inferred_network.npy', inferred_network)
np.save('edge_sign_predictions.npy', edge_sign)
json.dump({'hub_genes': hub_gene_indices.tolist()}, open('hub_genes.json', 'w'))
```