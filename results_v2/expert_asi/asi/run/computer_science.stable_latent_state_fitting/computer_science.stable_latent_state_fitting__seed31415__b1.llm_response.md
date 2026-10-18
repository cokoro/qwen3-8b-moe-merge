```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Load data and metadata
sequences = pd.read_csv('data/sequences.csv')
metadata = json.load(open('data/metadata.json'))
n_symbols = metadata['n_symbols']
n_sequences = len(sequences)
n_positions = sequences['n_positions'].max()

# Define helper functions
def forward_backward(seq, K, B, A, n_positions):
    # Implement forward-backward with scaling
    pass

def compute_posteriors(seq, K, B, A, n_positions):
    # Compute gamma, xi, and log-likelihood
    pass

def evaluate_candidate(K):
    # Evaluate a candidate state count using multiple criteria
    pass

# Main analysis
candidates = [3, 5, 10, 15, 20]
results = []
for K in candidates:
    best_params = evaluate_candidate(K)
    results.append({
        'candidate': K,
        'n_free_values': K*(K-1) + K,
        'objective': best_params['log_likelihood'],
        'criterion': 'weighted_likelihood_complexity',
        'chosen': 1 if K == best_params['selected'] else 0
    })

# Write outputs
os.makedirs('results', exist_ok=True)
pd.DataFrame(results).to_csv('results/candidate_summary.csv', index=False)
```