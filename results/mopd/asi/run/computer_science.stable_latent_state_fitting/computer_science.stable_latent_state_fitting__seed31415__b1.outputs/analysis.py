import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import os
import json
import time

# Load data
metadata = json.load(open('data/metadata.json'))
sequences = pd.read_csv('data/sequences.csv')
symbols = sorted(set(sequences['symbol']))
K_min = 1
K_max = 32
N = len(sequences)
N_positions = len(sequences.iloc[0]['sequence'])

# Initialize results directory
os.makedirs('results', exist_ok=True)

# Candidate evaluation
candidate_summary = []
for K in range(K_min, K_max + 1):
    # Run multiple restarts
    best_loglik = -np.inf
    best_A = None
    best_B = None
    best_pi = None
    best_gamma = None
    best_xi = None
    best_loglik_trace = []
    for restart in range(5):
        # Initialize A, B, pi
        A = np.random.rand(K, K)
        A = A / A.sum(axis=1, keepdims=True)
        B = np.random.rand(K, len(symbols))
        B = B / B.sum(axis=1, keepdims=True)
        pi = np.random.rand(K)
        pi = pi / pi.sum()
        
        # EM
        loglik_trace = []
        for step in range(200):
            # Forward
            alpha = np.zeros((N_positions, K))
            alpha[0] = pi * B[:, sequences.iloc[0]['sequence'][0]]
            for t in range(1, N_positions):
                alpha[t] = logsumexp(alpha[t-1][:, np.newaxis] @ A, axis=0)
                alpha[t] = alpha[t] * B[:, sequences.iloc[t]['sequence'][t]]
                alpha[t] = alpha[t] / alpha[t].sum()
            loglik = alpha.sum()
            loglik_trace.append(loglik)
            
            # Backward
            beta = np.zeros((N_positions, K))
            beta[-1] = np.ones(K)
            for t in range(N_positions-2, -1, -1):
                beta[t] = (alpha[t] * A) @ beta[t+1]
                beta[t] = beta[t] / beta[t].sum()
            
            # Gamma
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            # Xi
            xi = np.zeros((N_positions-1, K, K))
            for t in range(N_positions-1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, sequences.iloc[t+1]['sequence'][t+1]] * beta[t+1, j]
            xi = xi / xi.sum(axis=(1,2), keepdims=True)
            
            # Update A, B, pi
            A = np.zeros((K, K))
            B = np.zeros((K, len(symbols)))
            pi = np.zeros(K)
            for i in range(K):
                for j in range(K):
                    A[i, j] = xi[:, i, j].sum()
                A[i] = A[i] / A[i].sum()
            for i in range(K):
                for s in range(len(symbols)):
                    B[i, s] = gamma[:, i][np.where(sequences['sequence'] == s)].sum()
                B[i] = B[i] / B[i].sum()
            pi = gamma[0, :] / gamma[0, :].sum()
            
            # Log likelihood
            loglik = np.log(gamma.sum(axis=1)).sum()
            loglik_trace.append(loglik)
            if loglik > best_loglik:
                best_loglik = loglik
                best_A = A
                best_B = B
                best_pi = pi
                best_gamma = gamma
                best_xi = xi
                best_loglik_trace = loglik_trace
    candidate_summary.append({
        'candidate': K,
        'n_free_values': K * (K-1) + K * len(symbols),
        'objective': best_loglik,
        'criterion': 'log-likelihood',
        'chosen': 0
    })
candidate_summary = pd.DataFrame(candidate_summary)
candidate_summary.to_csv('results/candidate_summary.csv', index=False)

# Select best candidate
best_K = candidate_summary[candidate_summary['criterion'] == 'log-likelihood']['candidate'].iloc[0]
best_loglik = candidate_summary[candidate_summary['criterion'] == 'log-likelihood']['objective'].iloc[0]
best_A = best_A
best_B = best_B
best_pi = best_pi
best_gamma = best_gamma
best_xi = best_xi
best_loglik_trace = best_loglik_trace

# Save matrix A
matrix_A = []
for i in range(best_K):
    for j in range(best_K):
        matrix_A.append({'row': i, 'column': j, 'value': best_A[i,j]})
pd.DataFrame(matrix_A).to_csv('results/matrix_a.csv', index=False)

# Save matrix B
matrix_B = []
for i in range(best_K):
    for s in range(len(symbols)):
        matrix_B.append({'row': i, 'symbol': symbols[s], 'symbol_label': s, 'value': best_B[i,s]})
pd.DataFrame(matrix_B).to_csv('results/matrix_b.csv', index=False)

# Save vector A
vector_a = []
for i in range(best_K):
    vector_a.append({'row': i, 'value': best_pi[i]})
pd.DataFrame(vector_a).to_csv('results/vector_a.csv', index=False)

# Save fit trace
fit_trace = []
for run in range(5):
    for step, loglik in enumerate(best_loglik_trace):
        fit_trace.append({'run': run, 'step': step, 'objective': loglik, 'relative_change': np.nan})
pd.DataFrame(fit_trace).to_csv('results/fit_trace.csv', index=False)

# Save local weights
np.save('results/local_weights.npy', best_gamma)

# Save neighbor summary
neighbor_summary = []
for t in range(N_positions-1):
    for i in range(best_K):
        for j in range(best_K):
            neighbor_summary.append({
                'row': t,
                'column': j,
                'weight': best_xi[t, i, j],
                'joint_fraction': best_xi[t, i, j] / best_xi[t, i, j].sum(),
                'conditional_value': best_xi[t, i, j] / best_gamma[t, i]
            })
pd.DataFrame(neighbor_summary).to_csv('results/neighbor_summary.csv', index=False)

# Save local assignments
local_assignments = []
for seq_id in range(N):
    for pos in range(N_positions):
        for k in range(best_K):
            if best_gamma[pos, k] > 0.5:
                local_assignments.append({
                    'sequence_id': seq_id,
                    'position': pos,
                    'global_index': seq_id,
                    'label': k
                })
pd.DataFrame(local_assignments).to_csv('results/local_assignments.csv', index=False)

# Save stream diagnostics
stream_diagnostics = []
for seq_id in range(N):
    mean_objective = np.log(best_gamma[seq_id, :].sum())
    weight_entropy_mean = -np.sum(best_gamma[seq_id, :] * np.log(best_gamma[seq_id, :]))
    dominant_label_fraction = np.max(best_gamma[seq_id, :])
    stream_diagnostics.append({
        'sequence_id': seq_id,
        'n_positions': N_positions,
        'mean_objective': mean_objective,
        'weight_entropy_mean': weight_entropy_mean,
        'dominant_label_fraction': dominant_label_fraction
    })
pd.DataFrame(stream_diagnostics).to_csv('results/stream_diagnostics.csv', index=False)

# Plot diagnostics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(best_loglik_trace)
plt.title('Fit Trace')
plt.subplot(2, 2, 2)
plt.imshow(best_B, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Emission Matrix')
plt.subplot(2, 2, 3)
plt.plot(best_pi)
plt.title('Initial State Distribution')
plt.subplot(2, 2, 4)
plt.plot(np.sum(best_gamma, axis=1))
plt.title('Component Occupancy')
plt.tight_layout()
plt.savefig('results/diagnostics.png')
plt.close()