import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from collections import defaultdict

# Load data and metadata
sequences_path = 'data/sequences.csv'
metadata_path = 'data/metadata.json'

sequences = pd.read_csv(sequences_path)
metadata = json.load(open(metadata_path))

# Constants
num_symbols = 32
num_sequences = len(sequences)
sequence_lengths = sequences['length'].values
sequence_ids = sequences['sequence_id'].values
global_indices = sequences['global_index'].values

# Define function to compute scaled forward-backward recursions
def compute_scaled_forward_backward(observations, K):
    T = len(observations)
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    xi = np.zeros((T-1, K, K))
    gamma = np.zeros((T, K))
    log_likelihood = 0.0

    # Initialize alpha
    alpha[0, :] = np.log(1.0 / K) + np.log(metadata['initial_distribution'])
    alpha_raw = np.exp(alpha)
    c_0 = np.sum(alpha_raw[0, :])
    alpha[0, :] -= np.log(c_0)

    # Forward pass
    for t in range(1, T):
        alpha_raw_t = np.zeros(K)
        for i in range(K):
            alpha_raw_t[i] = np.sum(alpha[t-1, :] + np.log(metadata['transition_matrix'][i, :])) + np.log(metadata['emission_matrix'][i, observations[t]])
        c_t = np.sum(alpha_raw_t)
        alpha[t, :] = alpha_raw_t - np.log(c_t)
        log_likelihood += np.log(c_t)

    # Backward pass
    beta[T-1, :] = np.log(1.0 / K)
    for t in range(T-2, -1, -1):
        beta_raw_t = np.zeros(K)
        for j in range(K):
            beta_raw_t[j] = np.sum(alpha[t, :] + np.log(metadata['transition_matrix'][:, j])) + np.log(metadata['emission_matrix'][j, observations[t+1]]) + beta[t+1, j]
        c_t1 = np.sum(alpha_raw[t+1])
        beta[t, :] = beta_raw_t - np.log(c_t1)

    # Compute gamma and xi
    for t in range(T):
        gamma[t, :] = alpha[t, :] + beta[t, :]
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = alpha[t, i] + np.log(metadata['transition_matrix'][i, j]) + np.log(metadata['emission_matrix'][j, observations[t+1]]) + beta[t+1, j]
                xi[t, i, j] -= np.log(np.sum(np.exp(alpha[t, :] + np.log(metadata['transition_matrix'][:, j]) + np.log(metadata['emission_matrix'][j, observations[t+1]]) + beta[t+1, j])))

    # Normalize gamma and xi
    for t in range(T):
        gamma[t, :] -= np.log(np.sum(np.exp(gamma[t, :])))

    return alpha, beta, gamma, xi, log_likelihood

# Define function to compute objective for a given K
def objective_function(K, sequences, num_symbols):
    total_objective = 0.0
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        total_objective += log_likelihood
    return total_objective

# Define function to compare candidates
def compare_candidates(candidates, sequences, num_symbols):
    results = []
    for K in candidates:
        # Run multiple restarts
        restarts = 5
        objectives = []
        for _ in range(restarts):
            # Initial guess for transition matrix
            A = np.random.rand(K, K)
            A /= A.sum(axis=1, keepdims=True)
            # Initial guess for emission matrix
            B = np.random.rand(K, num_symbols)
            B /= B.sum(axis=1, keepdims=True)
            # Initial guess for initial distribution
            pi = np.random.rand(K)
            pi /= pi.sum()
            # Run EM
            for _ in range(100):
                # E-step
                alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(sequences, K)
                # M-step
                # Compute new transition matrix
                A_new = np.zeros((K, K))
                for i in range(K):
                    for j in range(K):
                        A_new[i, j] = np.sum(xi[:, i, j])
                A_new /= A_new.sum(axis=1, keepdims=True)
                # Compute new emission matrix
                B_new = np.zeros((K, num_symbols))
                for i in range(K):
                    for s in range(num_symbols):
                        B_new[i, s] = np.sum(gamma[:, i] * (observations == s))
                B_new /= B_new.sum(axis=1, keepdims=True)
                # Compute new initial distribution
                pi_new = np.zeros(K)
                for i in range(K):
                    pi_new[i] = np.sum(gamma[0, i])
                pi_new /= pi_new.sum()
                # Update parameters
                A, B, pi = A_new, B_new, pi_new
                # Compute objective
                objectives.append(log_likelihood)
        # Compute average objective
        avg_objective = np.mean(objectives)
        # Compute underfit and overfit diagnostics
        underfit = False
        overfit = False
        # Check for underfit evidence
        if np.any(np.abs(np.sum(gamma, axis=1) - np.ones_like(gamma)) > 1e-3):
            underfit = True
        # Check for overfit evidence
        if np.any(np.sum(gamma, axis=0) < 1e-3):
            overfit = True
        # Compute criterion
        criterion = avg_objective - 0.5 * np.log(K) * num_sequences
        results.append({
            'candidate': K,
            'n_free_values': K * (K + num_symbols) + K,
            'objective': avg_objective,
            'criterion': criterion,
            'chosen': 0
        })
    return results

# Define function to compute local weights
def compute_local_weights(sequences, K, A, B, pi):
    local_weights = []
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        local_weights.append(gamma)
    return np.stack(local_weights)

# Define function to compute neighbor summary
def compute_neighbor_summary(sequences, K, A, B, pi):
    neighbor_summary = []
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        for t in range(len(seq_obs) - 1):
            for i in range(K):
                for j in range(K):
                    weight = xi[t, i, j]
                    joint_fraction = np.sum(gamma[t, i] * gamma[t+1, j]) / np.sum(gamma[t, i] + gamma[t+1, j])
                    conditional_value = np.sum(gamma[t, i] * np.log(A[i, j])) + np.sum(gamma[t+1, j] * np.log(B[j, seq_obs[t+1]]))
                    neighbor_summary.append({
                        'row': i,
                        'column': j,
                        'weight': weight,
                        'joint_fraction': joint_fraction,
                        'conditional_value': conditional_value
                    })
    return neighbor_summary

# Define function to compute local assignments
def compute_local_assignments(sequences, K, A, B, pi):
    local_assignments = []
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        viterbi = np.zeros(len(seq_obs), dtype=int)
        viterbi[0] = np.argmax(gamma[0, :])
        for t in range(1, len(seq_obs)):
            viterbi[t] = np.argmax(gamma[t, :] + np.log(A[viterbi[t-1], :]))
        for t in range(len(seq_obs)):
            local_assignments.append({
                'sequence_id': seq_id,
                'position': t,
                'global_index': global_indices[seq_id],
                'label': viterbi[t]
            })
    return local_assignments

# Define function to compute stream diagnostics
def compute_stream_diagnostics(sequences, K, A, B, pi):
    diagnostics = []
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        mean_objective = log_likelihood
        weight_entropy_mean = -np.mean(gamma * np.log(gamma + 1e-12))
        dominant_label_fraction = np.max(gamma[0, :])
        diagnostics.append({
            'sequence_id': seq_id,
            'n_positions': len(seq_obs),
            'mean_objective': mean_objective,
            'weight_entropy_mean': weight_entropy_mean,
            'dominant_label_fraction': dominant_label_fraction
        })
    return diagnostics

# Define function to compute fit trace
def compute_fit_trace(sequences, K, A, B, pi):
    fit_trace = []
    for seq_id in range(num_sequences):
        seq_obs = sequences[seq_id]
        alpha, beta, gamma, xi, log_likelihood = compute_scaled_forward_backward(seq_obs, K)
        for t in range(len(seq_obs)):
            fit_trace.append({
                'run': 0,
                'step': t,
                'objective': log_likelihood,
                'relative_change': 0.0
            })
    return fit_trace

# Define function to compute matrix A
def compute_matrix_A(sequences, K, A, B, pi):
    matrix_A = []
    for i in range(K):
        for j in range(K):
            matrix_A.append({
                'row': i,
                'column': j,
                'value': A[i, j]
            })
    return matrix_A

# Define function to compute matrix B
def compute_matrix_B(sequences, K, A, B, pi):
    matrix_B = []
    for i in range(K):
        for s in range(num_symbols):
            matrix_B.append({
                'row': i,
                'symbol': s,
                'symbol_label': metadata['symbol_labels'][s],
                'value': B[i, s]
            })
    return matrix_B

# Define function to compute vector A
def compute_vector_A(sequences, K, A, B, pi):
    vector_A = []
    for i in range(K):
        vector_A.append({
            'row': i,
            'value': pi[i]
        })
    return vector_A

# Define function to compute diagnostics plot
def compute_diagnostics_plot(sequences, K, A, B, pi):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fit_trace)
    plt.title('Fit Trace')
    plt.subplot(2, 2, 2)
    plt.imshow(B, cmap='viridis', aspect='auto')
    plt.title('Emission Matrix')
    plt.subplot(2, 2, 3)
    plt.bar(range(K), np.sum(gamma, axis=0))
    plt.title('Component Occupancy')
    plt.subplot(2, 2, 4)
    plt.imshow(np.log(gamma), cmap='viridis', aspect='auto')
    plt.title('Assignment Uncertainty')
    plt.savefig('results/diagnostics.png')
    plt.close()

# Main function
def main():
    # Load sequences
    sequences = sequences.values
    # Generate candidates
    candidates = range(2, 20)
    # Compare candidates
    results = compare_candidates(candidates, sequences, num_symbols)
    # Select best candidate
    best_candidate = max(results, key=lambda x: x['criterion'])
    best_candidate['chosen'] = 1
    # Save candidate summary
    pd.DataFrame(results).to_csv('results/candidate_summary.csv', index=False)
    # Compute local weights
    local_weights = compute_local_weights(sequences, best_candidate['candidate'], A, B, pi)
    np.save('results/local_weights.npy', local_weights)
    # Compute neighbor summary
    neighbor_summary = compute_neighbor_summary(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(neighbor_summary).to_csv('results/neighbor_summary.csv', index=False)
    # Compute local assignments
    local_assignments = compute_local_assignments(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(local_assignments).to_csv('results/local_assignments.csv', index=False)
    # Compute stream diagnostics
    stream_diagnostics = compute_stream_diagnostics(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(stream_diagnostics).to_csv('results/stream_diagnostics.csv', index=False)
    # Compute fit trace
    fit_trace = compute_fit_trace(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(fit_trace).to_csv('results/fit_trace.csv', index=False)
    # Compute matrix A
    matrix_A = compute_matrix_A(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(matrix_A).to_csv('results/matrix_a.csv', index=False)
    # Compute matrix B
    matrix_B = compute_matrix_B(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(matrix_B).to_csv('results/matrix_b.csv', index=False)
    # Compute vector A
    vector_A = compute_vector_A(sequences, best_candidate['candidate'], A, B, pi)
    pd.DataFrame(vector_A).to_csv('results/vector_a.csv', index=False)
    # Compute diagnostics plot
    compute_diagnostics_plot(sequences, best_candidate['candidate'], A, B, pi)

if __name__ == '__main__':
    main()