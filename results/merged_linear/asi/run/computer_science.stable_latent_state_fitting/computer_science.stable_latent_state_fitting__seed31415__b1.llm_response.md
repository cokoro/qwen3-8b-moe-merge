```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data
data_path = 'data/sequences.csv'
metadata_path = 'data/metadata.json'

# Parse metadata
with open(metadata_path) as f:
    metadata = pd.read_json(f)

# Parse sequences
sequences = pd.read_csv(data_path)
n_sequences = len(sequences)
n_symbols = metadata['symbols']['size']
n_positions = sequences['length'].max()

# Candidate state counts
candidate_counts = range(1, 16)

# Function to compute scaled log-likelihood
def compute_log_likelihood(A, B, pi, sequence):
    T = len(sequence)
    alpha = np.zeros((T, A.shape[0]))
    beta = np.zeros((T, A.shape[0]))
    c = np.zeros(T)
    xi = np.zeros((T-1, A.shape[0], A.shape[0]))
    
    # Forward pass
    alpha[0, :] = pi * B[:, sequence[0]]
    c[0] = np.sum(alpha[0, :])
    alpha[0, :] /= c[0]
    
    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, sequence[t]]
        c[t] = np.sum(alpha[t, :])
        alpha[t, :] /= c[t]
    
    # Backward pass
    beta[T-1, :] = 1.0
    for t in range(T-2, -1, -1):
        beta[t, :] = np.dot(A, np.dot(B[:, sequence[t+1]], beta[t+1, :])) / c[t+1]
    
    # Compute log-likelihood
    log_likelihood = np.sum(np.log(c))
    
    # Compute xi
    for t in range(T-1):
        xi[t, :, :] = np.outer(alpha[t, :], alpha[t+1, :]) * B[:, sequence[t+1]]
    
    return log_likelihood, alpha, beta, xi

# Function to fit HMM with EM algorithm
def fit_hmm(A, B, pi, sequence):
    T = len(sequence)
    gamma = np.zeros((T, A.shape[0]))
    xi = np.zeros((T-1, A.shape[0], A.shape[0]))
    log_likelihood = 0.0
    
    # Initialize alpha, beta
    alpha = np.zeros((T, A.shape[0]))
    beta = np.zeros((T, A.shape[0]))
    c = np.zeros(T)
    alpha[0, :] = pi * B[:, sequence[0]]
    c[0] = np.sum(alpha[0, :])
    alpha[0, :] /= c[0]
    
    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, sequence[t]]
        c[t] = np.sum(alpha[t, :])
        alpha[t, :] /= c[t]
    
    beta[T-1, :] = 1.0
    for t in range(T-2, -1, -1):
        beta[t, :] = np.dot(A, np.dot(B[:, sequence[t+1]], beta[t+1, :])) / c[t+1]
    
    # Compute gamma and xi
    for t in range(T):
        gamma[t, :] = alpha[t, :] * beta[t, :]
    for t in range(T-1):
        xi[t, :, :] = np.outer(alpha[t, :], alpha[t+1, :]) * B[:, sequence[t+1]]
    
    # Compute new parameters
    A_new = np.zeros((A.shape[0], A.shape[0]))
    B_new = np.zeros((A.shape[0], n_symbols))
    pi_new = np.zeros(A.shape[0])
    
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            A_new[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
    
    for i in range(A.shape[0]):
        for k in range(n_symbols):
            B_new[i, k] = np.sum(gamma[:, i][sequence == k]) / np.sum(gamma[:, i])
    
    pi_new = np.sum(gamma[0, :]) / np.sum(gamma[0, :])
    
    # Update log-likelihood
    log_likelihood = np.sum(np.log(c))
    
    return A_new, B_new, pi_new, log_likelihood, alpha, beta, gamma, xi

# Function to evaluate model
def evaluate_model(A, B, pi, sequence):
    T = len(sequence)
    gamma = np.zeros((T, A.shape[0]))
    log_likelihood = 0.0
    
    # Forward pass
    alpha = np.zeros((T, A.shape[0]))
    c = np.zeros(T)
    alpha[0, :] = pi * B[:, sequence[0]]
    c[0] = np.sum(alpha[0, :])
    alpha[0, :] /= c[0]
    
    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, sequence[t]]
        c[t] = np.sum(alpha[t, :])
        alpha[t, :] /= c[t]
    
    log_likelihood = np.sum(np.log(c))
    
    # Compute gamma
    for t in range(T):
        gamma[t, :] = alpha[t, :] * beta[t, :]
    
    return A, B, pi, log_likelihood, gamma

# Function to compute candidate summary
def compute_candidate_summary(candidates):
    candidate_summary = []
    for k in candidates:
        A = np.random.rand(k, k)
        A /= A.sum(axis=1, keepdims=True)
        B = np.random.rand(k, n_symbols)
        B /= B.sum(axis=1, keepdims=True)
        pi = np.random.rand(k)
        pi /= pi.sum()
        
        # Run EM
        A_new, B_new, pi_new, log_likelihood, alpha, beta, gamma, xi = fit_hmm(A, B, pi, sequences.iloc[0])
        
        # Compute criterion
        criterion = log_likelihood - 0.5 * np.log(n_positions) * (k**2 + k)
        
        candidate_summary.append({
            'candidate': k,
            'n_free_values': k**2 + k,
            'objective': log_likelihood,
            'criterion': criterion,
            'chosen': 0
        })
    
    return pd.DataFrame(candidate_summary)

# Function to compute matrix A
def compute_matrix_a(A):
    return pd.DataFrame({
        'row': np.repeat(np.arange(A.shape[0]), A.shape[0]),
        'column': np.tile(np.arange(A.shape[0]), A.shape[0]),
        'value': A.flatten()
    })

# Function to compute matrix B
def compute_matrix_b(B):
    symbols = metadata['symbols']['labels']
    return pd.DataFrame({
        'row': np.repeat(np.arange(B.shape[0]), n_symbols),
        'symbol': np.tile(np.arange(n_symbols), B.shape[0]),
        'symbol_label': symbols,
        'value': B.flatten()
    })

# Function to compute vector A
def compute_vector_a(pi):
    return pd.DataFrame({
        'row': np.arange(pi.shape[0]),
        'value': pi
    })

# Function to compute fit trace
def compute_fit_trace(A, B, pi, sequence):
    T = len(sequence)
    gamma = np.zeros((T, A.shape[0]))
    log_likelihood = 0.0
    
    # Forward pass
    alpha = np.zeros((T, A.shape[0]))
    c = np.zeros(T)
    alpha[0, :] = pi * B[:, sequence[0]]
    c[0] = np.sum(alpha[0, :])
    alpha[0, :] /= c[0]
    
    for t in range(1, T):
        alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, sequence[t]]
        c[t] = np.sum(alpha[t, :])
        alpha[t, :] /= c[t]
    
    log_likelihood = np.sum(np.log(c))
    
    # Compute gamma
    for t in range(T):
        gamma[t, :] = alpha[t, :] * beta[t, :]
    
    return A, B, pi, log_likelihood, gamma

# Function to compute local weights
def compute_local_weights(gamma):
    return gamma

# Function to compute neighbor summary
def compute_neighbor_summary(xi):
    return pd.DataFrame({
        'row': np.repeat(np.arange(xi.shape[0]), xi.shape[1]),
        'column': np.tile(np.arange(xi.shape[1]), xi.shape[0]),
        'weight': xi.flatten(),
        'joint_fraction': xi.flatten() / np.sum(xi),
        'conditional_value': xi.flatten() / np.sum(xi, axis=1)
    })

# Function to compute local assignments
def compute_local_assignments(gamma, sequences):
    assignments = []
    for i in range(len(sequences)):
        sequence = sequences.iloc[i]
        for t in range(len(sequence)):
            assignments.append({
                'sequence_id': i,
                'position': t,
                'global_index': i,
                'label': np.argmax(gamma[t, :])
            })
    return pd.DataFrame(assignments)

# Function to compute stream diagnostics
def compute_stream_diagnostics(gamma, sequences):
    diagnostics = []
    for i in range(len(sequences)):
        sequence = sequences.iloc[i]
        n_positions = len(sequence)
        mean_objective = np.mean(gamma[i, :])
        weight_entropy_mean = -np.sum(gamma[i, :] * np.log(gamma[i, :]))
        dominant_label_fraction = np.max(gamma[i, :])
        diagnostics.append({
            'sequence_id': i,
            'n_positions': n_positions,
            'mean_objective': mean_objective,
            'weight_entropy_mean': weight_entropy_mean,
            'dominant_label_fraction': dominant_label_fraction
        })
    return pd.DataFrame(diagnostics)

# Function to plot diagnostics
def plot_diagnostics(log_likelihoods, A, B, gamma, sequences):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(log_likelihoods)
    plt.title('Fit Trace')
    
    plt.subplot(2, 2, 2)
    plt.imshow(B, cmap='viridis')
    plt.colorbar()
    plt.title('Emission Matrix')
    
    plt.subplot(2, 2, 3)
    plt.hist(gamma.flatten(), bins=100)
    plt.title('Component Occupancy')
    
    plt.subplot(2, 2, 4)
    plt.imshow(gamma, cmap='viridis')
    plt.colorbar()
    plt.title('Assignment Uncertainty')
    
    plt.tight_layout()
    plt.savefig('results/diagnostics.png')
    plt.close()

# Main analysis pipeline
def main():
    # Parse metadata
    with open(metadata_path) as f:
        metadata = pd.read_json(f)
    
    # Parse sequences
    sequences = pd.read_csv(data_path)
    n_sequences = len(sequences)
    n_symbols = metadata['symbols']['size']
    n_positions = sequences['length'].max()
    
    # Initialize candidate summary
    candidate_summary = []
    
    # Evaluate each candidate state count
    for k in candidate_counts:
        # Initialize A, B, pi
        A = np.random.rand(k, k)
        A /= A.sum(axis=1, keepdims=True)
        B = np.random.rand(k, n_symbols)
        B /= B.sum(axis=1, keepdims=True)
        pi = np.random.rand(k)
        pi /= pi.sum()
        
        # Run EM
        A_new, B_new, pi_new, log_likelihood, alpha, beta, gamma, xi = fit_hmm(A, B, pi, sequences.iloc[0])
        
        # Compute criterion
        criterion = log_likelihood - 0.5 * np.log(n_positions) * (k**2 + k)
        
        # Append to candidate summary
        candidate_summary.append({
            'candidate': k,
            'n_free_values': k**2 + k,
            'objective': log_likelihood,
            'criterion': criterion,
            'chosen': 0
        })
    
    # Convert to DataFrame
    candidate_summary = pd.DataFrame(candidate_summary)
    
    # Select the best candidate
    best_candidate = candidate_summary.loc[candidate_summary['criterion'].idxmax()]
    
    # Write candidate summary
    candidate_summary.to_csv('results/candidate_summary.csv', index=False)
    
    # Write matrix A
    matrix_a = compute_matrix_a(best_candidate['A'])
    matrix_a.to_csv('results/matrix_a.csv', index=False)
    
    # Write matrix B
    matrix_b = compute_matrix_b(best_candidate['B'])
    matrix_b.to_csv('results/matrix_b.csv', index=False)
    
    # Write vector A
    vector_a = compute_vector_a(best_candidate['pi'])
    vector_a.to_csv('results/vector_a.csv', index=False)
    
    # Run EM for the chosen model
    A = np.random.rand(best_candidate['k'], best_candidate['k'])
    A /= A.sum(axis=1, keepdims=True)
    B = np.random.rand(best_candidate['k'], n_symbols)
    B /= B.sum(axis=1, keepdims=True)
    pi = np.random.rand(best_candidate['k'])
    pi /= pi.sum()
    
    # Run EM
    A_new, B_new, pi_new, log_likelihood, alpha, beta, gamma, xi = fit_hmm(A, B, pi, sequences.iloc[0])
    
    # Compute fit trace
    fit_trace = compute_fit_trace(A_new, B_new, pi_new, sequences.iloc[0])
    fit_trace.to_csv('results/fit_trace.csv', index=False)
    
    # Compute local weights
    local_weights = compute_local_weights(gamma)
    np.save('results/local_weights.npy', local_weights)
    
    # Compute neighbor summary
    neighbor_summary = compute_neighbor_summary(xi)
    neighbor_summary.to_csv('results/neighbor_summary.csv', index=False)
    
    # Compute local assignments
    local_assignments = compute_local_assignments(gamma, sequences)
    local_assignments.to_csv('results/local_assignments.csv', index=False)
    
    # Compute stream diagnostics
    stream_diagnostics = compute_stream_diagnostics(gamma, sequences)
    stream_diagnostics.to_csv('results/stream_diagnostics.csv', index=False)
    
    # Plot diagnostics
    plot_diagnostics(log_likelihoods, A_new, B_new, gamma, sequences)

if __name__ == '__main__':
    main()
```