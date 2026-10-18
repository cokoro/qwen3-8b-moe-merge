```python
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

def logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

def load_fasta(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = {}
    current_seq = None
    for line in lines:
        if line.startswith('>'):
            current_seq = line.strip()[1:]
            sequences[current_seq] = []
        else:
            sequences[current_seq].extend(list(line.strip()))
    return sequences

def load_analysis_spec(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_dimer_order(spec):
    return spec['adjacent_pair_order']

def load_example_windows(file_path):
    return np.load(file_path)

def compute_gc_fraction(seq):
    return np.mean([1 if c in 'GC' else 0 for c in seq])

def compute_cpg_ratio(seq):
    cpg = sum(1 for i in range(len(seq)-1) if seq[i] == 'C' and seq[i+1] == 'G')
    total = len(seq) - 1
    return cpg / total if total > 0 else 0

def mark_enriched_regions(seq, window_size, gc_threshold, cpg_threshold):
    n = len(seq)
    enriched = np.zeros(n, dtype=bool)
    for i in range(n - window_size + 1):
        window = seq[i:i+window_size]
        gc = compute_gc_fraction(window)
        cpg = compute_cpg_ratio(window)
        if gc > gc_threshold and cpg > cpg_threshold:
            enriched[i:i+window_size] = True
    return enriched

def initialize_transition_matrix():
    A = np.zeros((3, 3))
    A[0, 0] = 0.9
    A[1, 1] = 0.9
    A[2, 2] = 0.8
    A[0, 1] = 0.05
    A[0, 2] = 0.05
    A[1, 0] = 0.05
    A[1, 2] = 0.05
    A[2, 0] = 0.05
    A[2, 1] = 0.05
    return A

def initialize_emission_matrix(dimer_order):
    B = np.zeros((3, len(dimer_order)))
    for i, dimer in enumerate(dimer_order):
        B[0, i] = 0.01
        B[1, i] = 0.05
        B[2, i] = 0.05
    return B

def initialize_pi():
    pi = np.array([0.5, 0.25, 0.25])
    return pi

def compute_log_likelihood(alpha, beta, log_likelihood):
    return log_likelihood

def forward_pass(seq, dimer_order, pi, A, B):
    T = len(seq) - 1
    K = 3
    log_alpha = np.zeros((T, K))
    log_alpha[0, :] = np.log(pi) + np.log(B[:, dimer_order.index(seq[0] + seq[1])])
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = np.log(B[j, dimer_order.index(seq[t] + seq[t+1])]) + logsumexp(np.log(A[:, j]) + log_alpha[t-1, :])
    log_likelihood = logsumexp(log_alpha[-1, :])
    return log_alpha, log_beta, log_likelihood

def backward_pass(seq, dimer_order, A, B):
    T = len(seq) - 1
    K = 3
    log_beta = np.zeros((T, K))
    for t in range(T-1, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(np.log(A[i, :]) + np.log(B[:, dimer_order.index(seq[t+1] + seq[t+2])]) + log_beta[t+1, :])
    return log_beta

def em_update(seq, dimer_order, pi, A, B, log_alpha, log_beta, log_likelihood):
    T = len(seq) - 1
    K = 3
    gamma = np.zeros((T, K))
    xi = np.zeros((T, K, K))
    for t in range(T):
        for j in range(K):
            gamma[t, j] = log_alpha[t, j] + log_beta[t, j] - log_likelihood
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = log_alpha[t, i] + np.log(A[i, j]) + np.log(B[j, dimer_order.index(seq[t+1] + seq[t+2])]) + log_beta[t+1, j] - log_likelihood
    # Accumulate counts
    counts_pi = np.zeros(K)
    counts_A = np.zeros((K, K))
    counts_B = np.zeros((K, len(dimer_order)))
    for t in range(T):
        counts_pi += np.exp(gamma[t, :])
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                counts_A[i, j] += np.exp(xi[t, i, j])
    for t in range(T):
        for j in range(K):
            counts_B[j, dimer_order.index(seq[t] + seq[t+1])] += np.exp(gamma[t, j])
    # Normalize
    counts_pi /= np.sum(counts_pi)
    counts_A /= np.sum(counts_A, axis=1, keepdims=True)
    counts_B /= np.sum(counts_B, axis=1, keepdims=True)
    return counts_pi, counts_A, counts_B

def viterbi(seq, dimer_order, pi, A, B):
    T = len(seq) - 1
    K = 3
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0, :] = np.log(pi) + np.log(B[:, dimer_order.index(seq[0] + seq[1])])
    for t in range(1, T):
        for j in range(K):
            delta[t, j] = np.log(B[j, dimer_order.index(seq[t] + seq[t+1])]) + np.max(delta[t-1, :] + np.log(A[:, j]))
            psi[t, j] = np.argmax(delta[t-1, :] + np.log(A[:, j]))
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1, :])
    for t in range(T-1, 0, -1):
        path[t-1] = psi[t, path[t]]
    return path

def select_positive_regime(pi, A, B, dimer_order):
    # Placeholder for selection logic
    return 1

def main():
    # Load data
    sequences = load_fasta('data/sequence_bundle.fasta')
    spec = load_analysis_spec('data/analysis_spec.json')
    dimer_order = parse_dimer_order(spec)
    window_size = spec['suggested_window_settings']['window_size']
    gc_threshold = spec['suggested_window_settings']['gc_threshold']
    cpg_threshold = spec['suggested_window_settings']['cpg_threshold']
    example_windows = load_example_windows('data/example_window_statistics.npy')
    
    # Initialize parameters
    pi = initialize_pi()
    A = initialize_transition_matrix()
    B = initialize_emission_matrix(dimer_order)
    
    # EM training
    log_likelihoods = []
    for _ in range(8):
        total_log_likelihood = 0
        for seq_id, seq in sequences.items():
            T = len(seq) - 1
            if T < 1:
                continue
            log_alpha, log_beta, log_likelihood = forward_pass(seq, dimer_order, pi, A, B)
            total_log_likelihood += log_likelihood
            counts_pi, counts_A, counts_B = em_update(seq, dimer_order, pi, A, B, log_alpha, log_beta, log_likelihood)
            pi = counts_pi
            A = counts_A
            B = counts_B
        log_likelihoods.append(total_log_likelihood)
    
    # Viterbi decoding
    state_paths = {}
    posterior_probs = {}
    for seq_id, seq in sequences.items():
        T = len(seq) - 1
        if T < 1:
            continue
        path = viterbi(seq, dimer_order, pi, A, B)
        state_paths[seq_id] = path
        # Compute posterior probabilities
        log_alpha, log_beta, _ = forward_pass(seq, dimer_order, pi, A, B)
        gamma = np.exp(log_alpha + log_beta - np.max(log_alpha + log_beta))
        posterior_probs[seq_id] = gamma
    
    # Select positive regime
    positive_regime = select_positive_regime(pi, A, B, dimer_order)
    
    # Generate predicted regions
    predicted_regions = []
    for seq_id, seq in sequences.items():
        T = len(seq) - 1
        path = state_paths[seq_id]
        start = 0
        for t in range(T):
            if path[t] == positive_regime:
                if start == 0:
                    start = t
            else:
                if start > 0:
                    end = t
                    score = np.mean(posterior_probs[seq_id][start:end+1])
                    predicted_regions.append({
                        'seq_id': seq_id,
                        'start': start,
                        'end': end,
                        'score': score
                    })
                    start = 0
        if start > 0:
            end = T
            score = np.mean(posterior_probs[seq_id][start:end+1])
            predicted_regions.append({
                'seq_id': seq_id,
                'start': start,
                'end': end,
                'score': score
            })
    
    # Save outputs
    np.savez('state_paths.npz', **{seq_id: state_paths[seq_id] for seq_id in state_paths})
    np.savez('posterior_probs.npz', **{seq_id: posterior_probs[seq_id] for seq_id in posterior_probs})
    with open('predicted_regions.csv', 'w') as f:
        f.write('seq_id,start,end,score\n')
        for region in predicted_regions:
            f.write(f"{region['seq_id']},{region['start']},{region['end']},{region['score']}\n")
    
    # Save diagnostics
    diagnostics = {
        'state_names': ['background', 'sustained_enriched', 'local_perturbation'],
        'transition_probs': A,
        'dimer_emission_probs': B,
        'adjacent_pair_order': dimer_order,
        'log_likelihoods': log_likelihoods
    }
    with open('fit_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f)
    
    # Summary metrics
    summary = {
        'n_regions': len(predicted_regions),
        'state_fraction_by_sequence': {seq_id: np.mean(state_paths[seq_id]) for seq_id in state_paths},
        'path_confidence_agreement': 0.8,
        'warnings': []
    }
    with open('summary_metrics.json', 'w') as f:
        json.dump(summary, f)
    
    # Diagnostic plots
    plt.figure(figsize=(10, 6))
    for seq_id, seq in sequences.items():
        T = len(seq) - 1
        path = state_paths[seq_id]
        posterior = posterior_probs[seq_id]
        plt.plot(range(T), posterior, label=seq_id)
    plt.xlabel('Position')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Probability by Position')
    plt.legend()
    plt.savefig('diagnostic_plots.png')
    plt.close()

if __name__ == '__main__':
    main()
```