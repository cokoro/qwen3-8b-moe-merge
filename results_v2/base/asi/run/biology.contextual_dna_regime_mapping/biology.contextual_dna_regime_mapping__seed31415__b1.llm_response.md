```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

def log_sum_exp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

def read_fasta(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sequences = {}
    current_seq = None
    for line in lines:
        if line.startswith('>'):
            current_seq = line.strip()[1:]
            sequences[current_seq] = []
        else:
            sequences[current_seq].append(line.strip())
    for seq in sequences:
        sequences[seq] = ''.join(sequences[seq])
    return sequences

def read_analysis_spec(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_adjacent_pairs(sequence):
    return [sequence[i] + sequence[i+1] for i in range(len(sequence)-1)]

def get_dinucleotide_order(spec):
    return spec['adjacent_pair_order']

def sliding_window_stats(sequence, window_size):
    n = len(sequence)
    stats = []
    for i in range(n - window_size + 1):
        window = sequence[i:i+window_size]
        gc = sum(1 for c in window if c in 'GC')
        cpg = sum(1 for i in range(len(window)-1) if window[i] == 'C' and window[i+1] == 'G')
        cpg_expected = (window.count('C') * window.count('G')) / (window_size * (window_size - 1))
        stats.append([i + window_size // 2, gc / window_size, cpg / (window_size - 1), cpg_expected])
    return np.array(stats)

def initialize_hmm(sequence, spec):
    window_size = spec['window_size']
    window_stats = sliding_window_stats(sequence, window_size)
    sorted_stats = window_stats[window_stats[:, 1].argsort()[::-1]]
    top_windows = sorted_stats[:int(len(sorted_stats) * 0.2)]
    
    enriched_windows = top_windows[top_windows[:, 1] > 0.6]
    perturbation_windows = top_windows[(top_windows[:, 1] > 0.4) & (top_windows[:, 1] <= 0.6)]
    background_windows = top_windows[top_windows[:, 1] <= 0.4]
    
    o = get_adjacent_pairs(sequence)
    o_set = set(o)
    
    B = np.zeros((3, 16))
    for i, obs in enumerate(o_set):
        idx = get_dinucleotide_order(spec).index(obs)
        B[0, idx] += 1
        B[1, idx] += 1
        B[2, idx] += 1
    
    A = np.zeros((3, 3))
    A[0, 0] = 0.9
    A[1, 1] = 0.8
    A[2, 2] = 0.7
    
    pi = np.array([0.2, 0.5, 0.3])
    
    return B, A, pi, o_set

def forward_backward(sequence, B, A, pi, o):
    T = len(o)
    K = B.shape[0]
    
    log_alpha = np.zeros((T, K))
    log_beta = np.zeros((T, K))
    
    log_alpha[0, :] = np.log(pi) + np.log(B[:, np.array([o[0]]).flatten()])
    
    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = np.log(B[j, np.array([o[t]]).flatten()]) + log_sum_exp(log_alpha[t-1, :] + np.log(A[:, j]))
    
    log_likelihood = log_sum_exp(log_alpha[-1, :])
    
    for t in range(T-1, -1, -1):
        for i in range(K):
            log_beta[t, i] = log_sum_exp(np.log(A[i, :]) + np.log(B[:, np.array([o[t+1]]).flatten()]) + log_beta[t+1, :])
    
    log_gamma = log_alpha + log_beta - log_likelihood
    log_xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = log_alpha[t, i] + np.log(A[i, j]) + np.log(B[j, np.array([o[t+1]]).flatten()]) + log_beta[t+1, j] - log_likelihood
    
    return log_gamma, log_xi, log_likelihood

def em_update(B, A, pi, o, log_gamma, log_xi):
    K = B.shape[0]
    T = len(o)
    
    gamma_counts = np.zeros((T, K))
    xi_counts = np.zeros((T-1, K, K))
    
    for t in range(T):
        for j in range(K):
            gamma_counts[t, j] = np.exp(log_gamma[t, j])
    
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                xi_counts[t, i, j] = np.exp(log_xi[t, i, j])
    
    B_counts = np.zeros((K, len(o)))
    for t in range(T):
        for j in range(K):
            B_counts[j, np.array([o[t]]).flatten()] += gamma_counts[t, j]
    
    B = B_counts / gamma_counts.sum(axis=1)[:, np.newaxis]
    
    A_counts = np.zeros((K, K))
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                A_counts[i, j] += xi_counts[t, i, j]
    
    A = A_counts / gamma_counts[:-1, :].sum(axis=0)[:, np.newaxis]
    
    pi_counts = gamma_counts[0, :]
    pi = pi_counts / np.sum(pi_counts)
    
    return B, A, pi

def viterbi(sequence, B, A, pi, o):
    K = B.shape[0]
    T = len(o)
    
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    
    delta[0, :] = np.log(pi) + np.log(B[:, np.array([o[0]]).flatten()])
    
    for t in range(1, T):
        for j in range(K):
            delta[t, j] = np.log(B[j, np.array([o[t]]).flatten()]) + np.max(delta[t-1, :] + np.log(A[:, j]))
            psi[t, j] = np.argmax(delta[t-1, :] + np.log(A[:, j]))
    
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(delta[-1, :])
    
    for t in range(T-1, 0, -1):
        best_path[t-1] = psi[t, best_path[t]]
    
    return best_path

def select_positive_regime(B, A, o):
    K = B.shape[0]
    regime_scores = np.zeros(K)
    
    for j in range(K):
        regime_scores[j] = B[j, np.array(['CG']).flatten()] + np.sum(B[j, [o.index(pair) for pair in o if 'G' in pair or 'C' in pair]])
    
    positive_regime = np.argmax(regime_scores)
    return positive_regime

def process_sequence(sequence, spec, B, A, pi, o):
    log_gamma, log_xi, log_likelihood = forward_backward(sequence, B, A, pi, o)
    B, A, pi = em_update(B, A, pi, o, log_gamma, log_xi)
    
    best_path = viterbi(sequence, B, A, pi, o)
    
    positive_regime = select_positive_regime(B, A, o)
    
    gamma = np.exp(log_gamma)
    posterior_probs = gamma
    
    return best_path, posterior_probs, positive_regime, log_likelihood

def main():
    spec = read_analysis_spec('data/analysis_spec.json')
    sequences = read_fasta('data/sequence_bundle.fasta')
    o_set = set(get_adjacent_pairs(sequences[next(iter(sequences))]))
    o_order = get_dinucleotide_order(spec)
    o = np.array([o_order.index(pair) for pair in o_set])
    
    B, A, pi, _ = initialize_hmm(sequences[next(iter(sequences))], spec)
    
    for seq_id in sequences:
        sequence = sequences[seq_id]
        o = get_adjacent_pairs(sequence)
        o = np.array([o_order.index(pair) for pair in o])
        best_path, posterior_probs, positive_regime, log_likelihood = process_sequence(sequence, spec, B, A, pi, o)
        
        # Save outputs
        np.savez('state_paths.npz', **{seq_id: best_path})
        np.savez('posterior_probs.npz', **{seq_id: posterior_probs})
        np.savez('fit_diagnostics.json', state_names=['background', 'enriched', 'perturbation'], transition_probs=A, dimer_emission_probs=B, adjacent_pair_order=o_order, log_likelihoods=[log_likelihood])
        
        # Generate predicted_regions.csv
        start = 0
        intervals = []
        for i in range(len(best_path)):
            if best_path[i] == positive_regime:
                if i == 0 or best_path[i-1] != positive_regime:
                    start = i
            else:
                if i > start:
                    end = i
                    score = np.mean(posterior_probs[start:end])
                    intervals.append([seq_id, start, end, score])
        if start < len(best_path) and best_path[-1] == positive_regime:
            end = len(best_path)
            score = np.mean(posterior_probs[start:end])
            intervals.append([seq_id, start, end, score])
        
        with open('predicted_regions.csv', 'a') as f:
            for interval in intervals:
                f.write(f"{interval[0]},{interval[1]},{interval[2]},{interval[3]}\n")
    
    # Generate diagnostic plots
    plt.figure(figsize=(10, 6))
    for seq_id in sequences:
        posterior_probs = np.load('posterior_probs.npz')[seq_id]
        plt.plot(posterior_probs, label=seq_id)
    plt.legend()
    plt.title('Posterior Probabilities')
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.savefig('diagnostic_plots.png')
    plt.close()

if __name__ == "__main__":
    main()
```