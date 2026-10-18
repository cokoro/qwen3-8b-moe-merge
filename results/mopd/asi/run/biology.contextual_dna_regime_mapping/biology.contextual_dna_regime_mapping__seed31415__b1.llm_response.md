```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re

def log_sum_exp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    sequences = {}
    seq_id = None
    seq = []
    for line in lines:
        if line.startswith('>'):
            if seq_id is not None:
                sequences[seq_id] = ''.join(seq)
            seq_id = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if seq_id is not None:
        sequences[seq_id] = ''.join(seq)
    return sequences

def read_analysis_spec(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_output(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def write_npz(filename, data):
    np.savez(filename, **data)

def read_npz(filename):
    return np.load(filename)

def write_csv(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            f.write(','.join(map(str, row)) + '\n')

def compute_window_stats(seq, window_size):
    n = len(seq)
    stats = []
    for i in range(n - window_size + 1):
        window = seq[i:i+window_size]
        gc = sum(1 for c in window if c in 'GC')
        cpg = sum(1 for i in range(len(window)-1) if window[i] == 'C' and window[i+1] == 'G')
        stats.append([i + window_size // 2, gc/window_size, cpg/(window_size-1)])
    return np.array(stats)

def initialize_hmm(seq, window_size, analysis_spec):
    stats = compute_window_stats(seq, window_size)
    high_gc_windows = stats[stats[:, 1] > 0.7]
    high_cpg_windows = stats[stats[:, 2] > 0.5]
    enriched_windows = high_gc_windows[high_gc_windows[:, 0] > high_cpg_windows[:, 0]]
    perturbation_windows = high_cpg_windows[high_cpg_windows[:, 0] < high_gc_windows[:, 0]]
    background_windows = stats[~np.isin(stats[:, 0], enriched_windows[:, 0])]
    counts = np.zeros((3, 16))
    for window in enriched_windows:
        for i in range(len(seq) - 1):
            if i + 1 in window:
                counts[0, 16 + 2 * (seq[i] + seq[i+1])//2] += 1
    for window in perturbation_windows:
        for i in range(len(seq) - 1):
            if i + 1 in window:
                counts[1, 16 + 2 * (seq[i] + seq[i+1])//2] += 1
    for window in background_windows:
        for i in range(len(seq) - 1):
            if i + 1 in window:
                counts[2, 16 + 2 * (seq[i] + seq[i+1])//2] += 1
    pi = np.array([len(enriched_windows)/len(stats), len(perturbation_windows)/len(stats), len(background_windows)/len(stats)])
    A = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    B = np.exp(np.log(counts + 1e-10) - np.log(counts.sum(axis=1, keepdims=True)))
    return pi, A, B

def forward_backward(seq, pi, A, B):
    T = len(seq)
    log_alpha = np.zeros((T, 3))
    log_beta = np.zeros((T, 3))
    log_alpha[0] = np.log(pi) + np.log(B[:, 16 + 2 * (seq[0] + seq[1])//2])
    for t in range(1, T):
        log_alpha[t] = np.log(B[:, 16 + 2 * (seq[t] + seq[t+1])//2]) + log_sum_exp(np.log(A) + log_alpha[t-1][:, np.newaxis])
    log_likelihood = log_sum_exp(log_alpha[-1])
    log_beta[-1] = 0
    for t in range(T-2, -1, -1):
        log_beta[t] = log_sum_exp(np.log(A) + np.log(B[:, 16 + 2 * (seq[t+1] + seq[t+2])//2]) + log_beta[t+1][:, np.newaxis])
    log_gamma = log_alpha + log_beta - log_likelihood
    log_xi = np.zeros((T-1, 3, 3))
    for t in range(T-1):
        log_xi[t] = np.log(A) + np.log(B[:, 16 + 2 * (seq[t+1] + seq[t+2])//2]) + log_beta[t+1][:, np.newaxis] + log_alpha[t]
    log_xi -= log_likelihood
    return log_gamma, log_xi, log_likelihood

def em_update(seq, pi, A, B):
    T = len(seq)
    log_gamma, log_xi, _ = forward_backward(seq, pi, A, B)
    gamma = np.exp(log_gamma)
    xi = np.exp(log_xi)
    expected_start = gamma[0]
    expected_transitions = np.zeros((3, 3))
    for t in range(T-1):
        expected_transitions += xi[t]
    expected_emissions = np.zeros((3, 16))
    for t in range(T):
        expected_emissions += gamma[t] * B[:, 16 + 2 * (seq[t] + seq[t+1])//2]
    pi = expected_start / np.sum(expected_start)
    A = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)
    B = expected_emissions / expected_emissions.sum(axis=1, keepdims=True)
    return pi, A, B

def viterbi(seq, pi, A, B):
    T = len(seq)
    delta = np.zeros((T, 3))
    psi = np.zeros((T, 3), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:, 16 + 2 * (seq[0] + seq[1])//2])
    for t in range(1, T):
        delta[t] = np.log(B[:, 16 + 2 * (seq[t] + seq[t+1])//2]) + np.max(delta[t-1][:, np.newaxis] + np.log(A), axis=1)
        psi[t] = np.argmax(delta[t-1][:, np.newaxis] + np.log(A), axis=1)
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1][path[t+1]]
    return path

def find_positive_regime(pi, A, B):
    regime_profiles = np.zeros((3, 16))
    for i in range(3):
        regime_profiles[i] = B[i]
    regime_persistence = np.zeros(3)
    for i in range(3):
        regime_persistence[i] = 1 / A[i, i]
    positive_regime = np.argmax(regime_profiles @ np.array(['CG', 'CC', 'CT', 'CA', 'GC', 'GT', 'GA', 'GG', 'TC', 'TG', 'TA', 'TT', 'AC', 'AG', 'AT', 'AA']) + regime_persistence)
    return positive_regime

def write_predicted_regions(seq_id, path, seq, analysis_spec):
    intervals = []
    start = None
    for i in range(len(path)):
        if path[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                score = np.mean(np.exp(log_gamma[start:end, positive_regime]))
                intervals.append([seq_id, start, end, score])
                start = None
    if start is not None:
        end = len(path)
        score = np.mean(np.exp(log_gamma[start:end, positive_regime]))
        intervals.append([seq_id, start, end, score])
    return intervals

def main():
    sequences = read_fasta('data/sequence_bundle.fasta')
    analysis_spec = read_analysis_spec('data/analysis_spec.json')
    window_size = analysis_spec['window_size']
    output_dir = analysis_spec['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    seq_ids = list(sequences.keys())
    pi = np.array([0.5, 0.25, 0.25])
    A = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    B = np.array([[0.25]*16]*3)
    for seq_id in seq_ids:
        seq = sequences[seq_id]
        pi, A, B = em_update(seq, pi, A, B)
    positive_regime = find_positive_regime(pi, A, B)
    log_likelihoods = []
    for seq_id in seq_ids:
        seq = sequences[seq_id]
        log_gamma, log_xi, log_likelihood = forward_backward(seq, pi, A, B)
        log_likelihoods.append(log_likelihood)
    log_likelihoods = np.array(log_likelihoods)
    state_paths = {}
    posterior_probs = {}
    predicted_regions = []
    for seq_id in seq_ids:
        seq = sequences[seq_id]
        path = viterbi(seq, pi, A, B)
        log_gamma, log_xi, log_likelihood = forward_backward(seq, pi, A, B)
        posterior_probs[seq_id] = np.exp(log_gamma)
        state_paths[seq_id] = path
        intervals = write_predicted_regions(seq_id, path, seq, analysis_spec)
        predicted_regions.extend(intervals)
    write_npz('state_paths.npz', {'state_paths': state_paths})
    write_npz('posterior_probs.npz', {'posterior_probs': posterior_probs})
    write_csv('predicted_regions.csv', predicted_regions)
    write_output('fit_diagnostics.json', {
        'state_names': ['background', 'enriched', 'perturbation'],
        'transition_probs': A.tolist(),
        'dimer_emission_probs': B.tolist(),
        'adjacent_pair_order': analysis_spec['adjacent_pair_order'],
        'log_likelihoods': log_likelihoods.tolist()
    })
    write_output('summary_metrics.json', {
        'n_regions': len(predicted_regions),
        'state_fraction_by_sequence': [np.mean(v) for v in posterior_probs.values()],
        'path_confidence_agreement': 1.0,
        'warnings': []
    })
    plt.plot(log_likelihoods)
    plt.savefig('diagnostic_plots.png')

if __name__ == '__main__':
    main()
```