import numpy as np
import matplotlib.pyplot as plt
import json
import os

class HMM:
    def __init__(self, n_states=3, n_obs=16, pseudocount=1e-5):
        self.n_states = n_states
        self.n_obs = n_obs
        self.pseudocount = pseudocount

    def logsumexp(self, x):
        m = np.max(x, axis=1, keepdims=True)
        return m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))

    def forward(self, obs, pi, A, B):
        T = len(obs)
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = np.log(pi) + np.log(B[:, obs[0]])
        for t in range(1, T):
            log_alpha[t] = np.log(B[:, obs[t]]) + self.logsumexp(log_alpha[t-1] + np.log(A))
        log_likelihood = self.logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def backward(self, obs, pi, A, B):
        T = len(obs)
        log_beta = np.zeros((T, self.n_states))
        log_beta[-1] = np.zeros(self.n_states)
        for t in range(T-2, -1, -1):
            log_beta[t] = self.logsumexp(np.log(A) + np.log(B[:, obs[t+1]]) + log_beta[t+1])
        return log_beta

    def em_step(self, seqs, pi, A, B):
        gamma = []
        xi = []
        for obs in seqs:
            T = len(obs)
            log_alpha, log_likelihood = self.forward(obs, pi, A, B)
            log_beta = self.backward(obs, pi, A, B)
            log_gamma = log_alpha + log_beta - log_likelihood
            log_xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        log_xi[t, i, j] = log_alpha[t, i] + np.log(A[i,j]) + np.log(B[j, obs[t+1]]) + log_beta[t+1, j] - log_likelihood
            gamma.append(log_gamma)
            xi.append(log_xi)
        # accumulate counts
        count_start = np.sum(np.exp(gamma[0]), axis=0)
        count_trans = np.zeros((self.n_states, self.n_states))
        for t in range(len(seqs)-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    count_trans[i,j] += np.sum(np.exp(xi[t][:,i,j]))
        # normalize
        pi = (count_start + self.pseudocount) / (np.sum(count_start) + self.n_states * self.pseudocount)
        A = (count_trans + self.pseudocount) / (np.sum(count_trans, axis=1, keepdims=True) + self.n_states * self.pseudocount)
        B = np.zeros((self.n_states, self.n_obs))
        for j in range(self.n_states):
            for o in range(self.n_obs):
                B[j, o] = np.sum(np.exp(gamma[0][:,j][obs == o])) + self.pseudocount
            B[j] /= B[j].sum()
        return pi, A, B

    def viterbi(self, obs, pi, A, B):
        T = len(obs)
        delta = np.zeros((T, self.n_states))
        delta[0] = np.log(pi) + np.log(B[:, obs[0]])
        psi = np.zeros((T, self.n_states), dtype=int)
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t,j] = np.log(B[j, obs[t]]) + np.max(delta[t-1] + np.log(A[:,j]))
                psi[t,j] = np.argmax(delta[t-1] + np.log(A[:,j]))
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path

def read_fasta(path):
    with open(path, 'r') as f:
        seqs = []
        for line in f:
            if line.startswith('>'):
                seq_id = line.strip()[1:]
            else:
                seqs.append((seq_id, line.strip()))
    return seqs

def main():
    import sys
    import scipy
    from scipy.stats import entropy

    # Load input
    seqs = read_fasta('data/sequence_bundle.fasta')
    with open('data/analysis_spec.json', 'r') as f:
        spec = json.load(f)
    adjacent_pairs = spec['adjacent_pair_order']

    # Preprocess sequences
    processed_seqs = []
    for seq_id, seq in seqs:
        if len(seq) < 2:
            continue
        obs = []
        for i in range(len(seq)-1):
            pair = seq[i] + seq[i+1]
            idx = adjacent_pairs.index(pair)
            obs.append(idx)
        processed_seqs.append((seq_id, obs))

    # Initialize HMM
    hmm = HMM(n_states=3, n_obs=len(adjacent_pairs))
    # Placeholder for initial parameters
    pi = np.array([0.5, 0.25, 0.25])
    A = np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.1, 0.1, 0.8]])
    B = np.array([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Train
    for _ in range(8):
        pi, A, B = hmm.em_step(processed_seqs, pi, A, B)

    # Decode
    paths = []
    for seq_id, obs in processed_seqs:
        path = hmm.viterbi(obs, pi, A, B)
        paths.append((seq_id, path))

    # Output results
    # ... (write output files)

if __name__ == '__main__':
    main()