```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def log_sum_exp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

class HMM:
    def __init__(self, n_states=3, n_emissions=16):
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.pi = np.zeros(n_states)
        self.A = np.zeros((n_states, n_states))
        self.B = np.zeros((n_states, n_emissions))
        self.pseudocount = 1e-10

    def fit(self, sequences, max_iter=8):
        for _ in range(max_iter):
            gamma = []
            xi = []
            log_likelihoods = []
            for seq in sequences:
                T = len(seq)
                log_alpha = np.zeros((T, self.n_states))
                log_beta = np.zeros((T, self.n_states))
                log_gamma = np.zeros((T, self.n_states))
                log_xi = np.zeros((T-1, self.n_states, self.n_states))
                
                # Forward pass
                log_alpha[0] = np.log(self.pi) + np.log(self.B[:, seq[0]])
                for t in range(1, T):
                    log_alpha[t] = np.log(self.B[:, seq[t]]) + log_sum_exp(
                        np.log(self.A.T) + log_alpha[t-1][:, np.newaxis]
                    ).T
                
                # Backward pass
                log_beta[-1] = 0
                for t in range(T-2, -1, -1):
                    log_beta[t] = log_sum_exp(
                        np.log(self.A.T) + np.log(self.B[:, seq[t+1]]) + log_beta[t+1]
                    )
                
                # Compute log_likelihood
                log_likelihood = log_sum_exp(log_alpha[-1])
                log_likelihoods.append(log_likelihood)
                
                # Compute gamma and xi
                for t in range(T):
                    log_gamma[t] = log_alpha[t] + log_beta[t] - log_likelihood
                
                for t in range(T-1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            log_xi[t, i, j] = log_alpha[t, i] + np.log(self.A[i,j]) + \
                                              np.log(self.B[j, seq[t+1]]) + log_beta[t+1, j] - log_likelihood
                
                gamma.append(log_gamma)
                xi.append(log_xi)
            
            # Update parameters
            self.pi = (np.sum([g[0] for g in gamma], axis=0) + self.pseudocount) / (len(sequences) + self.pseudocount * self.n_states)
            self.A = (np.sum([np.sum(xi[i], axis=0) for i in range(len(sequences))], axis=0) + self.pseudocount) / \
                     (np.sum([np.sum(gamma[i], axis=0) for i in range(len(sequences))], axis=0) + self.pseudocount * self.n_states)
            self.B = (np.sum([np.sum(gamma[i][t] * np.eye(self.n_emissions)[seq[t]] for t in range(len(seq))], axis=0) for seq in sequences], axis=0) + self.pseudocount) / \
                     (np.sum([np.sum(gamma[i], axis=0) for i in range(len(sequences))], axis=0) + self.pseudocount * self.n_states)

        return np.array(log_likelihoods)

def load_sequences(file_path):
    with open(file_path, 'r') as f:
        seqs = []
        current_seq = ''
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    seqs.append(current_seq)
                    current_seq = ''
            else:
                current_seq += line.strip()
        if current_seq:
            seqs.append(current_seq)
    return seqs

def main():
    # Load and preprocess data
    seqs = load_sequences('data/sequence_bundle.fasta')
    spec = json.load(open('data/analysis_spec.json'))
    
    # Initialize HMM
    hmm = HMM(n_states=3)
    hmm.fit(seqs, max_iter=8)
    
    # Predict and output results
    # ... (full implementation omitted for brevity)
    
    # Save outputs
    np.savez('state_paths.npz', paths=state_paths)
    np.savez('posterior_probs.npz', probs=posterior_probs)
    with open('predicted_regions.csv', 'w') as f:
        # ... (write predicted intervals)
    plt.savefig('diagnostic_plots.png')
    # ... (write other outputs)
```