import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def read_sequences(fn='data/sequences.csv'):
    df = pd.read_csv(fn)
    return df['sequence'].apply(lambda s: np.array([int(c) for c in s])).values

def read_metadata(fn='data/metadata.json'):
    import json
    with open(fn) as f:
        return json.load(f)

def write_result(df, path):
    df.to_csv(path, index=False)

def forward_backward(sequences, K, max_iter=100, tol=1e-4):
    N = len(sequences)
    T = [len(seq) for seq in sequences]
    log_likelihood = []
    gamma = []
    xi = []
    A = np.random.rand(K, K)
    A /= A.sum(axis=1, keepdims=True)
    B = np.random.rand(K, 32)
    B /= B.sum(axis=1, keepdims=True)
    pi = np.random.rand(K)
    pi /= pi.sum()
    
    for _ in range(max_iter):
        alpha = np.zeros((N, K, max(T)))
        beta = np.zeros((N, K, max(T)))
        log_c = np.zeros((N, max(T)))
        
        for n in range(N):
            t = 0
            alpha[n, :, t] = pi * B[:, sequences[n][t]]
            log_c[n, t] = np.log(alpha[n, :, t].sum())
            alpha[n, :, t] /= alpha[n, :, t].sum()
            
            for t in range(1, T[n]):
                raw_alpha = alpha[n, :, t-1].dot(A)
                alpha[n, :, t] = raw_alpha * B[:, sequences[n][t]]
                log_c[n, t] = np.log(alpha[n, :, t].sum())
                alpha[n, :, t] /= alpha[n, :, t].sum()
        
        for n in range(N):
            t = T[n] - 1
            beta[n, :, t] = 1.0
            for t in range(T[n]-2, -1, -1):
                beta[n, :, t] = beta[n, :, t+1] * A.T * B[:, sequences[n][t+1]]
                beta[n, :, t] /= beta[n, :, t].sum()
        
        for n in range(N):
            gamma_t = alpha[n, :, :T[n]] * beta[n, :, :T[n]]
            gamma_t /= gamma_t.sum(axis=1, keepdims=True)
            gamma.append(gamma_t)
            xi_t = np.zeros((K, K, T[n]-1))
            for t in range(T[n]-1):
                xi_t[:, :, t] = alpha[n, :, t].reshape(-1, 1) * A * B[:, sequences[n][t+1]].reshape(1, -1) * beta[n, :, t+1]
                xi_t[:, :, t] /= xi_t[:, :, t].sum(axis=1, keepdims=True)
            xi.append(xi_t)
        
        A_new = np.zeros((K, K))
        B_new = np.zeros((K, 32))
        pi_new = np.zeros(K)
        
        for n in range(N):
            for t in range(T[n]):
                A_new += xi[n, :, :, t].sum(axis=2)
                B_new += gamma[n, :, t].reshape(K, 1) * np.eye(32)[sequences[n][t]]
                pi_new += gamma[n, :, 0]
        
        A_new /= A_new.sum(axis=1, keepdims=True)
        B_new /= B_new.sum(axis=1, keepdims=True)
        pi_new /= pi_new.sum()
        
        log_likelihood.append(np.sum(log_c))
        
        if np.abs(log_likelihood[-1] - log_likelihood[-2]) < tol:
            break
        
        A, B, pi = A_new, B_new, pi_new
    
    return A, B, pi, log_likelihood, gamma, xi

def fit_model(K, sequences):
    A, B, pi, log_likelihood, gamma, xi = forward_backward(sequences, K)
    return A, B, pi, log_likelihood, gamma, xi

def evaluate_candidate(K, sequences, N_restarts=5):
    results = []
    for _ in range(N_restarts):
        A, B, pi, log_likelihood, gamma, xi = fit_model(K, sequences)
        results.append((A, B, pi, log_likelihood, gamma, xi))
    return results

def select_candidate(K_candidates, sequences):
    scores = []
    for K in K_candidates:
        results = evaluate_candidate(K, sequences)
        scores.append((K, np.mean([r[3][-1] for r in results]), np.std([r[3][-1] for r in results])))
    best = max(scores, key=lambda x: x[1])
    return best[0]

def main():
    sequences = read_sequences()
    metadata = read_metadata()
    K_candidates = range(2, 20)
    
    results = []
    for K in K_candidates:
        results.append((K, evaluate_candidate(K, sequences)))
    
    best_K = select_candidate(K_candidates, sequences)
    A, B, pi, log_likelihood, gamma, xi = best_K
    
    write_result(pd.DataFrame([{'candidate': K, 'n_free_values': K*(K-1) + K, 'objective': log_likelihood[-1], 'criterion': 'log_likelihood', 'chosen': 1} for K in K_candidates]), 'results/candidate_summary.csv')
    write_result(pd.DataFrame([(i, j, A[i,j]) for i in range(K) for j in range(K)]), 'results/matrix_a.csv')
    write_result(pd.DataFrame([(i, s, s, B[i,s]) for i in range(K) for s in range(32)]), 'results/matrix_b.csv')
    write_result(pd.DataFrame([(i, pi[i]) for i in range(K)]), 'results/vector_a.csv')
    write_result(pd.DataFrame([(run, step, obj, (obj - prev_obj)/obj) for run in range(5) for step in range(100) for obj in [log_likelihood[step]]]), 'results/fit_trace.csv')
    np.save('results/local_weights.npy', np.vstack(gamma))
    write_result(pd.DataFrame([(i, j, xi[i,j], xi[i,j]/gamma[i], xi[i,j]/gamma[i]/gamma[i].sum()) for i in range(K) for j in range(K)]), 'results/neighbor_summary.csv')
    write_result(pd.DataFrame([(seq, t, t, label) for seq in range(len(sequences)) for t in range(len(sequences[seq])) for label in range(K)]), 'results/local_assignments.csv')
    write_result(pd.DataFrame([(seq, len(sequences[seq]), np.mean(log_likelihood), np.std(log_likelihood), np.max(gamma[seq]) for seq in range(len(sequences))]), 'results/stream_diagnostics.csv')
    plt.plot(log_likelihood)
    plt.savefig('results/diagnostics.png')

if __name__ == '__main__':
    main()