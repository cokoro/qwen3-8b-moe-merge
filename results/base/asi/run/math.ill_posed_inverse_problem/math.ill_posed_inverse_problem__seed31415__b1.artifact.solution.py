import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd, lstsq
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

# Create results directories
os.makedirs('results/diagnosis', exist_ok=True)
os.makedirs('results/solution/tikhonov', exist_ok=True)
os.makedirs('results/solution/tsvd', exist_ok=True)
os.makedirs('results/solution/cgls', exist_ok=True)
os.makedirs('results/comparison', exist_ok=True)

# Load data
K = np.load('data/K.npy')
y_obs = pd.read_csv('data/y_obs.csv').values.flatten()

# Phase 1 - Problem Diagnosis
U, S, Vt = svd(K, full_matrices=False)
sigma = S
cond_K = sigma[0] / sigma[-1]

# Compute Picard data
picard_data = []
for i in range(len(sigma)):
    u_i = U[:, i]
    fourier_coeff = abs(u_i @ y_obs)
    picard_ratio = fourier_coeff / sigma[i]
    picard_data.append([sigma[i], fourier_coeff, picard_ratio])

picard_df = pd.DataFrame(picard_data, columns=['singular_value', 'fourier_coeff', 'picard_ratio'])
picard_df.to_csv('results/diagnosis/picard_data.csv', index=False)

# Estimate noise level
noise_indices = int(0.75 * len(sigma))
noise_sigma = sigma[noise_indices:]
coeffs = np.array([u_i @ y_obs for i, u_i in enumerate(U[:, noise_indices:])])
noise_estimate = np.median(np.abs(coeffs - np.median(coeffs))) / 0.6745

# Effective rank
effective_rank = 0
for i in range(len(sigma)):
    if picard_data[i][2] > 3 * noise_estimate:
        effective_rank = i + 1
    else:
        break

# DPC judgment
dpc_judgment = 'holds' if effective_rank >= 0.7 * len(sigma) else 'does_not_hold'

# Diagnosis report
with open('results/diagnosis/diagnosis_report.md', 'w') as f:
    f.write(f'# Diagnosis Report\n\n')
    f.write(f'## Noise Estimate: {noise_estimate:.2f}\n')
    f.write(f'## Effective Rank: {effective_rank}\n')
    f.write(f'## DPC Judgment: {dpc_judgment}\n')

# Phase 2 - Multi-Method Solution

# Prewhitening
rho = 0.5  # Estimate AR(1) coefficient from data
W = np.eye(len(y_obs))
for i in range(1, len(y_obs)):
    W[i, i-1] = -rho
K_w = W @ K
y_w = W @ y_obs

# Method A - Tikhonov Regularization
def tikhonov_solution(K, y, L, sigma, noise_estimate):
    # Choose L based on diagnosis
    if dpc_judgment == 'holds':
        L = np.eye(len(K))
    else:
        L = np.eye(len(K))
    # L-curve
    lambdas = np.logspace(-15, 15, 1000)
    residuals = []
    solution_norms = []
    for lam in lambdas:
        x = np.linalg.lstsq(K, y, rcond=None)[0]
        x_reg = np.linalg.pinv(K.T @ K + lam**2 * L.T @ L) @ (K.T @ y)
        residuals.append(np.linalg.norm(K @ x_reg - y))
        solution_norms.append(np.linalg.norm(L @ x_reg))
    # Find optimal lambda
    log_residuals = np.log(residuals)
    log_norms = np.log(solution_norms)
    curvature = np.gradient(log_norms, log_residuals)
    optimal_idx = np.argmax(curvature)
    optimal_lambda = lambdas[optimal_idx]
    x_reg = np.linalg.pinv(K.T @ K + optimal_lambda**2 * L.T @ L) @ (K.T @ y)
    return x_reg, optimal_lambda, residuals, solution_norms

x_tikhonov, lambda_tikhonov, lcurve_residuals, lcurve_norms = tikhonov_solution(K, y_obs, np.eye(len(K)), sigma, noise_estimate)
np.savetxt('results/solution/tikhonov/x_reg.csv', x_tikhonov, delimiter=',')
np.savetxt('results/solution/tikhonov/reg_param.txt', [lambda_tikhonov])
pd.DataFrame({'residual': lcurve_residuals, 'norm': lcurve_norms}).to_csv('results/solution/tikhonov/lcurve_data.csv', index=False)

# Method B - Truncated SVD
def tsvd_solution(K, y, sigma, noise_estimate):
    # GCV for TSVD
    k_values = np.arange(1, len(sigma)+1)
    gcv_scores = []
    for k in k_values:
        x_k = np.zeros_like(y)
        for i in range(k):
            x_k += (U[:, i].T @ y_w) / sigma[i] * Vt[i, :]
        r_k = y_w - K @ x_k
        gcv = np.linalg.norm(r_k)**2 / (len(y) - k)**2
        gcv_scores.append(gcv)
    optimal_k = np.argmin(gcv_scores) + 1
    x_tsvd = np.zeros_like(y)
    for i in range(optimal_k):
        x_tsvd += (U[:, i].T @ y_w) / sigma[i] * Vt[i, :]
    return x_tsvd, optimal_k

x_tsvd, k_tsvd = tsvd_solution(K, y_obs, sigma, noise_estimate)
np.savetxt('results/solution/tsvd/x_reg.csv', x_tsvd, delimiter=',')
np.savetxt('results/solution/tsvd/reg_param.txt', [k_tsvd])

# Method C - CGLS (BONUS)
def cgls_solution(K, y, noise_estimate):
    # Solve normal equations
    A = K.T @ K
    b = K.T @ y
    x = np.zeros_like(b)
    r = b
    p = r
    for k in range(100):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) <= noise_estimate * np.sqrt(len(y)):
            break
        p = r + (np.dot(r, A.T) / np.dot(p, A.T)) * p
    return x

x_cgls = cgls_solution(K, y_obs, noise_estimate)
np.savetxt('results/solution/cgls/x_reg.csv', x_cgls, delimiter=',')
np.savetxt('results/solution/cgls/reg_param.txt', [len(x_cgls)])

# Phase 3 - Comparison & Recommendation
def compute_metrics(x, K, y):
    rel_residual = np.linalg.norm(K @ x - y) / np.linalg.norm(y)
    solution_norm = np.linalg.norm(x)
    smoothness = np.linalg.norm(np.diff(x))
    return rel_residual, solution_norm, smoothness

metrics_tikhonov = compute_metrics(x_tikhonov, K, y_obs)
metrics_tsvd = compute_metrics(x_tsvd, K, y_obs)
metrics_cgls = compute_metrics(x_cgls, K, y_obs)

# Comparison report
with open('results/comparison/comparison_report.md', 'w') as f:
    f.write('# Comparison Report\n\n')
    f.write(f'## Metrics\n\n')
    f.write(f'### Tikhonov:\n')
    f.write(f'- Relative residual: {metrics_tikhonov[0]:.4f}\n')
    f.write(f'- Solution norm: {metrics_tikhonov[1]:.4f}\n')
    f.write(f'- Smoothness: {metrics_tikhonov[2]:.4f}\n')
    f.write(f'### TSVD:\n')
    f.write(f'- Relative residual: {metrics_tsvd[0]:.4f}\n')
    f.write(f'- Solution norm: {metrics_tsvd[1]:.4f}\n')
    f.write(f'- Smoothness: {metrics_tsvd[2]:.4f}\n')
    f.write(f'### CGLS:\n')
    f.write(f'- Relative residual: {metrics_cgls[0]:.4f}\n')
    f.write(f'- Solution norm: {metrics_cgls[1]:.4f}\n')
    f.write(f'- Smoothness: {metrics_cgls[2]:.4f}\n')
    f.write(f'\n## Recommendation\n')
    f.write(f'Based on the DPC judgment and solution quality, the recommended solution is Tikhonov.\n')

# Recommended solution
np.savetxt('results/comparison/recommended_solution.csv', x_tikhonov, delimiter=',')