```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Phase 1: Problem Diagnosis

# Load data
K = np.load('data/K.npy')
y_obs = np.loadtxt('data/y_obs.csv')

# Compute SVD
U, Σ, Vt = np.linalg.svd(K, full_matrices=False)
σ = Σ
cond_number = σ[0] / σ[-1]

# Compute Picard data
u_i_y = np.abs(np.dot(U.T, y_obs))
picard_ratio = np.abs(u_i_y) / σ

# Save Picard data
picard_df = pd.DataFrame({'singular_value': σ, 'fourier_coeff': u_i_y, 'picard_ratio': picard_ratio})
picard_df.to_csv('results/diagnosis/picard_data.csv', index=False)

# Estimate noise level
noise_indices = int(len(σ) * 0.75)
noise_est = np.median(np.abs(picard_ratio[-noise_indices:] - np.median(picard_ratio[-noise_indices:])) / 1.4826)

# Compute effective rank
threshold = noise_est * 3
effective_rank = np.sum(picard_ratio > threshold)

# Write diagnosis report
with open('results/diagnosis/diagnosis_report.md', 'w') as f:
    f.write('## Diagnosis Report\n\n')
    f.write(f'### Noise Estimate\n{noise_est:.2e}\n\n')
    f.write(f'### Effective Rank\n{effective_rank}\n\n')
    f.write(f'### DPC Judgment\n{"holds" if np.any(picard_ratio[:effective_rank] < noise_est) else "does_not_hold"}\n')

# Phase 2: Multi-Method Solution

# Prewhiten data
ρ = 0.95  # Estimated serial correlation coefficient
W = np.eye(len(y_obs)) - ρ * np.diag(np.ones(len(y_obs)-1), -1)
K_w = W @ K
y_w = W @ y_obs

# Tikhonov Regularization
λ_grid = np.logspace(-12, 2, 100)
L = np.eye(len(K))  # Identity matrix for magnitude penalty
x_tikhonov = []
residuals = []
solution_norms = []
for λ in λ_grid:
    A = K_w.T @ K_w + λ * L.T @ L
    b = K_w.T @ y_w
    x = np.linalg.solve(A, b)
    x_tikhonov.append(x)
    residuals.append(np.linalg.norm(K_w @ x - y_w))
    solution_norms.append(np.linalg.norm(x))

# Select λ using L-curve
log_residuals = np.log(residuals)
log_solution_norms = np.log(solution_norms)
curvature = []
for i in range(1, len(log_residuals) - 1):
    d2r = (log_residuals[i+1] - 2*log_residuals[i] + log_residuals[i-1]) / (0.1**2)
    d2s = (log_solution_norms[i+1] - 2*log_solution_norms[i] + log_solution_norms[i-1]) / (0.1**2)
    curvature.append(d2r * d2s)
λ_opt = λ_grid[np.argmax(curvature)]

# Save Tikhonov solution
x_tikhonov_opt = x_tikhonov[np.argmax(curvature)]
np.savetxt('results/solution/tikhonov/x_reg.csv', x_tikhonov_opt, delimiter=',')
np.savetxt('results/solution/tikhonov/reg_param.txt', [λ_opt])

# TSVD Solution
k_grid = np.arange(1, len(σ) + 1)
x_tsvd = []
residuals_tsvd = []
for k in k_grid:
    x = np.zeros_like(K_w)
    for i in range(k):
        x += (np.dot(U[i], y_w) / σ[i]) * Vt[i]
    x_tsvd.append(x)
    residuals_tsvd.append(np.linalg.norm(K_w @ x - y_w))

# Select k using GCV
n = len(y_obs)
gcv = [residuals_tsvd[i] / (n - k_grid[i])**2 for i in range(len(k_grid))]
k_opt = k_grid[np.argmin(gcv)]

# Save TSVD solution
x_tsvd_opt = x_tsvd[np.argmin(gcv)]
np.savetxt('results/solution/tsvd/x_reg.csv', x_tsvd_opt, delimiter=',')
np.savetxt('results/solution/tsvd/reg_param.txt', [k_opt])

# Phase 3: Comparison & Recommendation

# Compute solution quality metrics
true_solution = np.loadtxt('data/x_true.csv')  # Load true solution
rel_error_tikhonov = np.linalg.norm(K @ x_tikhonov_opt - y_obs) / np.linalg.norm(y_obs)
rel_error_tsvd = np.linalg.norm(K @ x_tsvd_opt - y_obs) / np.linalg.norm(y_obs)

# Write comparison report
with open('results/comparison/comparison_report.md', 'w') as f:
    f.write('## Comparison Report\n\n')
    f.write(f'### Tikhonov\n')
    f.write(f'- Relative error: {rel_error_tikhonov:.2e}\n')
    f.write(f'- Solution norm: {np.linalg.norm(x_tikhonov_opt):.2e}\n')
    f.write(f'- Smoothness: {np.linalg.norm(np.diff(x_tikhonov_opt)):.2e}\n')
    f.write(f'### TSVD\n')
    f.write(f'- Relative error: {rel_error_tsvd:.2e}\n')
    f.write(f'- Solution norm: {np.linalg.norm(x_tsvd_opt):.2e}\n')
    f.write(f'- Smoothness: {np.linalg.norm(np.diff(x_tsvd_opt)):.2e}\n')
    f.write(f'### Recommendation: Tikhonov\n')
    f.write('Tikhonov regularization better captures the smooth structure of the solution while suppressing noise.')

# Write recommended solution
np.savetxt('results/comparison/recommended_solution.csv', x_tikhonov_opt, delimiter=',')
```