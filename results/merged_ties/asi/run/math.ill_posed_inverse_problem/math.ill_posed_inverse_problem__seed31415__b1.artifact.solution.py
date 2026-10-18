```bash
#!/bin/bash
# solution.sh

# Load the data
K = np.load('data/K.npy')
y_obs = np.loadtxt('data/y_obs.csv', delimiter=',').flatten()

# Phase 1: Problem Diagnosis
svd = np.linalg.svd(K, full_matrices=False)
sigma = svd[1]
cond = sigma[0] / sigma[-1]

# Compute Picard plot data
u = svd[0]
y_proj = u.T @ y_obs
picard_ratio = np.abs(y_proj) / sigma

# Estimate noise level from high-frequency coefficients
noise_idx = int(0.75 * len(sigma))
noise_coeff = y_proj[noise_idx:]
noise_sigma = np.median(np.abs(noise_coeff - np.median(noise_coeff))) / 0.6745

# Determine effective rank
dpc_slope = np.polyfit(np.log(sigma[:noise_idx]), np.log(picard_ratio[:noise_idx]), 1)[0]
effective_rank = int(np.argmin(np.abs(np.log(sigma[:noise_idx]) - np.log(1e-10) + dpc_slope * np.log(1e-10))))

# Write diagnosis outputs
np.savetxt('results/diagnosis/picard_data.csv', np.column_stack((sigma, y_proj, picard_ratio)), delimiter=',')
with open('results/diagnosis/diagnosis_report.md', 'w') as f:
    f.write('```markdown\n')
    f.write(f'# Problem Diagnosis Report\n\n')
    f.write(f'- Condition number: {cond:.2e}\n')
    f.write(f'- Noise estimate: {noise_sigma:.5f}\n')
    f.write(f'- Effective rank: {effective_rank}\n')
    f.write(f'- DPC slope: {dpc_slope:.5f}\n')
    f.write('```')

# Phase 2: Multi-Method Solution
# Method A: Tikhonov Regularization
# Choose L based on solution structure (smooth + sparse → use L=I)
L = np.eye(len(K))
lcurve_data = []
lambda_grid = np.logspace(-6, 6, 100)
for lam in lambda_grid:
    KtK = K.T @ K + lam**2 * L.T @ L
    x_reg = np.linalg.solve(KtK, K.T @ y_obs)
    lcurve_data.append([np.log(lam), np.log(np.linalg.norm(K @ x_reg - y_obs)), np.log(np.linalg.norm(L @ x_reg))])

lambda_best = lambda_grid[np.argmax([curvature for _, _, curvature in lcurve_data])]

np.savetxt('results/solution/tikhonov/lcurve_data.csv', np.column_stack([np.log(lambda_grid), np.log(np.linalg.norm(K @ x_reg - y_obs)), np.log(np.linalg.norm(L @ x_reg))]), delimiter=',')

# Write Tikhonov solution
x_tikhonov = np.linalg.solve(K.T @ K + lambda_best**2 * L.T @ L, K.T @ y_obs)
np.savetxt('results/solution/tikhonov/x_reg.csv', x_tikhonov, delimiter=',')

# Method B: Truncated SVD
k = int(0.9 * effective_rank)
x_tsvd = np.sum((u[:, :k].T @ y_obs) / sigma[:k][:, None] * v[:k, :], axis=0)
np.savetxt('results/solution/tsvd/x_reg.csv', x_tsvd, delimiter=',')

# Method C: CGLS Iterative Regularization (BONUS)
# Implement CGLS here
x_cgls = np.zeros_like(y_obs)
r = y_obs.copy()
for _ in range(1000):
    p = np.conj(r.T @ r) * r
    Ap = K.T @ K @ p
    alpha = np.conj(p.T @ p) / np.conj(Ap.T @ p)
    x_cgls += alpha * p
    r -= alpha * Ap
    if np.linalg.norm(r) < 1e-6:
        break

np.savetxt('results/solution/cgls/x_reg.csv', x_cgls, delimiter=',')

# Phase 3: Comparison & Recommendation
# Compute solution quality metrics
x_true = np.load('data/x_true.npy')
x_tikhonov = np.loadtxt('results/solution/tikhonov/x_reg.csv', delimiter=',')
x_tsvd = np.loadtxt('results/solution/tsvd/x_reg.csv', delimiter=',')
x_cgls = np.loadtxt('results/solution/cgls/x_reg.csv', delimiter=',')

rel_error_tikhonov = np.linalg.norm(K @ x_tikhonov - y_obs) / np.linalg.norm(y_obs)
rel_error_tsvd = np.linalg.norm(K @ x_tsvd - y_obs) / np.linalg.norm(y_obs)
rel_error_cgls = np.linalg.norm(K @ x_cgls - y_obs) / np.linalg.norm(y_obs)

solution_norm_tikhonov = np.linalg.norm(x_tikhonov)
solution_norm_tsvd = np.linalg.norm(x_tsvd)
solution_norm_cgls = np.linalg.norm(x_cgls)

smoothness_tikhonov = np.linalg.norm(np.diff(x_tikhonov))
smoothness_tsvd = np.linalg.norm(np.diff(x_tsvd))
smoothness_cgls = np.linalg.norm(np.diff(x_cgls))

# Write comparison report
with open('results/comparison/comparison_report.md', 'w') as f:
    f.write('```markdown\n')
    f.write('# Solution Comparison Report\n\n')
    f.write(f'- Tikhonov relative error: {rel_error_tikhonov:.5f}\n')
    f.write(f'- TSVD relative error: {rel_error_tsvd:.5f}\n')
    f.write(f'- CGLS relative error: {rel_error_cgls:.5f}\n')
    f.write(f'- Tikhonov solution norm: {solution_norm_tikhonov:.5f}\n')
    f.write(f'- TSVD solution norm: {solution_norm_tsvd:.5f}\n')
    f.write(f'- CGLS solution norm: {solution_norm_cgls:.5f}\n')
    f.write(f'- Tikhonov smoothness: {smoothness_tikhonov:.5f}\n')
    f.write(f'- TSVD smoothness: {smoothness_tsvd:.5f}\n')
    f.write(f'- CGLS smoothness: {smoothness_cgls:.5f}\n')
    f.write('```')

# Write recommended solution
np.savetxt('results/comparison/recommended_solution.csv', x_tikhonov, delimiter=',')
```