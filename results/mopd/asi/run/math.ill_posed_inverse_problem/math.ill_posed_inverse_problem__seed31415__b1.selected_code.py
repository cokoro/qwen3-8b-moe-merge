import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LassoLarsCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Phase 1: Problem Diagnosis
K = np.load('data/K.npy')
y_obs = np.loadtxt('data/y_obs.csv')
U, S, Vt = np.linalg.svd(K, full_matrices=False)
sigma = np.array(S)
cond_number = sigma[0] / sigma[-1]
picard_data = []
for i in range(len(sigma)):
    fourier_coeff = abs(U[:, i].T @ y_obs)
    picard_ratio = fourier_coeff / sigma[i]
    picard_data.append((sigma[i], fourier_coeff, picard_ratio))
picard_df = pd.DataFrame(picard_data, columns=['singular_value', 'fourier_coeff', 'picard_ratio'])
picard_df.to_csv('results/diagnosis/picard_data.csv', index=False)
noise_estimate = np.median(np.abs(picard_df['fourier_coeff'].iloc[-25:]/sigma[-25:])) * 1.4826
effective_rank = len([x for x in sigma if x > noise_estimate * np.sqrt(10)])
dpc_judgment = 'holds' if effective_rank < len(sigma) * 0.7 else 'does_not_hold'
with open('results/diagnosis/diagnosis_report.md', 'w') as f:
    f.write(f'noise_estimate: {noise_estimate}\n')
    f.write(f'effective_rank: {effective_rank}\n')
    f.write(f'dpc_judgment: {dpc_judgment}\n')

# Phase 2: Multi-Method Solution
# Tikhonov Regularization
K_w = np.linalg.inv(np.eye(len(K)) - 0.5 * np.eye(len(K), k=1) - 0.5 * np.eye(len(K), k=-1)) @ K
y_w = np.linalg.inv(np.eye(len(K)) - 0.5 * np.eye(len(K), k=1) - 0.5 * np.eye(len(K), k=-1)) @ y_obs
l_curve_data = []
for lmbda in np.logspace(-15, 1, 100):
    x = np.linalg.inv(K_w.T @ K_w + lmbda**2 * np.eye(len(K))) @ K_w.T @ y_w
    res_norm = np.linalg.norm(K_w @ x - y_w)**2
    sol_norm = np.linalg.norm(x)**2
    l_curve_data.append((np.log(lmbda), res_norm, sol_norm))
l_curve_df = pd.DataFrame(l_curve_data, columns=['log_lambda', 'res_norm', 'sol_norm'])
l_curve_df.to_csv('results/solution/tikhonov/lcurve_data.csv', index=False)
curvature = []
for i in range(1, len(l_curve_df) - 1):
    d2r = l_curve_df['res_norm'].iloc[i+1] - 2*l_curve_df['res_norm'].iloc[i] + l_curve_df['res_norm'].iloc[i-1]
    d2s = l_curve_df['sol_norm'].iloc[i+1] - 2*l_curve_df['sol_norm'].iloc[i] + l_curve_df['sol_norm'].iloc[i-1]
    curvature.append((l_curve_df['log_lambda'].iloc[i], abs(d2r / d2s)))
curvature_df = pd.DataFrame(curvature, columns=['log_lambda', 'curvature'])
optimal_lambda = curvature_df['log_lambda'].iloc[curvature_df['curvature'].idxmax()]
with open('results/solution/tikhonov/reg_param.txt', 'w') as f:
    f.write(f'{optimal_lambda}\n')
x_tikhonov = np.linalg.inv(K_w.T @ K_w + optimal_lambda**2 * np.eye(len(K))) @ K_w.T @ y_w
np.savetxt('results/solution/tikhonov/x_reg.csv', x_tikhonov, delimiter=',')

# TSVD
k = 50
x_tsvd = np.zeros_like(y_obs)
for i in range(k):
    x_tsvd += (U[:, i].T @ y_w / sigma[i]) * Vt[i, :]
np.savetxt('results/solution/tsvd/x_reg.csv', x_tsvd, delimiter=',')
with open('results/solution/tsvd/reg_param.txt', 'w') as f:
    f.write(f'{k}\n')

# Phase 3: Comparison & Recommendation
x_true = np.load('data/x_true.npy')
rel_error_tikhonov = mean_squared_error(x_true, x_tikhonov) / mean_squared_error(x_true, np.zeros_like(x_true))
rel_error_tsvd = mean_squared_error(x_true, x_tsvd) / mean_squared_error(x_true, np.zeros_like(x_true))
comparison_df = pd.DataFrame({
    'method': ['Tikhonov', 'TSVD'],
    'relative_error': [rel_error_tikhonov, rel_error_tsvd],
    'solution_norm': [np.linalg.norm(x_tikhonov), np.linalg.norm(x_tsvd)],
    'smoothness': [np.linalg.norm(np.diff(x_tikhonov)), np.linalg.norm(np.diff(x_tsvd))],
    'computational_cost': [1, 1]
})
comparison_df.to_csv('results/comparison/comparison_report.md', index=False)
if rel_error_tikhonov < rel_error_tsvd:
    np.savetxt('results/comparison/recommended_solution.csv', x_tikhonov, delimiter=',')
else:
    np.savetxt('results/comparison/recommended_solution.csv', x_tsvd, delimiter=',')