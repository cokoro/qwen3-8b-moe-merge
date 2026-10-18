```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    K = np.load('data/K.npy')
    y = pd.read_csv('data/y_obs.csv').values
    return K, y

def svd_analysis(K):
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    cond_num = S[0] / S[-1]
    return U, S, Vt, cond_num

def picard_plot(U, S, y, output_path='results/diagnosis/picard_data.csv'):
    n = len(S)
    picard_data = []
    for i in range(n):
        u_i = U[:, i]
        fourier = np.abs(u_i @ y)
        picard = fourier / S[i]
        picard_data.append((S[i], fourier, picard))
    df = pd.DataFrame(picard_data, columns=['singular_value', 'fourier_coeff', 'picard_ratio'])
    df.to_csv(output_path, index=False)
    return df

def estimate_noise(y, U, S, effective_rank):
    n = len(S)
    noise_idx = int(0.75 * n)
    noise_coeffs = np.abs(U[:, noise_idx:] @ y)
    sigma_noise = np.median(np.abs(noise_coeffs - np.median(noise_coeffs)))
    return sigma_noise

def effective_rank(df, noise_level):
    k_eff = 0
    for i, row in df.iterrows():
        if row['picard_ratio'] > 1e-3 * noise_level:
            k_eff += 1
        else:
            break
    return k_eff

def diagnose(K, y):
    U, S, Vt, cond_num = svd_analysis(K)
    df = picard_plot(U, S, y)
    noise_level = estimate_noise(y, U, S, effective_rank(df, 1e-3))
    k_eff = effective_rank(df, noise_level)
    report = {
        'noise_estimate': noise_level,
        'effective_rank': k_eff,
        'dpc_judgment': 'holds' if k_eff < len(S) // 2 else 'does_not_hold'
    }
    with open('results/diagnosis/diagnosis_report.md', 'w') as f:
        f.write(f'# Diagnosis Report\n\nNoise estimate: {noise_level:.3e}\nEffective rank: {k_eff}\nDPC judgment: {report['dpc_judgment']}\n')
    return U, S, Vt, cond_num, noise_level, k_eff

def whiten(K, y, U, S, noise_level):
    n = len(S)
    noise_idx = int(0.75 * n)
    noise_coeffs = np.abs(U[:, noise_idx:] @ y)
    sigma_noise = np.median(np.abs(noise_coeffs - np.median(noise_coeffs)))
    rho = np.corrcoef(noise_coeffs[1:], noise_coeffs[:-1])[0, 1]
    W = np.eye(n) - np.diagflat([1 - rho] * n) + np.diagflat([rho] * (n - 1), k=1)
    K_w = W @ K
    y_w = W @ y
    return K_w, y_w, sigma_noise

def tikhonov(K_w, y_w, U, S, Vt, noise_level, k_eff):
    L = np.eye(len(K_w))  # Use identity for sparse structure
    lambda_grid = np.logspace(-15, 1, 100)
    lcurve_data = []
    for lam in lambda_grid:
        KtK = K_w.T @ K_w + lam**2 * L.T @ L
        Kty = K_w.T @ y_w
        x_reg = np.linalg.solve(KtK, Kty)
        lcurve_data.append((np.log(lam), np.log(np.linalg.norm(K_w @ x_reg - y_w)**2), np.log(np.linalg.norm(L @ x_reg)**2)))
    lcurve_df = pd.DataFrame(lcurve_data, columns=['log_lambda', 'log_residual', 'log_solution'])
    lcurve_df.to_csv('results/solution/tikhonov/lcurve_data.csv', index=False)
    # Select lambda with maximum curvature
    d2 = np.gradient(np.gradient(lcurve_df['log_residual'], lcurve_df['log_lambda']), lcurve_df['log_lambda'])
    lam_opt = lambda_grid[np.argmax(d2)]
    x_reg = np.linalg.solve(K_w.T @ K_w + lam_opt**2 * L.T @ L, K_w.T @ y_w)
    np.savetxt('results/solution/tikhonov/x_reg.csv', x_reg.reshape(-1, 1), delimiter=',')
    np.savetxt('results/solution/tikhonov/reg_param.txt', [lam_opt], delimiter=' ')
    return x_reg

def tsvid(K_w, y_w, U, S, Vt, noise_level, k_eff):
    k_grid = np.arange(1, len(S)+1)
    gcv_data = []
    for k in k_grid:
        x_k = U[:, :k] @ np.diag(1/S[:k]) @ U[:, :k].T @ y_w
        r_k = y_w - K_w @ x_k
        gcv = np.linalg.norm(r_k)**2 / (len(y_w) - k)**2
        gcv_data.append((k, gcv))
    k_opt = np.argmin(gcv_data[-1][1])
    x_reg = U[:, :k_opt] @ np.diag(1/S[:k_opt]) @ U[:, :k_opt].T @ y_w
    np.savetxt('results/solution/tsvd/x_reg.csv', x_reg.reshape(-1, 1), delimiter=',')
    np.savetxt('results/solution/tsvd/reg_param.txt', [k_opt], delimiter=' ')
    return x_reg

def compare_solutions(x_tikhonov, x_tsvd, K, y, x_true=None):
    rel_resid = np.linalg.norm(K @ x_tikhonov - y) / np.linalg.norm(y)
    norm_tikhonov = np.linalg.norm(x_tikhonov)
    smoothness_tikhonov = np.linalg.norm(np.diff(x_tikhonov))
    rel_resid_tsvd = np.linalg.norm(K @ x_tsvd - y) / np.linalg.norm(y)
    norm_tsvd = np.linalg.norm(x_tsvd)
    smoothness_tsvd = np.linalg.norm(np.diff(x_tsvd))
    report = {
        'rel_resid': rel_resid,
        'norm': norm_tikhonov,
        'smoothness': smoothness_tikhonov,
        'rel_resid_tsvd': rel_resid_tsvd,
        'norm_tsvd': norm_tsvd,
        'smoothness_tsvd': smoothness_tsvd
    }
    with open('results/comparison/comparison_report.md', 'w') as f:
        f.write('# Comparison Report\n\n')
        f.write(f'Tikhonov: rel_resid={report['rel_resid']:.3e}, norm={report['norm']:.3e}, smoothness={report['smoothness']:.3e}\n')
        f.write(f'TSVD: rel_resid={report['rel_resid_tsvd']:.3e}, norm={report['norm_tsvd']:.3e}, smoothness={report['smoothness_tsvd']:.3e}\n')
        f.write('Recommended solution: tikhonov\n')
    np.savetxt('results/comparison/recommended_solution.csv', x_tikhonov.reshape(-1, 1), delimiter=',')
    return report

def main():
    os.makedirs('results/diagnosis', exist_ok=True)
    os.makedirs('results/solution', exist_ok=True)
    os.makedirs('results/comparison', exist_ok=True)
    
    K, y = load_data()
    U, S, Vt, cond_num, noise_level, k_eff = diagnose(K, y)
    K_w, y_w, sigma_noise = whiten(K, y, U, S, noise_level)
    
    x_tikhonov = tikhonov(K_w, y_w, U, S, Vt, noise_level, k_eff)
    x_tsvd = tsvid(K_w, y_w, U, S, Vt, noise_level, k_eff)
    
    compare_solutions(x_tikhonov, x_tsvd, K, y)

if __name__ == '__main__':
    main()
```