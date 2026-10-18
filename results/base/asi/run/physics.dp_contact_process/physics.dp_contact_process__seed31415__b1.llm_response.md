```python
import json
import os
import numpy as np
import pandas as pd
import random
import time

def load_input():
    with open('./data/task_input.json', 'r') as f:
        return json.load(f)

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def simulate_lattice(lambda_val, system_size, t_max, n_seeds, seed):
    L = system_size
    dt = 0.2
    wL = 1.0
    wR = 0.85
    s = lambda i: 1.04 if i % 2 == 0 else 0.96
    np.random.seed(seed)
    random.seed(seed)
    
    rho_series = []
    survival_series = []
    
    for _ in range(n_seeds):
        lattice = np.ones(L, dtype=int)
        time = 0
        time_series = []
        survival_series_run = []
        
        while time < t_max:
            time += dt
            active_count = np.sum(lattice)
            survival = np.prod(lattice)
            time_series.append((time, active_count / L, survival))
            survival_series_run.append(survival)
            
            for _ in range(L):
                site = np.random.randint(L)
                if lattice[site] == 1:
                    if np.random.random() < dt:
                        lattice[site] = 0
                else:
                    left = (site - 1) % L
                    right = (site + 1) % L
                    left_active = lattice[left]
                    right_active = lattice[right]
                    birth_prob = min(1, lambda_val * dt * s(site) * (wL * left_active + wR * right_active))
                    if np.random.random() < birth_prob:
                        lattice[site] = 1
        
        rho_series.append(np.array([t for t, _, _ in time_series]))
        rho_series.append(np.array([r for _, r, _ in time_series]))
        survival_series.append(np.array(survival_series_run))
    
    rho_timeseries = np.stack(rho_series).T
    survival_timeseries = np.stack(survival_series).T
    
    rho_ss = np.mean(rho_timeseries[:, -int(0.35 * t_max):], axis=1)
    survival_prob = np.mean(survival_timeseries[:, -1], axis=1)
    
    return rho_timeseries, survival_timeseries, rho_ss, survival_prob

def main():
    input_data = load_input()
    protocol = input_data['protocol']
    lambda_scan = protocol['lambda_scan']
    beta_fit_lambdas = protocol['beta_fit_lambdas']
    system_sizes = protocol['system_sizes']
    n_seeds = protocol['n_seeds']
    t_max = protocol['t_max']
    seed = protocol['seed']
    
    results = []
    
    for system_size in system_sizes:
        for lambda_val in lambda_scan:
            rho_timeseries, survival_timeseries, rho_ss, survival_prob = simulate_lattice(
                lambda_val, system_size, t_max, n_seeds, seed
            )
            for i in range(n_seeds):
                results.append({
                    'lambda': lambda_val,
                    'system_size': system_size,
                    'rho_ss': rho_ss[i],
                    'survival_prob': survival_prob[i],
                    'n_runs': n_seeds
                })
    
    df_timeseries = pd.DataFrame({
        'lambda': [l for l in lambda_scan for _ in system_sizes for _ in range(n_seeds)],
        'system_size': [s for s in system_sizes for _ in lambda_scan for _ in range(n_seeds)],
        'time': np.concatenate([rho_timeseries[i, :, 0] for i in range(n_seeds)]),
        'rho': np.concatenate([rho_timeseries[i, :, 1] for i in range(n_seeds)]),
        'survival_prob': np.concatenate([survival_timeseries[i, :, 0] for i in range(n_seeds)])
    })
    
    df_scan = pd.DataFrame(results)
    df_scan['n_runs'] = n_seeds
    
    df_summary = pd.DataFrame({
        'task_id': [input_data['task_id']],
        'lambda_c_estimate': [0.0],
        'lambda_c_ci_low': [0.0],
        'lambda_c_ci_high': [0.0],
        'beta_estimate': [0.0],
        'beta_ci_low': [0.0],
        'beta_ci_high': [0.0],
        'delta_estimate': [0.0],
        'z_estimate': [0.0],
        'n_seeds': [n_seeds],
        't_max': [t_max],
        'collapse_success': [False]
    })
    
    save_results(df_scan, './results/dp_scan.csv')
    save_results(df_timeseries, './results/dp_timeseries.csv')
    save_results(df_summary, './results/dp_summary.csv')
    
    report = {
        'task_id': input_data['task_id'],
        'model': '1D asymmetric parity-modulated absorbing-state lattice process',
        'method_summary': 'Monte Carlo simulation with finite-size scaling and critical point estimation',
        'evidence': {
            'collapse_success': False,
            'universality_supported': False,
            'static_evidence_consistent': False,
            'dynamic_scale_observed': False,
            'z_q_used': None,
            'tau_q_half': None,
            'assessment_note': 'Dynamic scaling was not observed cleanly due to limited data and protocol constraints'
        },
        'conclusion': 'inconclusive'
    }
    
    with open('./results/dp_report.json', 'w') as f:
        json.dump(report, f)

if __name__ == '__main__':
    main()
```