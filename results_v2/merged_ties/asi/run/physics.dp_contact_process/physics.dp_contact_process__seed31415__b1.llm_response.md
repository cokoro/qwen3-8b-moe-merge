```bash
#!<abs_path>

cat > ./analysis.py << 'PYEOF'
import numpy as np
import pandas as pd
import json
import os
import math

def read_protocol():
    with open('./data/task_input.json') as f:
        data = json.load(f)
    return data['protocol']

def write_results(summary, scan, timeseries, report):
    os.makedirs('./results', exist_ok=True)
    summary.to_csv('./results/dp_summary.csv', index=False)
    scan.to_csv('./results/dp_scan.csv', index=False)
    timeseries.to_csv('./results/dp_timeseries.csv', index=False)
    with open('./results/dp_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def run_single_run(L, lambda_val, seed, t_max, dt=0.2):
    state = np.ones(L, dtype=int)
    t = 0.0
    time_series = []
    while t < t_max:
        t += dt
        site = np.random.randint(L)
        if state[site] == 1:
            if np.random.random() < dt:
                state[site] = 0
        else:
            left = state[(site - 1) % L]
            right = state[(site + 1) % L]
            s = 1.04 if site % 2 == 0 else 0.96
            wL = 1.0
            wR = 0.85
            p_birth = min(1.0, lambda_val * dt * s * (wL * left + wR * right))
            if np.random.random() < p_birth:
                state[site] = 1
        survival_prob = sum(state) / L
        time_series.append((t, np.sum(state) / L, survival_prob))
    return time_series

def compute_rho_ss(rho_series, t_threshold):
    rho_values = np.array([r for t, r, _ in rho_series])
    t_values = np.array([t for t, r, _ in rho_series])
    mask = t_values >= t_threshold
    return np.mean(rho_values[mask])

def compute_survival_prob(time_series):
    return time_series[-1][2]

def run_simulation(L, lambda_val, seed, t_max, n_seeds=1):
    all_rho_series = []
    all_survival_probs = []
    for seed in range(n_seeds):
        ts = run_single_run(L, lambda_val, seed, t_max)
        all_rho_series.append(ts)
        all_survival_probs.append(compute_survival_prob(ts))
    rho_series = pd.DataFrame(all_rho_series)
    rho_series.columns = ['time', 'rho', 'survival_prob']
    rho_ss = compute_rho_ss(rho_series.values, 0.65 * t_max)
    survival_prob = np.mean(all_survival_probs)
    return rho_ss, survival_prob

def run_full_protocol(protocol):
    lambda_scan = protocol['lambda_scan']
    beta_fit_lambdas = protocol['beta_fit_lambdas']
    system_sizes = protocol['system_sizes']
    n_seeds = protocol['n_seeds']
    t_max = protocol['t_max']
    seed = protocol['seed']

    timeseries = pd.DataFrame(columns=['lambda', 'system_size', 'time', 'rho', 'survival_prob'])
    scan = pd.DataFrame(columns=['lambda', 'system_size', 'rho_ss', 'survival_prob', 'n_runs'])

    for L in system_sizes:
        for lambda_val in lambda_scan:
            for seed_run in range(n_seeds):
                ts = run_single_run(L, lambda_val, seed_run, t_max)
                timeseries = pd.concat([timeseries, pd.DataFrame({
                    'lambda': lambda_val,
                    'system_size': L,
                    'time': ts[0],
                    'rho': ts[1],
                    'survival_prob': ts[2]
                })], ignore_index=True)

            rho_ss, survival_prob = run_simulation(L, lambda_val, seed, t_max, n_seeds)
            scan = pd.concat([scan, pd.DataFrame({
                'lambda': lambda_val,
                'system_size': L,
                'rho_ss': rho_ss,
                'survival_prob': survival_prob,
                'n_runs': n_seeds
            })], ignore_index=True)

    beta_estimates = {}
    for L in beta_fit_lambdas:
        beta_values = []
        for lambda_val in lambda_scan:
            if lambda_val in beta_fit_lambdas[L]:
                rho_series = scan[(scan['lambda'] == lambda_val) & (scan['system_size'] == L)]
                rho_ss = rho_series['rho_ss'].values[0]
                beta = np.log(rho_ss) / np.log(lambda_val - lambda_c_estimate)
                beta_values.append(beta)
        beta_estimates[L] = np.mean(beta_values)

    delta_estimates = {}
    for lambda_val in lambda_scan:
        rho_series = scan[(scan['lambda'] == lambda_val) & (scan['system_size'] == system_sizes[-1])]
        rho_ss = rho_series['rho_ss'].values[0]
        t_values = timeseries[(timeseries['lambda'] == lambda_val) & (timeseries['system_size'] == system_sizes[-1])]['time'].values
        rho_values = timeseries[(timeseries['lambda'] == lambda_val) & (timeseries['system_size'] == system_sizes[-1])]['rho'].values
        mask = t_values >= 0.65 * t_max
        rho_values = rho_values[mask]
        t_values = t_values[mask]
        delta = np.polyfit(np.log(t_values), np.log(rho_values), 1)[0]
        delta_estimates[lambda_val] = delta

    z_estimate = None
    z_q = 0.5
    tau_q_half = {}
    for L in system_sizes:
        survival_series = scan[(scan['system_size'] == L)]['survival_prob'].values
        t_values = scan[(scan['system_size'] == L)]['time'].values
        tau_q = np.log(1 - z_q) / np.log(survival_series)
        tau_q_half[L] = np.mean(tau_q)
        if z_estimate is None:
            z_estimate = tau_q_half[L] / system_sizes[-1]

    return scan, beta_estimates, delta_estimates, z_estimate, tau_q_half

if __name__ == "__main__":
    protocol = read_protocol()
    scan, beta_estimates, delta_estimates, z_estimate, tau_q_half = run_full_protocol(protocol)
    summary = pd.DataFrame({
        'task_id': ['task_1'],
        'lambda_c_estimate': [lambda_c_estimate],
        'lambda_c_ci_low': [lambda_c_ci_low],
        'lambda_c_ci_high': [lambda_c_ci_high],
        'beta_estimate': [beta_estimate],
        'beta_ci_low': [beta_ci_low],
        'beta_ci_high': [beta_ci_high],
        'delta_estimate': [delta_estimate],
        'z_estimate': [z_estimate],
        'n_seeds': [protocol['n_seeds']],
        't_max': [protocol['t_max']],
        'collapse_success': [collapse_success]
    })
    write_results(summary, scan, timeseries, report)
PYEOF
```