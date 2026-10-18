```python
import json
import numpy as np
import pandas as pd
import os
import random
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Load input
with open('./data/task_input.json', 'r') as f:
    protocol = json.load(f)

# Parameters from input
lambda_scan = protocol['lambda_scan']
beta_fit_lambdas = protocol['beta_fit_lambdas']
system_sizes = protocol['system_sizes']
n_seeds = protocol['n_seeds']
t_max = protocol['t_max']
seed = protocol['seed']

# Constants
wL = 1.0
wR = 0.85
s_even = 1.04
s_odd = 0.96
dt = 0.2

# Output directories
os.makedirs('./results', exist_ok=True)

# Simulation function
def simulate(lambda_val, system_size, n_seeds, t_max, seed):
    np.random.seed(seed)
    random.seed(seed)
    
    L = system_size
    timesteps = int(t_max / dt)
    time = np.linspace(0, t_max, timesteps)
    
    rho_timeseries = []
    survival_timeseries = []
    
    for _ in range(n_seeds):
        # Initialize fully active
        state = np.ones(L, dtype=int)
        
        rho = []
        survival = []
        
        for t in range(timesteps):
            # Random site visits
            sites = np.random.permutation(L)
            
            for i in sites:
                if state[i] == 1:
                    # Deactivation
                    if np.random.rand() < dt:
                        state[i] = 0
                else:
                    # Activation
                    left_active = state[(i - 1) % L]
                    right_active = state[(i + 1) % L]
                    if i % 2 == 0:
                        s = s_even
                    else:
                        s = s_odd
                    p_birth = min(1, lambda_val * dt * s * (wL * left_active + wR * right_active))
                    if np.random.rand() < p_birth:
                        state[i] = 1
            
            # Compute rho and survival
            rho_t = np.mean(state)
            survival_t = np.mean(state == 1)
            
            rho.append(rho_t)
            survival.append(survival_t)
        
        rho_timeseries.append(rho)
        survival_timeseries.append(survival)
    
    # Compute rho_ss and survival_prob
    rho_ss = []
    survival_prob = []
    
    for i in range(n_seeds):
        rho = rho_timeseries[i]
        survival = survival_timeseries[i]
        
        # Late-time window
        late_time = int(0.65 * timesteps)
        rho_ss.append(np.mean(rho[late_time:]))
        survival_prob.append(survival[-1])
    
    return time, np.array(rho_timeseries), np.array(survival_timeseries), np.array(rho_ss), np.array(survival_prob)

# Run simulations
results = {}
for system_size in system_sizes:
    for lambda_val in lambda_scan:
        print(f"Simulating for lambda={lambda_val}, system_size={system_size}")
        time, rho_timeseries, survival_timeseries, rho_ss, survival_prob = simulate(lambda_val, system_size, n_seeds, t_max, seed)
        
        # Save to dp_timeseries.csv
        df_timeseries = pd.DataFrame({
            'lambda': np.repeat(lambda_val, timesteps),
            'system_size': np.repeat(system_size, timesteps),
            'time': time,
            'rho': rho_timeseries.mean(axis=0),
            'survival_prob': survival_timeseries.mean(axis=0)
        })
        df_timeseries.to_csv('./results/dp_timeseries.csv', mode='a', header=False, index=False)
        
        # Save to dp_scan.csv
        df_scan = pd.DataFrame({
            'lambda': [lambda_val],
            'system_size': [system_size],
            'rho_ss': [rho_ss.mean()],
            'survival_prob': [survival_prob.mean()],
            'n_runs': [n_seeds]
        })
        df_scan.to_csv('./results/dp_scan.csv', mode='a', header=False, index=False)

# Estimate lambda_c and beta
lambda_c_estimate = 0.0
lambda_c_ci_low = 0.0
lambda_c_ci_high = 0.0
beta_estimate = 0.0
beta_ci_low = 0.0
beta_ci_high = 0.0
delta_estimate = 0.0
z_estimate = 0.0
collapse_success = False
universality_supported = False
static_evidence_consistent = False
dynamic_scale_observed = False
z_q_used = None
tau_q_half = None
assessment_note = ""

# Placeholder for the final report
dp_summary = {
    'task_id': 'task_001',
    'lambda_c_estimate': lambda_c_estimate,
    'lambda_c_ci_low': lambda_c_ci_low,
    'lambda_c_ci_high': lambda_c_ci_high,
    'beta_estimate': beta_estimate,
    'beta_ci_low': beta_ci_low,
    'beta_ci_high': beta_ci_high,
    'delta_estimate': delta_estimate,
    'z_estimate': z_estimate,
    'n_seeds': n_seeds,
    't_max': t_max,
    'collapse_success': collapse_success
}

dp_report = {
    'task_id': 'task_001',
    'model': 'asymmetric parity-modulated absorbing-state lattice process',
    'method_summary': 'Monte Carlo simulation with finite-size scaling analysis',
    'evidence': {
        'collapse_success': collapse_success,
        'universality_supported': universality_supported,
        'static_evidence_consistent': static_evidence_consistent,
        'dynamic_scale_observed': dynamic_scale_observed,
        'z_q_used': z_q_used,
        'tau_q_half': tau_q_half,
        'assessment_note': assessment_note
    },
    'conclusion': 'inconclusive'
}

# Save to CSV files
df_summary = pd.DataFrame([dp_summary])
df_summary.to_csv('./results/dp_summary.csv', index=False)
df_scan = pd.read_csv('./results/dp_scan.csv')
df_scan.to_csv('./results/dp_scan.csv', index=False)
df_timeseries = pd.read_csv('./results/dp_timeseries.csv')
df_timeseries.to_csv('./results/dp_timeseries.csv', index=False)

# Save to JSON
with open('./results/dp_report.json', 'w') as f:
    json.dump(dp_report, f, indent=2)
```