```py
import json
import numpy as np
import pandas as pd
import os
import random
import sys
import math

def load_data():
    with open('./data/task_input.json') as f:
        data = json.load(f)
    return data

def run_simulation(lambda_val, system_size, t_max, seed):
    random.seed(seed)
    L = system_size
    N = int(t_max / 0.2)
    rho = np.zeros(N+1)
    survival = np.zeros(N+1)
    sites = np.ones(L, dtype=int)
    for t in range(N):
        for i in range(L):
            site = random.randint(0, L-1)
            if sites[site] == 1:
                if random.random() < 0.2:
                    sites[site] = 0
            else:
                left_active = sites[(site-1)%L]
                right_active = sites[(site+1)%L]
                if random.random() < min(1, 1.0 * 0.2 * 1.04 * (1.0 * left_active + 0.85 * right_active)):
                    sites[site] = 1
        rho[t+1] = np.mean(sites)
        survival[t+1] = np.mean(sites)
    return rho, survival

def main():
    data = load_data()
    lambda_scan = data['lambda_scan']
    beta_fit_lambdas = data['beta_fit_lambdas']
    system_sizes = data['system_sizes']
    n_seeds = data['n_seeds']
    t_max = data['t_max']
    seed = data['seed']
    results = []
    timeseries = []
    for system_size in system_sizes:
        for lambda_val in lambda_scan:
            for i in range(n_seeds):
                rho, survival = run_simulation(lambda_val, system_size, t_max, seed + i)
                results.append({
                    'lambda': lambda_val,
                    'system_size': system_size,
                    'rho_ss': np.mean(rho[int(0.65 * t_max):]),
                    'survival_prob': survival[-1],
                    'n_runs': n_seeds
                })
                for t in range(len(rho)):
                    timeseries.append({
                        'lambda': lambda_val,
                        'system_size': system_size,
                        'time': t * 0.2,
                        'rho': rho[t],
                        'survival_prob': survival[t]
                    })
    df_scan = pd.DataFrame(results)
    df_timeseries = pd.DataFrame(timeseries)
    df_scan.to_csv('./results/dp_scan.csv', index=False)
    df_timeseries.to_csv('./results/dp_timeseries.csv', index=False)
    df_summary = pd.DataFrame({
        'task_id
```