import json
import numpy as np
import pandas as pd
from numba import njit, prange
import os

def read_input():
    with open('./data/task_input.json') as f:
        data = json.load(f)
    return data

def write_output(data, results):
    os.makedirs('./results', exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv('./results/fluid_summary.csv', index=False)
    df_force = pd.DataFrame(results)
    df_force.to_csv('./results/fluid_force_series.csv', index=False)
    df_centerline = pd.DataFrame(data_centerline)
    df_centerline.to_csv('./results/fluid_centerline.csv', index=False)
    df_transverse = pd.DataFrame(data_transverse)
    df_transverse.to_csv('./results/fluid_transverse_profile.csv', index=False)
    with open('./results/fluid_report.json', 'w') as f:
        json.dump({'task_id': '1', 'units': 'dimensionless', 'cd_mean_avg': np.mean([d['cd_mean'] for d in data]), 'method_summary': 'D2Q9 TRT-LBM solver'}, f)

def equilibrium_cell(rho, ux, uy):
    c = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cxy = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    feq = np.zeros(9)
    for i in range(9):
        ci = c[i]
        cix = cxy[i][0
        cixy

def initialize`