```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.linalg import qr
from scipy.optimize import least_squares
import json

def load_data(file_path):
    return pd.read_csv(file_path)

def interpolate_trajectory(probe_data, time_resolution=0.01):
    probe_data.sort_values('t', inplace=True)
    times = np.unique(probe_data['t'].values)
    num_points = len(times)
    full_times = np.arange(times[0], times[-1] + time_resolution, time_resolution)
    full_data = np.zeros((len(full_times), 4))
    for i, t in enumerate(full_times):
        valid_rows = probe_data[probe_data['t'] == t]
        for j, col in enumerate(['x1', 'x2', 'x3', 'x4']):
            if not valid_rows[col].isnull().all():
                full_data[i, j] = np.mean(valid_rows[col])
    return full_data

def compute_derivatives(data, time_resolution):
    dt = time_resolution
    dqdt = (data[1:] - data[:-1]) / dt
    dpdt = (data[1:] - data[:-1]) / dt
    return dqdt, dpdt

def compute_forces(data, k1, k2, a, b):
    q1, q2 = data[:, 0], data[:, 2]
    dqdt = (data[1:] - data[:-1]) / 0.01
    dpdt = (data[1:] - data[:-1]) / 0.01
    force1 = -k1*q1 - 2*a*q1*q2
</think>

```