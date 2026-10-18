```python
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import jackknife
from scipy.signal import find_peaks

# Load input files
with open('data/control_primary.json') as f:
    control = json.load(f)

scan_axis_a = np.load('data/scan_axis_a.npy')
scan_axis_b = np.load('data/scan_axis_b.npy')
probe_axis_a = np.load('data/probe_axis_a.npy')
state_tokens = np.load('data/state_tokens.npz')

# Constants
J = 1.0
T_values = scan_axis_a
L_values = scan_axis_b
r_values = probe_axis_a

# Initialize observables
obs_matrix_a = np.zeros((len(T_values), len(L_values)))  # Helicity modulus
obs_matrix_b = np.zeros((len(T_values), len(L_values)))  # Defect density
obs_tensor_c = np.zeros((len(control['focus_probe_rows']), len(r_values)))  # Radial correlations
obs_matrix_d = np.zeros((len(T_values), 2))  # eta(T)
obs_matrix_e = np.zeros((len(T_values), len
```