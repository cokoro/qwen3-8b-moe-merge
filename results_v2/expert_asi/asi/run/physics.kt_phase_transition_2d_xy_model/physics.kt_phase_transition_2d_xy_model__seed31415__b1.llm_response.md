```python
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

# Load control parameters
with open('data/control_primary.json') as f:
    control = json.load(f)

# Load temperature grid
temp_grid = np.load('data/scan_axis_a.npy')
lattice_sizes = np.load('data/scan_axis_b.npy')
probe_r = np.load('data/probe_axis_a.npy')

# Load initial state tokens
state_tokens = np.load('data/state_tokens.npz').keys()

# Simulate the XY model for each lattice size and temperature
obs_matrix_a = np.zeros((len(temp_grid), len(lattice_sizes)))
obs_matrix_b = np.zeros((len(temp_grid), len(lattice_sizes)))
obs_tensor_c = np.zeros((len(temp_grid), len(probe_r)))
obs_matrix_d = np.zeros((len(temp_grid), 2))
obs_matrix_e = np.zeros((len(temp_grid), len(lattice_sizes)))
obs_matrix_f = np.zeros((len(temp_grid), len(lattice_sizes)))

# Define the hybrid Monte Carlo schedule
def run_simulation

```