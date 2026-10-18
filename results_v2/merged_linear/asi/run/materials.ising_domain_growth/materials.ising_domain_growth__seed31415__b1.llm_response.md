```python
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy import signal

with open('data/parameters.json') as f:
    params = json.load(f)

L = params['grid_size']
T = params['control']
n_steps = params['n_steps']
n_snapshots = params['n_snapshots']
n_measurements = params['n_measurements']
snapshot_times = np.array(params['snapshot_times'])
measurement_times = np.array(params['measurement_times'])
max_r = params['max_r']
dynamics_seed = params['dynamics_seed']

# Read initial state
initial_state = np.load('data/initial_state.npy')

# Initialize RNG
rng = np.random.default_rng(dynamics_seed)

# Precompute checkerboard sublattice indices
ii, jj = np.mgrid[0:L, 0:L]
sublattice0_mask = (ii + jj) % 2 == 0
sublattice1_mask = (ii + jj) % 2 == 1

# Initialize spin configuration
spin_config = initial_state.copy()

# Precompute neighbor offsets
neighbors = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1)
]

```