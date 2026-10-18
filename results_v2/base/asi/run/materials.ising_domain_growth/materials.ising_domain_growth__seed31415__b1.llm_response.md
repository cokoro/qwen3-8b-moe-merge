```python
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import linregress
from scipy.interpolate import interp1d
import os

# Load parameters
with open('data/parameters.json') as f:
    params = json.load(f)
L = params['grid_size']
T = params['control']
n_steps = params['n_steps']
n_snapshots = params['n_snapshots']
n_measurements = params['n_measurements']
snapshot_times = params['snapshot_times']
measurement_times = params['measurement_times']
max_r = params['max_r']
dynamics_seed = params['dynamics_seed']

# Load initial state
initial_state = np.load('data/initial_state.npy')

# Initialize RNG
rng = np.random.default_rng(dynamics_seed)

# Initialize spin configuration
spin = initial_state.copy()

# Precompute checkerboard sublattices
ii, jj = np.mgrid[0:L, 0:L]
sublattice0 = (ii + jj) % 2 == 0
sublattice1 = (ii + jj) % 2 == 1

# Periodic boundary conditions helper
def neighbor(x, y):
    return [(x + dx) % L for dx in [-1, 1]], [(y + dy) % L for dy in [-1, 1]]

# Glauber dynamics simulation
def update_spin(spin, T, sublattice_mask, rng):
    L = spin.shape[0]
    indices = np.where(sublattice_mask)[0]
    for idx in indices:
        x, y = np.unravel_index(idx, (L, L))
        s = spin[x, y]
        neighbors = [spin[(x + dx) % L, (y + dy) % L] for dx, dy in neighbor(x, y)]
        delta_E = 2 * s * sum(neighbors)
        if T > 0:
            p = 1 / (1 + np.exp(delta_E / (T)))
            if rng.random() < p:
                spin[x, y] *= -1
        else:
            if delta_E < 0:
                spin[x, y] *= -1
            elif delta_E == 0:
                if rng.random() < 0.5:
                    spin[x, y] *= -1
    return spin

# Main simulation loop
spin_history = []
for step in range(n_steps):
    if step-1 in the step in measurement_times:
        spin_history.append(spin.copy())
    if step in snapshot_times:
        # Compute correlation function and structure factor
        pass
    # Update sublattices
    spin = update_spin(spin, T, sublattice0, rng)
    spin = update_spin(spin, T, sublattice1, rng)

# Compute correlation function
def compute_correlation(spin):
    L = spin.shape[0]
    fft_spin = fft2(spin)
    corr = ifft2(np.abs(1 * fft_spin)**2).real
    m = np.mean(spin)
    corr -= m**2
    return corr

# Compute structure factor
def compute_structure_factor(spin):
    L = spin.shape[0]
    fft_spin = fft2(spin)
    structure_factor = np.abs(fft_spin)**2
    k_x = np.fft.fftfreq(L) * 2 * np.pi
    k_y = np.fft.fftfreq(L) * 2 * np.pi
    k = np.sqrt(k_x[:, np.newaxis]**2 + k_y[np.newaxis, :]**2)
    return k, structure_factor

# Compute domain size
def compute_domain_size(corr):
    r_values = np.arange(0, max_r +  # 1)
    C_r = np.zeros(r_values.shape)
    for r in range(1, r_values.shape[0]):
        r_mask = np.sqrt((np.arange(L)[:, np.newaxis] - np.arange(L))**2 + (np.arange(L)[np.newaxis, :] - np.arange(L))**2) == r
        C_r[r] = np.mean(corr[r_mask])
    L_t = np.interp(0.5, C_r, r_values)
    return L_t

# Continue with the rest of the simulation, correlation, structure factor, and domain size calculations
```