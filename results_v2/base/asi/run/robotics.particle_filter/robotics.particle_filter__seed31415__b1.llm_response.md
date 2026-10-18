```python
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import norm, chi2
from scipy.interpolate import interp1d

# Load data
data_path = 'data/measurements.npz'
with np.load(data_path) as data:
    Y = data['Y']
    t = data['t']
    mask = data.get('mask', None)
    if Y.shape[0] == 2:
        Y = Y.T

# Determine time spacing
dt = np.mean(np.diff(t))
T = len(t)

# Initialize results directories
os.makedirs('results', exist_ok=True)

# Particle filter parameters
num_particles = 1500
process_noise = 0.1
measurement_noise = 0.1
angle_wrap = np.pi
angle_bias_drift = 0.01
resampling_threshold = 0.5
log_weight_threshold = -1000

# Initialize particles
particles = np.zeros((num_particles, 4))
log_weights = np.zeros(num_particles) - log_weight_threshold

# Initialize state estimates
state_estimates = []
filter_diagnostics = []
summary = {
    'pos_rmse': 0.0,
    'vel_rmse': 0.0,
    'mean_nees': 0.0,
    'median_nees': 0.0,
    'ess_min': 0.0,
    'ess_mean': 0.0,
    'resampling_count': 0,
    'weight_degeneracy_detected': False,
    'used_log_weights': False,
    'angle_wrapping_detected': False,
    'angle_wrap_event_count': 0
}

# Process each time step
for i in range(T):
    time = t[i]
    y = Y[i]
    mask_current = mask[i] if mask is not None else np.full_like(y, True)
    
    # Initialize particles if first time step
    if i == 0:
        # Invert range channel
        range_channel = y[0]
        if np.isfinite(range_channel):
            range_val = np.exp(range_channel - 0.08) - 1.5
            px = range_val * np.cos(y[1])
            py = range_val * np.sin(y[1])
        else:
            px, py = 0.0, 0.0
        # Initialize velocity
        vx, vy = 0.0, 0.0
        # Initialize particles
        for p in range(num_particles):
            particles[p, 0] = np.random.normal(px, 0.1)
            particles[p, 1] = np.random.normal(py, 0.1)
            particles[p, 2] = np.random.normal(vx, 0.1)
            particles[p, 3] = np.random.normal(vy, 0.1)
    else:
        # Propagate particles
        for p in range(num_particles):
            px, py, vx, vy = particles[p]
            px_new = px + vx * dt + np.random.normal(0, process_noise)
            py_new = py + vy * dt + np.random.normal(0, process_noise)
            vx_new = vx + 0.0 + np.random.normal(0, process_noise)
            vy_new = vy + 0.0 + np.random.normal(0, process_noise)
            particles[p] = [px_new, py_new, vx_new, vy_new]
    
    # Compute log weights
    for p in range(num_particles):
        px, py, vx, vy = particles[p]
        log_weight = 0.0
        # Range channel
        if np.isfinite(y[0]):
            range_val = np.exp(y[0] - 0.08) - 1.5
            range_pred = np.hypot(px, py)
            log_weight += np.log(norm.pdf(range_pred, range_val, measurement_noise))
```