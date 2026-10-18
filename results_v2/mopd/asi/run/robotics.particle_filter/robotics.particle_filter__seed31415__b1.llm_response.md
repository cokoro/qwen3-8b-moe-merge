```python
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json
import math

# Load data
data = np.load('data/measurements.npz')
Y = data['Y']
t = data['t']
mask = data.get('mask', None)
if Y.ndim == 2:
    Y = Y.T
if mask is not None and mask.ndim == 2:
    mask = mask.T

# Set up parameters
dt = np.mean(np.diff(t))
num_particles = 1500
process_noise = np.diag([0.1, 0.1, 0.1, 0.1])
measurement_noise = np.diag([0.1, 0.1])
angle_wrap = np.pi
angle_wrap_event_count = 0
used_log_weights = False
weight_degeneracy_detected = False
resampling_count = 0
angle_wrap_detected = False
max_log_weight = -np.inf
min_log_weight = np.inf
weight_entropy = 0
ess = 0
resampled = False
mean_nees = 0
median_nees = 0
pos_rmse = 0
vel_rmse = 0
ess_min = 0
ess_mean = 0
resampling_count = 0
weight_degeneracy_detected = False
used_log_weights = False
angle_wrap_detected = False
angle_wrap_event_count = 0

# Initialize particles
px = np.random.uniform(-100, 100, num_particles)
py = np.random.uniform(-100, 100, num_particles)
vx = np.random.uniform(-10, 10, num_particles)
vy = np.random.uniform(-10, 10, num_particles)
particles = np.column_stack((px, py, vx, vy))
log_weights = np.zeros(num_particles)

# Process measurements
state_estimates = []
filter_diagnostics = []

# Evaluate initial particles
for i in range(num_particles):
    if not np.isnan(Y[0, 0]):
        range_obs = np.exp(Y[0, 0] - 0.08) - 1.5
        if range_obs > 0:
            px[i] = range_obs * np.cos(Y[0, 1])
            py[i] = range_obs * np.sin(Y[0, 1])
    if not np.isnan(Y[0, 1]):
        bearing_obs = Y[0, 1] - np.arctan2(py[i], px[i])
        if bearing_obs > angle_wrap:
            bearing_obs -= 2 * np.pi
        elif bearing_obs < -angle_wrap:
            bearing_obs += 2 * np.pi
        if bearing_obs != 0:
            log_weights[i] = norm.logpdf(bearing_obs, 0, 1)

# Normalize log weights
log_weights -= np.log(np.sum(np.exp(log_weights)))

# Initialize results
state_estimates = []
filter_diagnostics = []

# Run particle filter
for i in range(len(t)):
    # Propagate particles
    px_new = px + vx * dt
    py_new = py + vy * dt
    vx_new = vx + np.random.normal(0, process_noise[0], num_particles)
    vy_new = vy + np.random.normal(0, process_noise[1], num_particles)
    particles = np.column_stack((px_new, py_new, vx_new, vy_new))
    
    # Evaluate likelihoods
    log_weights_new = np.zeros(num_particles)
    for j in range(num_particles):
        if not np.isnan(Y[i, 0]):
            range_obs = np.exp(Y[i, 0] - 0.08) - 1.5
            if range_obs > 0:
                range_pred = np.sqrt(px_new[j]**2 + py_new[j]**2)
                log_weights_new[j] += norm.logpdf(range_obs, range_pred, measurement_noise[0])
        if not np.isnan(Y[i, 1]):
            bearing_obs = Y[i, 1] - np.arctan2(py_new[j], px_new[j])
            if bearing_obs > angle_wrap:
                bearing_obs -= 2 * np.pi
            elif bearing_obs < -angle_wrap:
                bearing_obs += 2 * np.pi
            if bearing_obs != 0:
                log_weights_new[j] += norm.logpdf(bearing_obs, 0, measurement_noise[1])
    
```