```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import
import json
import os
import time
import warnings

# Load data
data = np.load('data/spacetime_observations.npz')
t = data['t']
x = data['x']
y = data['y']
u = data['u']

# Read grid description
with open('data/observation_grid.json') as f:
    grid_desc = json.load(f)
grid_shape = grid_desc['grid_shape']
```
```python
# 3a. Data exploration
n_observations = len(u)
u_min, u_max = np.min(u), np.max(u)
noise_estimate = np.sqrt(np.mean((u[::2] - u[1::2])**2))
coverage_estimate = np.median(np.sqrt((x[::2] - x[1::2])**2 + (y[::2] - y[1::2])**2))

# Histogram of u values
hist_bins = np.linspace(u_min, u_max, 10)
value_histogram = [[hist_bins[i], np.sum((u >= hist_bins[i]) & (u < hist_bins[i+1]))] for i in range(9)]

exploration_summary = {
    `n_observations`: n_observations,
    `u_range`: [u_min, u_max],
    `noise_estimate`: noise_estimate,
    `coverage_estimate`: coverage_estimate,
    `value_histogram`: value_histogram

```
```python
# Write exploration summary
np.savez('exploration_summary.npz', **exploration_summary)

# 3b. Dense field reconstruction
grid_desc = json.load(open('data/observation_grid.json'))
t_eval = np.array(grid_desc['evaluation_times'])
grid_shape = grid_desc['grid_shape']
x_eval = np.linspace(grid_desc['domain'][0][0], grid_desc['domain'][0][1], grid_shape[1])
y_eval = np.linspace(grid_desc['domain'][1][0], grid_desc['domain'][1][1], grid_shape[0])

reconstructed = np.zeros((len(t_eval), *grid_shape))
for k in range(len(t_eval)):
    t_k = t_eval[k]
    indices = np.argmin(np.abs(t - t_k))
    t_weights = np.exp(-0.5 * ((t - t_k) / 0.01)**2)  # Bandwidth 0.01
    x_weights = np.exp(-0.5 * ((x - x_eval[None, :]) / 0.1)**2)  # Bandwidth 0.1
    y_weights = np.exp(-0.5 * ((y - y_eval[None, :]) / 0.1)**2)
    w = t_weights[:, None, None] * x_weights[:, :, None] * y_weights[:, :, None]
    w /= w.sum(axis=(1, 2))[None, :, None]
    reconstructed[k] = (w * u[:, None, None]).sum(axis=0)

np.save('reconstructed_field.npy', reconstructed)

# 3c. Discontinuity detection
curves = []
normals = []
transition_mask = np.zeros((len(t_eval), *grid_shape), dtype=bool)
for k in range(len(t_eval)):
    grad = np.gradient(reconstructed[k])
    grad_magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
    threshold = np.percentile(grad_magnitude[grad_magnitude > 0.01], 85)
    mask = grad_magnitude > threshold
    contours = np.argwhere(mask)
    for contour in contours:
        curve = np.vstack([contour[:, 0], contour[:, 1]]).T
        n = np.gradient(curve, axis=0)
        n /= np.linalg.norm(n, axis=1)[:, None]
        normals_k = n
        curves_k = curve
        curves.append(curves_k)
        normals.append(normals_k`)

```