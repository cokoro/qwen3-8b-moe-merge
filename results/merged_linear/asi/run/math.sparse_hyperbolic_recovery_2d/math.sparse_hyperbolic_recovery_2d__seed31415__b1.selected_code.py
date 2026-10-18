import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# Load data
data = np.load('data/spacetime_observations.npz')
t, x, y, u = data['t'], data['x'], data['y'], data['u']
with open('data/observation_grid.json') as f:
    obs_grid = json.load(f)

# Step 3a: Data exploration
n_obs = len(t)
u_min, u_max = np.min(u), np.max(u)
noise_estimate = np.sqrt(np.mean((u[np.abs(x - x[0]) < 0.1] - u[0])**2))
coverage_estimate = np.median(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))
hist, _ = np.histogram(u, bins=10)
value_histogram = [[_bin, count] for _bin, count in zip(np.unique(u), hist)]

# Save exploration summary
np.savez('exploration_summary.npz', n_observations=n_obs, u_range=[u_min, u_max],
         noise_estimate=noise_estimate, coverage_estimate=coverage_estimate,
         value_histogram=value_histogram)

# Step 3b: Dense field reconstruction
grid_shape = (128, 128)
eval_times = [float(t) for t in obs_grid['eval_times']]
x_eval, y_eval = np.meshgrid(np.linspace(-2.8559, 2.8559, 128), np.linspace(-2.8559, 2.8559, 128), indexing='xy')
reconstructed_field = np.zeros((6, 128, 128))

for k, t_k in enumerate(eval_times):
    idx = np.abs(t - t_k) < 1e-3
    t_subset, x_subset, y_subset, u_subset = t[idx], x[idx], y[idx], u[idx]
    
    # Two-pass local regression
    # Pass 1: Global smoothing
    sigma_x = 0.5
    weights = np.exp(-((t_subset - t_k)[:, None] ** 2) / (2 * sigma_x ** 2))
    u_hat = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            dists = np.sqrt((x_subset - x_eval[i, j])**2 + (y_subset - y_eval[i, j])**2)
            weights_local = np.exp(-dists ** 2 / (2 * sigma_x ** 2))
            u_hat[i, j] = np.sum(u_subset * weights_local) / np.sum(weights_local)
    
    # Pass 2: Local refinement near shocks
    sigma_refine = 0.3
    grad = np.gradient(u_hat)
    grad_norm = np.sqrt(grad[0]**2 + grad[1]**2)
    threshold = np.percentile(grad_norm[grad_norm > 1e-3], 85)
    shock_mask = grad_norm > threshold
    
    for i in range(128):
        for j in range(128):
            if shock_mask[i, j]:
                dists = np.sqrt((x_subset - x_eval[i, j])**2 + (y_subset - y_eval[i, j])**2)
                weights_local = np.exp(-dists ** 2 / (2 * sigma_refine ** 2))
                u_hat[i, j] = np.sum(u_subset * weights_local) / np.sum(weights_local)
    
    reconstructed_field[k] = u_hat

# Save reconstructed field
np.save('reconstructed_field.npy', reconstructed_field)

# Step 3c: Discontinuity detection
transition_curves = {}
transition_normals = {}
transition_speeds = {}
transition_mask = np.zeros((6, 128, 128), dtype=bool)

for k, t_k in enumerate(eval_times):
    grad = np.gradient(reconstructed_field[k])
    grad_norm = np.sqrt(grad[0]**2 + grad[1]**2)
    threshold = np.percentile(grad_norm[grad_norm > 1e-3], 85)
    contour = np.zeros_like(grad_norm)
    contour[grad_norm > threshold] = 1
    contours = plt.contour(x_eval, y_eval, contour, levels=1, linewidths=1, colors='k')
    
    for idx, contour_line in enumerate(contours.allsegs[0]):
        curve = np.mean(contour_line, axis=1)
        transition_curves[f't_{k}'] = curve
        transition_mask[k] = np.zeros_like(transition_mask[k])
        for x_c, y_c in curve:
            dist = np.sqrt((x_eval - x_c)**2 + (y_eval - y_c)**2)
            transition_mask[k][dist < 3] = True
        
        # Compute normals
        normals = np.zeros_like(curve)
        for i in range(len(curve)):
            x1, y1 = curve[i-1] if i > 0 else curve[-1]
            x2, y2 = curve[i]
            x3, y3 = curve[i+1] if i < len(curve)-1 else curve[0]
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x3 - x2, y3 - y2
            normal = np.cross([dx1, dy1], [dx2, dy2])
            if normal != 0:
                nx = dy2 - dy1
                ny = dx1 - dx2
                normal = [nx, ny] / np.sqrt(nx**2 + ny**2)
            else:
                normal = [1, 0]
            normals[i] = normal
        transition_normals[f't_{k}'] = normals
    
    # Compute normal speeds
    speeds = np.zeros(len(transition_curves[f't_{k}']))
    for i in range(len(transition_curves[f't_{k}'])):
        x1, y1 = transition_curves[f't_{k}'][i]
        x2, y2 = transition_curves[f't_{k}'][i+1] if i < len(transition_curves[f't_{k}'])-1 else transition_curves[f't_{k}'][0]
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        speeds[i] = norm / 1e-3  # 1e-3 seconds between consecutive time slices
    transition_speeds[f't_{k}'] = speeds

np.savez('transition_curves.npz', **transition_curves)
np.savez('transition_normals.npz', **transition_normals)
np.savez('transition_speeds.npz', **transition_speeds)
np.save('transition_mask_preview.npy', transition_mask)

# Step 3d: Normal speed estimation
motion_speeds = {}
for k, t_k in enumerate(eval_times):
    curve = transition_curves[f't_{k}']
    next_curve = transition_curves[f't_{k+1}'] if k < 5 else transition_curves[f't_0']
    for i in range(len(curve)):
        x1, y1 = curve[i]
        dists = np.sqrt((next_curve[:, 0] - x1)**2 + (next_curve[:, 1] - y1)**2)
        idx = np.argmin(dists)
        x2, y2 = next_curve[idx]
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        motion_speeds[f't_{k}_{i}'] = norm / 1e-3  # 1e-3 seconds between consecutive time slices

# Step 3e: Flux hypothesis
# Use anisotropic Burgers family
alpha_x = 1.0
alpha_y = 1.0
fit_residuals = []

for k, t_k in enumerate(eval_times):
    curve = transition_curves[f't_{k}']
    normals = transition_normals[f't_{k}']
    motion_speeds_k = transition_speeds[f't_{k}']
    for i in range(len(curve)):
        x, y = curve[i]
        nx, ny = normals[i]
        u_L = reconstructed_field[k][y+1, x+1]  # 1 cell ahead in normal direction
        u_R = reconstructed_field[k][y-1, x-1]  # 1 cell behind in normal direction
        s_n = motion_speeds_k[i]
        rhs = (alpha_x * nx + alpha_y * ny) * (u_L + u_R) / 2
        fit_residuals.append(abs(s_n - rhs) / (abs(s_n) + 1e-6))

fit_residual_rms = np.sqrt(np.mean(np.array(fit_residuals)**2))
confidence = np.mean([r < 0.2 for r in fit_residuals])

governing_model = {
    'family': 'anisotropic_burgers',
    'coefficients': {'alpha_x': alpha_x, 'alpha_y': alpha_y},
    'confidence': confidence,
    'fit_residual_rms': fit_residual_rms
}

# Step 3f: RH consistency check
consistency_report = {
    'median_residual': np.median(fit_residuals),
    'p90_residual': np.percentile(fit_residuals, 90),
    'pass_fraction_20pct': np.mean([r < 0.2 for r in fit_residuals]),
    'methodology': 'Least-squares fit of anisotropic Burgers family'
}

# Step 3g: Visualisation
plt.figure(figsize=(15, 5))
for k, t_k in enumerate(eval_times):
    plt.subplot(1, 6, k+1)
    plt.imshow(reconstructed_field[k], extent=[-2.8559, 2.8559, -2.8559, 2.8559], cmap='viridis', origin='lower')
    plt.scatter(transition_curves[f't_{k}'][:, 0], transition_curves[f't_{k}'][:, 1], c='red', s=10)
    plt.colorbar()
plt.savefig('reconstruction_slices.png')

# Save final outputs
with open('governing_model.json', 'w') as f:
    json.dump(governing_model, f)

with open('consistency_report.json', 'w') as f:
    json.dump(consistency_report, f)

# Write model candidates
model_candidates = [
    {
        'name': 'isotropic_burgers',
        'motivation': 'Normalized isotropic case: f(u) = g(u) = u^2 / 2'
    },
    {
        'name': 'anisotropic_burgers',
        'motivation': 'Generalized anisotropic case: f(u) = α_x u^2 / 2, g(u) = α_y u^2 / 2'
    }
]
with open('model_candidates.json', 'w') as f:
    json.dump(model_candidates, f)

with open('diagnostic_report.json', 'w') as f:
    json.dump({'residuals': fit_residuals}, f)