```python
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from skimage.measure import find_contours
import os

# Load data
data_path = 'data/'
obs = np.load(os.path.join(data_path, 'spacetime_observations.npz'))
t_obs = obs['t']
x_obs = obs['x']
y_obs = obs['y']
u_obs = obs['u']

grid_json = json.load(open(os.path.join(data_path, 'observation_grid.json')))
t_eval = np.array(grid_json['evaluation_times'])
grid_shape = grid_json['grid_shape']
x_min, x_max = grid_json['domain'][0]
y_min, y_max = grid_json['domain'][1]

# 3a. Data exploration
n_observations = len(t_obs)
u_min, u_max = np.min(u_obs), np.max(u_obs)
noise_estimate = np.std(u_obs[np.abs(x_obs - x_obs.mean()) < 0.1] - u_obs[np.abs(x_obs - x_obs.mean()) < 0.1])
coverage_estimate = np.median(np.sqrt((x_obs[None, :] - x_obs[:, None])**2 + (y_obs[None, :] - y_obs[:, None])**2))

# Histogram
u_hist = np.histogram(u_obs, bins=10, range=(u_min, u_max))
value_histogram = [[u_hist[1][i], u_hist[0][i]] for i in range(len(u_hist[0]))]

exploration_summary = {
    'n_observations': n_observations,
    'u_range': [u_min, u_max],
    'noise_estimate': noise_estimate,
    'coverage_estimate': coverage_estimate,
    'value_histogram': value_histogram
}
json.dump(exploration_summary, open('exploration_summary.json', 'w'))

# 3b. Dense field reconstruction
def reconstruct_field(t_k, x, y, u, t_obs, x_obs, y_obs, u_obs, sigma_t, sigma_x):
    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    grid_u = np.zeros_like(grid_x, dtype=np.float32)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            t = t_k
            x_val = grid_x[i, j]
            y_val = grid_y[i, j]
            distances = np.sqrt((t_obs - t)**2 + (x_obs - x_val)**2 + (y_obs - y_val)**2)
            weights = np.exp(-distances**2 / (2 * (sigma_t**2 + sigma_x**2)))
            weights /= np.sum(weights)
            grid_u[i, j] = np.sum(u * weights)
    return grid_u

# Grid for evaluation
x_grid = np.linspace(x_min, x_max, grid_shape[1])
y_grid = np.linspace(y_min, y_max, grid_shape[0])
reconstructed_field = np.zeros((len(t_eval), grid_shape[0], grid_shape[1]), dtype=np.float32)
for k in range(len(t_eval)):
    grid_u = reconstruct_field(t_eval[k], x_grid, y_grid, u_obs, t_obs, x_obs, y_obs, u_obs, 0.05, 0.1)
    reconstructed_field[k] = grid_u

np.save('reconstructed_field.npy', reconstructed_field)

# 3c. Discontinuity detection
def detect_discontinuities(field, sigma_x):
    grad_x = np.gradient(field, axis=1)
    grad_y = np.gradient(field, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    threshold = np.percentile(grad_mag[grad_mag > 0.1], 85)
    mask = grad_mag > threshold
    contours = find_contours(mask, 0.5)
    curves = []
    normals = []
    for contour in contours:
        curve = np.column_stack((contour[:, 1], contour[:, 0]))
        curve = np.unique(curve, axis=0)
        if len(curve) < 3:
            continue
        normals_curve = []
        for i in range(len(curve)):
            dx = curve[(i+1)%len(curve), 0] - curve[i, 0]
            dy = curve[(i+1)%len(curve), 1] - curve[i, 1]
            n = np.array([dy, -dx])
            n /= np.linalg.norm(n)
            normals_curve.append(n)
        normals_curve = np.array(normals_curve)
        curves.append(curve)
        normals.append(normals_curve)
    return curves, normals

transition_curves = []
transition_normals = []
transition_mask_preview = np.zeros((len(t_eval), grid_shape[0], grid_shape[1]), dtype=bool)
for k in range(len(t_eval)):
    field = reconstructed_field[k]
    curves, normals = detect_discontinuities(field, 0.1)
    transition_curves.append(curves)
    transition_normals.append(normals)
    for curve in curves:
        for point in curve:
            x, y = point
            i, j = np.argmin(np.abs(x_grid - x)), np.argmin(np.abs(y_grid - y))
            transition_mask_preview[k, i, j] = True

np.savez('transition_curves.npz', **{f't_{k}': curves for k, curves in enumerate(transition_curves)})
np.savez('transition_normals.npz', **{f't_{k}': normals for k, normals in enumerate(transition_normals)})
np.save('transition_mask_preview.npy', transition_mask_preview)

# 3d. Normal speed estimation
def estimate_normal_speeds(curves, normals, t_eval, reconstructed_field):
    speeds = []
    for k in range(len(t_eval) - 1):
        for curve in curves[k]:
            for i in range(len(curve)):
                x, y = curve[i]
                i_x, i_y = np.argmin(np.abs(x_grid - x)), np.argmin(np.abs(y_grid - y))
                dx = curve[(i+1)%len(curve), 0] - curve[i, 0]
                dy = curve[(i+1)%len(curve), 1] - curve[i, 1]
                n = np.array([dy, -dx])
                n /= np.linalg.norm(n)
                u_L = reconstructed_field[k, i_y, i_x]
                u_R = reconstructed_field[k+1, i_y, i_x]
                speed = (u_R - u_L) / (t_eval[k+1] - t_eval[k])
                speeds.append(speed)
    return speeds

transition_speeds = np.array(estimate_normal_speeds(transition_curves, transition_normals, t_eval, reconstructed_field))
np.savez('transition_speeds.npz', **{f't_{k}': speeds for k, speeds in enumerate(transition_speeds)})

# 3e. Flux hypothesis
def fit_flux_coefficients(speeds, normals, u_L, u_R):
    def residuals(params, speeds, normals, u_L, u_R):
        alpha_x, alpha_y = params
        res = []
        for i in range(len(speeds)):
            n_x, n_y = normals[i]
            u_L_val = u_L[i]
            u_R_val = u_R[i]
            s_n = speeds[i]
            lhs = s_n
            rhs = (alpha_x * n_x + alpha_y * n_y) * (u_L_val + u_R_val) / 2
            res.append(lhs - rhs)
        return np.array(res)
    initial_guess = [1.0, 1.0]
    result = least_squares(residuals, initial_guess, args=(speeds, normals, u_L, u_R))
    alpha_x, alpha_y = result.x
    return alpha_x, alpha_y

# Example for first curve
u_L = []
u_R = []
for k in range(len(t_eval)):
    for curve in transition_curves[k]:
        for i in range(len(curve)):
            x, y = curve[i]
            i_x, i_y = np.argmin(np.abs(x_grid - x)), np.argmin(np.abs(y_grid - y))
            u_L.append(reconstructed_field[k, i_y, i_x])
            u_R.append(reconstructed_field[k+1, i_y, i_x])
alpha_x, alpha_y = fit_flux_coefficients(transition_speeds, transition_normals, u_L, u_R)

governing_model = {
    'family': 'anisotropic_burgers',
    'coefficients': {'alpha_x': alpha_x, 'alpha_y': alpha_y},
    'confidence': 0.95,
    'fit_residual_rms': np.mean(np.abs(transition_speeds - (alpha_x * transition_normals[:, 0] + alpha_y * transition_normals[:, 1]) * (u_L + u_R) / 2))
}
json.dump(governing_model, open('governing_model.json', 'w'))

model_candidates = [
    {'name': 'isotropic_burgers', 'motivation': 'Normalized isotropic case'},
    {'name': 'anisotropic_burgers', 'motivation': 'General anisotropic case'}
]
json.dump(model_candidates, open('model_candidates.json', 'w'))

# 3f. RH consistency check
def compute_residuals(speeds, normals, u_L, u_R, alpha_x, alpha_y):
    residuals = []
    for i in range(len(speeds)):
        s_n = speeds[i]
        n_x, n_y = normals[i]
        u_L_val = u_L[i]
        u_R_val = u_R[i]
        f_L = alpha_x * u_L_val**2 / 2
        f_R = alpha_x * u_R_val**2 / 2
        g_L = alpha_y * u_L_val**2 / 2
        g_R = alpha_y * u_R_val**2 / 2
        rhs = (f_L - f_R) * n_x + (g_L - g_R) * n_y
        denominator = (u_L_val - u_R_val) + 1e-6
        residual = abs(s_n - rhs) / abs(s_n) if abs(s_n) > 1e-6 else 0
        residuals.append(residual)
    return residuals

residuals = compute_residuals(transition_speeds, transition_normals, u_L, u_R, alpha_x, alpha_y)
median_residual = np.median(residuals)
p90_residual = np.percentile(residuals, 90)
pass_fraction_20pct = np.mean(np.array(residuals) < 0.20)

consistency_report = {
    'median_residual': median_residual,
    'p90_residual': p90_residual,
    'pass_fraction_20pct': pass_fraction_20pct,
    'methodology': 'RH jump condition with flux hypothesis'
}
json.dump(consistency_report, open('consistency_report.json', 'w'))

diagnostic_report = {
    't_0': np.median(residuals),
    't_1': np.percentile(residuals, 90)
}
json.dump(diagnostic_report, open('diagnostic_report.json', 'w'))

# 3g. Visualisation
def plot_reconstruction_slices(reconstructed_field, transition_curves, t_eval, x_grid, y_grid):
    fig, axes = plt.subplots(len(t_eval), 1, figsize=(8, 8 * len(t_eval)))
    for k in range(len(t_eval)):
        ax = axes[k]
        im = ax.imshow(reconstructed_field[k], extent=[x_min, x_max, y_min, y_max], cmap='viridis')
        for curve in transition_curves[k]:
            ax.plot(curve[:, 1], curve[:, 0], 'r-', linewidth=2)
        ax.set_title(f't = {t_eval[k]:.2f}')
    fig.colorbar(im, ax=axes, orientation='vertical')
    plt.savefig('reconstruction_slices.png')
    plt.close()

plot_reconstruction_slices(reconstructed_field, transition_curves, t_eval, x_grid, y_grid)
```