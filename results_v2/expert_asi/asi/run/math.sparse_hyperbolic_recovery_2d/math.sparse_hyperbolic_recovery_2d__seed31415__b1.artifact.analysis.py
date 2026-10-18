import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import least_squares
from sklearn.metrics import pairwise_distances_argmin_min

def load_data():
    obs = np.load('data/spacetime_observations.npz')
    t, x, y, u = obs['t'], obs['x'], obs['y'], obs['u']
    return t, x, y, u

def compute_noise_estimate(t, x, y, u, grid_spacing=0.1):
    # Use nearest neighbors within a small time and space window
    dist = np.sqrt((t[:, np.newaxis] - t[np.newaxis, :])**2 + 
                   (x[:, np.newaxis] - x[np.newaxis, :])**2 + 
                   (y[:, np.newaxis] - y[np.newaxis, :])**2)
    nearby = np.argsort(dist, axis=1)[:, :10]
    diff = u[nearby] - u[nearby[:, 0, np.newaxis]]
    return np.std(diff, axis=1).mean()

def fit_field(t, x, y, u, grid_shape=(128, 128), sigma_t=0.05, sigma_x=0.1):
    # Use kernel density estimation to fit the field
    grid = np.meshgrid(np.linspace(-2.8559, 2.8559, grid_shape[0]), 
                      np.linspace(-2.8559, 2.8559, grid_shape[1]), indexing='xy')
    grid_flat = np.stack([grid[0].ravel(), grid[1].ravel()], axis=1)
    interpolator = LinearNDInterpolator(np.column_stack((t, x, y)), u, rescale=True)
    return interpolator(grid_flat).reshape(grid_shape).astype(np.float32)

def detect_shocks(u, threshold=0.85):
    # Use contour detection to identify discontinuities
    grad = np.gradient(u)
    grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
    mask = grad_mag > np.percentile(grad_mag[grad_mag > 1e-6], threshold)
    return mask.astype(bool)

def main():
    t, x, y, u = load_data()
    # Step 3a: Exploration
    n_obs = len(t)
    u_min, u_max = np.min(u), np.max(u)
    noise_est = compute_noise_estimate(t, x, y, u)
    hist, bins = np.histogram(u, bins=10)
    with open('exploration_summary.json', 'w') as f:
        json.dump({
            'n_observations': n_obs,
            'u_range': [u_min, u_max],
            'noise_estimate': noise_est,
            'coverage_estimate': np.median(pairwise_distances_argmin_min(
                np.column_stack((x, y)), np.column_stack((x, y)))[0][:, 0])
        }, f)
    
    # Step 3b: Reconstruct field
    grid_shape = (128, 128)
    reconstructed = np.zeros((6, *grid_shape), dtype=np.float32)
    for k in range(6):
        t_k = np.linspace(0, 0.4637, 6)[k]
        points = np.column_stack((t_k * np.ones_like(x), x, y))
        interpolator = LinearNDInterpolator(points, u, rescale=True)
        grid = np.meshgrid(np.linspace(-2.8559, 2.8559, grid_shape[0]), 
                          np.linspace(-2.8559, 2.8559, grid_shape[1]), indexing='xy')
        reconstructed[k] = interpolator(np.column_stack((grid[0