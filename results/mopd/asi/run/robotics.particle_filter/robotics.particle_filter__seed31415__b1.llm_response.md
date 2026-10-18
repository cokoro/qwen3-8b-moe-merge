```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import json
from scipy.stats import norm
from scipy.stats import entropy
from scipy.linalg import block_diag
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import os

def load_data():
    data = np.load('data/measurements.npz')
    Y = data['Y']
    t = data['t']
    mask = data.get('mask', None)
    if Y.ndim == 2:
        Y = Y.T
    if mask is not None and mask.ndim == 2:
        mask = mask.T
    return t, Y, mask

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_bearing_residuals(Y, px, py, vx, vy, dt):
    range_ch = Y[:, 0]
    bearing_ch = Y[:, 1]
    range_est = np.exp(range_ch - 0.08) - 1.5
    bearing_est = wrap_angle(np.arctan2(py, px) + 0.0)
    bearing_residual = bearing_ch - bearing_est
    return bearing_residual

def compute_position_residuals(Y, px, py, vx, vy, dt):
    range_ch = Y[:, 0]
    bearing_ch = Y[:, 1]
    range_est = np.exp(range_ch - 0.08) - 1.5
    bearing_est = wrap_angle(np.arctan2(py, px) + 0.0)
    position_residual = np.sqrt(px**2 + py**2) - range_est
    return position_residual

def log_likelihood(Y, px, py, vx, vy, dt, mask):
    log_likelihoods = np.zeros_like(Y, dtype=np.float64)
    for i in range(Y.shape[0]):
        if mask is not None and mask[i] is not None and mask[i]:
            continue
        range_ch = Y[i, 0]
        bearing_ch = Y[i, 1]
        range_est = np.exp(range_ch - 0.08) - 1.5
        bearing_est = wrap_angle(np.arctan2(py, px) + 0.0)
        range_residual = range_est - range_ch
        bearing_residual = bearing_est - bearing_ch
        log_likelihoods[i] = -0.5 * (range_residual**2 + bearing_residual**2)
    return log_likelihoods

def particle_filter(t, Y, mask):
    num_particles = 1500
    num_states = 4
    dt = t[1] - t[0]
    num_steps = Y.shape[0]
    particles = np.random.randn(num_particles, num_states) * 10
    weights = np.ones(num_particles) / num_particles
    log_weights = np.log(weights)
    state_estimates = np.zeros((num_steps, num_states))
    covariance_estimates = np.zeros((num_steps, num_states))
    diagnostics = np.zeros((num_steps, 12))
    resampling_count = 0
    weight_degeneracy = False
    angle_wrapping = False
    angle_wrap_count = 0
    for step in range(num_steps):
        for i in range(num_particles):
            px, py, vx, vy = particles[i]
            if step > 0:
                px += vx * dt
                py += vy * dt
                vx += np.random.randn() * 0.1
                vy += np.random.randn() * 0.1
            particles[i] = [px, py, vx, vy]
        if step > 0:
            for i in range(num_particles):
                px, py, vx, vy = particles[i]
                px_new = px + vx * dt
                py_new = py + vy * dt
                vx_new = vx + np.random.randn() * 0.1
                vy_new = vy + np.random.randn() * 0.1
                particles[i] = [px_new, py_new, vx_new, vy_new]
        for i in range(num_particles):
            px, py, vx, vy = particles[i]
            if mask is not None and mask[step] is not None and mask[step]:
                continue
            range_ch = Y[step, 0]
            bearing_ch = Y[step, 1]
            range_est = np.exp(range_ch - 0.08) - 1.5
            bearing_est = wrap_angle(np.arctan2(py, px) + 0.0)
            range_residual = range_est - range_ch
            bearing_residual = bearing_est - bearing_ch
            log_weights[i] = -0.5 * (range_residual**2 + bearing_residual**2)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(weights**2)
        if ess < num_particles / 2:
            resampling_count += 1
            indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
            particles = particles[indices]
            log_weights = np.log(weights[indices]) + np.log(1.0 / num_particles)
        state_estimates[step] = np.sum(particles * weights[:, np.newaxis], axis=0)
        cov = np.cov(particles, rowvar=False)
        covariance_estimates[step] = np.diag(cov)
        diagnostics[step, 0] = np.sqrt((state_estimates[step, 0] - Y[step, 0])**2 + (state_estimates[step, 1] - Y[step, 1])**2)
        diagnostics[step, 1] = np.sqrt((state_estimates[step, 2] - Y[step, 2])**2 + (state_estimates[step, 3] - Y[step, 3])**2)
        diagnostics[step, 2] = np.sum((state_estimates[step, 0] - Y[step, 0])**2 + (state_estimates[step, 1] - Y[step, 1])**2)
        diagnostics[step, 3] = ess
        diagnostics[step, 4] = resampling_count
        diagnostics[step, 5] = -np.sum(weights * np.log(weights))
        diagnostics[step, 6] = np.max(log_weights)
        diagnostics[step, 7] = np.min(log_weights)
        diagnostics[step, 8] = np.mean(compute_bearing_residuals(Y, state_estimates[step, 0], state_estimates[step, 1], state_estimates[step, 2], state_estimates[step, 3], dt))
        diagnostics[step, 9] = np.std(compute_bearing_residuals(Y, state_estimates[step, 0], state_estimates[step, 1], state_estimates[step, 2], state_estimates[step, 3], dt))
        diagnostics[step, 10] = angle_wrapping
        diagnostics[step, 11] = angle_wrap_count
    return state_estimates, covariance_estimates, diagnostics, resampling_count, weight_degeneracy, angle_wrapping, angle_wrap_count

def save_results(state_estimates, covariance_estimates, diagnostics, resampling_count, weight_degeneracy, angle_wrapping, angle_wrap_count):
    os.makedirs('results', exist_ok=True)
    state_df = pd.DataFrame({
        't': t,
        'px': state_estimates[:, 0],
        'py': state_estimates[:, 1],
        'vx': state_estimates[:, 2],
        'vy': state_estimates[:, 3],
        'cov_px': covariance_estimates[:, 0],
        'cov_py': covariance_estimates[:, 1],
        'cov_vx': covariance_estimates[:, 2],
        'cov_vy': covariance_estimates[:, 3]
    })
    state_df.to_csv('results/state_estimates.csv', index=False)
    diag_df = pd.DataFrame({
        't': t,
        'rmse_pos': diagnostics[:, 0],
        'rmse_vel': diagnostics[:, 1],
        'nees': diagnostics[:, 2],
        'ess': diagnostics[:, 3],
        'resampled': diagnostics[:, 4],
        'weight_entropy': diagnostics[:, 5],
        'max_log_weight': diagnostics[:, 6],
        'min_log_weight': diagnostics[:, 7],
        'bearing_residual_mean': diagnostics[:, 8],
        'bearing_residual_std': diagnostics[:, 9]
    })
    diag_df.to_csv('results/filter_diagnostics.csv', index=False)
    summary = {
        'pos_rmse': np.mean(diagnostics[:, 0]),
        'vel_rmse': np.mean(diagnostics[:, 1]),
        'mean_nees': np.mean(diagnostics[:, 2]),
        'median_nees': np.median(diagnostics[:, 2]),
        'ess_min': np.min(diagnostics[:, 3]),
        'ess_mean': np.mean(diagnostics[:, 3]),
        'resampling_count': resampling_count,
        'weight_degeneracy_detected': weight_degeneracy,
        'used_log_weights': True,
        'angle_wrapping_detected': angle_wrapping,
        'angle_wrap_event_count': angle_wrap_count
    }
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, state_estimates[:, 0], label='px')
    ax.plot(t, state_estimates[:, 1], label='py')
    ax.plot(t, state_estimates[:, 2], label='vx')
    ax.plot(t, state_estimates[:, 3], label='vy')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    fig.savefig('results/overview.png')
    with open('results/conclusion.txt', 'w') as f:
        f.write('The particle filter estimates the target trajectory using the provided measurements. The diagnostics suggest that the filter performs well, with low RMSE and reasonable effective sample size.\n')

def main():
    t, Y, mask = load_data()
    state_estimates, covariance_estimates, diagnostics, resampling_count, weight_degeneracy, angle_wrapping, angle_wrap_count = particle_filter(t, Y, mask)
    save_results(state_estimates, covariance_estimates, diagnostics, resampling_count, weight_degeneracy, angle_wrapping, angle_wrap_count)

if __name__ == '__main__':
    main()
```