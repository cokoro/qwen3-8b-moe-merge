```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import
import os
import json

def load_data():
    data_path = os.path.join('data', 'measurements.npz')
    with np.load(data_path) as data:
        return data

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_angle_residual(estimated, observed, bearing_bias):
    predicted = wrap_angle(np.arctan2(estimated[1], estimated[0]) + bearing_bias)
    residual = wrap_angle(observed - predicted)
    return residual

def compute_nees(residual, covariance):
    return np.dot(residual.T, np.linalg.inv(covariance)).dot(residual)

def compute_ess(weights):
    weights = np.exp(weights)
    return 1.0 / np.sum(weights**2)

def compute_weight_entropy(weights):
    return -np.sum(weights * np.log(weights))

def compute_resampling_info(weights):
    weights = np.exp(weights)
    resampled = False
    max_log_weight = -np.inf
    min_log_weight = np.inf
    if compute_ess(weights) < 100:
        resampled = True
        weights = np.ones_like(weights) / len(weights)
        max_log_weight = np.log(np.max(weights))
        min_log_weight = np.log(np.min(weights))
    return resampled, max_log_weight, min_log_weight

def compute_angle_wrapping_info(estimated, observed, bearing_bias):
    residual = compute_angle_residual(estimated, observed, bearing_bias)
    wrapped = np.abs(residual) > np.pi / 2
    return wrapped.sum(), wrapped.any()

def main():
    data = load_data()
    Y = data['Y']
    t = data['t']
    mask = data.get('mask', None)
    dt = t[1] - t[0]

    num_particles = 1500
    num_steps = len(t)

    px = np.zeros((num_steps, num_particles))
    py = np.zeros((num_steps, num_particles))
    vx = np.zeros((num_steps, num_particles))
    vy = np.zeros((num_steps, num_particles))
    log_weights = np.zeros((num_steps, num_particles))
    ess = np.zeros(num_steps)
    resampled = np.zeros(num_steps, dtype=bool)
    weight_entropy = np.zeros(num_steps)
    max_log_weight = np.zeros(num_steps)
    min_log_weight = np.zeros(num_steps)
    bearing_residual_mean = np.zeros(num_steps)
    bearing_residual_std = np.zeros(num_steps)

    # Initialize particles
    if mask is not None:
        valid_indices = np.where(~np.isnan(Y))[0]
        Y_valid = Y[valid_indices]
        t_valid = t[valid_indices]
    else:
        valid_indices = np.arange(len(Y))
        Y_valid = Y
        t_valid = t

    if len(Y_valid) == 0:
        return

    range_obs = Y_valid[:, 0]
    bearing_obs = Y_valid[:, 1]

    range_est = np.exp(range_obs - 0.08) - 1.5
    bearing_est = wrap_angle(np.arctan2(range_est, 1.0))

    px[0, :] = np.random.normal(bearing_est, 1.0, num_particles)
    py[0, :] = np.random.normal(0.0, 1.0, num_particles)
    vx[0, :] = np.random.normal(0.0, 1.0 / dt, num_particles)
    vy[0, :] = np.random.normal(0.0, 1.0 / dt, num_particles)
    log_weights[0, :] = np.log(1.0 / num_particles)

    for i in range(1, num_steps):
        # Propagate particles
        px[i, :] = px[i-1, :] + vx[i-1, :] * dt
        py[i, :] = py[i-1, :] + vy[i-1, :] * dt
        vx[i, :] = vx[i-1, :] + np
        vy[i, :] = vy[i-1, :] + vy[i-1, :] * dt

        # Evaluate likelihood
        log_likelihood = np.zeros(num_particles)
        for j in range(num_particles):
            px_j = px[i, j]
            py_j = py[i, j]
            vx_j = vx[i, j]
            vy_j = vy[i, j]

            # Compute measurement
            range_j = np.hypot(px_j, py_j)
            range_meas = np.log(range_j + 1.5) + 0.08
            bearing_meas = wrap_angle(np.arctan2(py_j, px_j))

            # Compute residuals
            range_res = range_meas - Y_valid[i, 0]
            bearing_res = bearing_meas - Y_valid[i, 1]

            # Compute likelihood
            log_likelihood[j] = np.exp(-0.5 * (range_res**2 + bearing_res**2))

        # Normalize weights
        log_weights[i, :] = log_weights[i-1, :] + np.log(log_likelihood)
        log_weights[i, :] -= np.log(np.sum(np.exp(log_weights[i, :])))

        # Compute diagnostics
        ess[i] = compute_ess(log_weights[i, :])
        resampled[i], max_log_weight[i], min_log_weight[i] = compute_resampling_info(log_weights[i, :])
        weight_entropy[i] = compute_weight_entropy(log_weights[i, :])

        # Compute bearing residuals
        bearing_residual_mean[i] = np.mean(bearing_res)
        bearing_residual_std[i] = np.std(bearing_res)

        # Update particles
        if resampled[i]:
            weights = np.exp(log_weights[i, :])
            indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
            px[i, :] = px[i, indices]
            py[i, :] = py[i, indices]
            vx[i, :] = vx[i, indices]
            vy[i, :] = vy[i, indices]
            log_weights[i, :] = np.log(1.0 / num_particles)

    # Compute state estimates
    px_est = np.sum(px * np
    py_est = np.sum(py * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))
    vx_est = np.sum(vx * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))
    vy_est = np.sum(vy * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))

    # Compute covariance
    cov_px = np.sum((px[-1, :] - px_est) ** 2 * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))
    cov_py = np.sum((py[-1, :] - py_est) ** 2 * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))
    cov_vx = np.sum((vx[-1, :] - vx_est) ** 2 * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))
    cov_vy = np.sum((vy[-1,vy_est) ** 2 * np.exp(log_weights[-1, :])) / np.sum(np.exp(log_weights[-1, :]))

    # Compute diagnostics
    pos_rmse = np.sqrt(cov_px + cov_py)
    vel_rmse = np.sqrt(cov_vx + cov_vy)
    mean_nees = np.mean(np.abs(log_weights))
    median_nees = np.median(np.abs(log_weights))
    ess_min = np.min(ess)
    ess_mean = np.mean(ess)
    resampling_count = np.sum(resampled)
    weight_degeneracy_detected = np.any(ess < 100)
    used_log_weights = np.any(np.isnan(log_weights))
    angle_wrapping_detected = np.any(bearing
    angle_wrap_event_count = np.sum(angle_wrapping_detected)

    # Save results
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/state_estimates.csv', np.column_stack((t, px_est, py_est, vx_est, vy_est, cov_px, cov_py, cov_vx, cov_vy)), delimiter=',')
    np.savetxt('results/filter_diagnostics.csv', np.column_stack((t, pos_rmse, vel_rmse, mean_nees, ess, resampled, weight_entropy, max_log_weight, min_log_weight, bearing_residual_mean, bearing_residual_std)), delimiter=',')
    with open('results/summary.json', 'w') as f:
        json.dump({
            'pos_rmse': pos_rmse,
            'vel_rmse': vel_rmse,
            'mean_nees': mean_nees,
            'median_nees': median_nees,
            'ess_min': ess_min,
            'ess_mean': ess_mean,
            'resampling_count': resampling_count,
            'weight_degeneracy_detected': weight_degeneracy_detected,
            'used_log_weights': used_log_weights,
            'angle_wrapping_detected': angle_wrapping_detected,
            'angle_wrap_event_count': angle_wrap_event_count
        }, f)

    # Plot overview
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, px_est, label='Estimated x')
    ax.plot(t, py_est, label='Estimated y')
    ax.plot(t, vx_est, label='Estimated vx')
    ax.plot(t, vy_est, label='Estimated vy')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title('State Estimates')
    plt.savefig('results/overview.png')
    plt.close()

    with open('results/conclusion.txt', 'w') as f:
        f.write(f'Final state estimate: ({px_est:.4f}, {py_est:.4f}, {vx_est:.4f}, {vy_est:.4f})\n')
        f.write(f'Position RMSE: {pos_rmse:.4f}\n')
        f.write(f'Velocity RMSE: {vel_rmse:.4f}\n')
        f.write(f'Weight degeneracy detected: {weight_degeneracy_detected}\n')
        f.write(f'Angle wrapping detected: {angle_wrapping_detected
        f.write(f'Angle wrap event count: {angle_wrap_event_count}\n')

if __name__ == '__main__':
    main()
```