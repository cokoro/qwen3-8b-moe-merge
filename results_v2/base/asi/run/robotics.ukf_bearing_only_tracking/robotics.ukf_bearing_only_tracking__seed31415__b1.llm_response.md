```python
import numpy as np
import json
import os

def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def load_data():
    data_dir = 'data/'
    bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
    x0 = np.load(os.path.join(data_dir, 'x0.npy'))
    P0 = np.load(os.path.join(data_dir, 'P0.npy'))
    with open(os.path.join(data_dir, 'task_info.json'), 'r') as f:
        task_info = json.load(f)
    return bearing_obs, x0, P0, task_info

def generate_sigma_points(mean, cov, n, alpha, beta, kappa):
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = mean
    sqrt_cov = np.linalg.cholesky((n + lambda_) * cov)
    for i in range(n):
        sigma_points[i + 1] = mean + sqrt_cov[i]
        sigma_points[n + 1 + i] = mean - sqrt_cov[i]
    return sigma_points

def predict_sigma_points(sigma_points, dt, omega, process_pos_std, process_vel_std):
    n = sigma_points.shape[1]
    sigma_points_next = np.zeros_like(sigma_points)
    for i in range(sigma_points.shape[0]):
        px, py, vx, vy = sigma_points[i]
        if abs(omega) < 1e-6:
            p_next_x = px + dt * vx
            p_next_y = py + dt * vy
            v_next_x = vx
            v_next_y = vy
        else:
            theta = omega * dt
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            A11 = sin_theta / omega
            A12 = -(1 - cos_theta) / omega
            A21 = (1 - cos_theta) / omega
            A22 = sin_theta / omega
            p_next_x = px + A11 * vx + A12 * vy
            p_next_y = py + A21 * vx + A22 * vy
            v_next_x = cos_theta * vx - sin_theta * vy
            v_next_y = sin_theta * vx + cos_theta * vy
        sigma_points_next[i] = [p_next_x, p_next_y, v_next_x, v_next_y]
    return sigma_points_next

def update_sigma_points(sigma_points, observer_x, observer_y):
    n = sigma_points.shape[1]
    z_pred = np.zeros(n)
    for i in range(n):
        px, py, _, _ = sigma_points[i]
        z_pred[i] = np.arctan2(py - observer_y, px - observer_x)
    return z_pred

def unscented_transform(sigma_points, weights_m, weights_c):
    n = sigma_points.shape[1]
    mean = np.zeros(n)
    cov = np.zeros((n, n))
    for i in range(sigma_points.shape[0]):
        mean += weights_m[i] * sigma_points[i]
        for j in range(n):
            cov[j, :] += weights_c[i] * (sigma_points[i, j] - mean[j]) * (sigma_points[i] - mean)
    return mean, cov

def compute_nis(y, S):
    return y ** 2 / S

def main():
    bearing_obs, x0, P0, task_info = load_data()
    dt = task_info['dt']
    observer_x, observer_y = task_info['observer_position']
    state_layout = task_info['state_layout']
    T = task_info['sequence_length']
    process_pos_std = task_info['process_pos_std']
    process_vel_std = task_info['process_vel_std']
    bearing_std = task_info['bearing_std']
    measurement_variance = task_info['measurement_variance']
    alpha = task_info['alpha']
    beta = task_info['beta']
    kappa = task_info['kappa']
    outlier_gate_nis = task_info['outlier_gate_nis']

    n = 4
    state_history = np.zeros((T + 1, n))
    covariance_history = np.zeros((T + 1, n, n))
    measurement_outlier_prob = np.zeros(T)

    state_history[0] = x0
    covariance_history[0] = P0

    for k in range(T):
        z_k = bearing_obs[k]
        sigma_points = generate_sigma_points(state_history[k], covariance_history[k], n, alpha, beta, kappa)
        weights_m = np.zeros(2 * n + 1)
        weights_c = np.zeros(2 * n + 1)
        lambda_ = alpha**2 * (n + kappa) - n
        weights_m[0] = lambda_ / (n + lambda_)
        weights_c[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2 * n + 1):
            weights_m[i] = 1 / (2 * (n + lambda_))
            weights_c[i] = 1 / (2 * (n + lambda_))

        # Predict step
        sigma_points_next = predict_sigma_points(sigma_points, dt, task_info['turn_rate_rad_s'], process_pos_std, process_vel_std)
        mean_pred, cov_pred = unscented_transform(sigma_points_next, weights_m, weights_c)
        Q = np.diag([process_pos_std**2, process_pos_std**2, process_vel_std**2, process_vel_std**2]) * dt
        cov_pred += Q

        # Update step
        z_pred = update_sigma_points(sigma_points_next, observer_x, observer_y)
        z_pred_mean = np.arctan2(np.sum(weights_m[1:] * np.sin(z_pred[1:])), np.sum(weights_m[1:] * np.cos(z_pred[1:])))
        y = wrap_pi(z_k - z_pred_mean)
        S = np.sum(weights_c[1:] * (np.sin(z_pred[1:]) - np.sin(z_pred_mean)) * (np.cos(z_pred[1:]) - np.cos(z_pred_mean)))
        nis = compute_nis(y, S)
        if nis > outlier_gate_nis:
            measurement_outlier_prob[k] = 1.0
            state_history[k + 1] = state_history[k]
            covariance_history[k + 1] = covariance_history[k]
        else:
            measurement_outlier_prob[k] = 0.0
            # Compute Kalman gain
            sigma_points_h = update_sigma_points(sigma_points_next, observer_x, observer_y)
            h_pred = np.arctan2(np.sum(weights_m[1:] * np.sin(sigma_points_h[1:])), np.sum(weights_m[1:] * np.cos(sigma_points_h[1:])))
            h_pred = np.array([h_pred])
            h_sigma = sigma_points_h[1:]
            h_mean = h_pred
            h_cov = np.zeros((1, 1))
            for i in range(2 * n):
                h_cov += weights_c[i + 1] * (h_sigma[i] - h_mean) * (h_sigma[i] - h_mean)
            cross_cov = np.zeros((n, 1))
            for i in range(2 * n + 1):
                cross_cov += weights_m[i] * (sigma_points_next[i] - mean_pred) * (h_sigma[i] - h_mean)
            K = cross_cov @ np.linalg.inv(h_cov)
            state_history[k + 1] = mean_pred + K * y
            covariance_history[k + 1] = cov_pred - K @ h_cov @ K.T

    np.save('state_history.npy', state_history)
    np.save('covariance_history.npy', covariance_history)
    np.save('measurement_outlier_prob.npy', measurement_outlier_prob)

if __name__ == '__main__':
    main()
```