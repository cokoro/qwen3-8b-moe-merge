```python
import numpy as np
import os
import json

# Load data
data_dir = 'data'
bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
x0 = np.load(os.path.join(data_dir, 'x0.npy'))
P0 = np.load(os.path.join(data_dir, 'P0.npy'))
task_info = json.load(open(os.path.join(data_dir, 'task_info.json')))

# Parse task info
dt = task_info['dt']
observer_x, observer_y = task_info['observer_position']
state_layout = task_info['state_layout']
seq_len = task_info['sequence_length']
process_pos_std = task_info['process_pos_std']
process_vel_std = task_info['process_vel_std']
bearing_std = task_info['bearing_std']
measurement_variance = task_info['measurement_variance']
alpha = task_info['ukf_alpha']
beta = task_info['ukf_beta']
kappa = task_info['ukf_kappa']
outlier_gate_nis = task_info['outlier_gate_nis']

# UKF parameters
n = 4
lambda_ = alpha**2 * (n + kappa) - n
Wm = np.zeros(2*n + 1)
Wc = np.zeros(2*n + 1)
Wm[0] = lambda_ / (n + lambda_)
Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
Wm[1:] = Wc[1:] = 1 / (2 * (n + lambda_))

# Precompute rotation matrix for turn rate
omega = task_info['turn_rate_rad_s']
theta = omega * dt
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
Rot = np.array([[cos_theta, -sin_theta],
                [sin_theta, cos_theta]])

# Initialize output arrays
state_history = np.zeros((seq_len + 1, n))
covariance_history = np.zeros((seq_len + 1, n, n))
measurement_outlier_prob = np.zeros(seq_len)

# Initialize filter
x = x0.copy()
P = P0.copy()

# Add tiny jitter to ensure P is positive definite
np.fill_diagonal(P, np.sqrt(np.diag(P)))

state_history[0] = x
covariance_history[0] = P

for k in range(seq_len):
    z_k = bearing_obs[k]
    # Generate sigma points
    sqrt_P = np.linalg.cholesky((n + lambda_) * P)
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = x
    sigma_points[1:] = x + sqrt_P

    # Predict step
    x_pred = np.zeros(n)
    P_pred = np.zeros((n, n))
    for i in range(2*n + 1):
        sigma_point = sigma_points[i]
        px, py, vx, vy = sigma_point
        if np.abs(omega) < 1e-10:
            p_next = np.array([px + vx * dt, py + vy * dt])
            v_next = np.array([vx, vy])
        else:
            A = np.array([[sin_theta / omega, -(1 - cos_theta) / omega],
                          [(1 - cos_theta) / omega, sin_theta / omega]])
            p_next = np.array([px + A[0, 0] * vx + A[0, 1] * vy,
                               py + A[1, 0] * vx + A[1, 1] * vy])
            v_next = Rot @ np.array([vx, vy])
        x_pred += Wm[i] * np.concatenate([p_next, v_next])
        for j in range(n):
            for l in range(n):
                P_pred[j, l] += Wc[i] * (sigma_point - x_pred) @ (sigma_point - x_pred).T
    Q = np.diag([process_pos_std**2, process_pos_std**2,
                 process_vel_std**2, process_vel_std**2]) * dt
    P_pred += Q

    # Compute predicted measurement mean
    z_pred = 0.0
    for i in range(2*n + 1):
        px, py, vx, vy = sigma_points[i]
        z_i = np.arctan2(py - observer_y, px - observer_x)
        z_pred += Wm[i] * z_i

    # Compute predicted covariance
    S = np.zeros((1, 1))
    for i in range(2*n + 1):
        z_i = np.arctan2(sigma_points[i][1] - observer_y, sigma_points[i][0] - observer_x)
        S += Wc[i] * (z_i - z_pred) * (z_i - z_pred)

    # Update step
    nis = 0.0
    outlier_prob = 0.0
    if S < 1e-10:
        # Avoid division by zero
        outlier_prob = 1.0
    else:
        y = z_k - z_pred
        y = (y + np.pi) % (2 * np.pi) - np.pi
        nis = y * y / S
        if nis > outlier_gate_nis:
            outlier_prob = 1.0
        else:
            # Compute Kalman gain
            cross = np.zeros((n, 1))
            for i in range(2*n + 1):
                sigma_point = sigma_points[i]
                px, py, vx, vy = sigma_point
                z_i = np.arctan2(py - observer_y, px - observer_x)
                cross += Wc[i] * (sigma_point - x_pred) * (z_i - z_pred)
            K = cross @ np.linalg.inv(S)
            x = x_pred + K * y
            P = P_pred - K @ cross.T
            outlier_prob = 0.0

    state_history[k+1] = x
    covariance_history[k+1] = P
    measurement_outlier_prob[k] = outlier_prob

# Save outputs
np.save('state_history.npy', state_history)
np.save('covariance_history.npy', covariance_history)
np.save('measurement_outlier_prob.npy', measurement_outlier_prob)
```