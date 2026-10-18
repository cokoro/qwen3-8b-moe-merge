```bash
#!<abs_path>

cat > analysis.py << 'EOF'
import numpy as np
import os
import json
import sys
import time

def main():
    task_info_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'task_info.json')
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)

    dt = float(task_info['dt'])
    observer_x = float(task_info['observer_x'])
    observer_y = float(task_info['observer_y'])
    state_layout = task_info['state_layout']
    state_size = len(state_layout)
    seq_len = int(task_info['seq_len'])

    bearing_obs_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bearing_obs.npy')
    bearing_obs = np.load(bearing_obs_path)
    x0_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'x0.npy')
    x0 = np.load(x0_path)
    P0_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'P0.npy')
    P0 = np.load(P0_path)

    process_pos_std = float(task_info['process_pos_std'])
    process_vel_std = float(task_info['process_vel_std'])
    bearing_std = float(task_info['bearing_std'])
    measurement_variance = float(task_info['measurement_variance'])

    alpha = 0.001
    beta = 2
    kappa = 0

    outlier_gate_nis = float(task_info['outlier_gate_nis'])

    # Coordinated-turn model
    turn_rate_rad_s = float(task_info['turn_rate_rad_s'])

    def rotate(theta, p):
        # theta is turn rate * dt
        return np.array([
            p[0] * np.cos(theta) - p[1] * np.sin(theta),
            p[0] * np.sin(theta) + p[1] * np.cos(theta),
        ])

    def rotate_inv(theta, p):
        # theta is turn rate * dt
        return np.array([
            p[0] * np.cos(theta) + p[1] * np.sin(theta),
            -p[0] * np.sin(theta) + p[1] * np.cos(theta),
        ])

    def step(x):
        # x = [px, py, vx, vy]
        omega = turn_rate_rad_s
        theta = omega * dt
        if abs(omega) < 1e-6:
            p_next = x[0:2] + dt * x[2:4]
            v_next = x[2:4]
        else:
            A = np.array([
                [np.sin(theta)/omega, -(1 - np.cos(theta))/omega],
                [(1 - np.cos(theta))/omega, np.sin(theta)/omega],
            ])
            p_next = x[0:2] + A @ x[2:4]
            v_next = rotate(theta, x[2:4])
        x_next = np.concatenate((p_next, v_next))
        return x_next

    # Unscented transform
    n = state_size
    lambda_ = alpha**2 * (n + kappa) - n

    Wm = np.zeros((2*n+1, 1))
    Wm[0] = lambda_ / (n + lambda_)
    Wc = Wm.copy()
    Wc[0] = Wc[0] + (1 - alpha**2 + beta)
    for i in range(1, 2*n+1):
        Wm[i] = Wc[i] = 1/(2*(n + lambda_))

    # Covariance jitter
    eps = 1e-6

    # Unscented cross-covariance
    def unscented_crosscovar(x, y, state_indices, measurement_indices):
        # x is state vector, y is measurement vector
        # state_indices = [0, 1, 2, 3] for state layout "xy_vxvy"
        # measurement_indices = [0] for bearing measurement
        n = len(state_indices)
        m = len(measurement_indices)
        n_points = 2*n+1
        sigma_points = np.zeros((n_points, n))
        sigma_points[0] = x[state_indices]
        for i in range(n_points-1):
            sigma_points[i+1] = x[state_indices] + np.sqrt(n + lambda_) * np.sqrt(P0[state_indices, state_indices].diagonal())
        sigma_points = sigma_points[0]
        predicted_y = np.zeros((n_points, m))
        for i in range(n_points):
            predicted_y[i] = y(sigma_points[i])
        cross = np.zeros((n, m))
        for i in range(n_points):
            cross += Wm[i] * (sigma_points[i] - x[state_indices]) * (predicted_y[i] - y(x)[measurement_indices])
        return cross

    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def h(x):
        # x = [px, py, vx, vy]
        dx = x[0] - observer_x
        dy = x[1] - observer_y
        return np.arctan2(dy, dx)

    state_history = np.zeros((seq_len+1, state_size))
    covariance_history = np.zeros((seq_len+1, state_size, state_size))
    measurement_outlier_prob = np.zeros(seq_len)

    state_history[0] = x0
    covariance_history[0] = P0

    for k in range(seq_len):
        z_k = bearing_obs[k]
        x = state_history[k]
        P = covariance_history[k]

        # Generate sigma points
        sigma_points = np.zeros((2*n+1, n))
        sigma_points[0] = x
        for i in range(n_points-1):
            sigma_points[i+1] = x + np.sqrt(n + lambda_) * np.sqrt(P.diagonal())
        sigma_points = sigma_points[0]

        # Predict step
        x_pred = np.zeros(n)
        P_pred = np.zeros((n, n))
        for i in range(n_points):
            x_pred += Wm[i] * step(sigma_points[i])
        for i in range(n_points):
            delta = step(sigma_points[i]) - x_pred
            P_pred += Wm[i] * np.outer(delta, delta)
        P_pred += np.eye(n) * (process_pos_std**2 + process_vel_std**2) * dt

        # Update step
        # Propagate sigma points through h(x)
        predicted_z = np.zeros(n_points)
        for i in range(n_points):
            predicted_z[i] = h(sigma_points[i])
        z_pred = Wc.T @ predicted_z
        S = Wc.T @ (predicted_z - z_pred)**2

        # Innovation
        y = wrap_pi(z_k - z_pred)
        nis = y**2 / S

        if nis > outlier_gate_nis:
            # Save high outlier probability and carry forward predicted state
            measurement_outlier_prob[k] = 1.0
            state_history[k+1] = x_pred
            covariance_history[k+1] = P_pred
        else:
            # Compute Kalman gain and update
            cross = unscented_crosscovar(x, h, [0, 1, 2, 3], [0])
            K = cross @ np.linalg.inv(S)
            x_new = x_pred + K * y
            P_new = P_pred - K @ cross.T
            measurement_outlier_prob[k] = 0.0
            state_history[k+1] = x_new
            covariance_history[k+1] = P_new

    np.save(os.path.join(os.path.dirname(__file__), 'state_history.npy'), state_history)
    np.save(os.path.join(os.path.dirname(__file__), 'covariance_history.npy'), covariance_history)
    np.save(os.path.join(os.path.dirname(__file__), 'measurement_outlier_prob.npy'), measurement_outlier_prob)

if __name__ == '__main__':
    main()
```