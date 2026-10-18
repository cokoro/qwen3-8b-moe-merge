```python
import numpy as np
import json
import os

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def analysis():
    data_dir = 'data'
    task_info = json.load(open(os.path.join(data_dir, 'task_info.json')))
    dt = task_info['dt']
    observer_x = task_info['observer_x']
    observer_y = task_info['observer_y']
    bearing_std = task_info['bearing_std']
    measurement_variance = task_info['measurement_variance']
    process_pos_std = task_info['process_pos_std']
    process_vel_std = task_info['process_vel_std']
    alpha = task_info['alpha']
    beta = task_info['beta']
    kappa = task_info['kappa']
    outlier_gate_nis = task_info['outlier_gate_nis']

    bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
    x0 = np.load(os.path.join(data_dir, 'x0.npy'))
    P0 = np.load(os.path.join(data_dir, 'P0.npy'))

    T = len(bearing_obs)
    state_layout = task_info['state_layout']
    n = 4

    state_history = np.zeros((T+1, 4))
    covariance_history = np.zeros((T+1, 4, 4))
    measurement_outlier_prob = np.zeros(T)

    state_history[0] = x0
    covariance_history[0] = P0

    for k in range(T):
        z_k = bearing_obs[k]

        # Step 3: Scaled unscented transform
        lambda_ = alpha**2 * (n + kappa) - n
        Wm0 = lambda_ / (n + lambda_)
        Wc0 = Wm0 + (1 - alpha**2 + beta)
        Wm = np.zeros(2*n + 1)
        Wc = np.zeros(2*n + 1)
        Wm[0] = Wm0
        Wc[0] = Wc0
        for i in range(1, 2*n + 1):
            Wm[i] = Wc[i] = 1 / (2 * (n + lambda_))

        # Generate sigma points
        sqrt_chol = np.linalg.cholesky((n + lambda_) * covariance_history[k])
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = state_history[k]
        for i in range(1, 2*n + 1):
            sigma_points[i] = state_history[k] + sqrt_chol[i-1]

        # Step 1: Predict
        x_pred = np.zeros(n)
        P_pred = np.zeros((n, n))
        for i in range(2*n + 1):
            px, py, vx, vy = sigma_points[i]
            omega = task_info['turn_rate_rad_s']
            theta = omega * dt
            if abs(omega) < 1e-6:
                p_next = np.array([px + vx * dt, py + vy * dt])
                v_next = np.array([vx, vy])
            else:
                A = np.array([
                    [np.sin(theta) / omega, -(1 - np.cos(theta)) / omega],
                    [(1 - np.cos(theta)) / omega, np.sin(theta) / omega]
                ])
                p_next = np.array([px, py]) + A @ np.array([vx, vy])
                v_next = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.array([vx, vy])
            x_pred += Wm[i] * np.concatenate([p_next, v_next])
            for j in range(n):
                for l in range(n):
                    P_pred[j, l] += Wm[i] * (sigma_points[i] - x_pred) @ (sigma_points[i] - x_pred).T

        Q = np.diag([process_pos_std**2, process_pos_std**2, process_vel_std**2, process_vel_std**2]) * dt
        P_pred += Q

        # Step 2: Update
        z_pred = 0
        for i in range(2*n + 1):
            py_i = sigma_points[i][1]
            px_i = sigma_points[i][0]
            h_i = np.arctan2(py_i - observer_y, px_i - observer_x)
            z_pred += Wm[i] * h_i

        # Step 4: Outlier gate
        y = wrap_pi(z_k - z_pred)
        S = 0
        for i in range(2*n + 1):
            py_i = sigma_points[i][1]
            px_i = sigma_points[i][0]
            h_i = np.arctan2(py_i - observer_y, px_i - observer_x)
            S += Wm[i] * (h_i - z_pred) ** 2
        S += measurement_variance
        nis = y ** 2 / S
        if nis > outlier_gate_nis:
            measurement_outlier_prob[k] = 1.0
            state_history[k+1] = state_history[k]
            covariance_history[k+1] = P_pred
        else:
            measurement_outlier_prob[k] = 0.0
            # Step 5: Kalman gain
            cross = np.zeros((n, 1))
            for i in range(2*n + 1):
                py_i = sigma_points[i][1]
                px_i = sigma_points[i][0]
                h_i = np.arctan2(py_i - observer_y, px_i - observer_x)
                cross += Wm[i] * (sigma_points[i] - x_pred) * (h_i - z_pred)
            K = P_pred @ cross.T / S
            x_pred += K * y
            P_pred -= K @ P_pred @ cross.T

            state_history[k+1] = x_pred
            covariance_history[k+1] = P_pred

    np.save('state_history.npy', state_history)
    np.save('covariance_history.npy', covariance_history)
    np.save('measurement_outlier_prob.npy', measurement_outlier_prob)

analysis()
```