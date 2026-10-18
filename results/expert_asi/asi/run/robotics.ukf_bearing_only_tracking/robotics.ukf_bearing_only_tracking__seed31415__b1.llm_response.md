```python
import numpy as np
import json
import os

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def analysis():
    # Load data
    data_dir = 'data'
    bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
    x0 = np.load(os.path.join(data_dir, 'x0.npy'))
    P0 = np.load(os.path.join(data_dir, 'P0.npy'))
    with open(os.path.join(data_dir, 'task_info.json')) as f:
        task = json.load(f)
    dt = task['dt']
    observer_x = task['observer_x']
    observer_y = task['observer_y']
    state_layout = task['state_layout']
    T = task['sequence_length']
    process_pos_std = task['process_pos_std']
    process_vel_std = task['process_vel_std']
    bearing_std = task['bearing_std']
    measurement_variance = task['measurement_variance']
    alpha = task['alpha']
    beta = task['beta']
    kappa = task['kappa']
    outlier_gate_nis = task['outlier_gate_nis']

    # Initialize output arrays
    state_hist = np.zeros((T + 1, 4))
    cov_hist = np.zeros((T + 1, 4, 4))
    outlier_prob = np.zeros(T)

    # Initialize state and covariance
    x = x0.copy()
    P = P0.copy()
    state_hist[0] = x
    cov_hist[0] = P

    for k in range(T):
        z = bearing_obs[k]
        # Step 1: Predict
        n = 4
        lambda_ = alpha**2 * (n + kappa) - n
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        Wm[0] = Wc[0] = lambda_ / (n + lambda_)
        Wm[1:] = Wc[1:] = 1 / (2 * (n + lambda_))

        # Generate sigma points
        try:
            sqrt_chol = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            sqrt_chol = np.linalg.cholesky((n + lambda_) * (P + 1e-12 * np.eye(n)))
        sigma_points = x + sqrt_chol
        sigma_points = np.vstack((sigma_points, x - sqrt_chol)).T

        # Predict sigma points
        x_sigma = sigma_points.copy()
        for i in range(2 * n + 1):
            px, py, vx, vy = x_sigma[:, i]
            omega = 0.2455
            theta = omega * dt
            if abs(omega) < 1e-8:
                p_next = np.array([px + vx * dt, py + vy * dt])
                v_next = np.array([vx, vy])
            else:
                A = np.array([[np.sin(theta)/omega, -(1 - np.cos(theta))/omega],
                              [(1 - np.cos(theta))/omega, np.sin(theta)/omega]])
                p_next = np.array([px + A[0, 0] * vx + A[0, 1] * vy,
                                   py + A[1, 0] * vx + A[1, 1] * vy])
                v_next = np.array([vx * np.cos(theta) - vy * np.sin(theta),
                                   vx * np.sin(theta) + vy * np.cos(theta)])
            x_sigma[:, i] = np.concatenate([p_next, v_next])
        x_pred = np.sum(Wm[:, np.newaxis] * x_sigma, axis=1)
        P_pred = np.zeros((4, 4))
        for i in range(2 * n + 1):
            delta_x = x_sigma[:, i] - x_pred
            P_pred += Wc[i] * np.outer(delta_x, delta_x)
        Q = np.diag([process_pos_std**2, process_pos_std**2,
                     process_vel_std**2, process_vel_std**2]) * dt
        P_pred += Q

        # Step 2: Update
        # Generate sigma points for measurement function
        sigma_points_meas = x_pred + np.linalg.cholesky((n + lambda_) * P_pred)
        sigma_points_meas = np.vstack((sigma_points_meas, x_pred - np.linalg.cholesky((n + lambda_) * P_pred))).T

        # Compute predicted measurement
        z_pred = np.zeros(n + 1)
        for i in range(n + 1):
            px, py, _, _ = sigma_points_meas[:, i]
            h = np.arctan2(py - observer_y, px - observer_x)
            z_pred[i] = h
        z_pred = np.sum(Wm * z_pred)

        # Compute innovation
        y = wrap_pi(z - z_pred)
        S = 0
        for i in range(n + 1):
            delta_z = sigma_points_meas[:, i] - z_pred
            S += Wc[i] * np.outer(delta_z, delta_z)
        S += measurement_variance

        # Compute NIS
        nis = y @ np.linalg.solve(S, y)
        is_outlier = nis > outlier_gate_nis

        if is_outlier:
            outlier_prob[k] = 1.0
            # Skip update, carry forward predicted state
            x = x_pred.copy()
            P = P_pred.copy()
        else:
            # Compute Kalman gain
            K = np.zeros((4, 1))
            for i in range(n + 1):
                delta_x = sigma_points_meas[:, i] - x_pred
                delta_z = sigma_points_meas[:, i] - z_pred
                K += Wc[i] * np.outer(delta_x, delta_z)
            K = K / S
            x = x_pred + K * y
            P = P_pred - K @ S @ K.T

        # Save outputs
        state_hist[k + 1] = x
        cov_hist[k + 1] = P
        outlier_prob[k] = 1.0 if is_outlier else 0.0

    # Save outputs
    np.save('state_history.npy', state_hist)
    np.save('covariance_history.npy', cov_hist)
    np.save('measurement_outlier_prob.npy', outlier_prob)

analysis()
```