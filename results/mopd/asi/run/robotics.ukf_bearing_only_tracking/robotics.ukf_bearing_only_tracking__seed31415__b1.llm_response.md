```python
import numpy as np
import json
import os

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def analysis():
    data_dir = 'data'
    with open(os.path.join(data_dir, 'task_info.json'), 'r') as f:
        task_info = json.load(f)
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
    
    bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
    x = np.load(os.path.join(data_dir, 'x0.npy'))
    P = np.load(os.path.join(data_dir, 'P0.npy'))
    
    state_history = [x]
    covariance_history = [P]
    measurement_outlier_prob = np.zeros(T)
    
    n = 4
    lambda_ = alpha**2 * (n + kappa) - n
    Wm = np.zeros(2*n + 1)
    Wc = np.zeros(2*n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 1 / (2 * (n + lambda_))
    
    for k in range(T):
        z = bearing_obs[k]
        # Predict
        sigma_points = np.zeros((2*n+1, n))
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        for i in range(2*n+1):
            sigma_points[i] = x + sqrt_P[:, i]
        # Compute predicted mean and covariance
        x_pred = np.zeros(n)
        P_pred = np.zeros((n, n))
        for i in range(2*n+1):
            x_pred += Wm[i] * sigma_points[i]
        for i in range(2*n+1):
            for j in range(2*n+1):
                P_pred += Wm[i] * Wm[j] * (sigma_points[i] - x_pred) @ (sigma_points[j] - x_pred).T
        # Add process noise
        Q = np.diag([process_pos_std**2, process_pos_std**2, process_vel_std**2, process_vel_std**2]) * dt
        P_pred += Q
        
        # Update
        # Propagate sigma points through measurement function
        z_pred = np.zeros(1)
        for i in range(2*n+1):
            px = sigma_points[i, 0]
            py = sigma_points[i, 1]
            h = np.arctan2(py - observer_y, px - observer_x)
            z_pred += Wm[i] * h
        # Compute innovation
        y = wrap_pi(z - z_pred)
        S = measurement_variance
        nis = y**2 / S
        if nis > outlier_gate_nis:
            measurement_outlier_prob[k] = 1.0
            # Carry over predicted state
            x = x_pred
            P = P_pred
            state_history.append(x)
            covariance_history.append(P)
            continue
        else:
            measurement_outlier_prob[k] = 0.0
            # Compute Kalman gain
            cross_cov = np.zeros((n, 1))
            for i in range(2*n+1):
                cross_cov += Wm[i] * (sigma_points[i] - x_pred) @ (y - z_pred + np.arctan2(np.sin(z - z_pred), np.cos(z - z_pred))).T
            K = cross_cov / S
            x = x_pred + K * y
            P = P_pred - K * S * K.T
            state_history.append(x)
            covariance_history.append(P)
    
    state_history = np.array(state_history)
    covariance_history = np.array(covariance_history)
    np.save('state_history.npy', state_history)
    np.save('covariance_history.npy', covariance_history)
    np.save('measurement_outlier_prob.npy', measurement_outlier_prob)
    
analysis()
```