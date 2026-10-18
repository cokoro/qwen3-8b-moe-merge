```python
import numpy as np
import json
import os

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def analysis():
    data_dir = 'data/'
    bearing_obs = np.load(os.path.join(data_dir, 'bearing_obs.npy'))
    x0 = np.load(os.path.join(data_dir, 'x0.npy'))
    P0 = np.load(os.path.join(data_dir, 'x0.npy'))  # Placeholder, should load P0.npy
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
    
    n = 4
    lambda_ = alpha**2 * (n + kappa) - n
    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 1 / (2 * (n + lambda_))
    
    state_history = [x0]
    covariance_history = [P0]
    measurement_outlier_prob = np.zeros(T)
    
    x = x0
    P = P0
    
    for k in range(T):
        z = bearing_obs[k]
        # Generate sigma points
        sigma_points = np.random.randn(2 * n + 1, n)
        sigma_points = np.dot(np.sqrt((n + lambda_) * P), sigma_points.T).T
        sigma_points += x.reshape(-1, 1)
        
        # Predict step
        x_pred = np.zeros(n)
        P_pred = np.zeros((n, n))
        for i in range(2 * n + 1):
            sigma = sigma_points[i]
            px, py, vx, vy = sigma
            omega = turn_rate_rad_s
            theta = omega * dt
            if abs(omega) < 1e-6:
                p_next = np.array([px + vx * dt, py + vy * dt])
                v_next = np.array([vx, vy])
            else:
                A = np.array([[np.sin(theta)/omega, -(1 - np.cos(theta))/omega],
                              [(1 - np.cos(theta))/omega, np.sin(theta)/omega]])
                p_next = np.array([px + A[0, 0] * vx + A[0, 1] * vy,
                                   py + A[1, 0] * vx + A[1, 1] * vy])
                v_next = np.array([np.cos(theta) * vx - np.sin(theta) * vy,
                                   np.sin(theta) * vx + np.cos(theta) * vy])
            x_pred += Wm[i] * np.concatenate([p_next, v_next])
            for j in range(n):
                for l in range(n):
                    P_pred[j, l] += Wm[i] * (p_next[j] - x_pred[j]) * (p_next[l] - x_pred[l])
        Q = np.diag([process_pos_std**2, process_pos_std**2,
                     process_vel_std**2, process_vel_std**2]) * dt
        P_pred += Q
        
        # Update step
        # Propagate sigma points through measurement function
        z_pred = np.zeros(1)
        for i in range(2 * n + 1):
            sigma = sigma_points[i]
            px, py, _, _ = sigma
            h = np.arctan2(py - observer_y, px - observer_x)
            z_pred += Wm[i] * h
        
        # Compute innovation
        y = wrap_pi(z - z_pred)
        S = measurement_variance
        
        # Compute NIS
        nis = (y**2) / S
        if nis > outlier_gate_nis:
            measurement_outlier_prob[k] = 1.0
            # Skip update, carry predicted state forward
            x = x_pred
            P = P_pred
        else:
            measurement_outlier_prob[k] = 0.0
            # Compute cross-covariance
            cross = np.zeros((n, 1))
            for i in range(2 * n + 1):
                sigma = sigma_points[i]
                px, py, vx, vy = sigma
                h = np.arctan2(py - observer_y, px - observer_x)
                innovation = wrap_pi(z - h)
                for j in range(n):
                    cross[j, 0] += Wm[i] * (sigma[j] - x_pred[j]) * innovation
            # Compute Kalman gain
            K = np.dot(P_pred, cross.T) / S
            # Update state and covariance
            x = x_pred + K * y
            P = P_pred - np.dot(K, np.dot(cross, P_pred))
        
        state_history.append(x)
        covariance_history.append(P)
    
    np.save('state_history.npy', np.array(state_history))
    np.save('covariance_history.npy', np.array(covariance_history))
    np.save('measurement_outlier_prob.npy', measurement_outlier_prob)

analysis()
```