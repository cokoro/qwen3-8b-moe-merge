import numpy as np
import json
import os
import sys

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def analysis():
    # Load data
    data_dir = "data/"
    bearing_obs = np.load(os.path.join(data_dir, "bearing_obs.npy"))
    x0 = np.load(os.path.join(data_dir, "x0.npy"))
    P0 = np.load(os.path.join(data_dir, "P0.npy"))
    task_info = json.load(open(os.path.join(data_dir, "task_info.json")))
    dt = task_info["dt"]
    observer_x = task_info["observer_x"]
    observer_y = task_info["observer_y"]
    process_pos_std = task_info["process_pos_std"]
    process_vel_std = task_info["process_vel_std"]
    bearing_std = task_info["bearing_std"]
    measurement_variance = task_info["measurement_variance"]
    T = task_info["sequence_length"]
    
    # UKF parameters
    alpha = 0.001
    beta = 2
    kappa = 0
    outlier_gate_nis = 27.2907
    
    # State layout
    state_layout = task_info["state_layout"]
    n = 4
    
    # Initialize state and covariance
    x = x0
    P = P0
    state_history = [x]
    covariance_history = [P]
    outlier_prob = np.zeros(T)
    
    for k in range(T):
        # Step 1: Predict
        # Generate sigma points
        lambda_ = alpha**2 * (n + kappa) - n
        Wm = np.zeros(2*n + 1)
        Wc = np.zeros(2*n + 1)
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2*n + 1):
            Wm[i] = Wc[i] = 1 / (2 * (n + lambda_))
        
        # Generate sigma points
        sigma_points = np.random.multivariate_normal(x, P, 2*n + 1)
        sigma_points = sigma_points + np.random.randn(2*n + 1, n) * 1e-10
        
        # Predict sigma points
        sigma_points_pred = np.zeros((2*n + 1, n))
        for i in range(2*n + 1):
            px, py, vx, vy = sigma_points[i]
            omega = task_info["turn_rate_rad_s"]
            theta = omega * dt
            if abs(omega) < 1e-6:
                p_next = np.array([px + vx * dt, py + vy * dt])
                v_next = np.array([vx, vy])
            else:
                A = np.array([[np.sin(theta)/omega, -(1 - np.cos(theta))/omega],
                              [(1 - np.cos(theta))/omega, np.sin(theta)/omega]])
                p_next = np.array([px + A[0,0] * vx + A[0,1] * vy, py + A[1,0] * vx + A[1,1] * vy])
                v_next = np.array([np.cos(theta) * vx - np.sin(theta) * vy, np.sin(theta) * vx + np.cos(theta) * vy])
            sigma_points_pred[i] = np.concatenate((p_next, v_next))
        
        # Predict mean and covariance
        x_pred = np.sum(Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        P_pred = np.zeros((n, n))
        for i in range(2*n + 1):
            delta_x = sigma_points_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(delta_x, delta_x)
        P_pred += np.diag([process_pos_std**2, process_pos_std**2, process_vel_std**2, process_vel_std**2]) * dt
        
        # Step 2: Update
        # Generate sigma points for measurement
        sigma_points_meas = np.zeros((2*n + 1, n))
        for i in range(2*n + 1):
            px, py, vx, vy = sigma_points_pred[i]
            h = np.arctan2(py - observer_y, px - observer_x)
            sigma_points_meas[i] = np.array([px, py, vx, vy])
        
        # Predict measurement mean
        z_pred = np.zeros(1)
        for i in range(2*n + 1):
            z_pred += Wm[i] * np.arctan2(sigma_points_meas[i][1] - observer_y, sigma_points_meas[i][0] - observer_x)
        
        # Innovation
        z = bearing_obs[k]
        y = wrap_pi(z - z_pred)
        
        # Compute S
        S = 0
        for i in range(2*n + 1):
            delta_z = sigma_points_meas[i][1] - observer_y - np.sin(z_pred) * np.cos(y)
            delta_z += np.cos(z_pred) * np.sin(y)
            S += Wc[i] * delta_z**2
        S += measurement_variance
        
        # Robust outlier gate
        nis = y**2 / S
        if nis > outlier_gate_nis:
            outlier_prob[k] = 1.0
            # Skip update, carry forward predicted mean
            x = x_pred
            P = P_pred
        else:
            # Compute Kalman gain
            cross_cov = np.zeros((1, n))
            for i in range(2*n + 1):
                delta_z = sigma_points_meas[i][1] - observer_y - np.sin(z_pred) * np.cos(y)
                delta_z += np.cos(z_pred) * np.sin(y)
                delta_x = sigma_points_pred[i] - x_pred
                cross_cov += Wc[i] * delta_z * delta_x
            K = cross_cov / S
            
            # Update mean and covariance
            x = x_pred + K * y
            P = P_pred - K * S
            P = P - K * K.T * S
            P = P + np.diag([process_pos_std**2, process
            process_pos_std**2, process_vel_std**2, process_vel_std**2]) * dt
            
            # Save low outlier probability
            outlier_prob[k] = 0.0
        
        # Save state and covariance
        state_history.append(x)
        covariance_history.append(P)
    
    # Save outputs
    np.save("state_history.npy", np.array(state_history))
    np.save("covariance_history.npy", np.array(covariance_history))
    np.save("measurement_outlier_prob.npy", outlier_prob)

analysis()