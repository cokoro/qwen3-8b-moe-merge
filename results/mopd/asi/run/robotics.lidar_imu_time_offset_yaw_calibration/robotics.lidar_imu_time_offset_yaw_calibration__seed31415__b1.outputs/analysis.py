```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import json

def load_data():
    task_info = json.load(open('data/task_info.json'))
    imu = pd.read_csv('data/imu_gyro.csv')
    lidar = pd.read_csv('data/lidar_scan_matches.csv')
    segment_windows = pd.read_csv('data/segment_windows.csv')
    return task_info, imu, lidar, segment_windows

def interpolate_imu(imu, t):
    t_imu = imu['imu_time_s'].values
    omega = imu['omega_z_rad_s'].values
    f = interp1d(t_imu, omega, kind='linear', fill_value='extrapolate')
    return f(t)

def integrate_omega(omega, t_start, t_end):
    t = np.linspace(t_start, t_end, 1000)
    omega_int = np.trapz(omega(t), t)
    return omega_int

def compute_yaw_residuals(lidar_row, gyro_bias, time_offset):
    t0 = lidar_row['t0_lidar_s']
    t1 = lidar_row['t1_lidar_s']
    t_start = t0 - time_offset
    t_end = t1 - time_offset
    omega = interpolate_imu(imu, t)
    omega_int = integrate_omega(omega, t_start, t_end)
    yaw_residual = (omega_int - gyro_bias) % (2 * np.pi)
    return yaw_residual

def compute_trans_residuals(lidar_row, yaw_extrinsic):
    dx = lidar_row['dx_lidar_m']
    dy = lidar_row['dy_lidar_m']
    trans_residual = np.sqrt(dx**2 + dy**2) * np.sin(yaw_extrinsic)
    return trans_residual

def objective(params, lidar, imu, task_info):
    gyro_bias, time_offset, yaw_extrinsic = params
    residuals = []
    for index, row in lidar.iterrows():
        if row['split'] != 'train':
            continue
        yaw_residual = compute_yaw_residuals(row, gyro_bias, time_offset)
        trans_residual = compute_trans_residuals(row, yaw_extrinsic)
        residuals.append((yaw_residual, trans_residual))
    yaw_residuals = np.array([r[0] for r in residuals])
    trans_residuals = np.array([r[1] for r in residuals])
    return np.mean(yaw_residuals**2 + trans_residuals**2)

def optimize_calibration(lidar, imu, task_info):
    initial_guess = [0.0, 0.0, 0.0]
    result = minimize(objective, initial_guess, args=(lidar, imu, task_info))
    return result.x

def compute_observability(segment_windows, lidar):
    observability_scores = []
    for index, row in segment_windows.iterrows():
        segment_id = row['segment_id']
        t0 = row['t0_segment_s']
        t1 = row['t1_segment_s']
        yaw_rate = lidar[lidar['t0_lidar_s'].between(t0, t1)]['dyaw_lidar_rad'].values
        yaw_rate_var = np.var(yaw_rate)
        observability_scores.append({'segment_id': segment_id, 'observability_score': yaw_rate_var})
    return observability_scores

def main():
    task_info, imu, lidar, segment_windows = load_data()
    calibration_params = optimize_calibration(lidar, imu, task_info)
    observability = compute_observability(segment_windows, lidar)
    # Save calibration results
    calibration_df = pd.DataFrame([calibration_params], columns=['gyro_bias_rad_s', 'time_offset_s', 'yaw_extrinsic_rad'])
    calibration_df.to_csv('calibration.csv', index=False)
    # Save observability scores
    observability_df = pd.DataFrame(observability)
    observability_df.to_csv('observability.csv', index=False)
    # Save residuals
    heldout_lidar = lidar[lidar['split'] == 'heldout']
    residuals = []
    for index, row in heldout_lidar.iterrows():
        yaw_residual = compute_yaw_residuals(row, calibration_params[0], calibration_params[1])
        trans_residual = compute_trans_residuals(row, calibration_params[2])
        residuals.append({'scan_id': row['scan_id'], 'residual_yaw_rad': yaw_residual, 'residual_trans_m': trans_residual})
    residuals_df = pd.DataFrame(residuals)
    residuals_df.to_csv('residuals.csv', index=False)
    # Plot diagnostics
    plt.figure(figsize=(12, 6))
    plt.plot(lidar['t0_lidar_s'], lidar['dyaw_lidar_rad'], label='LiDAR yaw')
    plt.plot(lidar['t0_lidar_s'], [calibration_params[2]] * len(lidar), label='Estimated yaw extrinsic')
    plt.legend()
    plt.savefig('calibration_diagnostics.png')
    # Save metrics
    metrics = {'gyro_bias_rad_s': calibration_params[0], 'time_offset_s': calibration_params[1], 'yaw_extrinsic_rad': calibration_params[2]}
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
```