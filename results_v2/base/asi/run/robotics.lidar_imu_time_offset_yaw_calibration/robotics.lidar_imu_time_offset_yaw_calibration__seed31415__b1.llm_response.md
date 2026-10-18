```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json
import os

# Load data
task_info = json.load(open('data/task_info.json'))
data_dir = 'data/'

imu_data = pd.read_csv(os.path.join(data_dir, 'imu_gyro.csv'))
lidar_data = pd.read_csv(os.path.join(data_dir, 'lidar_scan_matches.csv'))
segment_windows = pd.read_csv(os.path.join(data_dir, 'segment_windows.csv'))

# Define units and split labels
units = task_info['units']
split_labels = task_info['split_labels']

# Extract training and held-out data
train_data = lidar_data[lidar_data['split'] == split_labels['train']]
held_out_data = lidar_data[lidar_data['split'] == split_labels['heldout']]

# Define the calibration function
def objective(params, train_data, imu_data, units):
    gyro_bias, time_offset, yaw_extrinsic = params
    residual_yaw = []
    residual_trans = []
    for _, row in train_data.iterrows():
        t0_lidar = row['t0_lidar_s']
        t1_lidar = row['t1_lidar_s']
        dx_lidar = row['dx_lidar_m']
        dy_lidar = row['dy_lidar_m']
        dyaw_lidar = row['dyaw_lidar_rad']
        sigma_trans = row['sigma_trans_m']
        sigma_yaw = row['sigma_yaw_rad']

        # Determine IMU time interval
        t0_imu = t0_lidar - time_offset
        t1_imu = t1_lidar - time_offset

        # Interpolate IMU data
        imu_times = imu_data['imu_time_s'].values
        omega_z = imu_data['omega_z_rad_s'].values
        omega_z_interp = interp1d(imu_times, omega_z, kind='linear', fill_value='extrapolate')

        # Integrate angular velocity over the interval
        t_vals = np.linspace(t0_imu, t1_imu, 1000)
        omega_z_vals = omega_z_interp(t_vals)
        integrated_yaw = np.trapz(omega_z_vals - gyro_bias, t_vals)
        residual_yaw.append((integrated_yaw - dyaw_lidar) % (2 * np.pi))

        # Compute body-frame translation
        dx_body = dx_lidar * np.cos(yaw_extrinsic) - dy_lidar * np.sin(yaw_extrinsic)
        dy_body = dx_lidar * np.sin(yaw_extrinsic) + dy_lidar * np.cos(yaw_extrinsic)
        residual_trans.append(dy_body)

    # Compute residuals
    residual_yaw = np.array(residual_yaw)
    residual_trans = np.array(residual_trans)

    # Wrap yaw residuals
    residual_yaw = np.where(residual_yaw > np.pi, residual_yaw - 2 * np.pi, residual_yaw)
    residual_yaw = np.where(residual_yaw < -np.pi, residual_yaw + 2 * np.pi, residual_yaw)

    # Compute weighted residuals
    weights_yaw = 1 / (sigma_yaw + 1e-6)
    weights_trans = 1 / (sigma_trans + 1e-6)

    weighted_residuals = (residual_yaw * weights_yaw) ** 2 + (residual_trans * weights_trans) ** 2
    return np.sum(weighted_residuals)

# Initial guess
initial_guess = [0.0, 0.0, 0.0]

# Optimize
result = minimize(objective, initial_guess, args=(train_data, imu_data, units), method='L-BFGS-B', bounds=[(-10, 10), (-1, 1), (-np.pi, np.pi)])

# Extract calibration results
gyro_bias_rad_s = result.x[0]
time_offset_s = result.x[1]
yaw_extrinsic_rad = result.x[2]

# Compute residuals on held-out data
residuals = []
for _, row in held_out_data.iterrows():
    t0_lidar = row['t0_lidar_s']
    t1_lidar = row['t1_lidar_s']
    dx_lidar = row['dx_lidar_m']
    dy_lidar = row['dy_lidar_m']
    dyaw_lidar = row['dyaw_lidar_rad']

    # Determine IMU time interval
    t0_imu = t0_lidar - time_offset_s
    t1_imu = t1_lidar - time_offset_s

    # Interpolate IMU data
    imu_times = imu_data['imu_time_s'].values
    omega_z = imu_data['omega_z_rad_s'].values
    omega_z_interp = interp1d(imu_times, omega_z, kind='linear', fill_value='extrapolate')

    # Integrate angular velocity over the interval
    t_vals = np.linspace(t0_imu, t1_imu, 1000)
    omega_z_vals = omega_z_interp(t_vals)
    integrated_yaw = np.trapz(omega_z_vals - gyro_bias_rad_s, t_vals)
    residual_yaw = (integrated_yaw - dyaw_lidar) % (2 * np.pi)

    # Wrap yaw residual
    residual_yaw = np.where(residual_yaw > np.pi, residual_yaw - 2 * np.pi, residual_yaw)
    residual_yaw = np.where(residual_yaw < -np.pi, residual_yaw + 2 * np.pi, residual_yaw)

    # Compute body-frame translation
    dx_body = dx_lidar * np.cos(yaw_extrinsic_rad) - dy_lidar * np.sin(yaw_extrinsic_rad)
    dy_body = dx_lidar * np.sin(yaw_extrinsic_rad) + dy_lidar * np.cos(yaw_extrinsic_rad)
    residual_trans = dy_body

    residuals.append({
        'scan_id': row['scan_id'],
        'residual_yaw_rad': residual_yaw,
        'residual_trans_m': residual_trans
    })

residuals_df = pd.DataFrame(residuals)

# Compute observability scores
observability_scores = []
for _, row in segment_windows.iterrows():
    segment_id = row['segment_id']
    start_time = row['start_time_s']
    end_time = row['end_time_s']

    # Filter relevant scan matches
    segment_data = lidar_data[
        (lidar_data['t0_lidar_s'] >= start_time) & (lidar_data['t0_lidar_s'] <= end_time)
    ]

    # Compute yaw rate variation
    yaw_rates = []
    for _, scan_row in segment_data.iterrows():
        t0_lidar = scan_row['t0_lidar_s']
        t1_lidar = scan_row['t1_lidar_s']
        dyaw_lidar = scan_row['dyaw_lidar_rad']
        yaw_rate = dyaw_lidar / (t1_lidar - t0_lidar)
        yaw_rates.append(yaw_rate)

    yaw_rate_var = np.var(yaw_rates) if len(yaw_rates) > 0 else 0

    # Compute translation-yaw coupling
    coupling = 0
    for _, scan_row in segment_data.iterrows():
        dx_lidar = scan_row['dx_lidar_m']
        dy_lidar = scan_row['dy_lidar_m']
        dyaw_lidar = scan_row['dyaw_lidar_rad']
        coupling += np.abs(dx_lidar * np.sin(dyaw_lidar) + dy_lidar * np.cos(dyaw_lidar))

    coupling /= len(segment_data) if len(segment_data) > 0 else 1

    observability_score = yaw_rate_var + coupling
    observability_scores.append({
        'segment_id': segment_id,
        'observability_score': observability_score
    })

observability_df = pd.DataFrame(observability_scores)

# Save outputs
calibration_df = pd.DataFrame({
    'gyro_bias_rad_s': [gyro_bias_rad_s],
    'time_offset_s': [time_offset_s],
    'yaw_extrinsic_rad': [yaw_extrinsic_rad]
})
calibration_df.to_csv('calibration.csv', index=False)

observability_df.to_csv('observability.csv', index=False)
residuals_df.to_csv('residuals.csv', index=False)

# Plot calibration diagnostics
plt.figure(figsize=(12, 6))
plt.plot(imu_data['imu_time_s'], imu_data['omega_z_rad_s'], label='IMU Angular Velocity')
plt.axhline(gyro_bias_rad_s, color='r', linestyle='--', label='Estimated Gyro Bias')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('IMU Angular Velocity and Estimated Gyro Bias')
plt.legend()
plt.savefig('calibration_diagnostics.png')

# Save metrics
metrics = {
    'gyro_bias_rad_s': gyro_bias_rad_s,
    'time_offset_s': time_offset_s,
    'yaw_extrinsic_rad': yaw_extrinsic_rad,
    'residual_yaw_mean': np.mean(residuals_df['residual_yaw_rad']),
    'residual_trans_mean': np.mean(residuals_df['residual_trans_m']),
    'observability_score_mean': np.mean(observability_df['observability_score'])
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
```