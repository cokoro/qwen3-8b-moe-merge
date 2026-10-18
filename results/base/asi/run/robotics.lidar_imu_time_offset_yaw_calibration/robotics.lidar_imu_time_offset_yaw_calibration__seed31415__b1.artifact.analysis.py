import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json

# Load data
task_info = json.load(open('data/task_info.json'))
data_dir = 'data/'

imu_data = pd.read_csv(f'{data_dir}imu_gyro.csv')
lidar_data = pd.read_csv(f'{data_dir}lidar_scan_matches.csv')
segment_windows = pd.read_csv(f'{data_dir}segment_windows.csv')

# Parse task info
units = task_info['units']
split_labels = task_info['split_labels']
output_schema = task_info['output_schema']
scoring = task_info['scoring']

# Define time offset and gyro bias function
def compute_yaw_residuals(bias, time_offset, yaw_extrinsic, lidar_data, imu_data):
    residuals = []
    for _, row in lidar_data.iterrows():
        scan_id = row['scan_id']
        t0_lidar = row['t0_lidar_s']
        t1_lidar = row['t1_lidar_s']
        dx_lidar = row['dx_lidar_m']
        dy_lidar = row['dy_lidar_m']
        dyaw_lidar = row['dyaw_lidar_rad']
        sigma_trans = row['sigma_trans_m']
        sigma_yaw = row['sigma_yaw_rad']
        split = row['split']

        if split != 'train':
            continue

        # Find IMU samples in the shifted interval
        t_start = t0_lidar - time_offset
        t_end = t1_lidar - time_offset
        imu_times = imu_data['imu_time_s']
        imu_omega_z = imu_data['omega_z_rad_s']

        # Interpolate IMU data
        f = interp1d(imu_times, imu_omega_z, kind='linear', fill_value='extrapolate')
        omega_z = f(np.linspace(t_start, t_end, 100))

        # Trapezoidal integration
        delta_t = (t_end - t_start) / 99
        integrated_yaw = np.trapz(omega_z - bias, dx=delta_t)

        # Wrap to [-pi, pi)
        integrated_yaw = np.mod(integrated_yaw + np.pi, 2 * np.pi) - np.pi

        # Compute lateral residual
        dx_body = dx_lidar * np.cos(yaw_extrinsic) - dy_lidar * np.sin(yaw_extrinsic)
        dy_body = dx_lidar * np.sin(yaw_extrinsic) + dy_lidar * np.cos(yaw_extrinsic)

        lateral_residual = dy_body

        residuals.append({
            'scan_id': scan_id,
            'residual_yaw_rad': integrated_yaw - dyaw_lidar,
            'residual_trans_m': lateral_residual
        })

    return residuals

# Define observability score
def compute_observability_score(segment_id, lidar_data):
    segment_data = lidar_data[lidar_data['segment_id'] == segment_id]
    yaw_rates = segment_data['dyaw_lidar_rad'].values
    translation_magnitudes = np.sqrt(segment_data['dx_lidar_m']**2 + segment_data['dy_lidar_m']**2)
    yaw_rate_var = np.var(yaw_rates)
    translation_var = np.var(translation_magnitudes)
    observability_score = yaw_rate_var * translation_var
    return observability_score

# Initial guesses
initial_guess = [0.0, 0.0, 0.0]

# Optimization
def objective_function(params):
    bias, time_offset, yaw_extrinsic = params
    residuals = compute_yaw_residuals(bias, time_offset, yaw_extrinsic, lidar_data, imu_data)
    yaw_residuals = [r['residual_yaw_rad'] for r in residuals]
    yaw_residuals_wrapped = np.mod(yaw_residuals + np.pi, 2 * np.pi) - np.pi
    return np.sum(yaw_residuals_wrapped**2)

result = minimize(objective_function, initial_guess, method='L-BFGS-B')

# Calibration results
calibration_results = {
    'gyro_bias_rad_s': result.x[0],
    'time_offset_s': result.x[1],
    'yaw_extrinsic_rad': result.x[2]
}

# Compute residuals on held-out data
held_out_residuals = []
for _, row in lidar_data.iterrows():
    scan_id = row['scan_id']
    t0_lidar = row['t0_lidar_s']
    t1_lidar = row['t1_lidar_s']
    dx_lidar = row['dx_lidar_m']
    dy_lidar = row['dy_lidar_m']
    dyaw_lidar = row['dyaw_lidar_rad']
    split = row['split']

    if split != 'heldout':
        continue

    t_start = t0_lidar - calibration_results['time_offset_s']
    t_end = t1_lidar - calibration_results['time_offset_s']
    imu_times = imu_data['imu_time_s']
    imu_omega_z = imu_data['omega_z_rad_s']

    f = interp1d(imu_times, imu_omega_z, kind='linear', fill_value='extrapolate')
    omega_z = f(np.linspace(t_start, t_end, 100))

    delta_t = (t_end - t_start) / 99
    integrated_yaw = np.trapz(omega_z - calibration_results['gyro_bias_rad_s'], dx=delta_t)
    integrated_yaw = np.mod(integrated_yaw + np.pi, 2 * np.pi) - np.pi

    dx_body = dx_lidar * np.cos(calibration_results['yaw_extrinsic_rad']) - dy_lidar * np.sin(calibration_results['yaw_extrinsic_rad'])
    dy_body = dx_lidar * np.sin(calibration_results['yaw_extrinsic_rad']) + dy_lidar * np.cos(calibration_results['yaw_extrinsic_rad'])

    held_out_residuals.append({
        'scan_id': scan_id,
        'residual_yaw_rad': integrated_yaw - dyaw_lidar,
        'residual_trans_m': dy_body
    })

# Observability scores
observability_scores = []
for _, row in segment_windows.iterrows():
    segment_id = row['segment_id']
    score = compute_observability_score(segment_id, lidar_data)
    observability_scores.append({
        'segment_id': segment_id,
        'observability_score': score
    })

# Save outputs
pd.DataFrame([calibration_results]).to_csv('calibration.csv', index=False)
pd.DataFrame(held_out_residuals).to_csv('residuals.csv', index=False)
pd.DataFrame(observability_scores).to_csv('observability.csv', index=False)

# Plot diagnostics
plt.figure(figsize=(12, 6))
plt.plot(lidar_data['t0_lidar_s'], lidar_data['dyaw_lidar_rad'], 'b.', label='LiDAR yaw')
plt.plot(lidar_data['t0_lidar_s'], [np.mod(r + np.pi, 2 * np.pi) - np.pi for r in [r['residual_yaw_rad'] for r in compute_yaw_residuals(0, 0, 0, lidar_data, imu_data)]], 'r.', label='Residual yaw')
plt.legend()
plt.xlabel('LiDAR time (s)')
plt.ylabel('Yaw (rad)')
plt.title('LiDAR vs. Residual Yaw')
plt.savefig('calibration_diagnostics.png')

# Save metrics
metrics = {
    'calibration_results': calibration_results,
    'observability_scores': observability_scores,
    'held_out_residuals': held_out_residuals
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)