```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def load_data(path):
    return pd.read_csv(path)

def wrap_angle(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi

def compute_observability(segment_ids, yaw_rates, translations):
    scores = np.zeros(len(segment_ids))
    for i, sid in enumerate(segment_ids):
        yaw_var = np.var(yaw_rates[sid])
        trans_var = np.var(translations[sid])
        scores[i] = (yaw_var + trans_var) / (1 + np.exp(-yaw_var * 10))
    return scores

def main():
    imu = load_data('data/imu_gyro.csv')
    lidar = load_data('data/lidar_scan_matches.csv')
    segments = load_data('data/segment_windows.csv')
    task_info = load_data('data/task_info.json')

    # Extract training and held-out indices
    train_mask = lidar['split'] == 'train'
    heldout_mask = lidar['split'] == 'heldout'

    # Preprocess IMU for interpolation
    imu_t = imu['imu_time_s'].values
    omega_z = imu['omega_z_rad_s'].values
    f = interp1d(imu_t, omega_z, kind='linear', fill_value='extrapolate')

    # Objective function for optimization
    def objective(params):
        gyro_bias, time_offset = params
        yaw_extrinsic = 0.0
        total_residual = 0.0
        for idx, row in lidar[train_mask].iterrows():
            t0 = row['t0_lidar_s']
            t1 = row['t1_lidar_s']
            omega = f(np.linspace(t0 - time_offset, t1 - time_offset, 100))
            yaw_pred = np.trapz(omega - gyro_bias, dx=(t1 - t0) / 100)
            yaw_res = wrap_angle(row['dyaw_lidar_rad'] - yaw_pred)
            total_residual += np.abs(yaw_res)
        return total_residual

    # Optimize gyro bias and time offset
    initial_guess = [0.0, 0.0]
    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(-1e-3, 1e-3), (-1e-3, 1e-3)])
    gyro_bias_opt, time_offset_opt = res.x

    # Estimate yaw extrinsic using lateral translation constraint
    def lateral_residual(yaw_extrinsic):
        lateral = 0.0
        for idx, row in lidar[train_mask].iterrows():
            dx, dy = row['dx_lidar_m'], row['dy_lidar_m']
            body_dx = dx * np.cos(yaw_extrinsic) - dy * np.sin(yaw_extrinsic)
            lateral += np.abs(body_dx)
        return lateral

    yaw_extrinsic_opt = minimize(lateral_residual, 0.0, method='L-BFGS-B', bounds=[(-np.pi, np.pi)]).x[0]

    # Compute residuals for held-out data
    heldout_residuals = []
    for idx, row in lidar[heldout_mask].iterrows():
        t0 = row['t0_lidar_s']
        t1 = row['t1_lidar_s']
        omega = f(np.linspace(t0 - time_offset_opt, t1 - time_offset_opt, 100))
        yaw_pred = np.trapz(omega - gyro_bias_opt, dx=(t1 - t0) / 100)
        yaw_res = wrap_angle(row['dyaw_lidar_rad'] - yaw_pred)
        heldout_residuals.append({
            'scan_id': row['scan_id'],
            'residual_yaw_rad': yaw_res,
            'residual_trans_m': np.sqrt(row['dx_lidar_m']**2 + row['dy_lidar_m']**2)
        })

    # Compute observability scores
    segment_ids = segments['segment_id'].values
    yaw_rates = []
    translations = []
    for sid in segment_ids:
        yaw_rate = np.zeros(len(lidar))
        trans = np.zeros(len(lidar))
        for idx, row in lidar.iterrows():
            if row['segment_id'] == sid:
                yaw_rate[idx] = row['dyaw_lidar_rad']
                trans[idx] = np.sqrt(row['dx_lidar_m']**2 + row['dy_lidar_m']**2)
        yaw_rates.append(yaw_rate)
        translations.append(trans)
    observability_scores = compute_observability(segment_ids, yaw_rates, translations)

    # Save outputs
    pd.DataFrame([{'gyro_bias_rad_s': gyro_bias_opt, 'time_offset_s': time_offset_opt, 'yaw_extrinsic_rad': yaw_extrinsic_opt}]).to_csv('calibration.csv', index=False)
    pd.DataFrame({'segment_id': segment_ids, 'observability_score': observability_scores}).to_csv('observability.csv', index=False)
    pd.DataFrame(heldout_residuals).to_csv('residuals.csv', index=False)

    # Plot diagnostics
    plt.figure(figsize=(12, 6))
    plt.plot(imu_t, omega_z, label='IMU omega_z')
    plt.plot(imu_t, f(imu_t), 'r--', label='Interpolated omega_z')
    plt.legend()
    plt.savefig('calibration_diagnostics.png')

    metrics = {
        'gyro_bias_opt': gyro_bias_opt,
        'time_offset_opt': time_offset_opt,
        'yaw_extrinsic_opt': yaw_extrinsic_opt,
        'heldout_residuals': heldout_residuals
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
```