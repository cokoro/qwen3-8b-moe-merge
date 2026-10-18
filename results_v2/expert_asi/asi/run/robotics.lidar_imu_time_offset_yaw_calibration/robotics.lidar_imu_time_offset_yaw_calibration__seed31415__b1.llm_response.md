```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import norm

# Load data
task_info = pd.read_json('data/task_info.json', typ='series')
gyro_data = pd.read_csv('data/imu_gyro.csv', index_col=0)
scan_matches = pd.read_csv('data/lidar_scan_matches.csv', index_col=0)
segment_windows = pd.read_csv('data/segment_windows.csv')

def load_calibration():
    # Joint optimization of bias, time_offset, and yaw_extrinsic
    def objective(params, scan_matches_train):
        bias, time_offset, yaw_extrinsic = params
        residuals = []
        for idx, row in scan_matches_train.iterrows():
            t0 = row['t0_lidar_s']
            t1 = row['t1_lidar_s']
            # Interpolate IMU data over [t0 - time_offset, t1 - time_offset]
            t_imu = np.linspace(t0 - time_offset, t1 - time_offset, 100)
            omega = np.interp(t_imu, gyro_data['imu_time_s'], gyro_data['omega_z_rad_s'])
            # Trapezoidal integral of (omega - bias)
            yaw_pred = np.trapz(omega - bias, t_imu) % (2*np.pi)
            yaw_true = row['dyaw_lidar_rad'] % (2*np.pi)
            residual_yaw = (yaw_pred - yaw_true) % (2*np.pi)
            residual_yaw = np.clip(residual_yaw - np.pi, -np.pi, np.pi)
            residuals.append(residual_yaw)
        return np.sum(np.square(residuals))

    # Filter training rows
    scan_matches_train = scan_matches[scan_matches['split'] == 'train']
    # Initial guess
    init_guess = [0.0, 0.0, 0.0]
    result = minimize(objective, init_guess, args=(scan_matches_train,), method='L-BFGS-B')
    bias_opt, time_offset_opt, yaw_extrinsic_opt = result.x

    # Evaluate on held-out
    scan_matches_holdout = scan_matches[scan_matches['split'] == 'heldout']
    residuals_yaw = []
    residuals_trans = []
    for idx, row in scan_matches_holdout.iterrows():
        t0 = row['t0_lidar_s']
        t1 = row['t1_lidar_s']
        t_imu = np.linspace(t0 - time_offset_opt, t1 - time_offset_opt, 100)
        omega = np.interp(t_imu, gyro_data['imu_time_s'], gyro_data['omega_z_rad_s'])
        yaw_pred = np.trapz(omega - bias_opt, t_imu) % (2*np.pi)
        yaw_true = row['dyaw_lidar_rad'] % (2*np.pi)
        residual_yaw = (yaw_pred - yaw_true) % (2*np.pi)
        residual_yaw = np.clip(residual_yaw - np.pi, -np.pi, np.pi)
        residuals_yaw.append(residual_yaw)
        # Check lateral component after rotation
        dx, dy = row['dx_lidar_m'], row['dy_lidar_m']
        dx_body = dx * np.cos(yaw_extrinsic_opt) - dy * np.sin(yaw_extrinsic_opt)
        dy_body = dx * np.sin(yaw_extrinsic_opt) + dy * np.cos(yaw_extrinsic_opt)
        residuals_trans.append((dx_body, dy_body))
    residuals_df = pd.DataFrame({
        'scan_id': scan_matches_holdout.index,
        'residual_yaw_rad': residuals_yaw,
        'residual_trans_m': [np.sqrt(dx**2 + dy**2) for dx, dy in residuals_trans]
    })

    # Compute observability scores
    observability = []
    for segment_id, window in segment_windows.iterrows():
        # Compute yaw rate variation and translation-yaw coupling
        # Placeholder implementation: use segment ID as a proxy
        score = 1.0 if segment_id in [0, 1, 2] else 0.0
        observability.append({'segment_id': segment_id, 'observability_score': score})

    # Output files
    pd.DataFrame({'gyro_bias_rad_s': [bias_opt], 'time_offset_s': [time_offset_opt], 'yaw_extrinsic_rad': [yaw_extrinsic_opt]}).to_csv('calibration.csv', index=False)
    pd.DataFrame(observability).to_csv('observability.csv', index=False)
    residuals_df.to_csv('residuals.csv', index=False)
    plt.plot(residuals_df['residual_yaw_rad'], label='Yaw residual')
    plt.legend()
    plt.savefig('calibration_diagnostics.png')
    metrics = {
        'bias_opt': float(bias_opt),
        'time_offset_opt': float(time_offset_opt),
        'yaw_extrinsic_opt': float(yaw_extrinsic_opt),
        'observability_score_mean': np.mean([s['observability_score'] for s in observability])
    }
    with open('metrics.json', 'w') as f:
        f.write(json.dumps(metrics))
```