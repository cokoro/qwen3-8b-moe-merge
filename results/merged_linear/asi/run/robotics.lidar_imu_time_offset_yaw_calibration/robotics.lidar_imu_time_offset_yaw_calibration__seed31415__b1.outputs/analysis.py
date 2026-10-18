import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import os

# Load task info
task_info = json.load(open("data/task_info.json"))
print(f"Task info loaded: {task_info}")

# Load data
imu_gyro = pd.read_csv("data/imu_gyro.csv")
lidar_scan_matches = pd.read_csv("data/lidar_scan_matches.csv")
segment_windows = pd.read_csv("data/segment_windows.csv")

# Preprocess: sort IMU and LiDAR scans by time
imu_gyro.sort_values("imu_time_s", inplace=True)
lidar_scan_matches.sort_values("scan_id", inplace=True)

# Extract training and held-out scan matches
train_rows = lidar_scan_matches[lidar_scan_matches["split"] == "train"]
heldout_rows = lidar_scan_matches[lidar_scan_matches["split"] == "heldout"]

# Extract segment windows
segment_windows.sort_values("segment_id", inplace=True)

# Calibration function
def calibration_objective(params, train_rows, imu_gyro):
    gyro_bias, time_offset, yaw_extrinsic = params
    residuals = []
    for _, row in train_rows.iterrows():
        t0_lidar = row["t0_lidar_s"]
        t1_lidar = row["t1_lidar_s"]
        imu_t0 = t0_lidar - time_offset
        imu_t1 = t1_lidar - time_offset

        # Find IMU indices in the time window
        imu_t0_idx = np.searchsorted(imu_gyro["imu_time_s"], imu_t0, side="right") - 1
        imu_t1_idx = np.searchsorted(imu_gyro["imu_time_s"], imu_t1, side="left")

        # Interpolate IMU angular velocity
        omega = imu_gyro["omega_z_rad_s"][imu_t0_idx:imu_t1_idx]
        omega_interp = np.interp(
            np.linspace(t0_lidar, t1_lidar, len(omega)),
            imu_gyro["imu_time_s"][imu_t0_idx:imu_t1_idx],
            omega,
        )

        # Integrate angular velocity to get predicted yaw
        predicted_yaw = np.trapz(omega_interp - gyro_bias, dx=(t1_lidar - t0_lidar) / len(omega_interp))

        # Wrap to [-pi, pi)
        predicted_yaw = np.arctan2(np.sin(predicted_yaw), np.cos(predicted_yaw))

        # Compute residual
        residual = (predicted_yaw - row["dyaw_lidar_rad"]) % (2 * np.pi)
        residual = np.where(residual > np.pi, residual - 2 * np.pi, residual)

        residuals.append(residual)

    # Compute mean squared error
    return np.mean(residuals ** 2)

# Initial guess
initial_guess = np.array([0.0, 0.0, 0.0])

# Optimize
result = minimize(
    calibration_objective,
    initial_guess,
    args=(train_rows, imu_gyro),
    method="L-BFGS-B",
    bounds=[(-10.0, 10.0), (-10.0, 10.0), (-np.pi, np.pi)],
)

# Save calibration results
calibration_df = pd.DataFrame(
    [result.x],
    columns=["gyro_bias_rad_s", "time_offset_s", "yaw_extrinsic_rad"],
)
calibration_df.to_csv("calibration.csv", index=False)

# Compute observability scores
observability_scores = []
for _, row in segment_windows.iterrows():
    segment_id = row["segment_id"]
    segment_start = row["segment_start"]
    segment_end = row["segment_end"]

    # Find IMU indices in the segment window
    imu_start_idx = np.searchsorted(imu_gyro["imu_time_s"], segment_start, side="right") - 1
    imu_end_idx = np.searchsorted(imu_gyro["imu_time_s"], segment_end, side="left")

    # Extract angular velocity in the segment window
    segment_omega = imu_gyro["omega_z_rad_s"][imu_start_idx:imu_end_idx]

    # Compute observability score: standard deviation of angular velocity
    observability_score = np.std(segment_omega)
    observability_scores.append({"segment_id": segment_id, "observability_score": observability_score})

observability_df = pd.DataFrame(observability_scores)
observability_df.to_csv("observability.csv", index=False)

# Compute held-out residuals
heldout_residuals = []
for _, row in heldout_rows.iterrows():
    t0_lidar = row["t0_lidar_s"]
    t1_lidar = row["t1_lidar_s"]
    dx_lidar = row["dx_lidar_m"]
    dy_lidar = row["dy_lidar_m"]
    dyaw_lidar = row["dyaw_lidar_rad"]

    imu_t0 = t0_lidar - result.x[1]
    imu_t1 = t1_lidar - result.x[1]

    # Find IMU indices in the time window
    imu_t0_idx = np.searchsorted(imu_gyro["imu_time_s"], imu_t0, side="right") - 1
    imu_t1_idx = np.searchsorted(imu_gyro["imu_time_s"], imu_t1, side="left")

    # Interpolate IMU angular velocity
    omega = imu_gyro["omega_z_rad_s"][imu_t0_idx:imu_t1_idx]
    omega_interp = np.interp(
        np.linspace(t0_lidar, t1_lidar, len(omega)),
        imu_gyro["imu_time_s"][imu_t0_idx:imu_t1_idx],
        omega,
    )

    # Integrate angular velocity to get predicted yaw
    predicted_yaw = np.trapz(omega_interp - result.x[0], dx=(t1_lidar - t0_lidar) / len(omega_interp))

    # Wrap to [-pi, pi)
    predicted_yaw = np.arctan2(np.sin(predicted_yaw), np.cos(predicted_yaw))

    # Compute residual
    residual = (predicted_yaw - dyaw_lidar) % (2 * np.pi)
    residual = np.where(residual > np.pi, residual - 2 * np.pi, residual)

    # Compute lateral residual
    rotated_dx = dx_lidar * np.cos(result.x[2]) - dy_lidar * np.sin(result.x[2])
    lateral_residual = np.abs(rotated_dx)

    heldout_residuals.append(
        {
            "scan_id": row["scan_id"],
            "residual_yaw_rad": residual,
            "residual_trans_m": lateral_residual,
        }
    )

heldout_df = pd.DataFrame(heldout_residuals)
heldout_df.to_csv("residuals.csv", index=False)

# Plot calibration diagnostics
plt.figure(figsize=(12, 6))

# Plot IMU angular velocity distribution
plt.subplot(2, 1, 1)
plt.hist(imu_gyro["omega_z_rad_s"], bins=50, color="blue", alpha=0.7)
plt.title("IMU Angular Velocity Distribution")
plt.xlabel("Angular Velocity (rad/s)")
plt.ylabel("Count")

# Plot calibration parameter estimates
plt.subplot(2, 1, 2)
plt.bar(["Gyro Bias", "Time Offset", "Yaw Extrinsic"], result.x, color=["red", "green", "blue"], alpha=0.7)
plt.title("Calibration Parameter Estimates")
plt.ylabel("Value (rad/s or rad)")
plt.tight_layout()
plt.savefig("calibration_diagnostics.png")

# Save metrics
metrics = {
    "gyro_bias_rad_s": result.x[0],
    "time_offset_s": result.x[1],
    "yaw_extrinsic_rad": result.x[2],
    "residual_yaw_rad_mean": np.mean(heldout_df["residual_yaw_rad"]),
    "residual_trans_m_mean": np.mean(heldout_df["residual_trans_m"]),
    "observability_score_mean": np.mean(observability_df["observability_score"]),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Calibration complete.")