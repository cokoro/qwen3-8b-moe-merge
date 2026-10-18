import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.stats import norm
import json

# Load data
imu_df = pd.read_csv("data/imu_gyro.csv", header=0)
lidar_df = pd.read_csv("data/lidar_scan_matches.csv", header=0)
segment_df = pd.read_csv("data/segment_windows.csv", header=0)
task_info = json.load(open("data/task_info.json"))

# Parse units and split labels
imu_time_s = task_info["units"]["imu_time_s"]
omega_z_rad_s = task_info["units"]["omega_z_rad_s"]
t0_lidar_s = task_info["units"]["t0_lidar_s"]
t1_lidar_s = task_info["units"]["t1_lidar_s"]
dx_lidar_m = task_info["units"]["dx_lidar_m"]
dy_lidar_m = task_info["units"]["dy_lidar_m"]
dyaw_lidar_rad = task_info["units"]["dyaw_lidar_rad"]
split_train = task_info["splits"]["train"]
split_heldout = task_info["splits"]["heldout"]

# Convert time units to seconds
imu_df["imu_time_s"] = imu_df["imu_time_s"] * imu_time_s
lidar_df["t0_lidar_s"] = lidar_df["t0_lidar_s"] * t0_lidar_s
lidar_df["t1_lidar_s"] = lidar_df["t1_lidar_s"] * t1_lidar_s

# Prepare training and held-out sets
train_df = lidar_df[lidar_df["split"] == split_train]
heldout_df = lidar_df[lidar_df["split"] == split_heldout]

# Interpolate IMU data
imu_times = imu_df["imu_time_s"].values
omega_z = imu_df["omega_z_rad_s"].values
f_omega_z = interp1d(imu_times, omega_z, kind="linear", fill_value="extrapolate")

# Define objective function for joint estimation
def objective(params, train_df):
    gyro_bias_rad_s, time_offset_s = params
    residuals = []
    for _, row in train_df.iterrows():
        t0_lidar_s = row["t0_lidar_s"]
        t1_lidar_s = row["t1_lidar_s"]
        dx_lidar_m = row["dx_lidar_m"]
        dy_lidar_m = row["dy_lidar_m"]
        dyaw_lidar_rad = row["dyaw_lidar_rad"]
        # Get IMU interval for this LiDAR interval
        t0_imu_s = t0_lidar_s - time_offset_s
        t1_imu_s = t1_lidar_s - time_offset_s
        # Integrate omega_z over the IMU interval
        omega_z_interp = f_omega_z(np.linspace(t0_imu_s, t1_imu_s, 100))
        omega_z_integrated = trapezoid(omega_z_interp, dx=(t1_imu_s - t0_imu_s)/99)
        # Compute residual
        residual_yaw_rad = (omega_z_integrated - gyro_bias_rad_s - dyaw_lidar_rad) % (2 * np.pi)
        # Normalize residual to [-pi, pi)
        residual_yaw_rad = (residual_yaw_rad + np.pi) % (2 * np.pi) - np.pi
        residuals.append(residual_yaw_rad)
    return np.array(residuals)

# Estimate gyro bias and time offset
def objective_func(params):
    return np.sum(objective(params, train_df)**2)

result = minimize(objective_func, x0=[0.0, 0.0], method="L-BFGS-B")
gyro_bias_rad_s, time_offset_s = result.x

# Estimate yaw extrinsic using translation constraints
def objective_yaw_extrinsic(yaw_extrinsic_rad):
    residuals = []
    for _, row in train_df.iterrows():
        t0_lidar_s = row["t0_lidar_s"]
        t1_lidar_s = row["t1_lidar_s"]
        dx_lidar_m = row["dx_lidar_m"]
        dy_lidar_m = row["dy_lidar_m"]
        # Rotate translation by yaw extrinsic
        dx_body_m = dx_lidar_m * np.cos(yaw_extrinsic_rad) - dy_lidar_m * np.sin(yaw_extrinsic_rad)
        dy_body_m = dx_lidar_m * np.sin(yaw_extrinsic_rad) + dy_lidar_m * np.cos(yaw_extrinsic_rad)
        # Compute residual
        residual_trans_m = np.sqrt(dx_body_m**2 + dy_body_m**2)
        residuals.append(residual_trans_m)
    return np.array(residuals)

# Weight rows with larger translations
train_df["weight"] = 1.0 / np.sqrt(train_df["dx_lidar_m"]**2 + train_df["dy_lidar_m"]**2)
train_df["weight"] = train_df["weight"] / train_df["weight"].mean()

# Optimize yaw extrinsic
result_yaw = minimize(objective_yaw_extrinsic, x0=0.0, method="L-BFGS-B")
yaw_extrinsic_rad = result_yaw.x[0]

# Compute held-out residuals
heldout_residuals = []
for _, row in heldout_df.iterrows():
    t0_lidar_s = row["t0_lidar_s"]
    t1_lidar_s = row["t1_lidar_s"]
    dx_lidar_m = row["dx_lidar_m"]
    dy_lidar_m = row["dy_lidar_m"]
    dyaw_lidar_rad = row["dyaw_lidar_rad"]
    # Get IMU interval for this LiDAR interval
    t0_imu_s = t0_lidar_s - time_offset_s
    t1_imu_s = t1_lidar_s - time_offset_s
    # Integrate omega_z over the IMU interval
    omega_z_interp = f_omega_z(np.linspace(t0_imu_s, t1_imu_s, 100))
    omega_z_integrated = trapezoid(omega_z_interp, dx=(t1_imu_s - t0_imu_s)/99)
    # Compute residual
    residual_yaw_rad = (omega_z_integrated - gyro_bias_rad_s - dyaw_lidar_rad) % (2 * np.pi)
    # Normalize residual to [-pi, pi)
    residual_yaw_rad = (residual_yaw_rad + np.pi) % (2 * np.pi) - np.pi
    # Compute translation residual
    dx_body_m = dx_lidar_m * np.cos(yaw_extrinsic_rad) - dy_lidar_m * np.sin(yaw_extrinsic_rad)
    dy_body_m = dx_lidar_m * np.sin(yaw_extrinsic_rad) + dy_lidar_m * np.cos(yaw_extrinsic_rad)
    residual_trans_m = np.sqrt(dx_body_m**2 + dy_body_m**2)
    heldout_residuals.append((row["scan_id"], residual_yaw_rad, residual_trans_m))

# Compute observability scores
observability_scores = []
for _, row in segment_df.iterrows():
    segment_id = row["segment_id"]
    t0 = row["t0_s"]