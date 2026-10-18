import json
import numpy as np
import os
import sys

# Read task info
task_info = json.load(open("data/task_info.json"))
n_dof = task_info["n_dof"]
a = np.array(task_info["a"])
alpha = np.array(task_info["alpha"])
d = np.array(task_info["d"])
theta_offset = np.array(task_info["theta_offset"])
gravity_base = np.array(task_info["gravity_base"])
mass = np.array(task_info["mass"])
com = np.array(task_info["com"])
inertia_diag = np.array(task_info["inertia_diag"])

# Read calibration data
calibration = np.load("data/calibration.npz")
q_calib = calibration["q"]
qd_calib = calibration["qd"]
qdd_calib = calibration["qdd"]
tau_calib = calibration["tau"]

# Read trajectory data
trajectory = np.load("data/trajectory.npz")
q_traj = trajectory["q"]
qd_traj = trajectory["qd"]
qdd_traj = trajectory["qdd"]

# Helper functions
def cross(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def rot_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def trans_x(a):
    return np.eye(4) + np.array([[0, 0, 0, a],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]])

def trans_z(d):
    return np.eye(4) + np.array([[0, 0, 0, 0],
                                 [0, 0, 0, -d*np.sin(alpha)],
                                 [0, 0, 0, d*np.cos(alpha)],
                                 [0, 0, 0, 0]])

def build_T(theta):
    T = np.eye(4)
    for i in range(n_dof):
        R = rot_x(alpha[i]) @ rot_z(theta[i] + theta_offset[i])
        T_i = trans_x(a[i]) @ trans_z(d[i]) @ R
        T = T @ T_i
    return T

def compute_R_i(q):
    R = np.eye(3)
    for i in range(n_dof):
        R_i = rot_x(alpha[i]) @ rot_z(q[i] + theta_offset[i])
        R = R @ R_i
    return R

def compute_p_com_i(q, i):
    T = build_T(q)
    com_i = com[i]
    p_com_i = T[:3, 3] + T[:3, :3] @ com_i
    return p_com_i

def compute_Jv_i(q):
    Jv = np.zeros((3, n_dof))
    for j in range(n_dof):
        eps = 1e-6
        dq = np.zeros(n_dof)
        dq[j] = eps
        q_plus = q + dq
        q_minus = q - dq
        p_plus = compute_p_com_i(q_plus, n_dof-1)
        p_minus = compute_p_com_i(q_minus, n_dof-1)
        Jv[:, j] = (p_plus - p_minus) / (2*eps)
    return Jv

def compute_Jw_i(q):
    Jw = np.zeros((3, n_dof))
    for j in range(n_dof):
        eps = 1e-6
        dq = np.zeros(n_dof)
        dq[j] = eps
        q_plus = q + dq
        q_minus = q - dq
        R_plus = compute_R_i(q_plus)
        R_minus = compute_R_i(q_minus)
        dR = (R_plus - R_minus) / (2*eps)
        Jw[:, j] = cross(dR @ R_plus.T)
    return Jw

def compute_M(q):
    Jv = compute_Jv_i(q)
    Jw = compute_Jw_i(q)
    M = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        I_world_i = np.eye(3)
        for k in range(n_dof):
            R = compute_R_i(q)
            I_com_i = np.diag(inertia_diag[i])
            I_world_i = R @ I_com_i @ R.T
        M += mass[i] * Jv[:, i] @ Jv[:, i].T
        M += Jw[:, i] @ I_world_i @ Jw[:, i].T
    M = 0.5 * (M + M.T)
    return M

def compute_dM(q):
    dM = np.zeros((n_dof, n_dof, n_dof))
    eps = 1e-5
    for k in range(n_dof):
        dq = np.zeros(n_dof)
        dq[k] = eps
        q_plus = q + dq
        q_minus = q - dq
        M_plus = compute_M(q_plus)
        M_minus = compute_M(q_minus)
        dM[:, :, k] = (M_plus - M_minus) / (2*eps)
    return dM

def compute_C(q, qd):
    C = np.zeros((n_dof, n_dof))
    dM = compute_dM(q)
    for i in range(n_dof):
        for j in range(n_dof):
            for k in range(n_dof):
                C[i, j] += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qd[j] * qd[k]
    return C

def compute_g(q):
    g = np.zeros(n_dof)
    for i in range(n_dof):
        p_com_i = compute_p_com_i(q, i)
        g += -mass[i] * (p_com_i - np.zeros(3)) @ gravity_base
    return g

def compute_tau(q, qd, qdd):
    M = compute_M(q)
    C = compute_C(q, qd)
    g = compute_g(q)
    tau = M @ qdd + C + g
    return tau

def compute_theta_id(calibration_data):
    n_samples = len(calibration_data["q"])
    theta0 = np.array([mass[-1]] + list(com[-1]) + inertia_diag[-1].tolist())
    A = []
    b = []
    for k in range(n_samples):
        q = calibration_data["q"][k]
        qd = calibration_data["qd"][k]
        qdd = calibration_data["qdd"][k]
        tau0 = compute_tau(q, qd, qdd)
        A_k = np.zeros((n_dof, 10))
        for j in range(10):
            theta = theta0.copy()
            theta[j] += 1
            tau = compute_tau(q, qd, qdd)
            A_k[:, j] = tau - tau0
        A.append(A_k)
        b.append(tau_calib[k] - tau0)
    A = np.vstack(A)
    b = np.hstack(b)
    delta, _ = np.linalg.lstsq(A, b, rcond=None)
    theta_id = theta0 + delta
    return theta_id

def compute_residuals(calibration_data, theta_id):
    n_samples = len(calibration_data["q"])
    residuals = []
    for k in range(n_samples):
        q = calibration_data["q"][k]
        qd = calibration_data["qd"][k]
        qdd = calibration_data["qdd"][k]
        tau0 = compute_tau(q, qd, qdd)
        A_k = np.zeros((n_dof, 10))
        for j in range(10):
            theta = theta0.copy()
            theta[j] += 1
            tau = compute_tau(q, qd, qdd)
            A_k[:, j] = tau - tau0
        residual = np.linalg.norm(tau_calib[k] - tau0 - A_k @ delta)
        residuals.append(residual)
    return residuals

def find_change_point(residuals, threshold=1e-3):
    n_samples = len(residuals)
    for s in range(1, n_samples):
        if np.linalg.norm(residuals[s]) > threshold:
            return s
    return n_samples

def compute_final_theta(calibration_data, change_point):
    n_samples = len(calibration_data["q"])
    theta0 = np.array([mass[-1]] + list(com[-1]) + inertia_diag[-1].tolist())
    A = []
    b = []
    for k in range(change_point, n_samples):
        q = calibration_data["q"][k]
        qd = calibration_data["qd"][k]
        qdd = calibration_data["qdd"][k]
        tau0 = compute_tau(q, qd, qdd)
        A_k = np.zeros((n_dof, 10))
        for j in range(10):
            theta = theta0.copy()
            theta[j] += 1
            tau = compute_tau(q, qd, qdd)
            A_k[:, j] = tau - tau0
        A.append(A_k)
        b.append(tau_calib[k] - tau0)
    A = np.vstack(A)
    b = np.hstack(b)
    delta, _ = np.linalg.lstsq(A, b, rcond=None)
    theta_id = theta0 + delta
    return theta_id

def compute_tau_history(theta_id):
    n_samples = len(trajectory["q"])
    tau_history = np.zeros((n_samples, n_dof))
    for k in range(n_samples):
        q = trajectory["q"][k]
        qd = trajectory["qd"][k]
        qdd = trajectory["qdd"][k]
        tau = compute_tau(q, qd, qdd)
        tau_history[k] = tau
    return tau_history

def compute_verification_data(theta_id):
    n_samples = len(trajectory["q"])
    max_accel_error = 0
    for k in range(n_samples):
        q = trajectory["q"][k]
        qd = trajectory["qd"][k]
        qdd = trajectory["qdd"][k]
        M = compute_M(q)
        C = compute_C(q, qd)
        g = compute_g(q)
        bias = C + g
        qdd_est = np.linalg.solve(M, tau_history[k] - bias)
        max_accel_error = max(max_accel_error, np.linalg.norm(qdd_est - qdd, ord=np.inf))
    return np.array([max_accel_error], dtype=np.float64)

# Main execution
theta_id = compute_theta_id(calibration)
residuals = compute_residuals(calibration, theta_id)
change_point = find_change_point(residuals)
theta_id = compute_final_theta(calibration, change_point)
tau_history = compute_tau_history(theta_id)
verification_data = compute_verification_data(theta_id)

np.save("tau_history.npy", tau_history)
np.save("verification_data_1.npy", verification_data)