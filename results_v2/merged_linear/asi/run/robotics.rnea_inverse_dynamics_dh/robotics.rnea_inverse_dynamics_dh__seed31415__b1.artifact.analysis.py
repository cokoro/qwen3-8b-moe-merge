import numpy as np
import json

with open("data/task_info.json", "r") as f:
    task_info = json.load(f)
n_dof = len(task_info["theta_offset"])

# Read calibration data
calib = np.load("data/calibration.npz")
q_calib = calib["q"]
qd_calib = calib["qd"]
qdd_calib = calib["qdd"]
tau_calib = calib["tau"]

# Read trajectory data
trajectory = np.load("data/trajectory.npz")
q_traj = trajectory["q"]
qd_traj = trajectory["qd"]
qdd_traj = trajectory["qdd"]

# Read task info
a = task_info["a"]
alpha = task_info["alpha"]
d = task_info["d"]
theta_offset = task_info["theta_offset"]
gravity_base = task_info["gravity_base"]
mass = np.array(task_info["mass"], dtype=float)
com = np.array(task_info["com"], dtype=float)
inertia_diag = np.array(task_info["inertia_diag"], dtype=float)

# Compute transforms for each joint position
def compute_T(q):
    T = np.zeros((n_dof, 4, 4))
    for i in range(n_dof):
        theta = q[i] + theta_offset[i]
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(alpha[i])
        sa = np.sin(alpha[i])
        T[i, 0, 0] = c
        T[i, 0, 1] = -s
        T[i, 0, 2] = 0
        T[i, 0, 3] = a[i]
        T[i, 1, 0] = s * ca
        T[i, 1, 1] = c * ca
        T[i, 1, 2] = -sa
        T[i, 1, 3] = -d[i] * sa
        T[i, 2, 0] = s * sa
        T[i, 2, 1] = c * sa
        T[i, 2, 2] = ca
        T[i, 2, 3] = d[i] * ca
        T[i, 3, 3] = 1
    return T

# Compute Jacobians by central finite differences
eps = 1e-6
n_calib = len(q_calib)
Jv = np.zeros((3, n_dof, n_calib))
Jw = np.zeros((3, n_dof, n_calib))
for j in range(n_dof):
    dq = np.zeros(n_calib)
    dq[:] = eps * (1 if j == 0 else 0)
    T_plus = compute_T(q_calib + dq)
    T_minus = compute_T(q_calib - dq)
    for i in range(n_dof):
        R_plus = T_plus[i, :3, :3]
        R_minus = T_minus[i, :3, :3]
        Jv[:, j, i] = (R_plus @ com[i] - R_minus @ com[i]) / (2 * eps)
        dR = (R_plus - R_minus) / (2 * eps)
        Jw[:, j, i] = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])

# Compute mass matrix
def compute_M(q):
    M = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        R_i = compute_T(q)[i, :3, :3]
        I_com_i = np.diag(inertia_diag[i])
        I_world_i = R_i @ I_com_i @ R_i.T
        Jv_i = Jv[:, :, i]
        Jw_i = Jw[:, :, i]
        M += mass[i] * Jv_i.T @ Jv_i + Jw_i.T @ I_world_i @ Jw_i
    return 0.5 * (M + M.T)

# Compute gravity term
def compute_g(q):
    g = np.zeros(n_dof)
    for i in range(n_dof):
        R_i = compute_T(q)[i, :3, :3]
        g += -mass[i] * Jv[:, :, i].T @ gravity_base
    return g

# Compute derivative of mass matrix
def compute_dM(q):
    dM = np.zeros((n_dof, n_dof, n_dof))
    for j in range(n_dof):
        dq = np.zeros(n_dof)
        dq[j] = 1e-5
        M_plus = compute_M(q + dq)
        M_minus = compute_M(q - dq)
        dM[:, :, j] = (M_plus - M_minus) / (2 * 1e-5)
    return dM

# Compute Coriolis term
def compute_C(q, qd):
    C = np.zeros(n_dof)
    dM = compute_dM(q)
    for i in range(n_dof):
        for j in range(n_dof):
            for k in range(n_dof):
                C[i] += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qd[j] * qd[k]
    return C

# Identify last-link parameters
theta0 = np.zeros(10)
theta0[0] = mass[-1]
theta0[1:4] = mass[-1] * com[-1]
theta0[4:7] = inertia_diag[-1]
theta0[7:10] = np.zeros(3)

def compute_tau(q, qd, qdd, theta):
    M = compute_M(q)
    C = compute_C(q, qd)
    g = compute_g(q)
    tau = M @ qdd + C + g
    return tau

def compute_A(q, qd, qdd, theta0):
    A = []
    for k in range(n_calib):
        tau0 = compute_tau(q_calib[k], qd_calib[k], qdd_calib[k], theta0)
        for j in range(10):
            theta = theta0.copy()
            theta[j] += 1
            tau = compute_tau(q_calib[k], qd_calib[k], qdd_calib[k], theta)
            A.append(tau - tau0)
    return np.array(A)

A = compute_A(q_calib, qd_calib, qdd_calib, theta0)
b = np.array([tau_calib[k][j] - compute_tau(q_calib[k], qd_calib[k], qdd_calib[k], theta0)[j] for k in range(n_calib) for j in range(n_dof)])
delta = np.linalg.lstsq(A, b, rcond=None)[0]
theta_id = theta0 + delta

# Detect change-point
def compute_resid(q, qd, qdd, theta):
    tau = compute_tau(q, qd, qdd, theta)
    return np.linalg.norm(tau - tau_calib, axis=1)

resid = compute_resid(q_calib, qd_calib, qdd_calib, theta0)
k_star = np.argmax(resid)

# Refit using post-attachment samples
A = compute_A(q_calib[k_star:], qd_calib[k_star:], qdd_calib[k_star:], theta0)
b = np.array([tau_calib[k][j] - compute_tau(q_calib[k], qd_calib[k], qdd_calib[k], theta0)[j] for k in range(k_star, n_calib) for j in range(n_dof)])
delta = np.linalg.lstsq(A, b, rcond=None)[0]
theta_id = theta0 + delta

# Convert theta_id to (m, c, I_com)
m = theta_id[0]
c = theta_id[...] / m
I_com = np.diag(inertia_diag[-1]) + np.outer(c, c) * m - np.diag(theta_id[4:7] + m * np.dot(c, c)) + theta_id[7:10] * m

# Compute test torques
tau_history = []
for k in range(len(q_traj)):
    q = q_traj[k]
    qd = qd_traj[k]
    qdd = qdd_traj[k]
    M = compute_M(q)
    C = compute_C(q, qd)
    g = compute_g(q)
    tau = M @ qdd + C + g
    tau_history.append(tau)
tau_history = np.array
np.save("tau_history.npy", tau_history)

# Compute forward-dynamics verification
verification_data = []
for k in range(len(q_traj)):
    q = q_traj[k]
    qd = qd_traj[k]
    qdd = qdd_traj[k]
    M = compute_M(q)
    C = compute_C(q, qd)
    g = compute_g(q)
    tau = M @ qdd + C + g
    bias = C + g
    qdd_est = np.linalg.solve(M, tau -1 - bias)
    verification_data.append(np.linalg.norm(qdd_est - qdd_traj[k], ord=np.inf))
verification_data = np.array([np.max(verification_data)], dtype=float64)
np.save("verification_data_1.npy", verification_data)