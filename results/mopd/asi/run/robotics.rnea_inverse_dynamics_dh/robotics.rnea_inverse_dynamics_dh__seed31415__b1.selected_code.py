import numpy as np
import json
import os

# Read data
task_info = json.load(open('data/task_info.json'))
calibration = np.load('data/calibration.npz')
trajectory = np.load('data/trajectory.npz')
n_dof = len(task_info['theta_offset'])

# Step 1: DH transform
def dh_transform(theta, a, alpha, d, theta_offset):
    theta_total = theta + theta_offset
    cos_theta = np.cos(theta_total)
    sin_theta = np.sin(theta_total)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    T = np.array([
        [cos_theta, -sin_theta, 0, a],
        [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -d * sin_alpha],
        [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, d * cos_alpha],
        [0, 0, 0, 1]
    ])
    return T

# Step 2: Kinematic Jacobians
def compute_jacobians(q, a, alpha, d, theta_offset, eps=1e-6):
    n = len(theta_offset)
    Jv = np.zeros((3, n))
    Jw = np.zeros((3, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = eps
        q_plus = q + dq
        q_minus = q - dq
        R_plus = np.eye(3)
        R_minus = np.eye(3)
        for i in range(n):
            T_plus = dh_transform(q_plus[i], a[i], alpha[i], d[i], theta_offset[i])
            R_plus = R_plus @ T_plus[:3, :3]
            T_minus = dh_transform(q_minus[i], a[i], alpha[i], d[i], theta_offset[i])
            R_minus = R_minus @ T_minus[:3, :3]
        R = R_plus
        p_plus = np.zeros(3)
        p_minus = np.zeros(3)
        for i in range(n):
            T_plus = dh_transform(q_plus[i], a[i], alpha[i], d[i], theta_offset[i])
            p_plus = p_plus + T_plus[:3, 3]
            T_minus = dh_transform(q_minus[i], a[i], alpha[i], d[i], theta_offset[i])
            p_minus = p_minus + T_minus[:3, 3]
        Jv[:, j] = (p_plus - p_minus) / (2 * eps)
        dR = (R_plus - R_minus) / (2 * eps)
        Jw[:, j] = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])
    return Jv, Jw

# Step 3: Mass matrix
def compute_mass_matrix(q, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base):
    Jv, Jw = compute_jacobians(q, a, alpha, d, theta_offset)
    n = len(theta_offset)
    M = np.zeros((n, n))
    for i in range(n):
        R = dh_transform(q[i], a[i], alpha[i], d[i], theta_offset[i])[:3, :3]
        I_com = np.diag(inertia_diag[i])
        I_world = R @ I_com @ R.T
        M += mass[i] * Jv[:, i].reshape(-1, 1) @ Jv[:, i].reshape(1, -1)
        M += Jw[:, i].reshape(-1, 1) @ (I_world @ Jw[:, i].reshape(1, -1))
    M = 0.5 * (M + M.T)
    return M

# Step 4: Gravity and Coriolis terms
def compute_gravity(q, a, alpha, d, theta_offset, mass, com, gravity_base):
    Jv, Jw = compute_jacobians(q, a, alpha, d, theta_offset)
    n = len(theta_offset)
    g = np.zeros(n)
    for i in range(n):
        R = dh_transform(q[i], a[i], alpha[i], d[i], theta_offset[i])[:3, :3]
        g[i] -= mass[i] * Jv[:, i].T @ gravity_base
    return g

def compute_coriolis(q, qd, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base):
    M = compute_mass_matrix(q, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
    n = len(theta_offset)
    dM = np.zeros((n, n, n))
    for k in range(n):
        q_plus = np.copy(q)
        q_plus[k] += 1e-5
        M_plus = compute_mass_matrix(q_plus, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
        dM[:, :, k] = (M_plus - M) / (2 * 1e-5)
    C = np.zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i] += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qd[j] * qd[k]
    return C

# Step 5: Last-link parameter identification
def parameterize_last_link(theta, mass, com, inertia_diag):
    m = theta[0]
    cx = theta[1] / m
    cy = theta[2] / m
    cz = theta[3] / m
    Ixx = theta[4]
    Iyy = theta[5]
    Izz = theta[6]
    Ixy = theta[7]
    Ixz = theta[8]
    Iyz = theta[9]
    I_com = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    return m, np.array([cx, cy, cz]), I_com

def unparameterize_last_link(theta, m, com, I_com):
    cx, cy, cz = com
    Ixx = theta[4]
    Iyy = theta[5]
    Izz = theta[6]
    Ixy = theta[7]
    Ixz = theta[8]
    Iyz = theta[9]
    I_com = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    return theta[0], np.array([cx, cy, cz]), I_com

def identify_last_link_parameters(calibration, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base):
    n_dof = len(theta_offset)
    q = calibration['q']
    qd = calibration['qd']
    qdd = calibration['qdd']
    tau = calibration['tau']
    n_samples = q.shape[0]
    Jv = np.zeros((3, n_samples, n_dof))
    Jw = np.zeros((3, n_samples, n_dof))
    for i in range(n_samples):
        Jv_i, Jw_i = compute_jacobians(q[i], a, alpha, d, theta_offset)
        Jv[:, i, :] = Jv_i
        Jw[:, i, :] = Jw_i
    M = np.zeros((n_samples, n_dof, n_dof))
    for i in range(n_samples):
        M[i] = compute_mass_matrix(q[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
    C = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        C[i] = compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
    g = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        g[i] = compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
    theta0 = np.array([mass[-1]] + [mass[-1] * c for c in com[-1]] + [inertia_diag[-1][0], inertia_diag[-1][1], inertia_diag[-1][2], 0, 0, 0])
    tau0 = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        tau0[i] = M[i] @ qdd[i] + C[i] + g[i]
    A = np.zeros((n_samples * n_dof, 10))
    b = np.zeros(n_samples * n_dof)
    for i in range(n_samples):
        for j in range(10):
            theta_pert = theta0.copy()
            theta_pert[j] += 1
            tau_pert = np.zeros(n_dof)
            for k in range(n_dof):
                tau_pert[k] = M[i, k, :] @ qdd[i] + compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base) + compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
            A[i * n_dof + k, j] = tau_pert[k] - tau0[i, k]
            b[i * n_dof + k] = tau[i, k] - tau0[i, k]
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    theta_id = theta0 + delta
    return theta_id

# Step 5b: Change-point detection
def find_change_point(calibration, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base):
    n_dof = len(theta_offset)
    q = calibration['q']
    qd = calibration['qd']
    qdd = calibration['qdd']
    tau = calibration['tau']
    n_samples = q.shape[0]
    Jv = np.zeros((3, n_samples, n_dof))
    Jw = np.zeros((3, n_samples, n_dof))
    for i in range(n_samples):
        Jv_i, Jw_i = compute_jacobians(q[i], a, alpha, d, theta_offset)
        Jv[:, i, :] = Jv_i
        Jw[:, i, :] = Jw_i
    M = np.zeros((n_samples, n_dof, n_dof))
    for i in range(n_samples):
        M[i] = compute_mass_matrix(q[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
    C = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        C[i] = compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
    g = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        g[i] = compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
    theta0 = np.array([mass[-1]] + [mass[-1] * c for c in com[-1]] + [inertia_diag[-1][0], inertia_diag[-1][1], inertia_diag[-1][2], 0, 0, 0])
    tau0 = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        tau0[i] = M[i] @ qdd[i] + C[i] + g[i]
    A = np.zeros((n_samples * n_dof, 10))
    b = np.zeros(n_samples * n_dof)
    for i in range(n_samples):
        for j in range(10):
            theta_pert = theta0.copy()
            theta_pert[j] += 1
            tau_pert = np.zeros(n_dof)
            for k in range(n_dof):
                tau_pert[k] = M[i, k, :] @ qdd[i] + compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base) + compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
            A[i * n_dof + k, j] = tau_pert[k] - tau0[i, k]
            b[i * n_dof + k] = tau[i, k] - tau0[i, k]
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    theta_id = theta0 + delta
    r = np.linalg.norm(tau - tau0 - A @ delta, axis=1)
    k_star = np.argmin(r)
    return k_star

# Step 6: Test torque history
def compute_test_torques(trajectory, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base, theta_id):
    q = trajectory['q']
    qd = trajectory['qd']
    qdd = trajectory['qdd']
    n_samples = q.shape[0]
    tau = np.zeros((n_samples, n_dof))
    for i in range(n_samples):
        Jv, Jw = compute_jacobians(q[i], a, alpha, d, theta_offset)
        M = compute_mass_matrix(q[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
        C = compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
        g = compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
        tau[i] = M @ qdd[i] + C + g
    return tau

# Step 7: Forward-dynamics verification
def verify_forward_dynamics(trajectory, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base, theta_id):
    q = trajectory['q']
    qd = trajectory['qd']
    qdd = trajectory['qdd']
    n_samples = q.shape[0]
    max_error = 0
    for i in range(n_samples):
        Jv, Jw = compute_jacobians(q[i], a, alpha, d, theta_offset)
        M = compute_mass_matrix(q[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
        C = compute_coriolis(q[i], qd[i], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
        g = compute_gravity(q[i], a, alpha, d, theta_offset, mass, com, gravity_base)
        tau = M @ qdd[i] + C + g
        M_inv = np.linalg.pinv(M)
        qdd_est = M_inv @ (tau - C - g)
        error = np.max(np.abs(qdd_est - qdd[i]))
        if error > max_error:
            max_error = error
    return max_error

# Main
task_info = json.load(open('data/task_info.json'))
calibration = np.load('data/calibration.npz')
trajectory = np.load('data/trajectory.npz')
n_dof = len(task_info['theta_offset'])
a = np.array(task_info['a'])
alpha = np.array(task_info['alpha'])
d = np.array(task_info['d'])
theta_offset = np.array(task_info['theta_offset'])
mass = np.array(task_info['mass'])
com = np.array(task_info['com'])
inertia_diag = np.array(task_info['inertia_diag'])
gravity_base = np.array(task_info['gravity_base'])

theta_id = identify_last_link_parameters(calibration, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
k_star = find_change_point(calibration, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)
theta_id = identify_last_link_parameters(calibration[k_star:], a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base)

tau_history = compute_test_torques(trajectory, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base, theta_id)
verification_data = verify_forward_dynamics(trajectory, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base, theta_id)

np.save('tau_history.npy', tau_history)
np.save('verification_data_1.npy', np.array([verification_data], dtype=np.float64))