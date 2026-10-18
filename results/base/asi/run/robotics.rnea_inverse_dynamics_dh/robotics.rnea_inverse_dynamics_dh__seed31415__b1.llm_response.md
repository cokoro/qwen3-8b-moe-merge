```python
import numpy as np
import json
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_npz(file_path):
    return np.load(file_path)

def save_npz(file_path, data):
    np.savez(file_path, **data)

def read_task_info(task_info_path):
    task_info = load_json(task_info_path)
    n_dof = len(task_info['theta_offset'])
    a = np.array(task_info['a'])
    alpha = np.array(task_info['alpha'])
    d = np.array(task_info['d'])
    theta_offset = np.array(task_info['theta_offset'])
    mass = np.array(task_info['mass'])
    com = np.array(task_info['com'])
    inertia_diag = np.array(task_info['inertia_diag'])
    gravity_base = np.array(task_info['gravity_base'])
    return n_dof, a, alpha, d, theta_offset, mass, com, inertia_diag, gravity_base

def compute_T_i(theta, a, alpha, d, theta_offset, i):
    theta_i = theta[i] + theta_offset[i]
    cos_theta = np.cos(theta_i)
    sin_theta = np.sin(theta_i)
    cos_alpha = np.cos(alpha[i])
    sin_alpha = np.sin(alpha[i])
    T = np.eye(4)
    T[0, 0] = cos_theta
    T[0, 1] = -sin_theta
    T[1, 0] = sin_theta * cos_alpha
    T[1, 1] = cos_theta * cos_alpha
    T[1, 2] = -sin_alpha
    T[1, 3] = -d[i] * sin_alpha
    T[2, 0] = sin_theta * sin_alpha
    T[2, 1] = cos_theta * sin_alpha
    T[2, 2] = cos_alpha
    T[2, 3] = d[i] * cos_alpha
    return T

def compute_R_i(theta, a, alpha, d, theta_offset, i):
    T = compute_T_i(theta, a, alpha, d, theta_offset, i)
    return T[:3, :3]

def compute_p_com_i(theta, a, alpha, d, theta_offset, com, i):
    T = compute_T_i(theta, a, alpha, d, theta_offset, i)
    p_com_i = T[:3, 3] + T[:3, :3] @ com[i]
    return p_com_i

def compute_Jv_i(theta, a, alpha, d, theta_offset, com, i, eps):
    n_dof = len(theta)
    Jv_i = np.zeros((3, n_dof))
    for j in range(n_dof):
        dq = np.zeros(n_dof)
        dq[j] = eps
        p_com_plus = compute_p_com_i(theta + dq, a, alpha, d, theta_offset, com, i)
        p_com_minus = compute_p_com_i(theta - dq, a, alpha, d, theta_offset, com, i)
        Jv_i[:, j] = (p_com_plus - p_com_minus) / (2 * eps)
    return Jv_i

def compute_Jw_i(theta, a, alpha, d, theta_offset, i, eps):
    n_dof = len(theta)
    Jw_i = np.zeros((3, n_dof))
    for j in range(n_dof):
        dq = np.zeros(n_dof)
        dq[j] = eps
        R_plus = compute_R_i(theta + dq, a, alpha, d, theta_offset, i)
        R_minus = compute_R_i(theta - dq, a, alpha, d, theta_offset, i)
        dR_dqj = (R_plus - R_minus) / (2 * eps)
        Jw_i[:, j] = np.array([dR_dqj[2, 1] - dR_dqj[1, 2],
                               dR_dqj[0, 2] - dR_dqj[2, 0],
                               dR_dqj[1, 0] - dR_dqj[0, 1]])
    return Jw_i

def compute_M(theta, a, alpha, d, theta_offset, mass, com, eps):
    n_dof = len(theta)
    Jv = np.zeros((3, n_dof, n_dof))
    Jw = np.zeros((3, n_dof, n_dof))
    for i in range(n_dof):
        Jv_i = compute_Jv_i(theta, a, alpha, d, theta_offset, com, i, eps)
        Jw_i = compute_Jw_i(theta, a, alpha, d, theta_offset, i, eps)
        Jv[:, :, i] = Jv_i
        Jw[:, :, i] = Jw_i
    M = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        I_world_i = np.eye(3)
        for j in range(3):
            for k in range(3):
                I_world_i[j, k] = np.dot(np.dot(com[i][j], np.eye(3)), com[i][k])
        I_world_i = np.diag(inertia_diag[i]) if i != n_dof - 1 else I_world_i
        M += mass[i] * np.dot(Jv[:, :, i].T, Jv[:, :, i]) + np.dot(Jw[:, :, i].T, np.dot(I_world_i, Jw[:, :, i]))
    M = 0.5 * (M + M.T)
    return M

def compute_dM(theta, a, alpha, d, theta_offset, mass, com, eps):
    n_dof = len(theta)
    dM = np.zeros((n_dof, n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            M_plus = compute_M(theta_plus, a, alpha, d, theta_offset, mass, com, eps)
            M_minus = compute_M(theta_minus, a, alpha, d, theta_offset, mass, com, eps)
            dM[i, j, :] = (M_plus - M_minus) / (2 * eps)
    return dM

def compute_C(theta, a, alpha, d, theta_offset, qd, dM, eps):
    n_dof = len(theta)
    C = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            for k in range(n_dof):
                C[i, j] += 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * qd[j] * qd[k]
    return C

def compute_g(theta, a, alpha, d, theta_offset, gravity_base, mass, com):
    n_dof = len(theta)
    g = np.zeros(n_dof)
    for i in range(n_dof):
        Jv_i = compute_Jv_i(theta, a, alpha, d, theta_offset, com, i, 1e-6)
        g += -mass[i] * np.dot(Jv_i.T, gravity_base)
    return g

def compute_tau(theta, qd, qdd, a, alpha, d, theta_offset, mass, com, eps):
    M = compute_M(theta, a, alpha, d, theta_offset, mass, com, eps)
    C = compute_C(theta, a, alpha, d, theta_offset, qd, compute_dM(theta, a, alpha, d, theta_offset, mass, com, eps), eps)
    g = compute_g(theta, a, alpha, d, theta_offset, gravity_base, mass, com)
    tau = np.dot(M, qdd) + np.dot(C, qd) + g
    return tau

def compute_theta_id(task_info, calibration_data, eps):
    n_dof = len(task_info['theta_offset'])
    a = np.array(task_info['a'])
    alpha = np.array(task_info['alpha'])
    d = np.array(task_info['d'])
    theta_offset = np.array(task_info['theta_offset'])
    mass = np.array(task_info['mass'])
    com = np.array(task_info['com'])
    inertia_diag = np.array(task_info['inertia_diag'])
    gravity_base = np.array(task_info['gravity_base'])

    theta0 = np.zeros(n_dof)
    for i in range(n_dof):
        if i == n_dof - 1:
            m = mass[i]
            c = com[i]
            I_com = np.diag(inertia_diag[i])
            theta0[i] = m
            theta0[i+1:i+4] = m * c
            theta0[i+4:i+7] = I_com.diagonal()
            theta0[i+7:i+10] = 0
        else:
            theta0[i] = mass[i]
            theta0[i+1:i+4] = mass[i] * com[i]
            theta0[i+4:i+7] = np.diag(inertia_diag[i])
            theta0[i+7:i+10] = 0

    q, qd, qdd, tau = calibration_data['q'], calibration_data['qd'], calibration_data['qdd'], calibration_data['tau']
    N = len(q)
    A = np.zeros((N * n_dof, 10))
    b = np.zeros(N * n_dof)
    for k in range(N):
        theta_k = q[k]
        tau0_k = compute_tau(theta_k, qd[k], qdd[k], a, alpha, d, theta_offset, mass, com, eps)
        for j in range(10):
            theta_pert = theta0.copy()
            theta_pert[j] += 1
            tau_pert_k = compute_tau(theta_pert, qd[k], qdd[k], a, alpha, d, theta_offset, mass, com, eps)
            A[k * n_dof + j, j] = 1
            A[k * n_dof + j, :] = tau_pert_k - tau0_k
            b[k * n_dof + j] = tau[k] - tau0_k
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    theta_id = theta0 + delta
    return theta_id

def compute_change_point(theta_id, task_info, calibration_data, eps):
    n_dof = len(task_info['theta_offset'])
    a = np.array(task_info['a'])
    alpha = np.array(task_info['alpha'])
    d = np.array(task_info['d'])
    theta_offset = np.array(task_info['theta_offset'])
    mass = np.array(task_info['mass'])
    com = np.array(task_info['com'])
    inertia_diag = np.array(task_info['inertia_diag'])
    gravity_base = np.array(task_info['gravity_base'])

    q, qd, qdd, tau = calibration_data['q'], calibration_data['qd'], calibration_data['qdd'], calibration_data['tau']
    N = len(q)
    A = np.zeros((N * n_dof, 10))
    b = np.zeros(N * n_dof)
    for k in range(N):
        theta_k = q[k]
        tau0_k = compute_tau(theta_k, qd[k], qdd[k], a, alpha, d, theta_offset, mass, com, eps)
        for j in range(10):
            theta_pert = theta0.copy()
            theta_pert[j] += 1
            tau_pert_k = compute_tau(theta_pert, qd[k], qdd[k], a, alpha, d, theta_offset, mass, com, eps)
            A[k * n_dof + j, j] = 1
            A[k * n_dof + j, :] = tau_pert_k - tau0_k
            b[k * n_dof + j] = tau[k] - tau0_k
    delta = np.linalg.lstsq(A, b, rcond=None)[0]
    theta_id = theta0 + delta
    r = np.linalg.norm(tau - compute_tau(theta_id, qd, qdd, a, alpha, d, theta_offset, mass, com, eps), axis=1)
    r_norm = np.linalg.norm(r, axis=1)
    s = np.argmin(r_norm)
    return s

def compute_tau_history(task_info, trajectory_data, theta_id, eps):
    n_dof = len(task_info['theta_offset'])
    a = np.array(task_info['a'])
    alpha = np.array(task_info['alpha'])
    d = np.array(task_info['d'])
    theta_offset = np.array(task_info['theta_offset'])
    mass = np.array(task_info['mass'])
    com = np.array(task_info['com'])
    inertia_diag = np.array(task_info['inertia_diag'])
    gravity_base = np.array(task_info['gravity_base'])

    q, qd, qdd = trajectory_data['q'], trajectory_data['qd'], trajectory_data['qdd']
    N = len(q)
    tau_history = np.zeros((N, n_dof))
    for k in range(N):
        theta_k = q[k]
        tau_k = compute_tau(theta_k, qd[k], qdd[k], a, alpha, d, theta_offset, mass, com, eps)
        tau_history[k] = tau_k
    return tau_history

def compute_verification_data(task_info, trajectory_data, theta_id, eps):
    n_dof = len(task_info['theta_offset'])
    a = np.array(task_info['a'])
    alpha = np.array(task_info['alpha'])
    d = np.array(task_info['d'])
    theta_offset = np.array(task_info['theta_offset'])
    mass = np.array(task_info['mass'])
    com = np.array(task_info['com'])
    inertia_diag = np.array(task_info['inertia_diag'])
    gravity_base = np.array(task_info['gravity_base'])

    q, qd, qdd = trajectory_data['q'], trajectory_data['qd'], trajectory_data['qdd']
    N = len(q)
    max_accel_error = 0
    for k in range(N):
        theta_k = q[k]
        M = compute_M(theta_k, a, alpha, d, theta_offset, mass, com, eps)
        C = compute_C(theta_k, a, alpha, d, theta_offset, qd[k], compute_dM(theta_k, a, alpha, d, theta_offset, mass, com, eps), eps)
        g = compute_g(theta_k, a, alpha, d, theta_offset, gravity_base, mass, com)
        bias = C + g
        qdd_est = np.linalg.solve(M, trajectory_data['tau'][k] - bias)
        error = np.linalg.norm(qdd_est - qdd[k], np.inf)
        if error > max_accel_error:
            max_accel_error = error
    return np.array([max_accel_error], dtype=np.float64)

def main():
    task_info_path = 'data/task_info.json'
    trajectory_path = 'data/trajectory.npz'
    calibration_path = 'data/calibration.npz'

    task_info = read_task_info(task_info_path)
    trajectory_data = load_npz(trajectory_path)
    calibration_data = load_npz(calibration_path)

    n_dof = task_info['n_dof']
    a = task_info['a']
    alpha = task_info['alpha']
    d = task_info['d']
    theta_offset = task_info['theta_offset']
    mass = task_info['mass']
    com = task_info['com']
    inertia_diag = task_info['inertia_diag']
    gravity_base = task_info['gravity_base']

    theta0 = np.zeros(n_dof)
    for i in range(n_dof):
        if i == n_dof - 1:
            m = mass[i]
            c = com[i]
            I_com = np.diag(inertia_diag[i])
            theta0[i] = m
            theta0[i+1:i+4] = m * c
            theta0[i+4:i+7] = I_com.diagonal()
            theta0[i+7:i+10] = 0
        else:
            theta0[i] = mass[i]
            theta0[i+1:i+4] = mass[i] * com[i]
            theta0[i+4:i+7] = np.diag(inertia_diag[i])
            theta0[i+7:i+10] = 0

    theta_id = compute_theta_id(task_info, calibration_data, 1e-6)
    s = compute_change_point(theta_id, task_info, calibration_data, 1e-6)
    theta_id = compute_theta_id(task_info, calibration_data, 1e-6)
    tau_history = compute_tau_history(task_info, trajectory_data, theta_id, 1e-6)
    verification_data = compute_verification_data(task_info, trajectory_data, theta_id, 1e-6)

    np.save('tau_history.npy', tau_history)
    np.save('verification_data_1.npy', verification_data)

if __name__ == '__main__':
    main()
```