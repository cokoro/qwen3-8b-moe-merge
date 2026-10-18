```python
import numpy as np
import json
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_npz(path):
    return np.load(path)

def save_npz(path, data):
    np.save(path, data)

def compute_mass_matrix(q, task_info):
    n = len(task_info['theta_offset'])
    a = task_info['a']
    alpha = task_info['alpha']
    d = task_info['d']
    theta_offset = task_info['theta_offset']
    gravity_base = task_info['gravity_base']
    mass = task_info['mass']
    com = task_info['com']
    inertia_diag = task_info['inertia_diag']

    # Compute R_i(q), p_com_i(q)
    R = [np.eye(3) for _ in range(n)]
    p = [np.zeros(3) for _ in range(n)]
    for i in range(n):
        theta = q[i] + theta_offset[i]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha[i])
        sin_alpha = np.sin(alpha[i])
        R_i = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha],
            [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha]
        ])
        R[i] = R_i
        p_i = np.array([
            [a[i]],
            [d[i] * sin_alpha],
            [d[i] * cos_alpha]
        ]).flatten()
        p[i] = R[i].T @ com[i] + p_i

    # Compute Jacobians via finite differences
    eps = 1e-6
    Jv = [np.zeros((3, n)) for _ in range(n)]
    Jw = [np.zeros((3, n)) for _ in range(n)]
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = eps
        for i in range(n):
            theta_j = q[j] + theta_offset[j]
            cos_theta_j = np.cos(theta_j)
            sin_theta_j = np.sin(theta_j)
            cos_alpha_j = np.cos(alpha[j])
            sin_alpha_j = np.sin(alpha[j])
            R_j_plus = np.array([
                [cos_theta_j, -sin_theta_j, 0],
                [sin_theta_j * cos_alpha_j, cos_theta_j * cos_alpha_j, -sin_alpha_j],
                [sin_theta_j * sin_alpha_j, cos_theta_j * sin_alpha_j, cos_alpha_j]
            ])
            R_j_minus = R_j_plus.copy()
            R_j_minus[j, :] = R_j_minus[j, :] - 2 * eps * (R_j_plus[j, :] - R_j_minus[j, :])
            dR_dqj = (R_j_plus - R_j_minus) / (2 * eps)
            Jw[i, j] = np.array([dR_dqj[0, 0], dR_dqj[1, 0], dR_dqj[2, 0]])

    # Compute mass matrix
    M = np.zeros((n, n))
    for i in range(n):
        I_com = np.diag(inertia_diag[i])
        I don't have the full inertia tensor for the last link yet
        # Wait, this is the step where we need to compute the full inertia tensor for the last link
        # But in the initial step, we're using the nominal values
        # So for now, proceed with nominal values
        # But wait, this is a placeholder. Need to recompute this after identifying the last link's parameters
        # So during the initial step, use the nominal inertia_diag for the last link
        # Then during the final step, use the identified parameters
</think>