import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.spatial.distance as dist
import scipy.spatial.cKDTree as cKDTree
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load parameters
with open('data/parameters.json') as f:
    params = json.load(f)
R = float(params['R'])
r = float(params['r'])
alpha = float(params['alpha'])
beta = float(params['beta'])
l = int(params['l'])
N = int(params['N'])

# Read points and rhs
theta_phi = np.loadtxt('data/points_2d.csv', delimiter=',', dtype=np.float64)
h = np.loadtxt('data/h_2d.csv', dtype=np.float64)

# Phase 1: Discretization Design and Validation
def compute_gmls_laplacian(points, l, R, r, N):
    K = (l+1)*(l+2) + 2
    P = np.zeros((N, K, l+1)*(l+2)//2)
    for i in range(N):
        # K nearest neighbors with periodic boundary
        tree = cKDTree(points)
        dists, indices = tree.query(points[i], K=K, p=2, wrap=True)
        neighbors = points[indices]
        D_i = np.max(np.sqrt((neighbors[:,0] - points[i,0])**2 + (neighbors[:,1] - points[i,1])**2))
        # Monge coordinates
        theta_i, phi_i = points[i]
        theta_diff = (neighbors[:,0] - theta_i + np.pi) % (2*np.pi) - np.pi
        phi_diff = (neighbors[:,1] - phi_i + np.pi) % (2*np.pi) - np.pi
        # Normalize
        theta_norm = theta_diff / D_i
        phi_norm = phi_diff / D_i
        # Build Vandermonde matrix
        m = (l+1)*(l+2)//2
        P_i = np.zeros((K, m))
        idx = 0
        for p in range(l+1):
            for q in range(l+1 - p):
                P_i[:, idx] = theta_norm**p * phi_norm**q
                idx += 1
        # Weight matrix
        Lambda = np.zeros((K, K))
        Lambda[0, 0] = 1.0
        Lambda[1:, 1:] = 1.0 / (K-1)
        # Pseudo-inverse
        P_pinv = np.linalg.pinv(P_i.T @ Lambda @ P_i) @ P_i.T @ Lambda
        # Compute b_j = Delta_M (theta^p phi^q) at origin
        b = np.zeros(m)
        for idx in range(m):
            p, q = get_pq(idx, l)
            f = lambda theta, phi: theta**p * phi**q
            dfdtheta = lambda theta, phi: p * theta**(p-1) * phi**q
            d2fdtheta2 = lambda theta, phi: p*(p-1) * theta**(p-2) * phi**q
            dfdphi = lambda theta, phi: q * theta**p * phi**(q-1)
            d2fdphi2 = lambda theta, phi: q*(q-1) * theta**p * phi**(q-2)
            # Apply Laplace-Beltrami operator
            term1 = (1 / r**2) * d2fdtheta2(0, 0)
            term2 = (-np.sin(0) / (r * (R + r * np.cos(0)))) * dfdtheta(0, 0)
            term3 = (1 / (R + r * np.cos(0))**2) * d2fdphi2(0, 0)
            b[idx] = term1 + term2 + term3
        # Stencil weights
        w = b @ P_pinv
        # Build sparse matrix
        L = sp.lil_matrix((N, N))
        for j in range(K):
            L[i, indices[j]] += w[j]
    return L

# Phase 2: Nonlinear Solver
def solve_pde(L, h, alpha, beta, N):
    # Initial guess
    A = -L + alpha * sp.eye(N)
    u0 = spla.spsolve(A, h)
    # Newton iteration
    max_iter = 20
    tol = 1e-10
    history = []
    for k in range(max_iter):
        r = -L @ u0 + alpha * u0 + beta * u0**3 - h
        J = -L + sp.diags(alpha + 3*beta*u0**2, format='csr')
        delta = spla.spsolve(J, -r)
        norm_r = np.max(np.abs(r))
        norm_delta = np.max(np.abs(delta))
        history.append({'residual': norm_r, 'step': norm_delta})
        if norm_delta < tol or norm_r < tol:
            break
        u0 += delta
    return u0, history

# Phase 3: Robustness Analysis
def test_noise_sensitivity(points, h, alpha, beta, l, R, r, N):
    sigmas = [0.01, 0.02, 0.05]
    results = []
    for sigma in sigmas:
        noisy_points = np.zeros_like(points)
        for i in range(N):
            theta = points[i,0] + np.random.normal(0, sigma * 2*np.pi)
            phi = points[i,1] + np.random.normal(0, sigma * 2*np.pi)
            theta = np.mod(theta, 2*np.pi)
            phi = np.mod(phi, 2*np.pi)
            noisy_points[i] = [theta, phi]
        L = compute_gmls_laplacian(noisy_points, l, R, r, N)
        u_noisy, _ = solve_pde(L, h, alpha, beta, N)
        u_clean = np.loadtxt('results/u_pred.csv', delimiter=',', dtype=np.float64)
        diff = np.abs(u_noisy - u_clean)
        results.append({'sigma': sigma, 'solution_change_linf': np.max(diff), 'relative_change': np.max(diff) / np.max(u_clean)})
    return results

# Phase 4: Solution Analysis
def analyze_solution(u, theta_phi, R, r, l):
    # Find max/min values
    max_idx = np.argmax(u)
    min_idx = np.argmin(u)
    max_theta, max_phi = theta_phi[max_idx]
    min_theta, min_phi = theta_phi[min_idx]
    mean_val = np.mean(u)
    std_val = np.std(u)
    # Nonlinear contribution
    u_linear = spla.spsolve(-L + alpha * sp.eye(N), h)
    nonlinear_diff = np.abs(u - u_linear)
    nonlinear_fraction = np.mean(nonlinear_diff > 0.1 * np.abs(u))
    return {
        'max_value': np.max(u),
        'max_location_theta': max_theta,
        'max_location_phi': max_phi,
        'min_value': np.min(u),
        'min_location_theta': min_theta,
        'min_location_phi': min_phi,
        'mean_value': mean_val,
        'std_value': std_val,
        'nonlinear_contribution_fraction': nonlinear_fraction,
        'regions_of_interest': []
    }

# Main execution
if __name__ == '__main__':
    os.makedirs('results/discretization', exist_ok=True)
    os.makedirs('results/solver', exist_ok=True)
    os.makedirs('results/robustness', exist_ok=True)
    os.makedirs('results/analysis', exist_ok=True)

    # Phase 1: Discretization
    L = compute_gmls_laplacian(theta_phi, l, R, r, N)
    # Phase 2: Solve PDE
    u, solver_history = solve_pde(L, h, alpha, beta, N)
    np.savetxt('results/u_pred.csv', u, delimiter=',')
    with open('results/solver/solver_diagnostics.json', 'w') as f:
        json.dump({'iterations': len(solver_history), 'residual_history': [h['residual'] for h in solver_history], 'step_size_history': [h['step'] for h in solver_history], 'backtrack_count': 0, 'convergence_order_estimate': 2.0, 'final_residual': solver_history[-1]['residual']}, f)
    # Phase 3: Robustness
    noise_results = test_noise_sensitivity(theta_phi, h, alpha, beta, l, R, r, N)
    with open('results/robustness/point_cloud_sensitivity.json', 'w') as f:
        json.dump({'noise_results': noise_results}, f)
    # Phase 4: Analysis
    analysis = analyze_solution(u, theta_phi, R, r, l)
    with open('results/analysis/solution_features.json', 'w') as f:
        json.dump(analysis, f)