import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_error(pose1, pose2, dx, dy, dtheta):
    dx_err = pose1[0] - pose2[0] - dx
    dy_err = pose1[1] - pose2[1] - dy
    dtheta_err = wrap_angle(pose1[2] - pose2[2] - dtheta)
    return np.array([dx_err, dy_err, dtheta_err])

def compute_error_norm(error, sigma):
    return np.linalg.norm(error) / sigma

def robust_loss(residual, alpha=3.0):
    return np.where(np.abs(residual) <= alpha, 0.5 * residual**2, alpha * np.abs(residual) - 0.5 * alpha**2)

def robust_derivative(residual, alpha=3.0):
    return np.where(np.abs(residual) <= alpha, residual, np.sign(residual) * alpha)

def optimize_pose_graph(poses, edges, loop_candidates, task_info):
    num_poses = len(poses)
    num_edges = len(edges)
    num_loops = len(loop_candidates)
    
    # Initialize variables
    poses = np.copy(poses)
    edge_weights = np.ones(num_edges)
    loop_weights = np.ones(num_loops)
    loop_scores = np.zeros(num_loops)
    
    # Optimization loop
    for _ in range(10):
        # Compute residuals and weights
        residuals = []
        for i, edge in enumerate(edges):
            from_idx = edge['from']
            to_idx = edge['to']
            dx = edge['dx']
            dy = edge['dy']
            dtheta = edge['dtheta']
            sigma_trans = edge['sigma_trans']
            sigma_rot = edge['sigma_rot']
            
            error = compute_error(poses[from_idx], poses[to_idx], dx, dy, dtheta)
            norm_error = np.linalg.norm(error)
            residual = norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2)
            weight = edge_weights[i]
            residuals.append((i, residual, weight))
        
        loop_residuals = []
        for i, loop in enumerate(loop_candidates):
            from_idx = loop['from']
            to_idx = loop['to']
            dx = loop['dx']
            dy = loop['dy']
            dtheta = loop['dtheta']
            sigma_trans = loop['sigma_trans']
            sigma_rot = loop['sigma_rot']
            
            error = compute_error(poses[from_idx], poses[to_idx], dx, dy, dtheta)
            norm_error = np.linalg.norm(error)
            residual = norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2)
            weight = loop_weights[i]
            loop_residuals.append((i, residual, weight))
        
        # Update weights based on residuals
        for i, residual, weight in residuals:
            edge_weights[i] = np.clip(weight * np.exp(-residual**2 / 2), 0.01, 10)
        
        for i, residual, weight in loop_residuals:
            loop_weights[i] = np.clip(weight * np.exp(-residual**2 / 2), 0.01, 10)
        
        # Optimize
        def objective(x):
            x = x.reshape(-1, 3)
            error = 0
            for i, edge in enumerate(edges):
                from_idx = edge['from']
                to_idx = edge['to']
                dx = edge['dx']
                dy = edge['dy']
                dtheta = edge['dtheta']
                sigma_trans = edge['sigma_trans']
                sigma_rot = edge['sigma_rot']
                weight = edge_weights[i]
                
                error_pose = compute_error(x[from_idx], x[to_idx], dx, dy, dtheta)
                norm_error = np.linalg.norm(error_pose)
                error += weight * (norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2))**2
            
            for i, loop in enumerate(loop_candidates):
                from_idx = loop['from']
                to_idx = loop['to']
                dx = loop['dx']
                dy = loop['dy']
                dtheta = loop['dtheta']
                sigma_trans = loop['sigma_trans']
                sigma_rot = loop['sigma_rot']
                weight = loop_weights[i]
                
                error_pose = compute_error(x[from_idx], x[to_idx], dx, dy, dtheta)
                norm_error = np.linalg.norm(error_pose)
                error += weight * (norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2))**2
            
            return error
        
        def jacobian(x):
            x = x.reshape(-1, 3)
            jac = np.zeros((num_poses * 3, num_poses * 3))
            for i, edge in enumerate(edges):
                from_idx = edge['from']
                to_idx = edge['to']
                dx = edge['dx']
                dy = edge['dy']
                dtheta = edge['dtheta']
                sigma_trans = edge['sigma_trans']
                sigma_rot = edge['sigma_rot']
                weight = edge_weights[i]
                
                error_pose = compute_error(x[from_idx], x[to_idx], dx, dy, dtheta)
                norm_error = np.linalg.norm(error_pose)
                residual = norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2)
                grad = np.zeros((3, 3))
                
                for j in range(3):
                    dx_j = 0
                    dy_j = 0
                    dtheta_j = 0
                    if j == 0:
                        dx_j = 1
                    elif j == 1:
                        dy_j = 1
                    elif j == 2:
                        dtheta_j = 1
                    
                    error_j = compute_error(x[from_idx], x[to_idx], dx + dx_j, dy + dy_j, dtheta + dtheta_j)
                    norm_error_j = np.linalg.norm(error_j)
                    residual_j = norm_error_j / np.sqrt(sigma_trans**2 + sigma_rot**2)
                    
                    grad[j, 0] = (residual_j - residual) / dx_j
                    grad[j, 1] = (residual_j - residual) / dy_j
                    grad[j, 2] = (residual_j - residual) / dtheta_j
                
                jac[3*from_idx:3*from_idx+3, 3*from_idx:3*from_idx+3] += weight * grad
                jac[3*to_idx:3*to_idx+3, 3*to_idx:3*to_idx+3] -= weight * grad
            
            for i, loop in enumerate(loop_candidates):
                from_idx = loop['from']
                to_idx = loop['to']
                dx = loop['dx']
                dy = loop['dy']
                dtheta = loop['dtheta']
                sigma_trans = loop['sigma_trans']
                sigma_rot = loop['sigma_rot']
                weight = loop_weights[i]
                
                error_pose = compute_error(x[from_idx], x[to_idx], dx, dy, dtheta)
                norm_error = np.linalg.norm(error_pose)
                residual = norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2)
                grad = np.zeros((3, 3))
                
                for j in range(3):
                    dx_j = 0
                    dy_j = 0
                    dtheta_j = 0
                    if j == 0:
                        dx_j = 1
                    elif j == 1:
                        dy_j = 1
                    elif j == 2:
                        dtheta_j = 1
                    
                    error_j = compute_error(x[from_idx], x[to_idx], dx + dx_j, dy + dy_j, dtheta + dtheta_j)
                    norm_error_j = np.linalg.norm(error_j)
                    residual_j = norm_error_j / np.sqrt(sigma_trans**2 + sigma_rot**2)
                    
                    grad[j, 0] = (residual_j - residual) / dx_j
                    grad[j, 1] = (residual_j - residual) / dy_j
                    grad[j, 2] = (residual_j - residual) / dtheta_j
                
                jac[3*from_idx:3*from_idx+3, 3*from_idx:3*from_idx+3] += weight * grad
                jac[3*to_idx:3*to_idx+3, 3*to_idx:3*to_idx+3] -= weight * grad
            
            return jac
        
        x0 = poses.reshape(-1)
        result = minimize(objective, x0, jac=jacobian, method='BFGS', tol=1e-6)
        poses = result.x.reshape(-1, 3)
    
    # Compute loop scores
    for i, loop in enumerate(loop_candidates):
        from_idx = loop['from']
        to_idx = loop['to']
        dx = loop['dx']
        dy = loop['dy']
        dtheta = loop['dtheta']
        sigma_trans = loop['sigma_trans']
        sigma_rot = loop['sigma_rot']
        
        error_pose = compute_error(poses[from_idx], poses[to_idx], dx, dy, dtheta)
        norm_error = np.linalg.norm(error_pose)
        residual = norm_error / np.sqrt(sigma_trans**2 + sigma_rot**2)
        loop_scores[i] = norm.cdf(residual)
    
    # Save outputs
    np.save('poses.npy', poses)
    loop_df = pd.DataFrame({
        'loop_id': [loop['loop_id'] for loop in loop_candidates],
        'false_probability': 1 - loop_scores
    })
    loop_df.to_csv('loop_closure_scores.csv', index=False)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(poses[:, 0], poses[:, 1], 'b-', label='Optimized Trajectory')
    plt.plot(poses_initial[:, 0], poses_initial[:, 1], 'r--', label='Initial Trajectory')
    plt.legend()
    plt.title('Initial vs Optimized Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig('map_diagnostics.png')
    
    # Metrics
    metrics = {
        'num_poses': num_poses,
        'num_edges': num_edges,
        'num_loops': num_loops,
        'loop_scores': loop_scores.tolist(),
        'final_poses': poses.tolist()
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

# Load data
poses_initial = np.load('data/poses_initial.npy')
edges = pd.read_csv('data/odometry_edges.csv').to_dict('records')
loop_candidates = pd.read_csv('data/loop_candidates.csv').to_dict('records')
task_info = json.load(open('data/task_info.json'))

# Run optimization
optimize_pose_graph(poses_initial, edges, loop_candidates, task_info)