```bash
#!/bin/bash
# Robust SE(2) pose-graph optimization with false loop closure rejection
# This script accepts candidate loop closures as measurement constraints
# but rejects those that are false perceptual aliases.
# Output: optimized poses, rejected loop candidates, map diagnostics plot

# Load input data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

task_info = json.load(open('task_info.json'))
pose_shape = task_info['pose_shape']
N = pose_shape[0]
E = task_info['num_edges']
L = task_info['num_loop_candidates']

# Load initial poses and edge measurements
poses_initial = np.load('data/poses_initial.npy')
odometry_edges = pd.read_csv('data/odometry_edges.csv', index_col='edge_id')
loop_candidates = pd.read_csv('data/loop_candidates.csv', index_col='loop_id')

# Build SE(2) optimization problem
from scipy.optimize import minimize
from scipy.stats import norm

def build_cost_function(poses, odometry_edges, loop_candidates, task_info):
    def cost(x, loop_mask=None):
        # x: flattened vector of pose deltas
        # loop_mask: boolean array of accepted loop constraints
        x = x.reshape(N, 3)
        cost = 0.0
        for edge_id, row in odometry_edges.iterrows():
            from_idx = row['from']
            to_idx = row['to']
            dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
            dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
            dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
            cost += 0.5 * np.sum((np.array([dx, dy, dtheta])**2) * np.array([row['sigma_trans'], row['sigma_rot'], row['sigma_rot']]))
        if loop_mask is not None:
            for loop_id, row in loop_candidates.iterrows():
                if loop_mask[loop_id]:
                    from_idx = row['from']
                    to_idx = row['to']
                    dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
                    dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
                    dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
                    cost += 0.5 * np.sum((np.array([dx, dy, dtheta])**2) * np.array([row['sigma_trans'], row['sigma_rot'], row['sigma_rot']]))
        return cost
    return cost

def build_jacobian(poses, odometry_edges, loop_candidates, task_info):
    def jacobian(x, loop_mask=None):
        x = x.reshape(N, 3)
        J = np.zeros((E, N*3))
        for edge_id, row in odometry_edges.iterrows():
            from_idx = row['from']
            to_idx = row['to']
            dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
            dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
            dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
            J[edge_id, from_idx*3:from_idx*3+3] = np.array([dx, dy, dtheta])
            J[edge_id, to_idx*3:to_idx*3+3] = -np.array([dx, dy, dtheta])
        if loop_mask is not None:
            for loop_id, row in loop_candidates.iterrows():
                if loop_mask[loop_id]:
                    from_idx = row['from']
                    to_idx = row['to']
                    dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
                    dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
                    dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
                    J[loop_id, from_idx*3:from_idx*3+3] = np.array([dx, dy, dtheta])
                    J[loop_id, to_idx*3:to_idx*3+3] = -np.array([dx, dy, dtheta])
        return J
    return jacobian

def build_hessian(poses, odometry_edges, loop_candidates, task_info):
    def hessian(x, loop_mask=None):
        x = x.reshape(N, 3)
        H = np.zeros((N*3, N*3))
        for edge_id, row in odometry_edges.iterrows():
            from_idx = row['from']
            to_idx = row['to']
            dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
            dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
            dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
            H[from_idx*3:from_idx*3+3, from_idx*3:from_idx*3+3] = np.array([[0, -dx, -dy], [dx, 0, -dtheta], [dy, dtheta, 0]])
            H[from_idx*3:from_idx*3+3, to_idx*3:to_idx*3+3] = np.array([[0, dx, dy], [-dx, 0, dtheta], [-dy, -dtheta, 0]])
            H[to_idx*3:to_idx*3+3, from_idx*3:from_idx*3+3] = np.array([[0, dx, dy], [-dx, 0, dtheta], [-dy, -dtheta, 0]])
            H[to_idx*3:to_idx*3+3, to_idx*3:to_idx*3+3] = np.array([[0, -dx, -dy], [dx, 0, -dtheta], [dy, dtheta, 0]])
        if loop_mask is not None:
            for loop_id, row in loop_candidates.iterrows():
                if loop_mask[loop_id]:
                    from_idx = row['from']
                    to_idx = row['to']
                    dx = row['dx'] + x[to_idx, 0] - x[from_idx, 0]
                    dy = row['dy'] + x[to_idx, 1] - x[from_idx, 1]
                    dtheta = row['dtheta'] + x[to_idx, 2] - x[from_idx, 2]
                    H[from_idx*3:from_idx*3+3, from_idx*3:from_idx*3+3] += np.array([[0, -dx, -dy], [dx, 0, -dtheta], [dy, dtheta, 0]])
                    H[from_idx*3:from_idx*3+3, to_idx*3:to_idx*3+3] += np.array([[0, dx, dy], [-dx, 0, dtheta], [-dy, -dtheta, 0]])
                    H[to_idx*3:to_idx*3+3, from_idx*3:from_idx*3+3] += np.array([[0, dx, dy], [-dx, 0, dtheta], [-dy, -dtheta, 0]])
                    H[to_idx*3:to_idx*3+3, to_idx*3:to_idx*3+3] += np.array([[0, -dx, -dy], [dx, 0, -dtheta], [dy, dtheta, 0]])
        return H
    return hessian

# Solve with Levenberg-Marquardt algorithm
x_initial = poses_initial.flatten()
x_opt, info = minimize(build_cost_function(poses_initial, odometry_edges, loop_candidates, task_info), x_initial, jac=build_jacobian(poses_initial, odometry_edges, loop_candidates, task_info), hess=build_hessian(poses_initial, odometry_edges, loop_candidates, task_info), method='L-BFGS-B')

# Output optimized poses
poses_opt = x_opt.reshape(N, 3)
np.save('poses.npy', poses_opt)

# Output loop closure scores
loop_scores = pd.DataFrame(0.0, index=loop_candidates.index, columns=['false_probability'])
for loop_id, row in loop_candidates.iterrows():
    from_idx = row['from']
    to_idx = row['to']
    dx = row['dx'] + poses_opt[to_idx, 0] - poses_opt[from_idx, 0]
    dy = row['dy'] + poses_opt[to_idx, 1] - poses_opt[from_idx, 1]
    dtheta = row['dtheta'] + poses_opt[to_idx, 2] - poses_opt[from_idx, 2]
    residual = np.sqrt(dx**2 + dy**2) + np.abs(dtheta)
    false_prob = 1.0 - norm.cdf(residual, loc=0.0, scale=np.sqrt(1.0))
    loop_scores.loc[loop_id, 'false_probability'] = false_prob
loop_scores.to_csv('loop_closure_scores.csv', index=False)

# Output metrics
with open('metrics.json', 'w') as f:
    json.dump({'num_loop_candidates': L, 'num_loop_constraints': loop_scores['false_probability'].lt(0.5).sum(), 'num_loop_rejected': loop_scores['false_probability'].gt(0.5).sum()}, f)

# Plot map diagnostics
plt.figure(figsize=(12, 8))
plt.plot(poses_initial[:, 0], poses_initial[:, 1], 'b-', alpha=0.5, label='Initial')
plt.plot(poses_opt[:, 0], poses_opt[:, 1], 'r-', alpha=0.5, label='Optimized')
plt.plot(poses_opt[loop_candidates['from'].astype(int), 0], poses_opt[loop_candidates['from'].astype(int), 1], 'ko', alpha=0.5, label='Loops')
plt.plot(poses_opt[loop_candidates['to'].astype(int), 0], poses_opt[loop_candidates['to'].astype(int), 1], 'ko', alpha=0.5)
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.savefig('map_diagnostics.png')
```