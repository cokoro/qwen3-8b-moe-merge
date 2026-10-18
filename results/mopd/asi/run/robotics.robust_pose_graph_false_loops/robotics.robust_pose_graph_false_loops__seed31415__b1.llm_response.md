```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json

# Load data
poses_initial = np.load('data/poses_initial.npy')
odometry_edges = pd.read_csv('data/odometry_edges.csv')
loop_candidates = pd.read_csv('data/loop_candidates.csv')
task_info = json.load(open('data/task_info.json'))

# Parse task info
N = task_info['num_poses']
K = task_info['num_loop_candidates']
num_odom_edges = task_info['num_odom_edges']
num_loop_candidates = task_info['num_loop_candidates']
pose_dim = task_info['pose_dim']
edge_dim = task_info['edge_dim']
output_file = task_info['output_file']
loop_file = task_info['loop_file']
metrics_file = task_info['metrics_file']
plot_file = task_info['plot_file']

# Set up graph
num_edges = num_odom_edges + num_loop_candidates
num_vars = N * pose_dim
x = poses_initial.flatten()
residuals = np.zeros(num_edges)
J = np.zeros((num_edges, num_vars))

# Define cost function
def cost_function(x, odometry_edges, loop_candidates):
    # Compute residuals
    for i, row in odometry_edges.iterrows():
        from_idx = row['from']
        to_idx = row['to']
        dx = row['dx']
        dy = row['dy']
        dtheta = row['dtheta']
        sigma_trans = row['sigma_trans']
        sigma_rot = row['sigma_rot']
        
        x1 = x[from_idx * pose_dim : (from_idx + 1) * pose_dim]
        x2 = x[to_idx * pose_dim : (to_idx + 1) * pose_dim]
        
        dx_est = x2[0] - x1[0]
        dy_est = x2[1] - x1[1]
        dtheta_est = x2[2] - x1[2]
        
        # Wrap angle to [-pi, pi)
        dtheta_est = (dtheta_est + np.pi) % (2 * np.pi) - np.pi
        
        # Compute residual
        residual = np.array([dx_est - dx, dy_est - dy, dtheta_est - dtheta])
        residual /= np.array([sigma_trans, sigma_rot, sigma_rot])
        
        # Add to residuals
        residuals[i] = residual[0] ** 2 + residual[1] ** 2 + residual[2] ** 2
        
        # Build Jacobian
        J[i, from_idx * pose_dim : (from_idx + 1) * pose_dim] = -np.array([1, 0, 0, 0, 1, 0])
        J[i, to_idx * pose_dim : (to_idx + 1) * pose_dim] = np.array([1, 0, 0, 0, 1, 0])
        
    for i, row in loop_candidates.iterrows():
        from_idx = row['from']
        to_idx = row['to']
        dx = row['dx']
        dy = row['dy']
        dtheta = row['dtheta']
        sigma_trans = row['sigma_trans']
        sigma_rot = row['sigma_rot']
        
        x1 = x[from_idx * pose_dim : (from_idx + 1) * pose_dim]
        x2 = x[to_idx * pose_dim : (to_idx + 1) * pose_dim]
        
        dx_est = x2[0] - x1[0]
        dy_est = x2[1] - x1[1]
        dtheta_est = x2[2] - x1[2]
        
        # Wrap angle to [-pi, pi)
        dtheta_est = (dtheta_est + np.pi) % (2 * np.pi) - np.pi
        
        # Compute residual
        residual = np.array([dx_est - dx, dy_est - dy, dtheta_est - dtheta])
        residual /= np.array([sigma_trans, sigma_rot, sigma_rot])
        
        # Add to residuals
        residuals[i + num_odom_edges] = residual[0] ** 2 + residual[1] ** 2 + residual[2] ** 2
        
        # Build Jacobian
        J[i + num_odom_edges, from_idx * pose_dim : (from_idx + 1) * pose_dim] = -np.array([1, 0, 0, 0, 1, 0])
        J[i + num_odom_edges, to_idx * pose_dim : (to_idx + 1) * pose_dim] = np.array([1, 0, 0, 0, 1, 0])
    
    return residuals

# Optimization
res = minimize(cost_function, x, args=(odometry_edges, loop_candidates), method='L-BFGS-B', jac=True)
x_opt = res.x
poses_opt = x_opt.reshape(N, pose_dim)

# Evaluate loop candidates
loop_scores = np.zeros(K)
for i, row in loop_candidates.iterrows():
    from_idx = row['from']
    to_idx = row['to']
    dx = row['dx']
    dy = row['dy']
    dtheta = row['dtheta']
    sigma_trans = row['sigma_trans']
    sigma_rot = row['sigma_rot']
    
    x1 = x_opt[from_idx * pose_dim : (from_idx + 1) * pose_dim]
    x2 = x_opt[to_idx * pose_dim : (to_idx + 1) * pose_dim]
    
    dx_est = x2[0] - x1[0]
    dy_est = x2[1] - x1[1]
    dtheta_est = x2[2] - x1[2]
    
    # Wrap angle to [-pi, pi)
    dtheta_est = (dtheta_est + np.pi) % (2 * np.pi) - np.pi
    
    # Compute residual
    residual = np.array([dx_est - dx, dy_est - dy, dtheta_est - dtheta])
    residual /= np.array([sigma_trans, sigma_rot, sigma_rot])
    
    # Compute score
    score = np.sqrt(residual[0] ** 2 + residual[1] ** 2 + residual[2] ** 2)
    loop_scores[i] = score

# Save outputs
np.save('poses.npy', poses_opt)
pd.DataFrame({'loop_id': loop_candidates['loop_id'], 'false_probability': loop_scores}).to_csv('loop_closure_scores.csv', index=False)
metrics = {'num_loop_candidates': num_loop_candidates, 'num_odom_edges': num_odom_edges, 'num_poses': N, 'pose_dim': pose_dim}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(poses_initial[:, 0], poses_initial[:, 1], 'r-', label='Initial')
plt.plot(poses_opt[:, 0], poses_opt[:, 1], 'b-', label='Optimized')
plt.legend()
plt.savefig('map_diagnostics.png')
```