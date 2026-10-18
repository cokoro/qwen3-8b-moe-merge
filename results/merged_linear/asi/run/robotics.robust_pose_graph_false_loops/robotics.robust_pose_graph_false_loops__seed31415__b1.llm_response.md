```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import json
from scipy import stats

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def build_graph(odometry_edges, loop_candidates):
    N = len(odometry_edges)
    M = len(loop_candidates)
    graph = {
        'edges': [{} for _ in range(N)],
        'loop_candidates': [{} for _ in range(M)]
    }
    for i, edge in enumerate(odometry_edges):
        from_id, to_id, dx, dy, dtheta, sigma_trans, sigma_rot = edge
        sigma = np.diag([sigma_trans**2, sigma_rot**2])
        graph['edges'][i] = {
            'from': from_id,
            'to': to_id,
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
            'sigma': sigma
        }
    for i, loop in enumerate(loop_candidates):
        loop_id, from_id, to_id, dx, dy, dtheta, sigma_trans, sigma_rot = loop
        sigma = np.diag([sigma_trans**2, sigma_rot**2])
        graph['loop_candidates'][i] = {
            'loop_id': loop_id,
            'from': from_id,
            'to': to_id,
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
            'sigma': sigma
        }
    return graph

def compute_odometry_residuals(poses, graph_edges):
    N = len(poses)
    residuals = np.zeros(len(graph_edges))
    for i, edge in enumerate(graph_edges):
        from_id, to_id, dx, dy, dtheta, sigma_trans, sigma_rot = edge
        x1, y1, theta1 = poses[from_id]
        x2, y2, theta2 = poses[to_id]
        dx_pred = x2 - x1
        dy_pred = y2 - y1
        dtheta_pred = wrap_angle(theta2 - theta1)
        dx_true = dx
        dy_true = dy
        dtheta_true = dtheta
        dx_res = dx_pred - dx_true
        dy_res = dy_pred - dy_true
        dtheta_res = wrap_angle(dtheta_pred - dtheta_true)
        residual = np.array([dx_res, dy_res, dtheta_res])
        residuals[i] = np.sqrt(residual.dot(np.linalg.inv(sigma_trans**2 + sigma_rot**2)) @ residual)
    return residuals

def compute_loop_residuals(poses, graph_loops):
    M = len(poses)
    residuals = np.zeros(len(graph_loops))
    for i, loop in enumerate(graph_loops):
        loop_id, from_id, to_id, dx, dy, dtheta, sigma_trans, sigma_rot = loop
        x1, y1, theta1 = poses[from_id]
        x2, y2, theta2 = poses[to_id]
        dx_pred = x2 - x1
        dy_pred = y2 - y1
        dtheta_pred = wrap_angle(theta2 - theta1)
        dx_true = dx
        dy_true = dy
        dtheta_true = dtheta
        dx_res = dx_pred - dx_true
        dy_res = dy_pred - dy_true
        dtheta_res = wrap_angle(dtheta_pred - dtheta_true)
        residual = np.array([dx_res, dy_res, dtheta_res])
        residuals[i] = np.sqrt(residual.dot(np.linalg.inv(sigma_trans**2 + sigma_rot**2)) @ residual)
    return residuals

def optimize_graph(poses, graph_edges, graph_loops, max_iterations=100):
    def objective_function(x):
        poses_opt = x.reshape(-1, 3)
        residuals_odom = compute_odometry_residuals(poses_opt, graph_edges)
        residuals_loop = compute_loop_residuals(poses_opt, graph_loops)
        return np.sum(residuals_odom**2 + residuals_loop**2)

    x0 = poses.reshape(-1)
    result = minimize(objective_function, x0, method='L-BFGS-B')
    return result.x.reshape(-1, 3)

def evaluate_loop_candidates(poses_opt, graph_loops):
    M = len(poses_opt)
    residuals = np.zeros(len(graph_loops))
    for i, loop in enumerate(graph_loops):
        loop_id, from_id, to_id, dx, dy, dtheta, sigma_trans, sigma_rot = loop
        x1, y1, theta1 = poses_opt[from_id]
        x2, y2, theta2 = poses_opt[to_id]
        dx_pred = x2 - x1
        dy_pred = y2 - y1
        dtheta_pred = wrap_angle(theta2 - theta1)
        dx_true = dx
        dy_true = dy
        dtheta_true = dtheta
        dx_res = dx_pred - dx_true
        dy_res = dy_pred - dy_true
        dtheta_res = wrap_angle(dtheta_pred - dtheta_true)
        residual = np.array([dx_res, dy_res, dtheta_res])
        residuals[i] = np.sqrt(residual.dot(np.linalg.inv(sigma_trans**2 + sigma_rot**2)) @ residual)
    return residuals

def compute_false_probability(residuals):
    return 1.0 - stats.norm.cdf(residuals)

def plot_trajectory(poses, title='Trajectory'):
    plt.figure(figsize=(10, 8))
    x = poses[:, 0]
    y = poses[:, 1]
    plt.plot(x, y, 'o-', markersize=2, linewidth=0.5)
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def save_results(poses, loop_closure_scores, metrics, output_dir):
    np.save(f'{output_dir}/poses.npy', poses)
    loop_closure_scores.to_csv(f'{output_dir}/loop_closure_scores.csv', index=False)
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f)
    plt.savefig(f'{output_dir}/map_diagnostics.png')

def main():
    data_dir = 'data'
    output_dir = 'output'
    
    poses_initial = np.load(f'{data_dir}/poses_initial.npy')
    odometry_edges = pd.read_csv(f'{data_dir}/odometry_edges.csv').values
    loop_candidates = pd.read_csv(f'{data_dir}/loop_candidates.csv').values
    task_info = json.load(open(f'{data_dir}/task_info.json'))
    
    graph = build_graph(odometry_edges, loop_candidates)
    
    # Initial optimization
    poses_opt = optimize_graph(poses_initial, graph['edges'], graph['loop_candidates'])
    
    # Evaluate loop candidates
    loop_residuals = evaluate_loop_candidates(poses_opt, graph['loop_candidates'])
    false_probabilities = compute_false_probability(loop_residuals)
    loop_closure_scores = pd.DataFrame({
        'loop_id': loop_candidates[:, 0],
        'false_probability': false_probabilities
    })
    
    # Plot results
    plot_trajectory(poses_initial, 'Initial Trajectory')
    plot_trajectory(poses_opt, 'Optimized Trajectory')
    
    # Save results
    save_results(poses_opt, loop_closure_scores, task_info, output_dir)

if __name__ == '__main__':
    main()
```