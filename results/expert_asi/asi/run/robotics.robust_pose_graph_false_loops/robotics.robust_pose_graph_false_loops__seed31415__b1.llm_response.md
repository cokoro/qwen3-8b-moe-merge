```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def build_graph(poses, edges, loops):
    n = len(poses)
    A = np.zeros((n, n), dtype=np.float64)
    for i, j, dx, dy, dtheta, sigma_trans, sigma_rot in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    for loop_id, i, j, dx, dy, dtheta, sigma_trans, sigma_rot in loops:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A

def cost_function(x, A, edges, loops, weights):
    n = len(x)
    cost = 0.0
    for i, j, dx, dy, dtheta, sigma_trans, sigma_rot in edges:
        if A[i, j] == 0:
            continue
        x_i, y_i, theta_i = x[i]
        x_j, y_j, theta_j = x[j]
        dx_est = x_j - x_i
        dy_est = y_j - y_i
        dtheta_est = wrap_angle(theta_j - theta_i)
        dx_res = dx_est - dx
        dy_res = dy_est - dy
        dtheta_res = wrap_angle(dtheta_est - dtheta)
        cost += (dx_res ** 2) / (sigma_trans ** 2) + (dy_res ** 2) / (sigma_trans ** 2) + (dtheta_res ** 2) / (sigma_rot ** 2)
    for i, j, dx, dy, dtheta, sigma_trans, sigma_rot in loops:
        if A[i, j] == 0:
            continue
        x_i, y_i, theta_i = x[i]
        x_j, y_j, theta_j = x[j]
        dx_est = x_j - x_i
        dy_est = y_j - y_i
        dtheta_est = wrap_angle(theta_j - theta_i)
        dx_res = dx_est - dx
        dy_res = dy_est - dy
        dtheta_res = wrap_angle(dtheta_est - dtheta)
        cost += (dx_res ** 2) / (sigma_trans ** 2) + (dy_res ** 2) / (sigma_trans ** 2) + (dtheta_res ** 2) / (sigma_rot ** 2)
    return cost

def optimize_poses(poses_initial, A, edges, loops, max_iter=100):
    x = poses_initial.copy()
    for _ in range(max_iter):
        def objective(x_flat):
            return cost_function(x_flat, A, edges, loops, 1.0)
        result = minimize(objective, x.flatten(), method='L-BFGS-B', jac=True)
        x = result.x.reshape((len(x), 3))
        if np.max(np.abs(x - poses_initial)) < 1e-5:
            break
    return x

def analyze_loops(loops, optimized_poses, edges):
    scores = []
    for loop_id, i, j, dx, dy, dtheta, sigma_trans, sigma_rot in loops:
        x_i, y_i, theta_i = optimized_poses[i]
        x_j, y_j, theta_j = optimized_poses[j]
        dx_est = x_j - x_i
        dy_est = y_j - y_i
        dtheta_est = wrap_angle(theta_j - theta_i)
        dx_res = dx_est - dx
        dy_res = dy_est - dy
        dtheta_res = wrap_angle(dtheta_est - dtheta)
        score = np.sqrt(dx_res ** 2 + dy_res ** 2 + dtheta_res ** 2)
        scores.append({
            'loop_id': loop_id,
            'score': score,
            'false_probability': 0.0
        })
    return scores

def plot_trajectories(poses_initial, optimized_poses, loop_candidates, loop_scores):
    plt.figure(figsize=(10, 10))
    for i in range(len(poses_initial)):
        plt.plot(poses_initial[i, 0], poses_initial[i, 1], 'r.')
    for i in range(len(optimized_poses)):
        plt.plot(optimized_poses[i, 0], optimized_poses[i, 1], 'b.')
    for loop_id, i, j, dx, dy, dtheta, sigma_trans, sigma_rot in loop_candidates:
        score = loop_scores[loop_id]['score']
        if score < 0.1:
            plt.plot([optimized_poses[i, 0], optimized_poses[j, 0]], [optimized_poses[i, 1], optimized_poses[j, 1]], 'g-')
        else:
            plt.plot([optimized_poses[i, 0], optimized_poses[j, 0]], [optimized_poses[i, 1], optimized_poses[j, 1]], 'r--')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Initial vs Optimized Trajectory with Loop Candidates')
    plt.show()

def main():
    poses_initial = np.load('data/poses_initial.npy')
    edges = pd.read_csv('data/odometry_edges.csv').values
    loops = pd.read_csv('data/loop_candidates.csv').values
    A = build_graph(poses_initial, edges, loops)
    optimized_poses = optimize_poses(poses_initial, A, edges, loops)
    loop_scores = analyze_loops(loops, optimized_poses, edges)
    plot_trajectories(poses_initial, optimized_poses, loops, loop_scores)
    pd.DataFrame(loop_scores).to_csv('loop_closure_scores.csv', index=False)
    np.save('poses.npy', optimized_poses)

if __name__ == '__main__':
    main()
```