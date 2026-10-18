```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json
import pandas as pd

# Load task info
task_info = json.load(open("data/task_info.json"))

# Load poses
poses_initial = np.load("data/poses_initial.npy")
N = poses_initial.shape[0]

# Load edge info
odometry_edges = pd.read_csv("data/odometry_edges.csv")
odometry_edges = odometry_edges.to_numpy()

# Load loop info
loop_candidates = pd.read_csv("data/loop_candidates.csv")
loop_candidates = loop_candidates.to_numpy()

# Build SE(2) factor graph
def build_factor_graph(poses, edges, loops):
    # Build factor graph with SE(2) edges
    # Returns: factor graph as list of factors, each factor is a tuple (type, index, data)
    factors = []
    for edge in edges:
        edge_id, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = edge
        if sigma_trans < 0:
            sigma_trans = 1e-6
        if sigma_rot < 0:
            sigma_rot = 1e-6
        # Build edge factor
        factors.append(("odometry", from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot))
    for loop in loops:
        loop_id, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = loop
        if sigma_trans < 0:
            sigma_trans = 1e-6
        if sigma_rot < 0:
            sigma_rot = 1e-6
        # Build loop closure factor
        factors.append(("loop", from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot))
    return factors

# Build initial factor graph
factors = build_factor_graph(poses_initial, odometry_edges, loop_candidates)

# Optimize factor graph
def optimize_factor_graph(factors, poses_initial):
    # Build initial poses
    poses = poses_initial.copy()
    # Build factor graph as list of factors
    factors = build_factor_graph(poses, odometry_edges, loop_candidates)
    # Build optimization objective
    def objective(x):
        # x is a vector of pose parameters
        # x has shape [N, 3]
        # x[0] is x0, x[1] is y0, x[2] is theta0
        # x[3] is x1, x[4] is y1, x[5] is theta1
        # ...
        # x[3*N-3] is xN-1, x[3*N-2] is yN-1, x[3*N-1] is thetaN-1
        # Reshape x to poses
        poses = x.reshape((N, 3))
        # Build objective
        objective = 0.0
        # Evaluate each factor
        for factor in factors:
            factor_type, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = factor
            # Get poses
            x1, y1, theta1 = poses[from_idx]
            x2, y2, theta2 = poses[to_idx]
            # Compute relative pose
            dx_rel = x2 - x1
            dy_rel = y2 - y1
            dtheta_rel = theta2 - theta1
            # Wrap angle to [-pi, pi)
            dtheta_rel = np.arctan2(np.sin(dtheta_rel), np.cos(dtheta_rel))
            # Compute residual
            if factor_type == "odometry":
                residual = np.array([dx_rel - dx, dy_rel - dy, dtheta_rel - dtheta])
            elif factor_type == "loop":
                residual = np.array([dx_rel - dx, dy_rel - dy, dtheta_rel - dtheta])
            else:
                raise ValueError("Unknown factor type")
            # Compute weight
            weight = np.array([1.0 / sigma_trans, 1.0 / sigma_rot])
            # Compute objective
            objective += np.sum(weight * residual**2)
        return objective
    # Optimize
    result = opt.minimize(objective, poses_initial.reshape(-1), method="L-BFGS-B")
    # Return optimized poses
    return result.x.reshape((N, 3))

# Optimize pose graph
poses_optimized = optimize_factor_graph(factors, poses_initial)

# Compute loop closure scores
def compute_loop_closure_scores(poses_optimized, loop_candidates):
    # Compute loop closure scores
    scores = []
    for loop in loop_candidates:
        loop_id, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = loop
        # Get poses
        x1, y1, theta1 = poses_optimized[from_idx]
        x2, y2, theta2 = poses_optimized[to_idx]
        # Compute relative pose
        dx_rel = x2 - x1
        dy_rel = y2 - y1
        dtheta_rel = theta2 - theta1
        # Wrap angle to [-pi, pi)
        dtheta_rel = np.arctan2(np.sin(dtheta_rel), np.cos(dtheta_rel))
        # Compute residual
        residual = np.array([dx_rel - dx, dy_rel - dy, dtheta_rel - dtheta])
        # Compute score
        score = np.sum(residual**2) / (sigma_trans**2 + sigma_rot**2)
        # Compute false probability
        false_probability = 1.0 / (1.0 + np.exp(-score))
        scores.append([loop_id, false_probability])
    return scores

# Compute loop closure scores
loop_closure_scores = compute_loop_closure_scores(poses_optimized, loop_candidates)

# Save loop closure scores
loop_closure_scores_df = pd.DataFrame(loop_closure_scores, columns=["loop_id", "false_probability"])
loop_closure_scores_df.to_csv("loop_closure_scores.csv", index=False)

# Save optimized poses
np.save("poses.npy", poses_optimized)

# Plot optimized trajectory
plt.figure(figsize=(10, 10))
plt.plot(poses_initial[:, 0], poses_initial[:, 1], "b-", label="Initial trajectory")
plt.plot(poses_optimized[:, 0], poses_optimized[:, 1], "r-", label="Optimized trajectory")
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Initial and optimized trajectories")
plt.show()

# Save map diagnostics
plt.savefig("map_diagnostics.png")
```