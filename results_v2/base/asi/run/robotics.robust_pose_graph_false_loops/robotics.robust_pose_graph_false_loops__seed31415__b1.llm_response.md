```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize
from scipy.spatial

# Load data
poses_initial = np.load('data/poses_initial.npy')
odometry_edges = pd.read_csv('data/odometry_edges.csv')
loop_candidates = pd.read_csv('data/loop_candidates.csv')
task_info = json.load(open('data/task_info.json'))

# Constants
N = task_info['num_poses']
num_loops = task_info['num_loop_candidates']
num_odom_edges = task_info['num_odom_edges']

# Define SE(2) transformation functions
def transform(x, y, theta):
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

def apply_transform(pose, T):
    x, y, theta = pose
    tx, ty, ttheta = T
    new_x = x * np.cos(ttheta) - y * np.sin(ttheta) + tx
    new_y = x * np.sin(ttheta) + y * np.cos(ttheta) + ty
    new_theta = theta + ttheta
    return np.array([new_x, new_y, new_theta % (2 * np.pi)])

def wrap_angle(angle):
    return angle % (2 * np.pi)

def calc_residuals(poses, edges, loop_candidates, loop_weights):
    residuals = []
    for _, row in edges.iterrows():
        from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = row
        pose_from = poses[from_idx]
        pose_to = poses[to_idx]
        T = apply_transform(pose_to, np.array([dx, dy, dtheta]))
        T_est = apply_transform(pose_to, np.array([0, 0, 0]))
        T_est = transform(pose_from[0], pose_from[1], pose_from[2]) @ T_est
        T_est = np.linalg.inv(transform(pose_from[0], pose_from[1], pose_from[2])) @ T_est
        dx_est = T_est[0, 2]
        dy_est = T_est[1, 2]
        dtheta_est = T_est[2, 2]
        dx_res = dx_est - dx
        dy_res = dy_est - dy
        dtheta_res = dtheta_est - dtheta
        dx_res = wrap_angle(dx_res)
        dy_res = wrap_angle(dy_res)
        dtheta_res = wrap_angle(dtheta_res)
        residual = np.array([dx_res, dy_res, dtheta_res])
        residuals.append(residual)
    
    loop_residuals = []
    for loop_id, _, _, dx, dy, dtheta, sigma_trans, sigma_rot in loop_candidates.values:
        from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = loop_id, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot
        pose_from = poses[from_idx]
        pose_to = poses[to_idx]
        T = apply_transform(pose_to, np.array([dx, dy, dtheta]))
        T_est = apply_transform(pose_to, np.array([0, 0, 0]))
        T_est = transform(pose_from[0], pose_from[1], pose_from[2]) @ T_est
        T_est = np.linalg.inv(transform(pose_from[0], pose_from[1], pose_from[2])) @ T_est
        dx_est = T_est[0, 2]
        dy_est = T_est[1, 2]
        dtheta_est = T_est[2, 2]
        dx_res = dx_est - dx
        dy_res = dy_est - dy
        dtheta_res = dtheta_est - dtheta
        dx_res = wrap_angle(dx_res)
        dy_res = wrap_angle(dy_res)
        dtheta_res = wrap_angle(dtheta_res)
        residual = np.array([dx_res, dy_res, dtheta_res])
        loop_residuals.append(residual)
    
    return np.array(residuals), np.array(loop_residuals)

def calc_chi_square(residuals, sigma):
    return np.sum((residuals ** 2) / (sigma ** 2))

def optimize_pose_graph(poses, edges, loop_candidates, loop_weights):
    def objective(poses_flat):
        poses = poses_flat.reshape((N, 3))
        residuals, loop_residuals = calc_residuals(poses, edges, loop_candidates, loop_weights)
        chi_square_odom = calc_chi_square(residuals, edges[['sigma_trans', 'sigma_rot']].values.T)
        chi_square_loops = calc_chi_square(loop_residuals, loop_candidates[['sigma_trans', 'sigma_rot']].values.T)
        return chi_square_odom + chi_square_loops
    
    result = minimize(objective, poses_initial.flatten(), method='L-BFGS-B')
    optimized_poses = result.x.reshape((N, 3))
    return optimized_poses

# Initial optimization with all loops
loop_weights = np.ones(num_loops)
optimized_poses = optimize_pose_graph(poses_initial, odometry_edges, loop_candidates, loop_weights)

# Loop closure score calculation
residuals, loop_residuals = calc_residuals(optimized_poses, odometry_edges, loop_candidates, loop_weights)
chi_square_loops = calc_chi_square(loop_residuals, loop_candidates[['sigma_trans', 'sigma_rot']].values.T)
loop_scores = chi_square_loops / np.sum(chi_square_loops)
loop_false_prob = 1 - loop_scores
loop_false_prob[loop_false_prob < 0.5] = 0
loop_false_prob[loop_false_prob > 0.5] = 1

# Save outputs
np.save('poses.npy', optimized_poses)
loop_candidates['false_probability'] = loop_false_prob
loop_candidates.to_csv('loop_closure_scores.csv', index=False)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(poses_initial[:, 0], poses_initial[:, 1], 'b-', label='Initial')
plt.plot(optimized_poses[:, 0], optimized_poses[:, 1], 'r-', label='Optimized')
plt.legend()
plt.title('Initial vs Optimized Trajectories')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.savefig('map_diagnostics.png')

# Save metrics
metrics = {
    'num_poses': N,
    'num_odom_edges': num_odom_edges,
    'num_loop_candidates': num_loops,
    'loop_weights': loop_weights.tolist(),
    'chi_square_odom': calc_chi_square(residuals, edges[['sigma_trans', 'sigma_rot']].values.T),
    'chi_square_loops': chi_square_loops
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save analysis script
with open('analysis.py', 'w') as f:
    f.write('import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport json\nfrom scipy.optimize import minimize\nfrom scipy.linalg import expm, logm\n\n# Load data\nposes_initial = np.load(\'data/poses_initial.npy\')\nodometry_edges = pd.read_csv(\'data/odometry_edges.csv\')\nloop_candidates = pd.read_csv(\'data/loop_candidates.csv\')\ntask_info = json.load(open(\'data/task_info.json\'))\n\n# Constants\nN = task_info[\'num_poses\']\nnum_loops = task_info[\'num_loop_candidates\']\nnum_odom_edges = task_info[\'num_odom_edges\']\n\n# Define SE(2) transformation functions\ndef transform(x, y, theta):\n    return np.array([[np.cos(theta), -np.sin(theta), x],\n                     [np.sin(theta), np.cos(theta), y],\n                     [0, 0, 1]])\n\ndef apply_transform(pose, T):\n    x, y, theta = pose\n    tx, ty, ttheta = T\n    new_x = x * np.cos(ttheta) - y * np.sin(ttheta) + tx\n    new_y = x * np.sin(ttheta) + y * np.cos(ttheta) + ty\n    new_theta = theta + ttheta\n    return np.array([new_x, new_y, new_theta % (2 * np.pi)])\n\ndef wrap_angle(angle):\n    return angle % (2 * np.pi)\n\ndef calc_residuals(poses, edges, loop_candidates, loop_weights):\n    residuals = []\n    for _, row in edges.iterrows():\n        from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = row\n        pose_from = poses[from_idx]\n        pose_to = poses[to_idx]\n        T = apply_transform(pose_to, np.array([dx, dy, dtheta]))\n        T_est = apply_transform(pose_to, np.array([0, 0, 0]))\n        T_est = transform(pose_from[0], pose_from[1], pose_from[2]) @ T_est\n        T_est = np.linalg.inv(transform(pose_from[0], pose_from[1], pose_from[2])) @ T_est\n        dx_est = T_est[0, 2]\n        dy_est = T_est[1, 2]\n        dtheta_est = T_est[2, 2]\n        dx_res = dx_est - dx\n        dy_res = dy_est - dy\n        dtheta_res = dtheta_est - dtheta\n        dx_res = wrap_angle(dx_res)\n        dy_res = wrap_angle(dy_res)\n        dtheta_res = wrap_angle(dtheta_res)\n        residual = np.array([dx_res, dy_res, dtheta_res])\n        residuals.append(residual)\n    \n    loop_residuals = []\n    for loop_id, _, _, dx, dy, dtheta, sigma_trans, sigma_rot in loop_candidates.values:\n        from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot = loop_id, from_idx, to_idx, dx, dy, dtheta, sigma_trans, sigma_rot\n        pose_from = poses[from_idx]\n        pose_to = poses[to_idx]\n        T = apply_transform(pose_to, np.array([dx, dy, dtheta]))\n        T_est = apply_transform(pose_to, np.array([0, 0, 0]))\n        T_est = transform(pose_from[0], pose_from[1], pose_from[2]) @ T_est\n        T_est = np.linalg.inv(transform(pose_from[0], pose_from[1], pose_from[2])) @ T_est\n        dx_est = T_est[0, 2]\n        dy_est = T_est[1, 2]\n        dtheta_est = T_est[2, 2]\n        dx_res = dx_est - dx\n        dy_res = dy_est - dy\n        dtheta_res = dtheta_est - dtheta\n        dx_res = wrap_angle(dx_res)\n        dy_res = wrap_angle(dy_res)\n        dtheta_res = wrap_angle(dtheta_res)\n        residual = np.array([dx_res, dy_res, dtheta_res])\n        loop_residuals.append(residual)\n    \n    return np.array(residuals), np.array(loop_residuals)\n\ndef calc_chi_square(residuals, sigma):\n    return np.sum((residuals ** 2) / (sigma ** 2))\n\ndef optimize_pose_graph(poses, edges, loop_candidates, loop_weights):\n    def objective(poses_flat):\n        poses = poses_flat.reshape((N, 3))\n        residuals, loop_residuals = calc_residuals(poses, edges, loop_candidates, loop_weights)\n        chi_square_odom = calc_chi_square(residuals, edges[[\'sigma_trans\', \'sigma_rot\']].values.T)\n        chi_square_loops = calc_chi_square(loop_residuals, loop_candidates[[\'sigma_trans\', \'sigma_rot\']].values.T)\n        return chi_square_odom + chi_square_loops\n    \n    result = minimize(objective, poses_initial.flatten(), method=\'L-BFGS-B\')\n    optimized_poses = result.x.reshape((N, 3))\n    return optimized_poses\n\n# Initial optimization with all loops\nloop_weights = np.ones(num_loops)\noptimized_poses = optimize_pose_graph(poses_initial, odometry_edges, loop_candidates, loop_weights)\n\n# Loop closure score calculation\nresiduals, loop_residuals = calc_residuals(optimized_poses, odometry_edges, loop_candidates, loop_weights)\nchi_square_loops = calc_chi_square(loop_residuals, loop_candidates[[\'sigma_trans\', \'sigma_rot\']].values.T)\nloop_scores = chi_square_loops / np.sum(chi_square_loops)\nloop_false_prob = 1 - loop_scores\nloop_false_prob[loop_false_prob < 0.5] = 0\nloop_false_prob[loop_false_prob > 0.5] = 1\n\n# Save outputs\nnp.save(\'poses.npy\', optimized_poses)\nloop_candidates[\'false_probability\'] = loop_false_prob\nloop_candidates.to_csv(\'loop_closure_scores.csv\', index=False)\n\n# Plot\nplt.figure(figsize=(12, 8))\nplt.plot(poses_initial[:, 0], poses_initial[:, 1], \'b-\', label=\'Initial\')\nplt.plot(optimized_poses[:, 0], optimized_poses[:, 1], \'r-\', label=\'Optimized\')\nplt.legend()\nplt.title(\'Initial vs Optimized Trajectories\')\nplt.xlabel(\'x [m]\')\nplt.ylabel(\'y [m]\')\nplt.axis(\'equal\')\nplt.savefig(\'map_diagnostics.png\')\n\n# Save metrics\nmetrics = {\n    \'num_poses\': N,\n    \'num_odom_edges\': num_odom_edges,\n    \'num_loop_candidates\': num_loops,\n    \'loop_weights\': loop_weights.tolist(),\n    \'chi_square_odom\': calc_chi_square(residuals, edges[[\'sigma_trans\', \'sigma_rot\']].values.T),\n    \'chi_square_loops\': chi_square_loops\n}\nwith open(\'metrics.json\', \'w\') as f:\n    json.dump(metrics, f)\n\n# Save analysis script\nwith open(\'analysis.py\', \'w\') as f:\n    f.write(open(__file__).read())')
```