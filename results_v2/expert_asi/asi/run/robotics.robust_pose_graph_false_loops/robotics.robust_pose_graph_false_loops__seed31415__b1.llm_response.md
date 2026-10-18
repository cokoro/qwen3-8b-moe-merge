```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def read_poses(path):
    return np.load(path)

def read_edges(path):
    return pd.read_csv(path)

def read_loops(path):
    return pd.read_csv(path)

def read_task_info(path):
    with open(path, 'r') as f:
        return json.load(f)

def optimize_pose_graph(initial_poses, edges, loops, task_info):
    # Placeholder for optimization logic
    # Return optimized poses, loop scores, and metrics
    pass

def plot_trajectories(poses_initial, poses_optimized, loops, loop_scores, output_path):
    plt.figure(figsize=(12, 8))
    plt.plot(poses_initial[:, 0], poses_initial[:, 1], 'r-', label='Initial')
    plt.plot(poses_optimized[:, 0], poses_optimized[:, 1], 'b-', label='Optimized')
    for _, row in loops.iterrows():
        if loop_scores.loc[row['loop_id']]['false_probability'] < 0.5:
            plt.plot([row['from'], row['to']], [row['to'], row['from']], 'g--')
        else:
            plt.plot([row['from'], row['to']], [row['to'], row['from']], 'r--')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def main():
    task_info = read_task_info('data/task_info.json')
    poses_initial = read_poses('data/poses_initial.npy')
    edges = read_edges('data/odometry_edges.csv')
    loops = read_loops('data/loop_candidates.csv')

    # Perform robust optimization
    optimized_poses, loop_scores, metrics = optimize_pose_graph(
        poses_initial, edges, loops, task_info
    )

    # Save outputs
    np.save('poses.npy', optimized_poses)
    loop_scores.to_csv('loop_closure_scores.csv', index=False)
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    plot_trajectories(poses_initial, optimized_poses, loops, loop_scores, 'map_diagnostics.png')

if __name__ == '__main__':
    main()
```