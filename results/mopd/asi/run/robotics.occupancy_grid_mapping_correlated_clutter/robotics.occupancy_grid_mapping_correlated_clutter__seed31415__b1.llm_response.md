```python
import numpy as np
import json
import os

# Read metadata
metadata = json.load(open('data/metadata.json'))
grid_size = metadata['grid_size']
origin = np.array(metadata['origin'])
cell_size = metadata['cell_size']
max_range = metadata['max_range']
grid_rows, grid_cols = grid_size

# Read data
poses = np.load('data/poses.npy')
ranges = np.load('data/ranges.npy')
scan_angles = np.load('data/scan_angles.npy')

# Convert to world coordinates
T, B = ranges.shape
cell_x = np.floor((poses[:, 0] - origin[0]) / cell_size).astype(int)
cell_y = np.floor((poses[:, 1] - origin[1]) / cell_size).astype(int)

# Initialize occupancy grid
occupancy = np.zeros((grid_rows, grid_cols), dtype=np.float64)
clutter_likelihood = np.zeros((grid_rows, grid_cols), dtype=np.float64)
uncertainty = np.zeros((grid_rows, grid_cols), dtype=np.float64)

# Ray trace from robot to endpoints
for t in range(T):
    for b in range(B):
        beam_angle = scan_angles[b]
        range = ranges[t, b]
        if range > max_range:
            continue
        x = poses[t, 0] + range * np.cos(poses[t, 2] + beam_angle)
        y = poses[t, 1] + range * np.sin(poses[t, 2] + beam_angle)
        cell_x_end = np.floor((x - origin[0]) / cell_size).astype(int)
        cell_y_end = np.floor((y - origin[1]) / cell_size).astype(int)
        if 0 <= cell_x_end < grid_cols and 0 <= cell_y_end < grid_rows:
            for i in range(1, 10):
                cell_x_mid = cell_x[t] + i * (cell_x_end - cell_x[t]) // 10
                cell_y_mid = cell_y[t] + i * (cell_y_end - cell_y[t]) // 10
                if 0 <= cell_x_mid < grid_cols and 0 <= cell_y_mid < grid_rows:
                    occupancy[cell_y_mid, cell_x_mid] = 0.95
                    clutter_likelihood[cell_y_mid, cell_x_mid] = 0.1
                    uncertainty[cell_y_mid, cell_x_mid] = 0.05

# Evaluate persistent endpoints
endpoint_counts = np.zeros((grid_rows, grid_cols), dtype=int)
endpoint_times = np.zeros((grid_rows, grid_cols), dtype=int)
for t in range(T):
    for b in range(B):
        beam_angle = scan_angles[b]
        range = ranges[t, b]
        if range > max_range:
            continue
        x = poses[t, 0] + range * np.cos(poses[t, 2] + beam_angle)
        y = poses[t, 1] + range * np.sin(poses[t, 2] + beam_angle)
        cell_x_end = np.floor((x - origin[0]) / cell_size).astype(int)
        cell_y_end = np.floor((y - origin[1]) / cell_size).astype(int)
        if 0 <= cell_x_end < grid_cols and 0 <= cell_y_end < grid_rows:
            endpoint_counts[cell_y_end, cell_x_end] += 1
            endpoint_times[cell_y_end, cell_x_end] = t

# Evaluate moving clutter
moving_clutter_core = np.zeros((grid_rows, grid_cols), dtype=bool)
for t in range(T):
    for b in range(B):
        beam_angle = scan_angles[b]
        range = ranges[t, b]
        if range > max_range:
            continue
        x = poses[t, 0] + range * np.cos(poses[t, 2] + beam_angle)
        y = poses[t, 1] + range * np.sin(poses[t, 2] + beam_angle)
        cell_x_end = np.floor((x - origin[0]) / cell_size).astype(int)
        cell_y_end = np.floor((y - origin[1]) / cell_size).astype(int)
        if 0 <= cell_x_end < grid_cols and 0 <= cell_y_end < grid_rows:
            # Compute local background
            t_start = max(0, t - 2)
            t_end = min(T, t + 3)
            b_start = max(0, b - 3)
            b_end = min(B, b + 4)
            local_background = np.quantile(ranges[t_start:t_end, b_start:b_end], 0.85)
            if range < local_background - max(0.55, 0.12 * local_background):
                # Mark as short-return candidate
                moving_clutter_core[cell_y_end, cell_x_end] = True

# Evaluate persistent neighbourhood
persistent_neighbourhood = np.zeros((grid_rows, grid_cols), dtype=bool)
for y in range(grid_rows):
    for x in range(grid_cols):
        if endpoint_counts[y, x] > 3 and np.ptp(endpoint_times[y, x]) > max(18, T // 8):
            persistent_neighbourhood[y, x] = True

# Evaluate core moving clutter
moving_clutter_core &= ~persistent_neighbourhood

# Evaluate uncertainty map
uncertainty[moving_clutter_core] = 0.9
uncertainty[endpoint_counts == 0] = 0.9
uncertainty[endpoint_counts > 0] = 0.1

# Evaluate occupancy probability
occupancy[endpoint_counts == 0] = 0.05
occupancy[moving_clutter_core] = 0.25
occupancy[persistent_neighbourhood] = 0.95

# Save outputs
np.save('occupancy_prob.npy', occupancy)
np.save('clutter_likelihood.npy', clutter_likelihood)
np.save('uncertainty_map.npy', uncertainty)
```