```python
import numpy as np
import json
import os

# Load metadata
metadata_path = 'data/metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

grid_size = metadata['grid_size']
origin = np.array(metadata['origin'])
cell_size = metadata['cell_size']
max_range = metadata['max_range']

# Load data
poses = np.load('data/poses.npy')
ranges = np.load('data/ranges.npy')
scan_angles = np.load('data/scan_angles.npy')

# Convert to world coordinates
T, B = ranges.shape
occupancy_prob = np.full((grid_size, grid_size), 0.5, dtype=np.float64)
clutter_likelihood = np.zeros((grid_size, grid_size), dtype=np.float64)
uncertainty_map = np.zeros((grid_size, grid_size), dtype=np.float64)

# Helper functions
def world_to_grid(x, y):
    grid_x = int((x - origin[0]) / cell_size)
    grid_y = int((y - origin[1]) / cell_size)
    return (grid_x, grid_y)

def is_in_grid(x, y):
    grid_x, grid_y = world_to_grid(x, y)
    return 0 <= grid_x < grid_size and 0 <= grid_y < grid_size

# Step 1: Convert beam endpoints to world coordinates
beam_endpoints = []
for t in range(T):
    x, y, yaw = poses[t]
    for b in range(B):
        angle = scan_angles[b] + yaw
        end_x = x + ranges[t, b] * np.cos(angle)
        end_y = y + ranges[t, b] * np.sin(angle)
        beam_endpoints.append((t, b, end_x, end_y))

# Step 2: Ray trace and mark traversed cells
endpoint_cells = {}
for t, b, end_x, end_y in beam_endpoints:
    if ranges[t, b] > max_range:
        continue
    start_x, start_y = poses[t, :2]
    grid_start = world_to_grid(start_x, start_y)
    grid_end = world_to_grid(end_x, end_y)
    if not is_in_grid(start_x, start_y) or not is_in_grid(end_x, end_y):
        continue
    # Ray tracing logic (simplified)
    dx = end_x - start_x
    dy = end_y - start_y
    steps = int(np.hypot(dx, dy) / cell_size)
    for i in range(steps + 1):
        x = start_x + dx * i / steps
        y = start_y + dy * i / steps
        grid_x, grid_y = world_to_grid(x, y)
        if is_in_grid(x, y):
            if (grid_x, grid_y) not in endpoint_cells:
                endpoint_cells[(grid_x, grid_y)] = []
            endpoint_cells[(grid_x, grid_y)].append(t)

# Step 3: Mark persistent endpoint cells
persistent_cells = set()
for cell, scan_indices in endpoint_cells.items():
    if len(scan_indices) >= 3 and len(set(scan_indices)) > max(18, T // 8):
        persistent_cells.add(cell)

# Step 4: Compute moving clutter core
moving_clutter_core = set()
for t in range(T):
    for b in range(B):
        if ranges[t, b] > max_range:
            continue
        # Compute local background
        window = np.clip(np.arange(t - 2, t + 3), 0, T - 1)
        window_b = np.clip(np.arange(b - 3, b + 4), 0, B - 1)
        local_background = np.quantile(ranges[window, window_b], 0.85)
        if local_background - ranges[t, b] > max(0.55, 0.12 * local_background):
            end_x = poses[t, 0] + ranges[t, b] * np.cos(scan_angles[b] + poses[t, 2])
            end_y = poses[t, 1] + ranges[t, b] * np.sin(scan_angles[b] + poses[t, 2])
            grid_end = world_to_grid(end_x, end_y)
            if is_in_grid(end_x, end_y) and grid_end not in persistent_cells:
                moving_clutter_core.add(grid_end)

# Step 5: Update occupancy probability
for cell in persistent_cells:
    occupancy_prob[cell[0], cell[1]] = 0.95

for cell in moving_clutter_core:
    occupancy_prob[cell[0], cell[1]] = 0.25

# Step 6: Update clutter likelihood
clutter_core = np.zeros((grid_size, grid_size), dtype=bool)
for cell in moving_clutter_core:
    clutter_core[cell[0], cell[1]] = True
clutter_core = np.lib.pad(clutter_core, 1, 'constant', constant_values=0)
clutter_likelihood[clutter_core] = 0.9

# Step 7: Update uncertainty map
for cell in persistent_cells:
    uncertainty_map[cell[0], cell[1]] = 0.1

for cell in moving_clutter_core:
    uncertainty_map[cell[0], cell[1]] = 0.8

# Save outputs
np.save('occupancy_prob.npy', occupancy_prob)
np.save('clutter_likelihood.npy', clutter_likelihood)
np.save('uncertainty_map.npy', uncertainty_map)
```