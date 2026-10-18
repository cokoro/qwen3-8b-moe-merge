import json
import numpy as np
import matplotlib.pyplot as plt

# Load metadata
with open('data/metadata.json') as f:
    metadata = json.load(f)
    grid_size = metadata['grid_size']
    origin = np.array(metadata['origin'])
    cell_size = metadata['cell_size']
    max_range = metadata['max_range']

# Load data
poses = np.load('data/poses.npy')
ranges = np.load('data/ranges.npy')
scan_angles = np.load('data/scan_angles.npy')

# Convert scan angles to radians
scan_angles = np.deg2rad(scan_angles)

# Calculate the number of beams
num_beams = len(scan_angles)

# Initialize occupancy probability grid
occupancy_prob = np.full(grid_size, 0.5, dtype=np.float64)

# Initialize clutter likelihood grid
clutter_likelihood = np.full(grid_size, 0.0, dtype=np.float64)

# Initialize uncertainty map
uncertainty_map = np.full(grid_size, 0.5, dtype=np.float64)

# Convert poses to world coordinates
world_poses = np.zeros((poses.shape[0], 2))
for t in range(poses.shape[0]):
    x, y, yaw = poses[t]
    world_poses[t, 0] = x
    world_poses[t, 1] = y

# Initialize endpoint tracking
endpoint_hits = np.zeros(grid_size, dtype=int)
endpoint_scans = np.zeros((grid_size, 100), dtype=int)
endpoint_scan_indices = np.zeros((grid_size, 100), dtype=int)
endpoint_cells = np.zeros(grid_size, dtype=int)

# Process each scan
for t in range(poses.shape[0]):
    pose = poses[t]
    for b in range(num_beams):
        angle = scan_angles[b]
        beam_angle = angle + pose[2]
        beam_end_x = pose[0] + ranges[t, b] * np.cos(beam_angle)
        beam_end_y = pose[1] + ranges[t, b] * np.sin(beam_angle)
        beam_end_cell = ((beam_end_x - origin[0]) / cell_size, (beam_end_y - origin[1]) / cell_size)
        beam_end_cell = tuple(beam_end_cell.astype(int))
        if 0 <= beam_end_cell[0] < grid_size[0] and 0 <= beam_end_cell[1] < grid_size[1]:
            endpoint_cells[beam_end_cell] += 1
            endpoint_scans[beam_end_cell, endpoint_hits[beam_end_cell]] = t
            endpoint_scan_indices[beam_end_cell, endpoint_hits[beam_end_cell]] = b
            endpoint_hits[beam_end_cell] += 1

# Calculate persistent endpoint cells
persistent_cells = np.zeros(grid_size, dtype=bool)
for cell in range(grid_size[0] * grid_size[1]):
    if endpoint_cells[cell] >= 3:
        scan_indices = endpoint_scans[cell, :endpoint_cells[cell]]
        if np.max(scan_indices) - np.min(scan_indices) > max(18, poses.shape[0] // 8):
            persistent_cells[cell] = True

# Calculate short-return endpoint candidates
short_return_cells = np.zeros(grid_size, dtype=bool)
for t in range(poses.shape[0]):
    for b in range(num_beams):
        if ranges[t, b] <= max_range:
            window = ranges[max(0, t-2):min(poses.shape[0], t+3), max(0, b-3):min(num_beams, b+4)]
            clipped = np.clip(window, 0, max_range)
            local_background = np.quantile(clipped, 0.85

            if local_background - ranges[t, b] > max(0.55, 0.12 * local_background):
                beam_end_x = poses[t, 0] + ranges[t, b] * np.cos(scan_angles[b] + poses[t, 2])
                beam_end_y = poses[t, 1] + ranges[t, b] * np.sin(scan_angles[b] + poses[t, 2])
                beam_end_cell = ((beam_end_x - origin[0]) / cell_size, (beam_end_y - origin[1]) / cell_size)
                beam_end_cell = tuple(beam_end_cell.astype(int))
                if 0 <= beam_end_cell[0] < grid_size[0] and 0 <= beam_end_cell[1] < grid_size[1]:
                    short_return_cells[beam_end_cell] = True

# Calculate persistent neighbourhood
persistent_neighbourhood = np.zeros(grid_size, dtype=bool)
for cell in range(grid_size[0] * grid_size[1]):
    if persistent_cells[cell]:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= cell[0] + dx < grid_size[0] and 0 <= cell[1] + dy < grid_size[1]:
                    persistent_neighbourhood[cell[0] + dx, cell[1] + dy] = True

# Calculate moving clutter core
moving_clutter_core = np.zeros(grid_size, dtype=bool)
for cell in range(grid_size[0] * grid_size[1]):
    if short_return_cells[cell] and not persistent_neighbourhood[cell]:
        moving_clutter_core[cell] = True

# Set clutter likelihood
clutter_likelihood[moving_clutter_core] = 0.9
clutter_likelihood[dilate(moving_clutter_core, radius=1)] = 0.9

# Set occupancy probability
occupancy_prob[persistent_cells] = 0.95
occupancy_prob[short_return_cells] = 0.25

# Set uncertainty map
uncertainty_map[~persistent_cells] = 0.5
uncertainty_map[moving_clutter_core] = 0.5

# Save outputs
np.save('occupancy_prob.npy', occupancy_prob)
np.save('clutter_likelihood.npy', clutter_likelihood)
np.save('uncertainty_map.npy', uncertainty_map)