import numpy as np
import json

# Load metadata
with open('data/metadata.json') as f:
    meta = json.load(f)
    grid_size = meta['grid_size']
    origin = meta['origin']
    cell_size = meta['cell_size']
    max_range = meta['max_range']

# Load input data
poses = np.load('data/poses.npy')
ranges = np.load('data/ranges.npy')
scan_angles = np.load('data/scan_angles.npy')

# Initialize maps
occupancy_prob = np.full((grid_size, grid_size), 0.5, dtype=np.float64)
clutter_likelihood = np.zeros((grid_size, grid_size), dtype=np.float64)
uncertainty_map = np.zeros((grid_size, grid_size), dtype=np.float64)

# Convert scan angles to radians
scan_angles = np.deg2rad(scan_angles)

# Map cell coordinates to world coordinates
def world_to_cell(x, y):
    x_cell = int(np.floor((x - origin[0]) / cell_size))
    y_cell = int(np.floor((y - origin[1]) / cell_size))
    return (x_cell, y_cell)

# Convert beam endpoints to world coordinates
def beam_to_world(t, b):
    angle = scan_angles[b] + poses[t, 2]
    x = poses[t, 0] + ranges[t, b] * np.cos(angle)
    y = poses[t, 1] + ranges[t, b] * np.sin(angle)
    return (x, y)

# Ray trace from robot cell to endpoint cell
def ray_trace(robot_cell, endpoint_cell):
    x0, y0 = robot_cell
    x1, y1 = endpoint_cell
    dx = x1 - x0
    dy = y1 - y0
    steps = np.arange(0, 1.01, 1.0 / max(1, np.sqrt(dx**2 + dy**2)))
    for step in steps:
        x = x0 + step * dx
        y = y0 + step * dy
        cell = world_to_cell(x, y)
        if cell[0] < 0 or cell[0] >= grid_size or cell[1] < 0 or cell[1] >= grid_size:
            break
        yield cell

# Process each scan
endpoint_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
persistent_cells = np.zeros((grid_size, grid_size), dtype=bool)
short_return_candidates = np.zeros((grid_size, grid_size), dtype=bool)
for t in range(poses.shape[0]):
    for b in range(ranges.shape[1]):
        x, y = beam_to_world(t, b)
        if ranges[t, b] > max_range:
            continue
        cell = world_to_cell(x, y)
        if cell[0] < 0 or cell[0] >= grid_size or cell[1] < 0 or cell[1] >= grid_size:
            continue
        endpoint_counts[cell[0], cell[1]] += 1
        for cell in ray_trace(world_to_cell(poses[t, 0], poses[t, 1]), cell):
            if 0 < cell[0] < grid_size - 1 and 0 < cell[1] < grid_size - 1:
                occupancy_prob[cell[0], cell[1]] = 0.95

# Identify persistent cells
for x in range(grid_size):
    for y in range(grid_size):
        if endpoint_counts[x, y] >= 3 and np.sum(endpoint_counts[max(0, x-2):min(grid_size, x+3), max(0, y-2):min(grid_size, y+3)]) > 18:
            persistent_cells[x, y] = True

# Identify short-return candidates
for t in range(poses.shape[0]):
    for b in range(ranges.shape[1]):
        x, y = beam_to_world(t, b)
        if ranges[t, b] > max_range:
            continue
        cell = world_to_cell(x, y)
        if cell[0] < 0 or cell[0] >= grid_size or cell[1] < 0 or cell[1] >= grid_size:
            continue
        local_background = np.quantile(ranges[max(0, t-2):min(poses.shape[0], t+3), max(0, b-3):min(ranges.shape[1], b+4)], 0.85)
        if local_background - ranges[t, b] > max(0.55, 0.12 * local_background):
            short_return_candidates[cell[0], cell[1]] = True

# Identify moving-clutter core
moving_clutter_core = np.zeros((grid_size, grid_size), dtype=bool)
for x in range(grid_size):
    for y in range(grid_size):
        if short_return_candidates[x, y] and not persistent_cells[x, y]:
            moving_clutter_core[x, y] = True

# Set high clutter likelihood
for x in range(grid_size):
    for y in range(grid_size):
        if moving_clutter_core[x, y]:
            clutter_likelihood[x, y] = 0.9
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    clutter_likelihood[nx, ny] = 0.9

# Set uncertainty map
for x in range(grid_size):
    for y in range(grid_size):
        if endpoint_counts[x, y] == 0:
            uncertainty_map[x, y] = 1.0
        elif endpoint_counts[x, y] >= 3 and np.sum(endpoint_counts[max(0, x-2):min(grid_size, x+3), max(0, y-2):min(grid_size, y+3)]) > 18:
            uncertainty_map[x, y] = 0.0
        elif short_return_candidates[x, y]:
            uncertainty_map[x, y] = 1.0
        else:
            uncertainty_map[x, y] = 0.5

# Save outputs
np.save('occupancy_prob.npy', occupancy_prob)
np.save('clutter_likelihood.npy', clutter_likelihood)
np.save('uncertainty_map.npy', uncertainty_map)