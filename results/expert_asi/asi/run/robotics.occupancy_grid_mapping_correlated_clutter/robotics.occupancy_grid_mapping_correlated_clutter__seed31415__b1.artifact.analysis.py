import numpy as np
import json
import os

def main():
    # Read metadata
    with open('data/metadata.json') as f:
        meta = json.load(f)
    grid_size = meta['grid_size']
    origin = np.array(meta['origin'])
    cell_size = meta['cell_size']
    max_range = meta['max_range']
    grid = np.zeros((grid_size[0], grid_size[1]), dtype=np.float64)

    # Load data
    poses = np.load('data/poses.npy')
    ranges = np.load('data/ranges.npy')
    scan_angles = np.load('data/scan_angles.npy')

    # Step 1: Convert each beam endpoint to world coordinates
    T, B = ranges.shape
    beam_endpoints = []
    for t in range(T):
        pose = poses[t]
        angles = scan_angles + pose[2]
        for b in range(B):
            beam_length = ranges[t, b]
            if beam_length < max_range:
                x = pose[0] + beam_length * np.cos(angles[b])
                y = pose[1] + beam_length * np.sin(angles[b])
                beam_endpoints.append((x, y, t, b))
    
    # Step 2: Ray trace from robot cell to endpoint cell
    robot_cells = []
    for t in range(T):
        pose = poses[t]
        x, y = pose[0], pose[1]
        cell_x = int((x - origin[0]) / cell_size)
        cell_y = int((y - origin[1]) / cell_size)
        robot_cells.append((cell_x, cell_y))
    
    # Step 3: Mark traversed cells as free
    free_cells = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for (x, y, t, b) in beam_endpoints:
        # Ray trace from robot cell to endpoint cell
        robot_cell = robot_cells[t]
        endpoint_cell = (int(x / cell_size), int(y / cell_size))
        for dx in range(robot_cell[0], endpoint_cell[0]):
            for dy in range(robot_cell[1], endpoint_cell[1]):
                free_cells[dy, dx] = True
        free_cells[endpoint_cell[1], endpoint_cell[0]] = True
    
    # Step 4: Mark endpoint cells
    endpoint_cells = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for (x, y, t, b) in beam_endpoints:
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if (cx, cy) == (endpoint_cell[0], endpoint_cell[1]):
            endpoint_cells[cy, cx] = True
    
    # Step 5: Accumulate free evidence
    free_evidence = np.zeros((grid_size[0], grid_size[1]), dtype=np.float64)
    for (x, y, t, b) in beam_endpoints:
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if free_cells[cy, cx]:
            free_evidence[cy, cx] += 1
    
    # Step 6: Determine persistent endpoints
    persistent = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for (x, y, t, b) in beam_endpoints:
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if endpoint_cells[cy, cx] and free_evidence[cy, cx] >= 3:
            persistent[cy, cx] = True
    
    # Step 7: Estimate moving clutter
    moving_clutter = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for t in range(T):
        for b in range(B):
            r = ranges[t, b]
            if r < max_range:
                # Compute local background
                window = np.zeros((5, 5), dtype=np.float64)
                for i in range(5):
                    for j in range(5):
                        t_idx = min(max(t - 2 + i, 0), T - 1)
                        b_idx = min(max(b - 3 + j, 0), B - 1)
                        window[i, j] = ranges[t_idx, b_idx]
                window = np.sort(window)
                q = np.quantile(window, 0.85)
                if r < q - max(0.55, 0.12 * q):
                    cx = int((pose[0] + r * np.cos(scan_angles[b])) / cell_size)
                    cy = int((pose[1] + r * np.sin(scan_angles[b])) / cell_size)
                    moving_clutter[cy, cx] = True
    
    # Step 8: Suppress moving clutter near static structure
    persistent_neighborhood = np.zeros((grid_size[0], grid_size[1]), dtype=bool)
    for (x, y, t, b) in beam_endpoints:
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if persistent[cy, cx]:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= cy + dy < grid_size[0] and 0 <= cx + dx < grid_size[1]:
                        persistent_neighborhood[cy + dy, cx + dx] = True
    moving_clutter_core = moving_clutter & ~persistent_neighborhood
    
    # Step 9: Set output values
    occupancy_prob = np.zeros((grid_size[0], grid_size[1]), dtype=np.float64)
    occupancy_prob.fill(0.5)
    occupancy_prob[free_cells] = 0.05
    occupancy_prob[persistent] = 0.95
    occupancy_prob[moving_clutter_core] = 0.25
    
    clutter_likelihood = np.zeros((grid_size[0], grid_size[1]), dtype=np.float64)
    clutter_likelihood[dilate(moving_clutter_core, radius=1)] = 0.9
    
    uncertainty_map = np.zeros((grid_size[0], grid_size[1]), dtype=np.float64)
    uncertainty_map[~free_cells & ~persistent] = 0.8
    uncertainty_map[moving_clutter_core] = 0.7

    # Save outputs
    np.save('occupancy_prob.npy', occupancy_prob)
    np.save('clutter_likelihood.npy', clutter_likelihood)
    np.save('uncertainty_map.npy', uncertainty_map)

def dilate(mask, radius=1):
    result = np.zeros_like(mask, dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if 0 <= dx < mask.shape[0] and 0 <= dy < mask.shape[1]:
                result[dx, dy] |= mask[dx, dy]
    return result

if __name__ == '__main__':
    main()