import numpy as np
import json
import os

def main():
    data_dir = 'data'
    poses = np.load(os.path.join(data_dir, 'poses.npy'))
    ranges = np.load(os.path.join(data_dir, 'ranges.npy'))
    scan_angles = np.load(os.path.join(data_dir, 'scan_angles.npy'))
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    width = metadata['width']
    height = metadata['height']
    origin_x = metadata['origin'][0]
    origin_y = metadata['origin'][1]
    cell_size = metadata['cell_size']
    max_range = metadata['max_range']

    # Convert to world coordinates
    T = poses.shape[0]
    B = ranges.shape[1]
    cell_grid = np.zeros((width, height), dtype=int)
    for t in range(T):
        x, y, yaw = poses[t]
        for b in range(B):
            angle = scan_angles[b] + yaw
            dist = ranges[t, b]
            if dist > max_range:
                continue
            end_x = x + dist * np.cos(angle)
            end_y = y + dist * np.sin(angle)
            cx = int((end_x - origin_x) / cell_size)
            cy = int((end_y - origin_y) / cell_size)
            if 0 <= cx < width and 0 <= cy < height:
                cell_grid[cx, cy] += 1

    # Mark persistent endpoint cells
    persistent = np.zeros((width, height), dtype=bool)
    for cx in range(width):
        for cy in range(height):
            if cell_grid[cx, cy] >= 3:
                for t in range(T):
                    if cell_grid[cx, cy] >= 3 and t >= 18 and t <= T - 18:
                        persistent[cx, cy] = True
                        break

    # Compute short-return candidates
    short_return = np.zeros((width, height), dtype=bool)
    for t in range(T):
        for b in range(B):
            dist = ranges[t, b]
            if dist > max_range:
                continue
            cx = int((poses[t, 0] + dist * np.cos(scan_angles[b] + poses[t, 2]) - origin_x) / cell_size)
            cy = int((poses[t, 1] + dist * np.sin(scan_angles[b] + poses[t, 2]) - origin_y) / cell_size)
            if 0 <= cx < width and 0 <= cy < height:
                local_background = np.quantile(ranges[max(0, t-2):min(T, t+3), max(0, b-3):min(B, b+4)], 0.85, axis=None)
                if dist < local_background + max(0.55, 0.12 * local_background):
                    short_return[cx, cy] = True

    # Build persistent neighbourhood
    persistent_neighbourhood = np.zeros((width, height), dtype=bool)
    for cx in range(width):
        for cy in range(height):
            if persistent[cx, cy]:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx = cx + dx
                        ny = cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            persistent_neighbourhood[nx, ny] = True

    # Compute moving-clutter core
    moving_clutter_core = np.zeros((width, height), dtype=bool)
    for cx in range(width):
        for cy in range(height):
            if short_return[cx, cy] and not persistent_neighbourhood[cx, cy]:
                moving_clutter_core[cx, cy] = True

    # Build clutter likelihood
    clutter_likelihood = np.zeros((width, height), dtype=np.float64)
    for cx in range(width):
        for cy in range(height):
            if moving_clutter_core[cx, cy]:
                clutter_likelihood[cx, cy] = 0.9

    # Build uncertainty map
    uncertainty_map = np.zeros((width, height), dtype=np.float64)
    for cx in range(width):
        for cy in range(height):
            if cell_grid[cx, cy] == 0:
                uncertainty_map[cx, cy] = 1.0
            elif cell_grid[cx, cy] >= 3 and persistent[cx, cy]:
                uncertainty_map[cx, cy] = 0.0
            elif cell_grid[cx, cy] >= 1 and not persistent[cx, cy]:
                uncertainty_map[cx, cy] = 0.8
            else:
                uncertainty_map[cx, cy] = 0.5

    # Build occupancy probability
    occupancy_prob = np.zeros((width, height), dtype=np.float64)
    for cx in range(width):
        for cy in range(height):
            if cell_grid[cx, cy] == 0:
                occupancy_prob[cx, cy] = 0.05
            elif cell_grid[cx, cy] >= 3 and persistent[cx, cy]:
                occupancy_prob[cx, cy] = 0.95
            elif cell_grid[cx, cy] >= 1 and not persistent[cx, cy]:
                occupancy_prob[cx, cy] = 0.25
            else:
                occupancy_prob[cx, cy] = 0.5

    # Save outputs
    np.save('occupancy_prob.npy', occupancy_prob)
    np.save('clutter_likelihood.npy', clutter_likelihood)
    np.save('uncertainty_map.npy', uncertainty_map)

if __name__ == '__main__':
    main()