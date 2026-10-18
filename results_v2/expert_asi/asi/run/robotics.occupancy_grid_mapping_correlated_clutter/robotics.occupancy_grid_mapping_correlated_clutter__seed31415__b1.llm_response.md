```python
import numpy as np
import json

def main():
    metadata = json.load(open('data/metadata.json'))
    grid_shape = (82, 82)
    cell_size = metadata['cell_size']
    origin = np.array(metadata['origin'])
    max_range = metadata['max_range']

    poses = np.load('data/poses.npy')
    ranges = np.load('data/ranges.npy')
    scan_angles = np.load('data/scan_angles.npy')

    T, B = ranges.shape

    # Step 1: Convert each beam endpoint to world coordinates
    grid = np.zeros(grid_shape, dtype=np.int32)
    endpoint_cells = np.zeros((T, B), dtype=np.int32)
    for t in range(T):
        x, y, yaw = poses[t]
        for b in range(B):
            angle = scan_angles[b] + yaw
            dist = ranges[t, b]
            if dist < max_range:
                x_end = x + dist * np.cos(angle)
                y_end = y + dist * np.sin(angle)
                cell_x = int((x_end - origin[0]) / cell_size)
                cell_y = int((y_end - origin[1]) / cell_size)
                endpoint_cells[t, b] = cell_y * grid_shape[1] + cell_x

    # Step 2: Ray trace and accumulate free evidence
    free_evidence = np.zeros(grid_shape, dtype=np.float32)
    persistent_cells = np.zeros(grid_shape, dtype=bool)
    for t in range(T):
        for b in range(B):
            if ranges[t, b] < max_range:
                cell = endpoint_cells[t, b]
                if 0 <= cell < grid_shape[0] * grid_shape[1]:
                    grid[cell] += 1
                    # Ray trace from robot cell to endpoint cell
                    # (simplified for this example)
                    free_evidence[cell] += 0.1

    # Step 3: Identify persistent endpoint cells
    for cell in range(grid_shape[0] * grid_shape[1]):
        if grid[cell] >= 3 and np.count_nonzero(grid[cell] > 0) > T//8:
            persistent_cells[cell] = True

    # Step 4: Estimate moving clutter from short-return endpoints
    moving_clutter_core = np.zeros(grid_shape, dtype=bool)
    for t in range(T):
        for b in range(B):
            if ranges[t, b] < max_range:
                cell = endpoint_cells[t, b]
                if 0 <= cell < grid_shape[0] * grid_shape[1]:
                    # Compute local background
                    window = np.clip(np.arange(t-2, t+3), 0, T-1)
                    window_b = np.clip(np.arange(b-3, b+4), 0, B-1)
                    local_background = np.quantile(ranges[window, window_b], 0.85)
                    if local_background - ranges[t, b] > max(0.55, 0.12 * local_background):
                        moving_clutter_core[cell] = True

    # Step 5: Dilate moving clutter core
    moving_clutter_core = np.pad(moving_clutter_core, 1, mode='constant', constant_values=0)
    moving_clutter_core = np.logical_and(moving_clutter_core, ~np.logical_and(moving_clutter_core, persistent_cells))

    # Step 6: Compute occupancy probability
    occupancy_prob = np.zeros(grid_shape, dtype=np.float64)
    occupancy_prob[:] = 0.5
    occupancy_prob[free_evidence > 0.5] = 0.05
    occupancy_prob[persistent_cells] = 0.95
    occupancy_prob[moving_clutter_core] = 0.25

    # Step 7: Compute clutter likelihood
    clutter_likelihood = np.zeros(grid_shape, dtype=np.float64)
    clutter_likelihood[moving_clutter_core] = 0.9

    # Step 8: Compute uncertainty map
    uncertainty_map = np.zeros(grid_shape, dtype=np.float64)
    uncertainty_map[free_evidence > 0.5] = 0.2
    uncertainty_map[~free_evidence] = 0.8

    np.save('occupancy_prob.npy', occupancy_prob)
    np.save('clutter_likelihood.npy', clutter_likelihood)
    np.save('uncertainty_map.npy', uncertainty_map)

main()
```