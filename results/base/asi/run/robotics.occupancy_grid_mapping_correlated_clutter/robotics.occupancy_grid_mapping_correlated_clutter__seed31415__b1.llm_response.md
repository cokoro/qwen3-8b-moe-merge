```python
import numpy as np
import json
import os

def load_data():
    with open('data/metadata.json', 'r') as f:
        metadata = json.load(f)
    poses = np.load('data/poses.npy')
    ranges = np.load('data/ranges.npy')
    scan_angles = np.load('data/scan_angles.npy')
    return metadata, poses, ranges, scan_angles

def main():
    metadata, poses, ranges, scan_angles = load_data()
    grid_size = metadata['grid_size']
    origin = np.array(metadata['origin'])
    cell_size = metadata['cell_size']
    max_range = metadata['max_range']
    num_cells = grid_size[0], grid_size[1]
    occupancy_prob = np.full(num_cells, 0.5, dtype=np.float64)
    clutter_likelihood = np.zeros(num_cells, dtype=np.float64)
    uncertainty_map = np.zeros(num_cells, dtype=np.float64)
    
    # Convert poses to world coordinates
    T, B = ranges.shape
    endpoint_cells = {}
    for t in range(T):
        x, y, yaw = poses[t]
        for b in range(B):
            angle = scan_angles[b] + yaw
            r = ranges[t, b]
            if r > max_range:
                continue
            # Convert to world coordinates
            x_end = x + r * np.cos(angle)
            y_end = y + r * np.sin(angle)
            # Convert to grid cell
            cell_x = int((x_end - origin[0]) / cell_size)
            cell_y = int((y_end - origin[1]) / cell_size)
            if 0 <= cell_x < num_cells[0] and 0 <= cell_y < num_cells[1]:
                if cell_x not in endpoint_cells:
                    endpoint_cells[cell_x] = {}
                if cell_y not in endpoint_cells[cell_x]:
                    endpoint_cells[cell_x][cell_y] = []
                endpoint_cells[cell_x][cell_y].append(t)
    
    # Compute persistent endpoint cells
    persistent_cells = set()
    for cell_x in endpoint_cells:
        for cell_y in endpoint_cells[cell_x]:
            hits = endpoint_cells[cell_x][cell_y]
            if len(hits) >= 3 and len(set(hits)) > max(18, T // 8):
                persistent_cells.add((cell_x, cell_y))
    
    # Compute short return endpoint candidates
    short_return_cells = set()
    for t in range(T):
        for b in range(B):
            r = ranges[t, b]
            if r > max_range:
                continue
            # Compute local background
            t_window = np.clip(t - 2, 0, T - 1)
            b_window = np.clip(b - 3, 0, B - 1)
            local_background = np.quantile(ranges[t_window:t_window + 5, b_window:b_window + 5], 0.85)
            if local_background - r > max(0.55, 0.12 * local_background):
                # Convert to grid cell
                x_end = poses[t, 0] + r * np.cos(scan_angles[b] + poses[t, 2])
                y_end = poses[t, 1] + r * np.sin(scan_angles[b] + poses[t, 2])
                cell_x = int((x_end - origin[0]) / cell_size)
                cell_y = int((y_end - origin[1]) / cell_size)
                if 0 <= cell_x < num_cells[0] and 0 <= cell_y < num_cells[1]:
                    short_return_cells.add((cell_x, cell_y))
    
    # Compute persistent neighbourhood
    persistent_neighbourhood = set()
    for (cell_x, cell_y) in persistent_cells:
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if 0 <= cell_x + dx < num_cells[0] and 0 <= cell_y + dy < num_cells[1]:
                    persistent_neighbourhood.add((cell_x + dx, cell_y + dy))
    
    # Compute moving clutter core
    moving_clutter_core = set()
    for (cell_x, cell_y) in short_return_cells:
        if (cell_x, cell_y) not in persistent_neighbourhood:
            moving_clutter_core.add((cell_x, cell_y))
    
    # Update occupancy_prob
    for (cell_x, cell_y) in persistent_cells:
        occupancy_prob[cell_x, cell_y] = 0.95
    for (cell_x, cell_y) in moving_clutter_core:
        occupancy_prob[cell_x, cell_y] = 0.25
    
    # Update clutter_likelihood
    for (cell_x, cell_y) in moving_clutter_core:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= cell_x + dx < num_cells[0] and 0 <= cell_y + dy < num_cells[1]:
                    clutter_likelihood[cell_x + dx, cell_y + dy] = 0.9
    
    # Update uncertainty_map
    for cell_x in range(num_cells[0]):
        for cell_y in range(num_cells[1]):
            if (cell_x, cell_y) in persistent_cells:
                uncertainty_map[cell_x, cell_y] = 0.1
            elif (cell_x, cell_y) in moving_clutter_core:
                uncertainty_map[cell_x, cell_y] = 0.8
            else:
                uncertainty_map[cell_x, cell_y] = 0.5
    
    np.save('occupancy_prob.npy', occupancy_prob)
    np.save('clutter_likelihood.npy', clutter_likelihood)
    np.save('uncertainty_map.npy', uncertainty_map)

if __name__ == '__main__':
    main()
```