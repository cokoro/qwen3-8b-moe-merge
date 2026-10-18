```python
import numpy as np
import sys
import os
import time

def read_tsp_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('DIMENSION'):
            n = int(line.split()[-1])
    coords = np.zeros((n, 2), dtype=np.int32)
    for i, line in enumerate(lines):
        if line.startswith('NODE_COORD_SECTION'):
            break
    for i, line in enumerate(lines):
        if line.startswith('EOF'):
            break
        if i >= len(lines):
            break
        if line.strip() == '':
            continue
        parts = line.strip().split()
        if parts[0] == 'EOF':
            break
        coords[i-1] = [int(parts[1]), int(parts[2])]
    return coords

def euc_2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(np.floor(np.sqrt(dx*dx + dy*dy) + 0.5))

def tsp_solver(input_path, output_path, seed, time_limit):
    np.random.seed(seed)
    coords = read_tsp_file(input_path)
    n = coords.shape[0]
    tour = np.arange(n, dtype=np.int32)
    np.random.shuffle(tour)
    best_tour = tour.copy()
    best_length = 0
    start_time = time.time()
    while time.time() < start_time + time_limit:
        for i in range(n):
            for j in range(i+1, n):
                if i == 0 and j == n-1:
                    continue
                if i == n-1 and j == 0:
                    continue
                if np.random.rand() < 0.05:
                    tour[i], tour[j] = tour[j], tour[i]
                    current_length = best_length
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        current_length += euc_2d(coords[tour[k]], coords[tour[(k+1)%n]])
                    if current_length < best_length:
                        best_length = current_length
                        best_tour = tour.copy()
        if np.random.rand() < 0.1:
            tour = np.random.permutation(n)
    np.save(output_path, best_tour, allow_pickle=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--time-limit-seconds', type=int)
    args = parser.parse_args()
    tsp_solver(args.input, args.output, args.seed, args.time_limit_seconds)
```