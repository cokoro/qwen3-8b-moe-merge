```python
import numpy as np
import sys
import time
import math

def parse_tsp_file(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    coords = []
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            for l in lines[lines.index(line)+1:]:
                if l.startswith('EOF'):
                    break
                _, x, y = l.strip().split()
                coords.append((float(x), float(y)))
    return np.array(coords, dtype=np.float64)

def euc_2d(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return int(math.floor(math.sqrt(dx*dx + dy*dy) + 0.5))

def solve_tsp(coords):
    n = coords.shape[0]
    if n <= 2:
        return np.arange(n)
    # Use nearest neighbor heuristic to construct an initial tour
    tour = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    current = 0
    visited[current] = True
    tour[0] = current
    for _ in range(n-1):
        next_node = np.argmin([euc_2d(coords[current], coords[i]) for i in range(n) if not visited[i]])
        tour[_+1] = next_node
        visited[next_node] = True
        current = next_node
    # Apply 2-opt local search
    for _ in range(1000):
        best = -1
        for i in range(1, n-1):
            for j in range(i+1, n):
                if tour[i] == tour[j]:
                    continue
                cost = euc_2d(coords[tour[i-1]], coords[tour[i]]) + euc_2d(coords[tour[j]], coords[tour[j+1]])
                cost -= euc_2d(coords[tour[i-1]], coords[tour[j]]) + euc_2d(coords[tour[i]], coords[tour[j+1]])
                if cost < best:
                    best = cost
                    best_i, best_j = i, j
        if best == -1:
            break
        tour[best_i:best_j+1] = tour[best_i:best_j+1][::-1]
    return tour

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()
    np.random.seed(args.seed)
    coords = parse_tsp_file(args.input)
    start_time = time.monotonic()
    tour = solve_tsp(coords)
    end_time = time.monotonic()
    if end_time - start_time < args.time_limit_seconds:
        tour = np.arange(coords.shape[0])
    np.save(args.output, tour, allow_pickle=False)

if __name__ == '__main__':
    main()
```