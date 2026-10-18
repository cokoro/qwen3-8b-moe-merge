```python
import sys
import numpy as np
import time
import argparse

def parse_tsp(input_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].split()[1])
    coords = []
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            for l in lines[lines.index(line)+1:]:
                if l.strip() == 'EOF':
                    break
                x, y = map(float, l.strip().split()[1:])
                coords.append((x, y))
    return np.array(coords, dtype=np.float32)

def tsp_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx*dx + dy*dy)

def nearest_neighbor(coords, start=0):
    n = len(coords)
    tour = [start]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    for _ in range(n-1):
        nearest = np.argmin([tsp_distance(coords[current], coords[i]) for i in range(n) if not visited[i]])
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour

def two_opt(tour, coords):
    n = len(tour)
    best = tour.copy()
    for i in range(n):
        for j in range(i+2, n):
            if tsp_distance(coords[tour[i]], coords[tour[j]]) + tsp_distance(coords[tour[j]], coords[tour[i+1]]) < \
               tsp_distance(coords[tour[i]], coords[tour[i+1]]) + tsp_distance(coords[tour[j-1]], coords[tour[j]]):
                best[i:j] = best[i:j][::-1]
                tour = best.copy()
    return tour

def lkh_heuristic(coords, seed, time_limit):
    np.random.seed(seed)
    n = len(coords)
    tour = nearest_neighbor(coords)
    best = tour.copy()
    start_time = time.monotonic()
    while time.monotonic() - start_time < time_limit:
        for _ in range(50):
            tour = two_opt(tour, coords)
            if np.array_equal(tour, best):
                break
        if np.array_equal(tour, best):
            break
        best = tour.copy()
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()

    coords = parse_tsp(args.input)
    n = len(coords)
    if n <= 2:
        tour = np.arange(n)
    else:
        tour = lkh_heuristic(coords, args.seed, args.time_limit_seconds)

    tour = np.array(tour, dtype=np.int32)
    assert len(tour) == n
    assert np.unique(tour).size == n
    np.save(args.output, tour, allow_pickle=False)

if __name__ == '__main__':
    main()
```