```python
import numpy as np
import sys
import time

def read_tsp(path):
    with open(path, 'r') as f:
        n = int([l for l in f if l.strip().startswith('DIMENSION:')][0].split()[-1])
        coords = np.array([list(map(int, l.split()))[1:] for l in f if l.strip().startswith('NODE_COORD_SECTION')])
    return n, coords

def euc_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return int(np.sqrt(dx*dx + dy*dy) + 0.5)

def solve_tsp(n, coords, seed, time_limit):
    np.random.seed(seed)
    start_time = time.monotonic()
    tour = np.arange(n)
    np.random.shuffle(tour)
    tour = np.argsort(tour)  # Ensure consistent ordering

    # Use Morton order as initial heuristic
    morton = np.packbits((coords[:, 0] << 16) | coords[:, 1], axis=1).flatten()
    tour = np.argsort(morton)

    # Use nearest neighbor heuristic
    tour = np.zeros(n, dtype=int)
    tour[0] = np.random.randint(n)
    visited = np.zeros(n, dtype=bool)
    visited[tour[0]] = True
    for i in range(1, n):
        dists = np.array([euc_dist(coords[tour[-1], 0], coords[tour[-1], 1], coords[j], coords[j, 1]) for j in range(n) if not visited[j]])
        next_idx = np.argmin(dists)
        tour[i] = next_idx
        visited[next_idx] = True

    # Use 2-opt local search
    best_tour = tour.copy()
    best_length = 0
    for i in range(n):
        best_length += euc_dist(coords[best_tour[i], 0], coords[best_tour[i], 1], coords[best_tour[(i+1)%n], 0], coords[best_tour[(i+1)%n], 1])

    for _ in range(1000):
        if time.monotonic() - start_time > time_limit:
            break
        for i in range(n):
            for j in range(i+2, n):
                if i < j:
                    a, b, c, d = best_tour[i], best_tour[i+1], best_tour[j], best_tour[j+1]
                    cost = best_length - euc_dist(coords[a, 0], coords[a, 1], coords[b, 0], coords[b, 1]) - euc_dist(coords[c, 0], coords[c, 1], coords[d, 0], coords[d, 1]) + euc_dist(coords[a, 0], coords[a, 1], coords[c, 0], coords[c, 1]) + euc_dist(coords[b, 0], coords[b, 1], coords[d, 0], coords[d, 1])
                    if cost < best_length:
                        best_length = cost
                        best_tour[i+1:j+1] = best_tour[j::-1]
                        break
            else:
                continue
            break

    return best_tour

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()

    n, coords = read_tsp(args.input)
    tour = solve_tsp(n, coords, args.seed, args.time_limit_seconds)
    tour = np.array(tour, dtype=np.int32)
    np.save(args.output, tour)

if __name__ == '__main__':
    main()
```