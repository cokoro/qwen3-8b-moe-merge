import numpy as np
import argparse
import time
import sys
import os

def parse_tsp_file(input_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    dimension = None
    coords = []
    for line in lines:
        if line.strip().startswith('DIMENSION'):
            dimension = int(line.strip().split()[-1])
        elif line.strip().startswith('NODE_COORD_SECTION'):
            coords = []
        elif line.strip().startswith('EOF'):
            break
        elif line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                coords.append((int(parts[1]), int(parts[2])))
    
    if dimension != len(coords):
        raise ValueError("Dimension mismatch in TSP file")
    
    return np.array(coords, dtype=np.int32)

def euc_2d_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(np.floor(np.sqrt(dx * dx + dy * dy) + 0.5))

def calculate_tour_length(tour, coords):
    length = 0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        length += euc_2d_distance(coords[a], coords[b])
    return length

def nearest_neighbor_tour(coords):
    n = len(coords)
    tour = [0] * n
    visited = [False] * n
    current = 0
    visited[current] = True
    tour[0] = current
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for i in range(n):
            if not visited[i]:
                dist = euc_2d_distance(coords[current], coords[i])
                if dist < min_dist:
                    min_dist = dist
                    next_node = i
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def two_opt(tour, coords):
    n = len(tour)
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour, coords)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_tour = tour.copy()
            new_tour[i:j+1] = tour[j:i-1:-1]
            new_length = calculate_tour_length(new_tour, coords)
            if new_length < best_length:
                best_tour = new_tour
                best_length = new_length
    return best_tour

def three_opt(tour, coords):
    n = len(tour)
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour, coords)
    for i in range(1, n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                new_tour = tour.copy()
                new_tour[i:j+1] = tour[j:i-1:-1]
                new_tour[j:k+1] = tour[k:j-1:-1]
                new_length = calculate_tour_length(new_tour, coords)
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
    return best_tour

def lin_kernighan(tour, coords):
    n = len(tour)
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour, coords)
    for _ in range(100):
        for i in range(n):
            for j in range(n):
                if i != j:
                    new_tour = tour.copy()
                    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                    new_length = calculate_tour_length(new_tour, coords)
                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
    return best_tour

def solve_tsp(coords):
    n = len(coords)
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    # Initial tour using nearest neighbor
    initial_tour = nearest_neighbor_tour(coords)
    best_tour = initial_tour.copy()
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply 2-opt
    best_tour = two_opt(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply 3-opt
    best_tour = three_opt(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply Lin-Kernighan
    best_tour = lin_kernighan(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    return best_tour

def main():
    parser = argparse.ArgumentParser(description='Euclidean TSP Solver')
    parser.add_argument('--input', required=True, help='Path to TSP input file')
    parser.add_argument('--output', required=True, help='Path to output file')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--time-limit-seconds', type=int, required=True, help='Time limit in seconds')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    start_time = time.time()
    
    coords = parse_tsp_file(args.input)
    n = len(coords)
    
    # Initial tour using nearest neighbor
    initial_tour = nearest_neighbor_tour(coords)
    best_tour = initial_tour.copy()
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply 2-opt
    best_tour = two_opt(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply 3-opt
    best_tour = three_opt(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    # Apply Lin-Kernighan
    best_tour = lin_kernighan(best_tour, coords)
    best_length = calculate_tour_length(best_tour, coords)
    
    # Final check and output
    end_time = time.time()
    if end_time - start_time < args.time_limit_seconds - 1:
        # Perform additional iterations if time allows
        pass
    
    # Ensure the tour is valid
    if len(set(best_tour)) != n:
        raise ValueError("Tour contains duplicate nodes")
    
    # Save the tour
    np.save(args.output, best_tour, allow_pickle=False)
    
if __name__ == "__main__":
    main()