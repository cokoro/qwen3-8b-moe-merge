import numpy as np
import sys
import argparse
import time
import math

def euc_2d_distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.floor(math.sqrt(dx * dx + dy * dy) + 0.5)

def read_tsp_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    nodes = []
    for line in lines:
        if line.strip().startswith('NODE_COORD_SECTION'):
            break
    for line in lines:
        if line.strip().startswith('EOF'):
            break
        if line.strip().startswith('NAME'):
            continue
        if line.strip().startswith('TYPE'):
            continue
        if line.strip().startswith('EDGE_WEIGHT_TYPE'):
            continue
        if line.strip().startswith('DIMENSION'):
            continue
        if line.strip().startswith('COMMENT'):
            continue
        if line.strip().startswith('TOUR_SECTION'):
            continue
        if line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        nodes.append((x, y))
    return np.array(nodes, dtype=np.float32)

def create_initial_tour(nodes):
    n = len(nodes)
    tour = np.arange(n, dtype=np.int32)
    tour = np.roll(tour, 1)
    return tour

def nearest_neighbor_tour(nodes):
    n = len(nodes)
    tour = np.zeros(n, dtype=np.int32)
    tour[0] = 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    for i in range(1, n):
        next_node = -1
        min_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                dist = euc_2d_distance(nodes[current][0], nodes[current][1], nodes[j][0], nodes[j][1])
                if dist < min_dist:
                    min_dist = dist
                    next_node = j
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def strip_ordering(nodes):
    n = len(nodes)
    y_coords = nodes[:, 1]
    sorted_indices = np.argsort(y_coords)
    tour = np.zeros(n, dtype=np.int32)
    tour[0] = sorted_indices[0]
    visited = np.zeros(n, dtype=bool)
    visited[sorted_indices[0]] = True
    current = sorted_indices[0]
    for i in range(1, n):
        next_node = -1
        min_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                dist = euc_2d_distance(nodes[current][0], nodes[current][1], nodes[j][0], nodes[j][1])
                if dist < min_dist:
                    min_dist = dist
                    next_node = j
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def angular_ordering(nodes):
    n = len(nodes)
    angles = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dx = nodes[i][0]
        dy = nodes[i][1]
        if dx == 0 and dy == 0:
            angles[i] = 0
        else:
            angles[i] = math.atan2(dy, dx)
    sorted_indices = np.argsort(angles)
    tour = np.zeros(n, dtype=np.int32)
    tour[0] = sorted_indices[0]
    visited = np.zeros(n, dtype=bool)
    visited[sorted_indices[0]] = True
    current = sorted_indices[0]
    for i in range(1, n):
        next_node = -1
        min_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                dist = euc_2d_distance(nodes[current][0], nodes[current][1], nodes[j][0], nodes[j][1])
                if dist < min_dist:
                    min_dist = dist
                    next_node = j
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def get_arguments():
    parser = argparse.ArgumentParser(description='Euclidean TSP Solver')
    parser.add_argument('--input', type=str, help='Input TSP file path')
    parser.add_argument('--output', type=str, help='Output tour file path')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--time-limit-seconds', type=int, help='Time limit in seconds')
    return parser.parse_args()

def solve_tsp(nodes):
    n = len(nodes)
    if n <= 1:
        return np.array([], dtype=np.int32)
    
    # Create multiple initial tours
    initial_tours = []
    initial_tours.append(create_initial_tour(nodes))
    initial_tours.append(nearest_neighbor_tour(nodes))
    initial_tours.append(strip_ordering(nodes))
    initial_tours.append(angular_ordering(nodes))
    
    best_tour = initial_tours[0]
    best_length = compute_tour_length(best_tour, nodes)
    
    # Use a random seed to vary the search
    random.seed(args.seed)
    
    # Run multiple restarts
    for restart in range(10):
        # Construct a new initial tour
        initial_tour = random.choice(initial_tours)
        current_tour = np.copy(initial_tour)
        current_length = best_length
        
        # Apply 2-opt until no improvement
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if current_tour[i] == current_tour[j]:
                        continue
                    # Compute the cost of the 2-opt move
                    cost = 0
                    for k in range(i, j):
                        cost += euc_2d_distance(nodes[current_tour[k]][0], nodes[current_tour[k]][1], nodes[current_tour[k+1]][0], nodes[current_tour[k+1]][1])
                    cost -= euc_2d_distance(nodes[current_tour[i]][0], nodes[current_tour[i]][1], nodes[current_tour[j]][0], nodes[current_tour[j]][1])
                    cost -= euc_2d_distance(nodes[current_tour[j]][0], nodes[current_tour[j]][1], nodes[current_tour[i]][0], nodes[current_tour[i]][1])
                    if cost < 0:
                        # Apply the 2-opt move
                        current_tour[i:j+1] = current_tour[j:i-1:-1]
                        current_length += cost
                        improved = True
        
        # Update the best tour if necessary
        if current_length < best_length:
            best_length = current_length
            best_tour = current_tour
    
    return best_tour

def compute_tour_length(tour, nodes):
    length = 0
    n = len(tour)
    for i in range(n - 1):
        x1, y1 = nodes[tour[i]]
        x2, y2 = nodes[tour[i+1]]
        length += euc_2d_distance(x1, y1, x2, y2)
    x1, y1 = nodes[tour[-1]]
    x2, y2 = nodes[tour[0]]
    length += euc_2d_distance(x1, y1, x2, y2)
    return length

def main():
    args = get_arguments()
    nodes = read_tsp_file(args.input)
    start_time = time.time()
    
    tour = solve_tsp(nodes)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Write the tour to the output file
    np.save(args.output, tour, allow_pickle=False)

if __name__ == '__main__':
    main()