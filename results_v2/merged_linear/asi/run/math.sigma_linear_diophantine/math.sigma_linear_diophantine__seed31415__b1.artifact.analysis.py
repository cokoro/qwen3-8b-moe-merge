import json
import math
import heapq
import sys
import itertools

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    solutions = []
    
    def solve_helper(n, a, b, c, U, last_prime):
        if n > U:
            return []
        
        # Base case: check if current n satisfies the equation
        if a * sum_of_divisors(n) == b * n + c:
            solutions.append(n)
            return []
        
        # If we've reached the search bound, stop
        if n > U:
            return []
        
        # Find the next prime to try
        next_prime = find_next_prime(last_prime, U // n)
        if next_prime is None:
            return []
        
        # Try multiplying by powers of the next prime
        for power in range(1, 10):  # Try up to p^9 to stay under U
            next_n = n * (next_prime ** power)
            if next_n > U:
                break
            solve_helper(next_n, a, b, c, U, next_prime)
        
        return solutions
    
    def find_next_prime(last_prime, bound):
        if last_prime == 0:
            return 2
        current = last_prime + 1
        while current <= bound:
            if is_prime(current):
                return current
            current += 1
        return None
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        max_div = int(math.sqrt(n)) + 1
        for i in range(3, max_div, 2):
            if n % i == 0:
                return False
        return True
    
    def sum_of_divisors(n):
        divisors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sum(divisors)
    
    solve_helper(1, a, b, c, U, 0)
    return sorted(solutions)

if __name__ == "__main__":
    with open("data/problem_instances.json") as f:
        instances = json.load(f)
    
    results = []
    for instance in instances:
        instance_id = instance["instance_id"]
        a = instance["a"]
        b = instance["b"]
        c = instance["c"]
        U = instance["U"]
        solutions = solve(a, b, c, U)
        results.extend([(instance_id, n) for n in solutions])
    
    with open("results/solutions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "n"])
        for instance_id, n in sorted(results):
            writer.writerow([instance_id, n])
    
    instance_summary = []
    for instance in instances:
        instance_id = instance["instance_id"]
        a = instance["a"]
        b = instance["b"]
        c = instance["c"]
        U = instance["U"]
        solutions = solve(a, b, c, U)
        instance_summary.append((instance_id, len(solutions), max(solutions) if solutions else 0))
    
    with open("results/instance_summary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "solution_count", "largest_solution"])
        for instance_id, count, largest in sorted(instance_summary):
            writer.writerow([instance_id, count, largest])