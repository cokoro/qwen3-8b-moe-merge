```python
import json
import math
import sys
import os
from typing import List, Tuple, Set

def solve(a: int, b: int, c: int, U: int) -> List[int]:
    solutions = []
    visited = set()
    
    def dfs(n: int, current_factors: List[int], current_gcd: int, current_sum: int, current_max_prime: int):
        if n > U:
            return
        
        if n in visited:
            return
        
        visited.add(n)
        
        # Check if current n satisfies the equation
        if a * current_sum == b * n + c:
            solutions.append(n)
        
        # Generate next primes to multiply with
        next_primes = []
        current_max_prime_squared = current_max_prime ** 2
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                next_primes.append(p)
        
        next_primes = list(set(next_primes))
        next_primes.sort()
        
        for p in next_primes:
            if p > current_max_prime:
                new_n = n * p
                if new_n > U:
                    continue
                new_factors = current_factors + [p]
                new_sum = current_sum * (p + 1)
                new_max_prime = p
                dfs(new_n, new_factors, current_gcd, new_sum, new_max_prime)
    
    dfs(1, [], 1, 1, 1)
    
    return solutions

def main():
    if not os.path.exists('data/problem_instances.json'):
        print("Error: data/problem_instances.json not found")
        return
    
    with open('data/problem_instances.json', 'r') as f:
        instances = json.load(f)
    
    solutions = []
    instance_summary = []
    
    for instance in instances:
        instance_id = instance['instance_id']
        a = instance['a']
        b = instance['b']
        c = instance['c']
        U = instance['U']
        
        results = solve(a, b, c, U)
        
        solutions.extend([(instance_id, n) for n in results])
        instance_summary.append((instance_id, len(results), max(results) if results else 0))
    
    # Write solutions to CSV
    with open('results/solutions.csv', 'w') as f:
        f.write('instance_id,n\n')
        for instance_id, n in solutions:
            f.write(f"{instance_id},{n}\n")
    
    # Write instance summary to CSV
    with open('results/instance_summary.csv', 'w') as f:
        f.write('instance_id,solution_count,largest_solution\n')
        for instance_id, count, largest in instance_summary:
            f.write(f"{instance_id},{count},{largest}\n")

if __name__ == "__main__":
    main()
```