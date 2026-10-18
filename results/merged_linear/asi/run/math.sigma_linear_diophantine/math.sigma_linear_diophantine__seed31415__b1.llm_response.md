```python
import json
import sys
import math
import heapq
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict

def main():
    with open('data/problem_instances.json') as f:
        instances = json.load(f)
    
    results = []
    for inst in instances:
        instance_id = inst['instance_id']
        a = inst['a']
        b = inst['b']
        c = inst['c']
        U = inst['U']
        solutions = solve(a, b, c, U)
        for n in sorted(solutions):
            results.append((instance_id, n))
    
    with open('results/solutions.csv', 'w') as f:
        f.write('instance_id,n\n')
        for line in results:
            f.write(','.join(map(str, line)) + '\n')
    
    summary = []
    for inst in instances:
        instance_id = inst['instance_id']
        a = inst['a']
        b = inst['b']
        c = inst['c']
        U = inst['U']
        solutions = solve(a, b, c, U)
        count = len(solutions)
        largest = max(solutions) if count else 0
        summary.append((instance_id, count, largest))
    
    with open('results/instance_summary.csv', 'w') as f:
        f.write('instance_id,solution_count,largest_solution\n')
        for line in sorted(summary, key=lambda x: x[0]):
            f.write(','.join(map(str, line)) + '\n')

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    @lru_cache(maxsize=None)
    def solve_reduced(a, b, c, U, max_factor):
        if U == 0:
            return []
        if a == 0:
            if b == 0:
                return [] if c != 0 else list(range(1, U+1))
            else:
                return [] if c % b != 0 else [c//b]
        if b == 0:
            return [] if c != 0 else list(range(1, U+1))
        
        # Reduce the equation
        gcd_val = math.gcd(a, b)
        a //= gcd_val
        b //= gcd_val
        c //= gcd_val
        
        # Check for trivial solutions
        if a == 1 and b == 1:
            return list(range(1, U+1)) if c == 0 else []
        
        # Try to find solutions using tree search
        return dfs(1, 1, a, b, c, U, max_factor)
    
    def dfs(current_n, current_max_factor, a, b, c, U, max_factor):
        if current_n > U:
            return []
        
        # Check if current_n satisfies the equation
        if a * sum_of_divisors(current_n) == b * current_n + c:
            result = [current_n]
        else:
            result = []
        
        # Generate next factors
        next_factors = []
        next_max_factor = max_factor
        for p in primes:
            if p > next_max_factor:
                break
            if p > current_max_factor:
                next_max_factor = p
                next_factors.append(p)
                break
            if p > next_max_factor:
                break
            next_factors.append(p)
        
        # Continue the search
        for p in next_factors:
            next_n = current_n * p
            next_max_factor = max(current_max_factor, p)
            if next_n > U:
                break
            result += dfs(next_n, next_max_factor, a, b, c, U, max_factor)
        
        return result
    
    def sum_of_divisors(n):
        result = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                result *= (i**(count + 1) - 1) // (i - 1)
            i += 1
        if n > 1:
            result *= (n + 1)
        return result
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    return solve_reduced(a, b, c, U, primes[-1])

if __name__ == "__main__":
    main()
```