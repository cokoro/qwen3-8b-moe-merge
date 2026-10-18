```python
import json
import sys
import time
import math
from math import gcd
from collections import defaultdict

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    """Solve the equation a*sigma(n) = b*n + c for n in [1, U]."""
    def is_valid(n):
        s = sum(divisors(n))
        return a * s == b * n + c

    def generate_candidate(n):
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                yield p * n

    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]

    primes = sieve(1000)
    candidates = set()
    for n in range(1, U+1):
        if is_valid(n):
            candidates.add(n)
    return sorted(candidates)

def main():
    with open('data/problem_instances.json') as f:
        instances = json.load(f)
    results = []
    for inst in instances:
        instance_id = inst['instance_id']
        a, b, c, U = inst['a'], inst['b'], inst['c'], inst['U']
        solutions = solve(a, b, c, U)
        results.extend((instance_id, n) for n in solutions)
    with open('results/solutions.csv', 'w') as f:
        for row in sorted(results):
            f.write(f"{row[0]},{row[1]}\n")
    with open('results/instance_summary.csv', 'w') as f:
        for inst in instances:
            solutions = solve(inst['a'], inst['b'], inst['c'], inst['U'])
            f.write(f"{inst['instance_id']},{len(solutions)},{max(solutions) if solutions else 0}\n")

if __name__ == '__main__':
    main()
```