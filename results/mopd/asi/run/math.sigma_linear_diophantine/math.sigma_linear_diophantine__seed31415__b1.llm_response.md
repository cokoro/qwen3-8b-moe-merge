```python
import json
import math
import sys
import csv

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    def dfs(n, max_factor, current_n, current_a, current_b, current_c, solutions):
        if current_n > U:
            return
        if current_a == 0:
            if current_b * current_n + current_c == 0:
                solutions.append(current_n)
            return
        if current_a == 1:
            if current_b * current_n + current_c == 0:
                solutions.append(current_n)
            return
        if current_b == 0:
            if current_a * current_n == current_c:
                solutions.append(current_n)
            return
        if current_b == 1:
            if current_a * current_n == current_c + current_n:
                solutions.append(current_n)
            return
        if current_c == 0:
            if current_a * current_n == current_b * current_n:
                solutions.append(current_n)
            return
        if current_c == 1:
            if current_a * current_n == current_b * current_n + 1:
                solutions.append(current_n)
            return
        if current_a * current_n == current_b * current_n + current_c:
            solutions.append(current_n)
        # Next prime factor
        for p in range(2, int(math.isqrt(current_n)) + 1):
            if current_n % p == 0:
                # Check if p is a new prime factor
                new_n = current_n * p
                if new_n > U:
                    continue
                new_a = current_a
                new_b = current_b
                new_c = current_c
                if current_a % p == 0:
                    new_a //= p
                if current_b % p == 0:
                    new_b //= p
                if current_c % p == 0:
                    new_c //= p
                dfs(new_n, p, new_n, new_a, new_b, new_c, solutions)
        return solutions
    
    solutions = []
    dfs(1, 1, 1, a, b, c, solutions)
    return solutions

def main():
    with open('data/problem_instances.json', 'r') as f:
        instances = json.load(f)
    with open('results/solutions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_id', 'n'])
        for instance in instances:
            solutions = solve(instance['a'], instance['b'], instance['c'], instance['U'])
            for n in solutions:
                writer.writerow([instance['instance_id'], n])
    with open('results/instance_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_id', 'solution_count', 'largest_solution'])
        for instance in instances:
            solutions = solve(instance['a'], instance['b'], instance['c'], instance['U'])
            solution_count = len(solutions)
            largest_solution = max(solutions) if solution_count > 0 else 0
            writer.writerow([instance['instance_id'], solution_count, largest_solution])

if __name__ == '__main__':
    main()
```