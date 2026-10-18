import json
import math
import sys
import heapq
from collections import defaultdict

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    def sum_of_divisors(n):
        total = 1
        factor = 2
        while factor * factor <= n:
            if n % factor == 0:
                count = 0
                while n % factor == 0:
                    n //= factor
                    count += 1
                total *= (factor**(count + 1) - 1) // (factor - 1)
            factor += 1
        if n > 1:
            total *= (n + 1)
        return total

    def solve_helper(n, current_n, current_factors, current_gcd, remaining, max_n):
        if current_n > max_n:
            return []
        if current_n == 0:
            return []
        if remaining == 0:
            if a * sum_of_divisors(current_n) == b * current_n + c:
                return [current_n]
            return []
        res = []
        if current_n > 1:
            res += solve_helper(n, current_n * 2, current_factors + [2], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 3, current_factors + [3], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 5, current_factors + [5], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 7, current_factors + [7], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 11, current_factors + [11], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 13, current_factors + [13], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 17, current_factors + [17], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 19, current_factors + [19], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 23, current_factors + [23], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 29, current_factors + [29], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 31, current_factors + [31], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 37, current_factors + [37], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 41, current_factors + [41], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 43, current_factors + [43], current_gcd, remaining - 1, max_n)
            res += solve_helper(n, current_n * 47, current_factors + [47], current_gcd, remaining - 1, max_n)
        return res

    solutions = []
    for n in range(1, U + 1):
        if a * sum_of_divisors(n) == b * n + c:
            solutions.append(n)
    return solutions

def main():
    import os
    import csv

    if not os.path.exists('data/problem_instances.json'):
        print("Error: data/problem_instances.json not found")
        return

    with open('data/problem_instances.json', 'r') as f:
        instances = json.load(f)

    solutions = []
    for instance in instances:
        instance_id = instance['instance_id']
        a = instance['a']
        b = instance['b']
        c = instance['c']
        U = instance['U']
        results = solve(a, b, c, U)
        solutions.extend([(instance_id, n) for n in results])

    with open('results/solutions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_id', 'n'])
        for instance_id, n in solutions:
            writer.writerow([instance_id, n])

    instance_summary = defaultdict(lambda: {'solution_count': 0, 'largest_solution': 0})
    for instance_id, n in solutions:
        instance_summary[instance_id]['solution_count'] += 1
        if n > instance_summary[instance_id]['largest_solution']:
            instance_summary[instance_id]['largest_solution'] = n

    with open('results/instance_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_id', 'solution_count', 'largest_solution'])
        for instance_id in sorted(instance_summary.keys()):
            writer.writerow([instance_id, instance_summary[instance_id]['solution_count'], instance_summary[instance_id]['largest_solution']])

if __name__ == "__main__":
    main()