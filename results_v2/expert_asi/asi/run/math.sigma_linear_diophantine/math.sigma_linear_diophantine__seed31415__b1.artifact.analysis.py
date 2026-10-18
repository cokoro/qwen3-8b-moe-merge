import sys
import json
import math
import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

@dataclass
class Node:
    value: int
    primes: List[int]
    max_prime: int
    current_n: int
    a_coeff: int
    b_coeff: int
    c_coeff: int
    depth: int

def solve(a: int, b: int, c: int, U: int) -> List[int]:
    results = []
    initial = Node(
        value=1,
        primes=[],
        max_prime=0,
        current_n=1,
        a_coeff=a,
        b_coeff=b,
        c_coeff=c,
        depth=0
    )
    stack = [initial]
    
    while stack:
        node = stack.pop()
        if node.current_n > U:
            continue
        if node.value == 0:
            continue
        
        # Check if current node is a solution
        if node.a_coeff * (node.current_n * (node.current_n + 1) // 2) == node.b_coeff * node.current_n + node.c_coeff:
            results.append(node.current_n)
            continue
        
        # Generate next primes
        next_primes = []
        max_p = node.max_prime
        if not node.primes:
            # First prime
            next_primes = [2]
        else:
            for p in range(node.max_prime + 1, int(math.isqrt(node.current_n)) + 1):
                if p not in node.primes:
                    next_primes.append(p)
        
        # Try each next prime
        for p in next_primes:
            # Compute next n and coefficients
            new_n = node.current_n * p
            if new_n > U:
                continue
            new_a = node.a_coeff * p
            new_b = node.b_coeff * p
            new_c = node.c_coeff * p
            
            # Check if new coefficients are reducible
            g = math.gcd(new_a, new_b)
            if g != 1:
                new_a //= g
                new_b //= g
                new_c //= g
            
            # Check for early pruning
            if new_a == 0:
                if new_b == 0:
                    if new_c == 0:
                        results.append(new_n)
                    continue
                else:
                    if new_c % new_b == 0:
                        if new_c // new_b == new_n:
                            results.append(new_n)
                    continue
            
            # Add to stack
            new_node = Node(
                value=new_n,
                primes=node.primes + [p],
                max_prime=p,
                current_n=new_n,
                a_coeff=new_a,
                b_coeff=new_b,
                c_coeff=new_c,
                depth=node.depth + 1
            )
            stack.append(new_node)
    
    return results

def main():
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    with open(input_path) as f:
        data = json.load(f)
    
    solutions = []
    summaries = []
    for instance in data['instances']:
        instance_id = instance['instance_id']
        a = instance['a']
        b = instance['b']
        c = instance['c']
        U = instance['U']
        solutions_list = solve(a, b, c, U)
        solutions.extend([(instance_id, n) for n in solutions_list])
        summaries.append((instance_id, len(solutions_list), max(solutions_list) if solutions_list else 0))
    
    # Write output files
    with open('results/solutions.csv', 'w') as f:
        for instance_id, n in sorted(solutions, key=lambda x: (x[0], x[1])):
            f.write(f"{instance_id},{n}\n")
    
    with open('results/instance_summary.csv', 'w') as f:
        for instance_id, count, largest in sorted(summaries, key=lambda x: x[0]):
            f.write(f"{instance_id},{count},{largest}\n")

if __name__ == "__main__":
    main()