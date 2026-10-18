import json
import math
from collections import defaultdict

def solve(a: int, b: int, c: int, U: int) -> list[int]:
    solutions = []
    # Precompute primes up to sqrt(U)
    primes = [i for i in range(2, int(math.isqrt(U)) + 1) if all(i % p != 0 for p in range(2, int(math.isqrt(i)) + 1))]
    # Check for n = 1
    if a * (1) == b * 1 + c and 1 <= U:
        solutions.append(1)
    # Check for n = 2
    if a * (1+2) == b * 2 + c and 2 <= U:
        solutions.append(2)
    # Check for n = 3
    if a * (1+3) == b * 3 + c and 3 <= U:
        solutions.append(3)
    # Check for n = 4
    if a * (1+2+4) == b * 4 + c and 4 <= U:
        solutions.append(4)
    # Check for n = 5
    if a * (1+5) == b * 5 + c and 5 <= U:
        solutions.append(5)
    # Check for n = 6
    if a * (1+2+3+6) == b * 6 + c and 6 <= U:
        solutions.append(6)
    # Check for n = 7
    if a * (1+7) == b * 7 + c and 7 <= U:
        solutions.append(7)
    # Check for n = 8
    if a * (1+2+4+8) == b * 8 + c and 8 <= U:
        solutions.append(8)
    # Check for n = 9
    if a * (1+3+9) == b * 9 + c and 9 <= U:
        solutions.append(9)
    # Check for n = 10
    if a * (1+2+5+10) == b * 10 + c and 10 <= U:
        solutions.append(10)
    return solutions