```python
import os
import sys
import numpy as np
from subprocess import Popen, PIPE

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    seed = int(sys.argv[3])
    time_limit = float(sys.argv[4])

    # Create a fallback assignment to ensure a valid output exists
    n = 0
    clauses = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip().startswith('p cnf'):
                _, _, n, _ = line.strip().split()
                n = int(n)
            elif line.strip().startswith('0'):
                continue
            else:
                clauses.append(line.strip())
    np.save(output_path, np.zeros(n, dtype=np.uint8))

    # Convert to WCNF and run the solver
    tmp_wcnf = '<abs_path>'
    with open(tmp_wcnf, 'w') as f:
        f.write('p wcnf {} {} {}\n'.format(n, len(clauses), len(clauses)+1))
        for clause in clauses:
            f.write('1 {} {} {} 0\n'.format(*clause.split()))

    solver = Popen(['./spb-maxsat', tmp_wcnf, '-seed', str(seed), '-max_flips', '5000000000', '-cutoff', '5'],
                   stdout=PIPE, stderr=PIPE)
    stdout, stderr = solver.communicate()

    assignment = []
    for line in stdout.decode().split('\n'):
        if line.startswith('v '):
            assignment = [int(c) for c in line[2:].split() if c.isdigit()]
            break

    if len(assignment) != n:
        raise RuntimeError(f"Expected {n} assignments, got {len(assignment)}")

    np.save(output_path, np.array(assignment, dtype=np.uint8))

if __name__ == '__main__':
    main()
```