```python
import sys
import os
import numpy as np
import subprocess
import tempfile
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_vars = int(header[2])
    n_clauses = int(header[3])

    tmp_dir = tempfile.mkdtemp()
    wcnf_path = os.path.join(tmp_dir, 'input.wcnf')
    with open(wcnf_path, 'w') as f:
        f.write(f'p wcnf {n_vars} {n_clauses} {n_clauses}\n')
        for line in lines[1:]:
            if line.strip().startswith('c'):
                continue
            parts = line.strip().split()
            clause = [int(x) for x in parts if x]
            clause.append(0)
            f.write(' '.join(map(str, clause)) + '\n')

    solver_path = os.path.join(tmp_dir, 'spb-maxsat')
    if not os.path.exists(solver_path):
        raise RuntimeError("Precompiled solver binary not found")

    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        solver_path,
        '-seed', str(args.seed),
        '-max_flips', '5000000000',
        '-cutoff', str(args.time_limit_seconds - 10),
        '-log', '0',
        '-output', output `output_path`,
        wcnf_path
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Solver failed: {stderr.decode()}")

    assignment = np.load(output_path)
    np.save(args.output, assignment)

if __name__ == '__main__':
    main()
```