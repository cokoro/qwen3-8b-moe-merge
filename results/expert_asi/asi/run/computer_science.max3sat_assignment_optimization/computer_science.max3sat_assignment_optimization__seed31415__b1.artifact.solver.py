import numpy as np
import subprocess
import sys
import os

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Max-SAT solver wrapper')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        header = f.readline()
        while header.strip() == '':
            header = f.readline()
        _, _, n_vars, n_clauses = map(int, header.split())
        clauses = []
        for _ in range(n_clauses):
            line = f.readline()
            while line.strip() == '':
                line = f.readline()
            clauses.append(line.strip())

    # Write fallback assignment
    np.save(args.output, np.zeros(n_vars, dtype=np.uint8))

    # Write WCNF
    tmp_wcnf = '/tmp/instance.wcnf'
    with open(tmp_wcnf, 'w') as f:
        f.write(f'p wcnf {n_vars} {n_clauses} {n_clauses + 1}\n')
        for clause in clauses:
            f.write(f'1 {clause} 0\n')

    # Run SPB-MaxSAT
    cmd = ['./spb-maxsat', tmp_wcnf, f'-seed {args.seed}', '-max_flips 5000000000', '-cutoff 500']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    # Parse assignment
    assignment = np.array([int(c) for c in stdout.strip().split()], dtype=np.uint8)
    np.save(args.output, assignment)

if __name__ == '__main__':
    main()