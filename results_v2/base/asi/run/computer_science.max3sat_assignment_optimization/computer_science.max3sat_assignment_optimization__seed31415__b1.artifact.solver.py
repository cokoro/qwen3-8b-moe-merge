import sys
import os
import numpy as np
import subprocess
import json
import base64
import zlib
import argparse
from pathlib import Path

def decode_source():
    source_data = sys.stdin.read()
    source_json = json.loads(zlib.decompress(base64.b64decode(source_data)))
    return source_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--time-limit-seconds', type=int, required=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    seed = args.seed
    time_limit = args.time_limit_seconds

    # Parse DIMACS input
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Find n_variables and n_clauses
    n_vars = 0
    n_clauses = 0
    for line in lines:
        line = line.strip()
        if line.startswith('p cnf'):
            parts = line.split()
            n_vars = int(parts[2])
            n_clauses = int(parts[3])
            break

    # Write fallback assignment
    fallback = np.zeros(n_vars, dtype=np.uint8)
    np.save(output_path, fallback)

    # Convert to WCNF
    wcnf_path = os.path.join('/tmp', 'instance.wcnf')
    with open(wcnf_path, 'w') as f:
        f.write(f'p wcnf {n_vars} {n_clauses} {n_clauses + 1}\n')
        for line in lines:
            line = line.strip()
            if line.startswith('c') or line.startswith('p'):
                continue
            parts = line.split()
            clause = ' '.join(parts) + ' 0\n'
            f.write(clause)

    # Compile and run the solver
    source_json = decode_source()
    source_dir = '/tmp/spb_source'
    os.makedirs(source_dir, exist_ok=True)
    for filename, content in source_json.items():
        with open(os.path.join(source_dir, filename), 'w') as f:
</think>