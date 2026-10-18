import sys
import os
import numpy as np
import subprocess
import tempfile
import json
import base64
import zlib
from pathlib import Path

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    seed = int(sys.argv[3])
    time_limit = int(sys.argv[4])

    # Parse DIMACS
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Extract n_vars and n_clauses
    header = lines[0].strip()
    n_vars = int(header.split()[-2])
    n_clauses = int(header.split()[-1])

    # Write fallback assignment
    fallback = np.zeros(n_vars, dtype=np.uint8)
    np.save(output_path, fallback)

    # Convert to WCNF
    tmp_dir = tempfile.mkdtemp()
    wcnf_path = os.path.join(tmp_dir, 'instance.wcnf')

    with open(wcnf_path, 'w') as f:
        f.write(f'p wcnf {n_vars} {n_clauses} {n_clauses + 1}\n')
        for line in lines[1:]:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            clause = [int(x) for x in parts if x != '0']
            f.write(' '.join(['1'] + [str(x) for x in clause]) + ' 0\n')

    # Compile and run the solver
    solver_path = os.path.join(tmp_dir, 'spb-maxsat')
    with open('exact_spb_source.json', 'r') as f:
        source = json.load(f)

    # Extract and write the source files
    for filename, content in source.items():
        with open(os.path.join(tmp_dir, filename), 'w') as f:
            f.write(content)

    # Compile the solver
    compile_cmd = [
        'g++', '-std=c++17', '-O3', '-DNDEBUG',
        os.path.join(tmp_dir, 'pms.cpp'), '-o', solver_path
    ]
    subprocess.run(compile_cmd, check=True)

    # Run the solver
    solver_cmd = [
        solver_path,
        '-seed', str(seed),
        '-max_flips', '5000000000',
        '-cutoff', str(time_limit - 10),  # Reserve 10 seconds for final export
        wcnf_path
    ]
    result = subprocess.run(solver_cmd, capture_output=True, text=True)

    # Extract the assignment from the solver output
    assignment_str = None
    for line in result.stdout.split('\n'):
        if line.startswith('v '):
            assignment_str = line[2:]
            break

    if assignment_str:
        assignment = np.array([int(bit) for bit in assignment_str[:n_vars]], dtype=np.uint8)
        np.save(output_path, assignment)

if __name__ == '__main__':
    main()