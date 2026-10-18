```python
import subprocess, shutil, os
import json, csv

def run_ngspice(netlist_file, work_dir=None):
    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)
    if shutil.which("ngspice"):
        cmd = ["ngspice", "-b", netlist_file]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120, cwd=work_dir)
    else:
        raise RuntimeError("ngspice is missing from the task runtime")
    if result.returncode != 0 or "Error" in result.stderr:
        print(f"ngspice FAILED for {netlist_file}:")
        print(result.stderr[:2000])
    else:
        print(f"ngspice OK: {netlist_file}")
    return result

def parse_wrdata(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return rows

def normalize_ac_rows(raw):
    normalized = []
    for row in raw:
        if len(row) >= 3:
            normalized.append([row[0], row[1], row[2]])
    return normalized

def normalize_tran_rows(raw):
    normalized = []
    for row in raw:
        if len(row) >= 2:
            normalized.append([row[0], row[1]])
    return normalized

def normalize_noise_rows(raw):
    normalized = []
    for row in raw:
        if len(row) >= 2:
            normalized.append([row[0], row[1]])
    return normalized

def normalize_swing_rows(raw):
    normalized = []
    for row in raw:
        if len(row) >= 4:
            normalized.append([row[0], row[1], row[2
```