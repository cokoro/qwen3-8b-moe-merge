```python
import subprocess, shutil, os
import json, csv
import numpy as np

def run_ngspice(netlist_file, work_dir=None):
    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)
    if shutil.which("ngspice"):
        cmd = ["ngspice", "-b", netlist_file]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=120, cwd=work_dir)
    else:
        raise RuntimeError("ng

```