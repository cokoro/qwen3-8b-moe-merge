```bash
#!<abs_path>

# simulation.py
import subprocess, shutil, os

def run_ngspice(netlist_file, work_dir=None):
    """Run ngspice in batch mode using the native binary in the task runtime."""
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

# Example usage:
run_ngspice("ac_testbench.cir")
```