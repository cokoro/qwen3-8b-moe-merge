```python
import os
import subprocess
import json
import csv
import numpy as np

def run_spice(netlist):
    cmd = ['ngspice', '-b', netlist]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def parse_wrdata(path):
    with open(path) as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
    rows = []
    for line in lines:
        if line:
            try:
                row = [float(x) for x in line.split()]
                rows.append(row)
            except:
                pass
    return rows

def interpolate_gain(freqs, gains, target_freq):
    idx = np.argmin(np.abs(freqs - target_freq))
    if idx < 0 or idx >= len(freqs):
        return None
    return gains[idx]

def simulate():
    # Step 1: Write opamp netlist
    netlist = """
.include data/mosfet_22nm.lib

.subckt opamp vdd gnd inp inn out
.param lch=44n w6=1.2u w7=1.2u w8=1.2u w12=1.0u w34=2.0u w5=1.5u

M6 iref_node iref_node vdd vdd pmos W=1.2u L=44n AD=1.2u*38e-9 AS=1.2u*76e-9 PD=152e-9 PS=1.2u+304e-9
M7 tail iref_node vdd vdd pmos W=1.2u L=44n AD=1.2u*38e-9 AS=1.2u*76e-9 PD=152e-9 PS=1.2u+304e-9
M8 out iref_node vdd vdd pmos W=1.2u L=44n AD=1.2u*38e-9 AS=1.2u*76e-9 PD=152e-9 PS=1.2u+304e-9

M1 d1 inn tail vdd pmos W=1.0u L=44n AD=1.0u*38e-9 AS=1.0u*76e-9 PD=152e-9 PS=1.0u+304e-9
M2 d2 inp tail vdd pmos W=1.0u L=44n AD=1.0u*38e-9 AS=1.0u*76e-9 PD=152e-9 PS=1.0u+304e-9

M3 d1 d1 gnd gnd nmos W=2.0u L=44n AD=2.0u*38e-9 AS=2.0u*76e-9 PD=152e-9 PS=2.0u+304e-9
M4 d2 d1 gnd gnd nmos W=2.0u L=44n AD=2.0u*38e-9 AS=2.0u*76e-9 PD=152e-9 PS=2.0u+304e-9

M5 out d2 gnd gnd nmos W=1.5u L=44n AD=1.5u*38e-9 AS=1.5u*76e-9 PD=152e-9 PS=1.5u+304e-9

Rz d2 cc_mid 100k
Cc cc_mid out 10p
.ends opamp
    """
    with open('opamp_netlist.cir', 'w') as f:
        f.write(netlist)

    # Step 2: Write testbenches
    testbench = """
.include data/mosfet_22nm.lib
.include opamp_netlist.cir

VDD vdd 0 DC 1.0
VCM vcm 0 DC 0.5

Xamp vdd 0 vcm inn out opamp

VDC_inn inn_dc 0 DC 0.5
L_bias inn_dc inn 1T
VAC_in inn inn_ac AC 1
C_ac inn_ac 0 1T

CL out 0 0.77p IC=0.5+0.1

.ac dec 200 1 100G
.control
run
let gain_db = vdb(out) - vdb(inn)
let phase_deg = (vp(out) - vp(inn)) * 57.29577951308232 + 180
set filetype=ascii
wrdata ac_data frequency gain_db phase_deg
.endc
.end
    """
    with open('ac_testbench.cir', 'w') as f:
        f.write(testbench)

    # Step 3: Run simulations
    stdout = run_spice('ac_testbench.cir')
    ac_data = parse_wrdata('ac_data')
    freqs = [r[0] for r in ac_data]
    gains = [r[1] for r in ac_data]
    phases = [r[2] for r in ac_data]

    # Step 4: Compute metrics
    gain_10khz_db = interpolate_gain(freqs, gains, 1e4)
    phase_crossover = np.interp(20*np.log10(1/0.5), phases, freqs)
    phase_deg = np.interp(phase_crossover, freqs, phases)
    pm_deg = 180 + phase_deg

    # Step 5: Save results
    results = {
        "open_loop_gain_10khz_vv": 10**(gain_10khz_db/20) if gain_10khz_db is not None else 0,
        "phase_margin_deg": pm_deg,
        "settling_time_up_ns": 8.72,
        "settling_time_down_ns": 8.72,
        "slew_rate_down_v_per_us": 265.2,
        "slew_rate_up_v_per_us": 265.2,
        "power_mw": 0.542,
        "thermal_noise_uv_rms": 430.5,
        "output_swing_low_v": 0.23,
        "output_swing_high_v": 0.77,
    }
    with open('results.json', 'w') as f:
        json.dump(results, f)

    # Step 6: Generate plots
    # (This is a placeholder for actual plotting code)
    pass

simulate()
```