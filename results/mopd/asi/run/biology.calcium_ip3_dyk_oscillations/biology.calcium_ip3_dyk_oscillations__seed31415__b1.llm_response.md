```python
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

# Load system parameters
with open('data/system_info.json') as f:
    sys_params = json.load(f)
a1, d1 = float(sys_params['a1']), float(sys_params['d1'])
a2, d2 = float(sys_params['a2']), float(sys_params['d2'])
a3, d3 = float(sys_params['a3']), float(sys_params['d3'])
a4, d4 = float(sys_params['a4']), float(sys_params['d4'])
a5, d5 = float(sys_params['a5']), float(sys_params['d5'])
c0 = float(sys_params['c0'])
c1 = float(sys_params['c1'])
v1 = float(sys_params['v1'])
v2 = float(sys_params['v2'])
v3 = float(sys_params['v3'])
K3 = float(sys_params['K3'])

# Load scan parameters
with open('data/scan_spec.json') as f:
    scan_params = json.load(f)
ip3_levels = [float(x) for x in scan_params['ip3_levels_micromolar']]
channel_counts = [int(x) for x in scan_params['channel_counts']]
n_replicates = int(scan_params['n_replicates'])
T_obs = float(scan_params['T_obs_seconds'])
dt_out = float(scan_params['dt_out_seconds'])
n_samples = int(scan_params['n_samples'])

# Simulation parameters
dt_sim = 0.005
burn_in = int(0.15 * n_samples)
dt_out_sec = dt_out

# Precompute constants
c0 = float(sys_params['c0'])
c1 = float(sys_params['c1'])
c = 0.1
c_L = (c0 - c) / c1

def run_simulation(ip3, N):
    n_sub = 3 * N
    # Initialize subunit states from stationary distribution
    # (This is a placeholder; actual implementation would require solving stationary distribution)
    subunit_states = np.random.randint(0, 8, size=n_sub)
    c = 0.1
    c_L = (c0 - c) / c1
    time = 0.0
    ca_trace = []
    p_open_trace = []
    conservation_error = []
    
    for step in range(n_samples):
        # Build rate matrix Q
        Q = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                # Compute transition rate from state i to j
                # (This is a placeholder; actual implementation would require computing transition rates)
                rate = 0.0
                Q[i, j] = rate
        # Compute transition matrix
        P = scipy.linalg.expm(Q * dt_sim)
        # Sample next subunit states
        subunit_states = np.random.choice(range(8), size=n_sub, p=P[subunit_states, :])
        # Group into channels
        channels = subunit_states.reshape(N, 3)
        # Compute P_open
        p_open = np.mean(channels == 3).item()
        # Update c and c_L
        J_IP3R = c1 * v1 * p_open * (c_L - c)
        J_leak = c1 * v2 * (c_L - c)
        J_SERCA = v3 * c * c / (K3 * K3 + c * c)
        dc = J_IP3R + J_leak - J_SERCA
        dc_L = -(J_IP3R + J_leak - J_SERCA) / c1
        c += dc * dt_sim
        c_L += dc_L * dt_sim
        # Track conservation drift
        conservation_error.append(abs(c + c1 * c_L - c0) / c0)
        # Save c and p_open
        ca_trace.append(c)
        p_open_trace.append(p_open)
        time += dt_sim
        if time >= T_obs:
            break
    # Remove burn-in
    ca_trace = ca_trace[burn_in:]
    p_open_trace = p_open_trace[burn_in:]
    return np.array(ca_trace), np.array(p_open_trace), np.array(conservation_error)

# Run simulations for each grid point
ca_traces = {}
p_open_traces = {}
conservation_errors = {}
for ip3 in ip3_levels:
    for N in channel_counts:
        ca_traces[f'ip3{ip3}_N{N}'] = []
        p_open_traces[f'ip3{ip3}_N{N}'] = []
        conservation_errors[f'ip3{ip3}_N{N}'] = []
        for replicate in range(n_replicates):
            ca, p_open, conservation_error = run_simulation(ip3, N)
            ca_traces[f'ip3{ip3}_N{N}'].append(ca)
            p_open_traces[f'ip3{ip3}_N{N}'].append(p_open)
            conservation_errors[f'ip3{ip3}_N{N}'].append(conservation_error)

# Save results
np.savez('ca_traces.npz', **ca_traces)
np.savez('p_open_traces.npz', **p_open_traces)
np.savez('conservation_errors.npz', **conservation_errors)

# Compute summary statistics
summary = {}
summary['ip3_levels'] = ip3_levels
summary['channel_counts'] = channel_counts
summary['cr_ip3_idx_used'] = np.argmin(np.abs(np.array(ip3_levels) - 0.28))
period_vs_ip3 = []
cv_isi_vs_N = []
p_open_vs_ip3 = []
conservation_error_max = []
for ip3 in ip3_levels:
    for N in channel_counts:
        ca = ca_traces[f'ip3{ip3}_N{N}']
        # Compute period
        peaks = np.where(np.diff(np.sign(ca - np.mean(ca))))[0]
        period = np.median(np.diff(peaks))
        period_vs_ip3.append(period)
        # Compute CV(ISI)
        isi = np.diff(peaks)
        cv_isi = np.std(isi) / np.mean(isi)
        cv_isi_vs_N.append(cv_isi)
        # Compute P_open
        p_open = p_open_traces[f'ip3{ip3}_N{N}']
        p_open_vs_ip3.append(np.mean(p_open))
        # Compute conservation error
        conservation_error = np.max(conservation_errors[f'ip3{ip3}_N{N}'])
        conservation_error_max.append(conservation_error)
summary['period_vs_ip3'] = period_vs_ip3
summary['cv_isi_vs_N'] = cv_isi_vs_N
summary['p_open_vs_ip3'] = p_open_vs_ip3
summary['conservation_error_max'] = conservation_error_max
with open('summary_stats.json', 'w') as f:
    json.dump(summary, f)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(ca_traces['ip30.28_N800'][0])
plt.title('Representative Trace')
plt.subplot(2, 2, 2)
plt.plot(ip3_levels, period_vs_ip3)
plt.title('Period vs IP3')
plt.subplot(2, 2, 3)
plt.plot(channel_counts, cv_isi_vs_N)
plt.title('CV(ISI) vs Channel Count')
plt.subplot(2, 2, 4)
plt.plot(ip3_levels, p_open_vs_ip3)
plt.title('P_open vs IP3')
plt.savefig('calcium_summary.png')
```