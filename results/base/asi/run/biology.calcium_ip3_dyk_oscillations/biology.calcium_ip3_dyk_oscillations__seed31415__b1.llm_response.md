```python
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import pearsonr

# Load system parameters
with open('data/system_info.json') as f:
    sys_params = json.load(f)

# Load scan specification
with open('data/scan_spec.json') as f:
    scan_spec = json.load(f)

# Constants
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
dt_sim = 0.005
dt_out = float(scan_spec['dt_out_seconds'])
T_obs = float(scan_spec['T_obs_seconds'])
n_replicates = int(scan_spec['n_replicates'])
ip3_levels = [float(x) for x in scan_spec['ip3_levels_micromolar']]
channel_counts = [int(x) for x in scan_spec['channel_counts']]
n_samples = int(scan_spec['n_samples'])

# Precompute IP3 levels and channel counts
ip3_levels_sorted = sorted(ip3_levels)
cr_ip3_idx_used = np.argmin(np.abs(np.array(ip3_levels_sorted) - 0.28))

# Define subunit state transitions
def get_transition_rates(c, ip3, subunit_state):
    i, j, k = subunit_state
    rates = []
    if k == 0:
        rates.append((a1 * ip3, a1 * d1, 'i'))
    if k == 1:
        rates.append((a3 * ip3, a3 * d3, 'i'))
    rates.append((a5 * c, a5 * d5, 'j'))
    if i == 0:
        rates.append((a4 * c, a4 * d4, 'k'))
    if i == 1:
        rates.append((a2 * c, a2 * d2, 'k'))
    return rates

def build_rate_matrix(c, ip3, n_subunits):
    Q = np.zeros((8, 8))
    for s in range(8):
        i, j, k = s // 4, (s % 4) // 2, s % 2
        transitions = get_transition_rates(c, ip3, (i, j, k))
        for rate, reverse_rate, type_ in transitions:
            for delta in [-1, 1]:
                next_s = s + delta
                if 0 <= next_s < 8:
                    if type_ == 'i':
                        if (i + delta) % 2 == 1:
                            Q[s, next_s] += rate
                            Q[next_s, s] += reverse_rate
                    elif type_ == 'j':
                        if (j + delta) % 2 == 1:
                            Q[s, next_s] += rate
                            Q[next_s, s] += reverse_rate
                    elif type_ == 'k':
                        if (k + delta) % 2 == 1:
                            Q[s, next_s] += rate
                            Q[next_s, s] += reverse_rate
    return Q

def simulate(ip3, N, c0, c1, v1, v2, v3, K3, dt_sim, dt_out, T_obs, n_replicates):
    n_subunits = 3 * N
    c = 0.1
    c_L = (c0 - c) / c1
    ca_total_0 = c + c1 * c_L
    times = []
    c_traces = []
    p_open_traces = []
    conservation_errors = []

    for _ in range(n_replicates):
        current_states = np.random.randint(0, 8, n_subunits)
        time = 0.0
        c_trace = []
        p_open_trace = []
        conservation_error_trace = []
        while time < T_obs:
            Q = build_rate_matrix(c, ip3, n_subunits)
            P = expm(Q * dt_sim)
            next_states = []
            for s in current_states:
                next_states.append(np.random.choice(8, p=P[s, :]))
            current_states = np.array(next_states)
            channels = np.reshape(current_states, (N, 3))
            p_open = np.mean(channels[:, 0] == 1 and channels[:, 1] == 1 and channels[:, 2] == 0 for _ in channels)
            p_open_trace.append(p_open)
            # Calculate fluxes
            J_IP3R = c1 * v1 * p_open * (c_L - c)
            J_leak = c1 * v2 * (c_L - c)
            J_SERCA = v3 * c * c / (K3 * K3 + c * c)
            dc = J_IP3R + J_leak - J_SERCA
            dc_L = -(J_IP3R + J_leak - J_SERCA) / c1
            c += dc * dt_sim
            c_L += dc_L * dt_sim
            # Track conservation error
            conservation_error = abs(c + c1 * c_L - ca_total_0) / ca_total_0
            conservation_error_trace.append(conservation_error)
            # Save data
            if time >= 0.0 and (time - int(time)) < dt_out:
                c_trace.append(c)
                times.append(time)
            time += dt_sim
        c_trace = np.array(c_trace)
        p_open_trace = np.array(p_open_trace)
        conservation_error_trace = np.array(conservation_error_trace)
        c_traces.append(c_trace)
        p_open_traces.append(p_open_trace)
        conservation_errors.append(conservation_error_trace)
    return c_traces, p_open_traces, conservation_errors

# Run simulations
ca_traces = {}
p_open_traces = {}
conservation_errors = {}
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        print(f"Simulating IP3={ip3} uM, N={N}")
        c_traces, p_open_traces, conservation_errors = simulate(ip3, N, c0, c1, v1, v2, v3, K3, dt_sim, dt_out, T_obs, n_replicates)
        ca_traces[f'ip3{ip3}_N{N}'] = np.array(c_traces)
        p_open_traces[f'ip3{ip3}_N{N}'] = np.array(p_open_traces)
        conservation_errors[f'ip3{ip3}_N{N}'] = np.array(conservation_errors)

# Save results
np.savez('ca_traces.npz', **ca_traces)

# Compute summary statistics
period_vs_ip3 = []
cv_isi_vs_N = []
p_open_vs_ip3 = []
conservation_error_max = 0.0

# Period vs IP3
max_N = max(channel_counts)
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        if N == max_N:
            traces = ca_traces[f'ip3{ip3}_N{N}']
            periods = []
            for trace in traces:
                time = np.arange(0, T_obs, dt_out)
                filtered_trace = trace[int(0.15 * len(trace)):]
                peaks, _ = find_peaks(filtered_trace, prominence=0.1)
                if len(peaks) >= 3:
                    spacing = np.diff(time[peaks])
                    valid_spacing = spacing[(spacing >= 2) & (spacing <= 50)]
                    if len(valid_spacing) >= 2:
                        periods.append(np.median(valid_spacing))
                else:
                    filtered_trace -= np.mean(filtered_trace)
                    corr = np.correlate(filtered_trace, filtered_trace, mode='full')
                    lags = np.arange(-len(filtered_trace) + 1, len(filtered_trace))
                    valid_lags = lags[(lags >= 2) & (lags <= 50)]
                    valid_corr = corr[valid_lags]
                    if len(valid_corr) >= 2:
                        max_corr = np.max(valid_corr)
                        if max_corr > 0.2:
                            periods.append(np.argwhere(valid_corr == max_corr)[0][0])
            if periods:
                period_vs_ip3.append(np.median(periods))
            else:
                period_vs_ip3.append(None)

# CV(ISI) vs N
target_ip3 = ip3_levels_sorted[cr_ip3_idx_used]
for N in channel_counts:
    traces = ca_traces[f'ip3{target_ip3}_N{N}']
    isis = []
    for trace in traces:
        time = np.arange(0, T_obs, dt_out)
        filtered_trace = trace[int(0.15 * len(trace)):]
        baseline = np.percentile(filtered_trace, 30)
        peak_max = np.percentile(filtered_trace, 98)
        threshold = baseline + 0.5 * (peak_max - baseline)
        peaks, _ = find_peaks(filtered_trace, height=threshold)
        isis.extend(np.diff(time[peaks]))
    if len(isis) >= 3:
        cv_isi = np.std(isis) / np.mean(isis)
        cv_isi_vs_N.append(cv_isi)
    else:
        cv_isi_vs_N.append(None)

# P_open vs IP3
max_N = max(channel_counts)
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        if N == max_N:
            traces = p_open_traces[f'ip3{ip3}_N{N}']
            p_open = np.mean([np.mean(trace[int(0.15 * len(trace)):]) for trace in traces])
            p_open_vs_ip3.append(p_open)

# Conservation error
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        errors = conservation_errors[f'ip3{ip3}_N{N}']
        conservation_error_max = max(conservation_error_max, np.max(errors))

# Save summary statistics
summary_stats = {
    'ip3_levels': ip3_levels_sorted,
    'channel_counts': channel_counts,
    'period_vs_ip3': period_vs_ip3,
    'cv_isi_vs_N': cv_isi_vs_N,
    'p_open_vs_ip3': p_open_vs_ip3,
    'conservation_error_max': conservation_error_max,
    'cr_ip3_idx_used': cr_ip3_idx_used
}
with open('summary_stats.json', 'w') as f:
    json.dump(summary_stats, f)

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        if N == max(channel_counts):
            traces = ca_traces[f'ip3{ip3}_N{N}']
            for trace in traces:
                plt.plot(np.arange(0, T_obs, dt_out), trace, alpha=0.3)
plt.title('Representative Calcium Trace')

plt.subplot(2, 2, 2)
periods = [p for p in period_vs_ip3 if p is not None]
ips = [ip for ip in ip3_levels_sorted if period_vs_ip3[ip3_levels_sorted.index(ip)] is not None]
plt.plot(ips, periods, marker='o')
plt.title('Period vs IP3')

plt.subplot(2, 2, 3)
for i, N in enumerate(channel_counts):
    if cv_isi_vs_N[i] is not None:
        plt.plot(N, cv_isi_vs_N[i], marker='o')
plt.title('CV(ISI) vs Channel Count')

plt.subplot(2, 2, 4)
for ip3 in ip3_levels_sorted:
    for N in channel_counts:
        if N == max(channel_counts):
            p_open = p_open_vs_ip3[ip3_levels_sorted.index(ip3)]
            plt.plot(ip3, p_open, marker='o')
plt.title('P_open vs IP3')

plt.tight_layout()
plt.savefig('calcium_summary.png')
plt.close()
```