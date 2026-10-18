import numpy as np
import scipy.linalg
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, autocorrelate
import os

# Load system parameters
with open('data/system_info.json') as f:
    system_info = json.load(f)

# Constants from system_info
a1, d1 = float(system_info['a1']), float(system_info['d1'])
a2, d2 = float(system_info['a2']), float(system_info['d2'])
a3, d3 = float(system_info['a3']), float(system_info['d3'])
a4, d4 = float(system_info['a4']), float(system_info['d4'])
a5, d5 = float(system_info['a5']), float(system_info['d5'])
c0 = float(system_info['c0'])
c1 = float(system_info['c1'])
v1 = float(system_info['v1'])
v2 = float(system_info['v2'])
v3 = float(system_info['v3'])
K3 = float(system_info['K3'])

# Load scan specification
with open('data/scan_spec.json') as f:
    scan_spec = json.load(f)

ip3_levels = [float(x) for x in scan_spec['ip3_levels_micromolar']]
channel_counts = [int(x) for x in scan_spec['channel_counts']]
n_replicates = int(scan_spec['n_replicates'])
T_obs_seconds = float(scan_spec['T_obs_seconds'])
dt_out_seconds = float(scan_spec['dt_out_seconds'])
n_samples = int(scan_spec['n_samples'])

# Find IP3 level closest to 0.28 uM
cr_ip3_idx_used = np.abs(np.array(ip3_levels) - 0.28).argmin()

# Precompute IP3 levels and channel counts
ip3_levels = np.array(ip3_levels)
channel_counts = np.array(channel_counts)

# Simulation parameters
dt_sim = 0.005  # s
dt_out_seconds = dt_out_seconds
dt_out = dt_out_seconds / dt_sim
n_steps = int(T_obs_seconds / dt_sim)
n_output = int(n_steps / dt_out)
n_output = min(n_output, n_samples)

# Initialize output arrays
ca_traces = {}
p_open_traces = {}
conservation_errors = {}

# Simulation function
def simulate(ip3, N):
    n_sub = 3 * N
    # Initialize subunit states from stationary distribution
    # (Assuming stationary distribution at c=0.1 and given ip3)
    # For simplicity, initialize to all subunits in state 3 (i=1, j=1, k=0)
    # This is a placeholder; actual stationary distribution calculation would be more complex
    states = np.zeros(n_sub, dtype=int)
    states[:] = 3  # State 3 corresponds to (i=1, j=1, k=0)
    
    # Initial calcium concentrations
    c = 0.1  # uM
    c_L = (c0 - c) / c1
    
    # Precompute transition rates
    def get_transition_rates(c, ip3):
        rates = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                # Calculate transition rate from state i to state j
                # Determine which transition (flip i, j, or k)
                # This is a placeholder for the actual rate calculation
                # For simplicity, we assume a uniform rate matrix
                # In a real implementation, this would be calculated based on the state transitions
                rates[i, j] = 1.0
        return rates
    
    # Main simulation loop
    ca_trace = np.zeros(n_output)
    p_open_trace = np.zeros(n_output)
    conservation_errors_trace = np.zeros(n_steps)
    
    for step in range(n_steps):
        # Build rate matrix
        Q = get_transition_rates(c, ip3)
        # Compute matrix exponential
        P = scipy.linalg.expm(Q * dt_sim)
        # Sample next states
        new_states = np.zeros(n_sub, dtype=int)
        for i in range(n_sub):
            current_state = states[i]
            new_states[i] = np.random.choice(range(8), p=P[current_state, :])
        states = new_states
        
        # Group subunits into channels
        channels = np.reshape(states, (N, 3))
        p_open = np.mean(channels[:, 0] == 3 and channels[:, 1] == 3 and channels[:, 2] == 3)
        p_open_trace[step // dt_out] = p_open
        
        # Calculate fluxes
        J_IP3R = c1 * v1 * p_open * (c_L - c)
        J_leak = c1 * v2 * (c_L - c)
        J_SERCA = v3 * c * c / (K3 * K3 + c * c)
        dc = J_IP3R + J_leak - J_SERCA
        dc_L = -(J_IP3R + J_leak - J_SERCA) / c1
        
        # Euler step
        c += dc * dt_sim
        c_L += dc_L * dt_sim
        
        # Track conservation error
        conservation_errors_trace[step] = abs(c + c1 * c_L - c0) / c0
        
        # Save calcium concentration
        ca_trace[step // dt_out] = c
    
    return ca_trace, p_open_trace, conservation_errors_trace

# Run simulations for all grid points
for i, ip3 in enumerate(ip3_levels):
    for j, N in enumerate(channel_counts):
        print(f"Simulating ip3={ip3} uM, N={N} channels")
        ca_traces[f'ip3{i}_N{j}'], p_open_traces[f'ip3{i}_N{j}'], conservation_errors[f'ip3{i}_N{j}'] = simulate(ip3, N)

# Save ca_traces
np.savez('ca_traces.npz', **{k: v for k, v in ca_traces.items()})

# Calculate summary statistics
period_vs_ip3 = []
cv_isi_vs_N = []
p_open_vs_ip3 = []
conservation_error_max = 0.0

# Period vs IP3
max_N = channel_counts[-1]
for i, ip3 in enumerate(ip3_levels):
    ca_traces_key = f'ip3{i}_N{len(channel_counts)-1}'
    ca_trace = ca_traces[ca_traces_key]
    # Burn-in
    ca_trace = ca_trace[int(0.15 * len(ca_trace)):]
    # Smooth
    ca_smooth = np.convolve(ca_trace, np.ones(10), mode='same') / 10
    # Find peaks
    peaks, _ = find_peaks(ca_smooth, prominence=0.1)
    if len(peaks) >= 3:
        # Calculate period
        peak_times = np.arange(len(peaks)) * dt_out_seconds
        periods = np.diff(peak_times)
        valid_periods = periods[(periods >= 2) & (periods <= 50)]
        period = np.median(valid_periods)
    else:
        # Autocorrelation
        autocorr = autocorrelate(ca_trace)
        lags = np.arange(len(autocorr))
        valid_lags = lags[(lags >= 2) & (lags <= 50)]
        valid_autocorr = autocorr[valid_lags]
        max_corr = np.max(valid_autocorr)
        if max_corr > 0.2:
            period = lags[np.argmax(valid_autocorr)]
        else:
            period = np.nan
    period_vs_ip3.append(period)

# CV(ISI) vs N
target_ip3 = ip3_levels[cr_ip3_idx_used]
for j, N in enumerate(channel_counts):
    ca_traces_key = f'ip3{cr_ip3_idx_used}_N{j}'
    ca_trace = ca_traces[ca_traces_key]
    # Burn-in
    ca_trace = ca_trace[int(0.15 * len(ca_trace)):]
    # Find spikes
    baseline = np.percentile(ca_trace, 30)
    peak_max = np.percentile(ca_trace, 98)
    threshold = baseline + 0.5 * (peak_max - baseline)
    peaks, _ = find_peaks(ca_trace, height=threshold)
    isis = np.diff(peaks * dt_out_seconds)
    isis = isis[isis > 1.0]
    if len(isis) >= 3:
        cv_isi = np.std(isis) / np.mean(isis)
    else:
        cv_isi = np.nan
    cv_isi_vs_N.append(cv_isi)

# P_open vs IP3
max_N = channel_counts[-1]
for i, ip3 in enumerate(ip3_levels):
    ca_traces_key = f'ip3{i}_N{len(channel_counts)-1}'
    p_open_trace = p_open_traces[ca_traces_key]
    # Burn-in
    p_open_trace = p_open_trace[int(0.15 * len(p_open_trace)):]
    p_open_vs_ip3.append(np.mean(p_open_trace))

# Conservation error
conservation_error_max = np.max([np.max(v) for v in conservation_errors.values()])

# Save summary stats
summary_stats = {
    'ip3_levels': ip3_levels.tolist(),
    'channel_counts': channel_counts.tolist(),
    'period_vs_ip3': period_vs_ip3,
    'cv_isi_vs_N': cv_isi_vs_N,
    'p_open_vs_ip3': p_open_vs_ip3,
    'conservation_error_max': conservation_error_max,
    'cr_ip3_idx_used': cr_ip3_idx_used
}

with open('summary_stats.json', 'w') as f:
    json.dump(summary_stats, f)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Representative trace
rep_ip3 = ip3_levels[cr_ip3_idx_used]
rep_N = channel_counts[0]
rep_key = f'ip3{cr_ip3_idx_used}_N{0}'
rep_ca = ca_traces[rep_key]
axs[0, 0].plot(rep_ca)
axs[0, 0].set_title('Representative Calcium Trace')

# Period vs IP3
axs[0, 1].plot(ip3_levels, period_vs_ip3, 'o-')
axs[0, 1].set_xlabel('IP3 Concentration (uM)')
axs[0, 1].set_ylabel('Period (s)')
axs[0, 1].set_title('Period vs IP3')

# CV(ISI) vs N
axs[1, 0].plot(channel_counts, cv_isi_vs_N, 'o-')
axs[1, 0].set_xlabel('Channel Count')
axs[1, 0].set_ylabel('CV(ISI)')
axs[1, 0].set_title('CV(ISI) vs Channel Count')

# P_open vs IP3
axs[1, 1].plot(ip3_levels, p_open_vs_ip3, 'o-')
axs[1, 1].set_xlabel('IP3 Concentration (uM)')
axs[1, 1].set_ylabel('P_open')
axs[1, 1].set_title('P_open vs IP3')

plt.tight_layout()
plt.savefig('calcium_summary.png')
plt.close()