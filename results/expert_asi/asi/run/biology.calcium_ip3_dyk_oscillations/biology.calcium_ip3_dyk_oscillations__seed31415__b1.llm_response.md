```python
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json

# Load parameters
with open('data/system_info.json') as f:
    sys_params = json.load(f)

ip3_levels = [0.1854, 0.2434, 0.28, 0.3886, 0.4987, 0.6616]
channel_counts = [5, 20, 80, 250, 800]
n_replicates = 2
T_obs = 120.0
dt_out = 0.1
n_samples = int(T_obs / dt_out)

# Initialize output arrays
ca_traces = {}
summary = {}

def simulate(ip3, N):
    # Initialize subunits
    n_sub = 3 * N
    states = np.random.randint(0, 8, n_sub)
    c = 0.1
    c_L = (sys_params['c0'] - c) / sys_params['c1']
    time = 0.0
    outputs = []
    burn_in = int(0.15 * n_samples)
    t_out = np.arange(0, T_obs, dt_out)
    t_idx = 0
    
    for step in range(int(T_obs / dt_sim)):
        # Build rate matrix
        Q = build_rate_matrix(c, ip3)
        P = la.expm(Q * dt_sim)
        # Sample next states
        new_states = np.random.choice(8, n_sub, p=P[states, :].T)
        # Group into channels
        channels = np.reshape(new_states, (N, 3))
        # Compute P_open
        p_open = np.mean(np.prod(channels == 3, axis=1))
        # Update calcium concentrations
        J_IP3R = sys_params['c1'] * sys_params['v1'] * p_open * (c_L - c)
        J_leak = sys_params['c1'] * sys_params['v2'] * (c_L - c)
        J_SERCA = sys_params['v3'] * c**2 / (sys_params['K3']**2 + c**2)
        dc = (J_IP3R + J_leak - J_SERCA) * dt_sim
        dc_L = -(J_IP3R + J_leak - J_SERCA) / (sys_params['c1'] * dt_sim)
        c += dc
        c_L += dc_L
        # Track conservation error
        cons_error = abs(c + sys_params['c1']*c_L - sys_params['c0']) / sys_params['c0']
        # Record output
        if step % int(dt_out / dt_sim) == 0:
            outputs.append((time, c, p_open, cons_error))
        time += dt_sim
    
    outputs = np.array(outputs)
    return outputs[:, 1], outputs[:, 2]

def build_rate_matrix(c, ip3):
    a1, d1 = sys_params['a1'], sys_params['d1']
    a3, d3 = sys_params['a3'], sys_params['d3']
    a5, d5 = sys_params['a5'], sys_params['d5']
    a2, d2 = sys_params['a2'], sys_params['d2']
    a4, d4 = sys_params['a4'], sys_params['d4']
    
    rate = np.zeros((8, 8))
    for s in range(8):
        i = s % 2
        j = (s // 2) % 2
        k = (s // 4) % 2
        
        # Flip i
        if k == 0:
            rate[s, s ^ 1] += a1 * ip3 - a1 * d1
        else:
            rate[s, s ^ 1] += a3 * ip3 - a3 * d3
        
        # Flip j
        rate[s, s ^ 2] += a5 * c - a5 * d5
        
        # Flip k
        if i == 0:
            rate[s, s ^ 4] += a4 * c - a4 * d4
        else:
            rate[s, s ^ 4] += a2 * c - a2 * d2
    
    return rate

def compute_period(trace):
    peaks = find_peaks(trace, prominence=0.1)
    if len(peaks) < 3:
        # Use autocorrelation
        ac = np.correlate(trace, trace, mode='full')
        ac = ac[len(ac)//2:]
        lags = np.arange(1, len(ac))
        ac = ac[1:]
        corr = ac / np.sqrt(np.var(trace)**2)
        peaks = find_peaks(corr, height=0.2)
        if not peaks:
            return np.nan
        peaks = peaks[1:]
    else:
        peaks = peaks[1:]
    periods = np.diff(peaks)
    return np.median(periods)

def compute_cv_isi(trace):
    peaks = find_peaks(trace, prominence=0.1)
    if len(peaks) < 3:
        return np.nan
    isis = np.diff(peaks)
    return np.std(isis) / np.mean(isis)

def find_peaks(signal, prominence=0.1):
    # Simple peak detection
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)

# Run simulations
for ip3_idx, ip3 in enumerate(ip3_levels):
    for N in channel_counts:
        for rep in range(n_replicates):
            c_trace, p_open = simulate(ip3, N)
            key = f'ip3{ip3_idx}_N{N}'
            if key not in ca_traces:
                ca_traces[key] = []
            ca_traces[key].append(c_trace)

# Compute summary statistics
period_vs_ip3 = []
cv_isi_vs_N = []
p_open_vs_ip3 = []
conservation_error_max = 0.0

for ip3_idx, ip3 in enumerate(ip3_levels):
    for N in channel_counts:
        c_traces = ca_traces[f'ip3{ip3_idx}_N{N}']
        # Compute period
        period = np.nan
        for c_trace in c_traces:
            peak_idx = np.argmax(c_trace[500:])
            burn_idx = int(0.15 * len(c_trace))
            trace = c_trace[burn_idx:]
            if len(trace) < 200:
                continue
            period = compute_period(trace)
        period_vs_ip3.append(period)
        # Compute CV(ISI)
        cv = np.nan
        for c_trace in c_traces:
            peak_idx = np.argmax(c_trace[500:])
            burn_idx = int(0.15 * len(c_trace))
            trace = c_trace[burn_idx:]
            if len(trace) < 200:
                continue
            cv = compute_cv_isi(trace)
        cv_isi_vs_N.append(cv)
        # Compute P_open
        p_open = np.mean([c_trace[500:] for c_trace in c_traces], axis=0)
        p_open_vs_ip3.append(np.mean(p_open))
        # Compute conservation error
        max_error = np.max(np.abs(c + sys_params['c1']*c_L - sys_params['c0']) / sys_params['c0'])
        conservation_error_max = max(conservation_error_max, max_error)

# Save outputs
np.savez('ca_traces.npz', **ca_traces)
summary['ip3_levels'] = ip3_levels
summary['channel_counts'] = channel_counts
summary['period_vs_ip3'] = period_vs_ip3
summary['cv_isi_vs_N'] = cv_isi_vs_N
summary['p_open_vs_ip3'] = p_open_vs_ip3
summary['conservation_error_max'] = conservation_error_max
summary['cr_ip3_idx_used'] = np.argmin(cv_isi_vs_N)

with open('summary_stats.json', 'w') as f:
    json.dump(summary, f)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(ip3_levels, period_vs_ip3, 'o-')
plt.xlabel('IP3 [μM]')
plt.ylabel('Period [s]')
plt.subplot(2, 2, 2)
plt.plot(channel_counts, cv_isi_vs_N, 'o-')
plt.xlabel('Channel Count')
plt.ylabel('CV(ISI)')
plt.subplot(2, 2, 3)
plt.plot(ip3_levels, p_open_vs_ip3, 'o-')
plt.xlabel('IP3 [μM]')
plt.ylabel('P_open')
plt.subplot(2, 2, 4)
plt.plot(ip3_levels, period_vs_ip3, 'o-')
plt.xlabel('IP3 [μM]')
plt.ylabel('Period [s]')
plt.savefig('calcium_summary.png')
```