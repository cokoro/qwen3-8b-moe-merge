import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
import json
import os

# Load data
with open("data/system_info.json") as f:
    system_info = json.load(f)
with open("data/scan_spec.json") as f:
    scan_spec = json.load(f)

# Constants
a1, d1 = float(system_info["a1"]), float(system_info["d1"])
a2, d2 = float(system_info["a2"]), float(system_info["d2"])
a3, d3 = float(system_info["a3"]), float(system_info["d3"])
a4, d4 = float(system_info["a4"]), float(system_info["d4"])
a5, d5 = float(system_info["a5"]), float(system_info["d5"])
c0 = float(system_info["c0"])
c1 = float(system_info["c1"])
v1 = float(system_info["v1"])
v2 = float(system_info["v2"])
v3 = float(system_info["v3"])
K3 = float(system_info["K3"])
dt_sim = 0.005
dt_out = scan_spec["dt_out_seconds"]
T_obs = scan_spec["T_obs_seconds"]
n_replicates = scan_spec["n_replicates"]
ip3_levels = np.array(scan_spec["ip3_levels_micromolar"])
channel_counts = np.array(scan_spec["channel_counts"])
n_samples = int(T_obs / dt_out)
n_grid = len(ip3_levels) * len(channel_counts)
ca_traces = np.zeros((n_replicates, n_samples, n_grid))
period_vs_ip3 = np.zeros(len(ip3_levels))
cv_isi_vs_N = np.zeros(len(channel_counts))
p_open_vs_ip3 = np.zeros(len(ip3_levels))
conservation_error_max = 0.0

# Main simulation loop
for idx, (ip3, N) in enumerate(zip(ip3_levels, channel_counts)):
    for replicate in range(n_replicates):
        # Initialize states
        states = np.random.randint(0, 8, size=3*N)
        c = 0.1
        c_L = (c0 - c) / c1
        c_history = np.zeros(n_samples)
        p_open_history = np.zeros(n_samples)
        conservation_error_history = np.zeros(n_samples)
        burn_in = int(0.15 * n_samples)
        for t in range(n_samples):
            # Build rate matrix
            Q = np.zeros((8, 8))
            for s in range(8):
                for s_new in range(8):
                    if s == s_new:
                        continue
                    rate = 0.0
                    # Transition rules
                    i = s % 2
                    j = (s // 2) % 2
                    k = (s // 4) % 2
                    # Flip i
                    if i == 0:
                        rate += a1 * ip3 if k == 0 else a3 * ip3
                    else:
                        rate += a1 * d1 if k == 0 else a3 * d3
                    # Flip j
                    rate += a5 * c if j == 0 else a5 * d5
                    # Flip k
                    if i == 0:
                        rate += a4 * c if k == 0 else a2 * c
                    else:
                        rate += a4 * d4 if k == 0 else a2 * d2
                    Q[s, s_new] = rate
            # Propagate
            P = expm(Q * dt_sim)
            states = np.random.choice(np.arange(8), size=3*N, p=P[states, :])
            # Group into channels
            channels = np.reshape(states, (N, 3))
            p_open = np.mean(channels == 3).astype(float)
            # Update c and c_L
            J_IP3R = c1 * v1 * p_open * (c_L - c)
            J_leak = c1 * v2 * (c_L - c)
            J_SERCA = v3 * c * c / (K3*K3 + c*c)
            dc = J_IP3R + J_leak - J_SERCA
            dc_L = -(J_IP3R + J_leak - J_SERCA) / c1
            c += dc * dt_sim
            c_L += dc_L * dt_sim
            # Track conservation error
            conservation_error = abs(c + c1*c_L - c0) / c0
            if conservation_error > conservation_error_max:
                conservation_error_max = conservation_error
            # Save
            if t >= burn_in:
                c_history[t - burn_in] = c
                p_open_history[t - burn_in] = p_open
                conservation_error_history[t - burn_in] = conservation_error
        # Save trace
        ca_traces[replicate, :, idx] = c_history
        # Save p_open
        if N == max(channel_counts):
            p_open_vs_ip3[idx] = np.mean(p_open_history)

# Compute summary stats
# Period vs IP3
max_N = max(channel_counts)
idx_max_N = np.where(channel_counts == max_N)[0][0]
for idx, ip3 in enumerate(ip3_levels):
    # Get all traces for this ip3
    traces = ca_traces[:, :, idx * len(channel_counts) + idx_max_N]
    # Burn-in
    burn_in = int(0.15 * n_samples)
    data = traces[burn_in:]
    # Smooth
    data_smooth = np.zeros_like(data)
    for i in range(len(data)):
        data_smooth[i] = np.mean(data[max(0, i-10):i+11])
    # Find peaks
    peaks = []
    for i in range(1, len(data_smooth) - 1):
        if data_smooth[i] > data_smooth[i-1] and data_smooth[i] > data_smooth[i+1]:
            peaks.append(i)
    # Check for period
    if len(peaks) >= 3:
        periods = []
        for i in range(2, len(peaks)):
            if peaks[i] - peaks[i-1] > 2:
                periods.append(peaks[i] - peaks[i-1])
        period_vs_ip3[idx] = np.median(periods)
    else:
        # Autocorrelation
        data_norm = (data - np.mean(data)) / np.std(data)
        acf = np.correlate(data_norm, data_norm, mode="full")[len(data_norm)-1:]
        acf = acf / np.max(acf)
        # Find peak
        lags = np.arange(1, len(acf))
        peak_idx = np.argmax(acf[1:] > 0.2)
        period_vs_ip3[idx] = lags[peak_idx]
    # cv_isi_vs_N
    if ip3 == 0.28:
        idx_N = np.where(channel_counts == N)[0][0]
        # Get all traces for this N
        traces = ca_traces[:, :, idx_N]
        # Burn-in
        burn_in = int(0.15 * n_samples)
        data = traces[burn_in:]
        # Find ISIs
        isis = []
        for i in range(1, len(data)):
            if data[i] > data[i-1] and data[i] > data[i-2]:
                isis.append(data[i] - data[i-1])
        if len(isis) >= 3:
            cv_isi_vs_N[idx_N] = np.std(isis) / np.mean(isis)
        else:
            cv_isi_vs_N[idx_N] = np.nan

# Save output
np.savez("ca_traces.npz", ca_traces=ca_traces)
with open("summary_stats.json", "w") as f:
    json.dump({
        "ip3_levels": ip3_levels.tolist(),
        "channel_counts": channel_counts.tolist(),
        "period_vs_ip3": period_vs_ip3.tolist(),
        "cv_isi_vs_N": cv_isi_vs_N.tolist(),
        "p_open_vs_ip3": p_open_vs_ip3.tolist(),
        "conservation_error_max": conservation_error_max,
        "cr_ip3_idx_used": np.where(ip3_levels == 0.28)[0][0]
    }, f)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# Representative trace
axs[0, 0].plot(ca_traces[0, :, idx_max_N * len(ip3_levels) + np.where(ip3_levels == 0.28)[0][0]])
axs[0, 0].set_title("Representative Trace")
# Period vs IP3
axs[0, 1].plot(ip3_levels, period_vs_ip3)
axs[0, 1].set_title("Period vs IP3")
# CV(ISI) vs N
axs[1, 0].plot(channel_counts, cv_isi_vs_N)
axs[1, 0].set_title("CV(ISI) vs N")
# P_open vs IP3
axs[1, 1].plot(ip3_levels, p_open_vs_ip3)
axs[1, 1].set_title("P_open vs IP3")
plt.tight_layout()
plt.savefig("calcium_summary.png")