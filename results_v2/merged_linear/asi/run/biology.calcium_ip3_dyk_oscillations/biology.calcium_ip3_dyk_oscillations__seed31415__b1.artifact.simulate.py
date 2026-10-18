import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

# Load system parameters
with open("data/system_info.json") as f:
    sys_params = json.load(f)
    a1, d1 = float(sys_params["a1"]), float(sys_params["d1"])
    a2, d2 = float(sys_params["a2"]), float(sys_params["d2"])
    a3, d3 = float(sys_params["a3"]), float(sys_params["d3"])
    a4, d4 = float(sys_params["a4"]), float(sys_params["d4"])
    a5, d5 = float(sys_params["a5"]), float(sys_params["d5"])
    c0 = float(sys_params["c0"])
    c1 = float(sys_params["c1"])
    v1 = float(sys_params["v1"])
    v2 = float(sys_params["v2"])
    v3 = float(sys_params["v3"])
    K3 = float(sys_params["K3"])
    dt_sim = float(sys_params["dt_sim"])
    dt_out = float(sys_params["dt_out"])

with open("data/scan_spec.json") as f:
    scan_params = json.load(f)
    ip3_levels = [float(x) for x in scan_params["ip3_levels_micromolar"]]
    channel_counts = [int(x) for x in scan_params["channel_counts"]]
    n_replicates = int(scan_params["n_replicates"])
    T_obs = float(scan_params["T_obs_seconds"])
    dt_out_seconds = float(scan_params["dt_out_seconds"])

# Load example trajectory for sanity check
try:
    example_traj = np.load("data/example_trajectory.npy")
except FileNotFoundError:
    example_traj = None

def build_Q(c, ip3, n_sub):
    """Build the 8x8 rate matrix Q for n_sub subunits."""
    rates = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            # Get the current state's binding/unbinding rates
            i_current = (i >> 0) & 1
            j_current = (i >> 1) & 1
            k_current = (i >> 2) & 1
            i_next = (j >> 0) & 1
            j_next = (j >> 1) & 1
            k_next = (j >> 2) & 1

            # Compute the rate from state i to j
            if i_current == 1 and j_current == 0:
                # Unbinding IP3 (i)
                if k_current == 0:
                    rate = a1 * d1
                else:
                    rate = a3 * d3
            elif i_current == 0 and j_current == 1:
                # Binding IP3 (i)
                if k_current == 0:
                    rate = a1 * ip3
                else:
                    rate = a3 * ip3
            elif i_current == 0 and j_current == 0:
                # Binding activating Ca (j)
                rate = a5 * c
            elif i_current == 1 and j_current == 1:
                # Unbinding activating Ca (j)
                rate = a5 * d5
            elif i_current == 0 and j_current == 1:
                # Binding inhibitory Ca (k)
                rate = a2 * c if i_current == 1 else a4 * c
            elif i_current == 1 and j_current == 0:
                # Unbinding inhibitory Ca (k)
                rate = a2 * d2 if i_current == 1 else a4 * d4
            else:
                rate = 0.0
            rates[i, j] = rate
    return rates

def simulate(ip3, N, c0, c1, v1, v2, v3, K3, dt_sim, T_obs, dt_out_seconds, n_replicates):
    """Simulate the IP3R channel dynamics for a given IP3 level and channel count."""
    n_sub = 3 * N
    # Initialize subunit states from stationary distribution
    # (for simplicity, assume all subunits are in state 0)
    substates = np.zeros((n_replicates, n_sub), dtype=int)
    c = np.zeros((n_replicates, int(T_obs / dt_out_seconds) + 1))
    c_L = np.zeros((n_replicates, int(T_obs / dt_out_seconds) + 1))
    c[:, 0] = 0.1
    c_L[:, 0] = (c0 - c[:, 0]) / c1
    P_open = np.zeros((n_replicates, int(T_obs / dt_out_seconds) + 1))
    conservation_error = np.zeros((n_replicates, int(T_obs / dt_sim) + 1))

    for rep in range(n_replicates):
        substates[rep] = np.zeros(n_sub, dtype=int)
        c[rep, 0] = 0.1
        c_L[rep, 0] = (c0 - c[rep, 0]) / c1
        for t in range(int(T_obs / dt_sim) + 1):
            # Build rate matrix
            Q = build_Q(c[rep, t], ip3, n_sub)
            # Compute transition matrix
            P = la.expm(Q * dt_sim)
            # Sample next states
            next_states = np.zeros(n_sub, dtype=int)
            for i in range(n_sub):
                next_states[i] = np.random.choice(range(8), p=P[substates[rep, i], :])
            substates[rep] = next_states
            # Group into N channels of 3 subunits
            channels = np.array_split(substates[rep], N)
            # Count channels in state 3
            P_open[rep, t] = np.mean([1 if all(s == 3 for s in ch) else 0 for ch in channels])
            # Euler-step c and c_L
            J_IP3R = c1 * v1 * P_open[rep, t] * (c_L[rep, t] - c[rep, t])
            J_leak = c1 * v2 * (c_L[rep, t] - c[rep, t])
            J_SERCA = v3 * c[rep, t] * c[rep, t] / (K3 * K3 + c[rep, t] * c[rep, t])
            dc = J_IP3R + J_leak - J_SERCA
            dc_L = -(J_IP3R + J_leak - J_SERCA) / c1
            c[rep, t + 1] = c[rep, t] + dc * dt_sim
            c_L[rep, t + 1] = c_L[rep, t] + dc_L * dt_sim
            # Track conservation error
            conservation_error[rep, t] = abs(c[rep, t] + c1 * c_L[rep, t] - c0) / c0

    # Extract the time-averaged P_open
    avg_P_open = np.mean(P_open[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    # Extract the time-averaged c and c_L
    avg_c = np.mean(c[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    avg_c_L = np.mean(c_L[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    # Extract the time-averaged conservation error
    avg_conservation_error = np.mean(conservation_error[:, int(0.15 * T_obs / dt_sim):], axis=1)
    # Extract the time-averaged c and c_L
    avg_c = np.mean(c[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    avg_c_L = np.mean(c_L[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    # Extract the time-averaged conservation error
    avg_conservation_error = np.mean(conservation_error[:, int(0.15 * T_obs / dt_sim):], axis=1)
    # Extract the time-averaged c and c_L
    avg_c = np.mean(c[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    avg_c_L = np.mean(c_L[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    # Extract the time-averaged conservation error
    avg_conservation_error = np.mean(conservation_error[:, int(0.15 * T_obs / dt_sim):], axis=1)
    # Extract the time-averaged c and c_L
    avg_c = np.mean(c[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    avg_c_L = np.mean(c_L[:, int(0.15 * T_obs / dt_out_seconds):], axis=1)
    # Extract the time-averaged conservation error
    avg_conservation_error = np.mean(conservation_error[:, int(0.15 * T_obs / dt_sim):], axis=1)

    return {
        "c": c,
        "c_L": c_L,
        "P_open": P_open,
        "avg_P_open": avg_P_open,
        "avg_c": avg_c,
        "avg_c_L": avg_c_L,
        "avg_conservation_error": avg_conservation_error
    }

# Run simulations for all grid points
results = {}
for ip3_idx, ip3 in enumerate(ip3_levels):
    for N_idx, N in enumerate(channel_counts):
        grid_key = f"ip3{ip3_idx}_N{N_idx}"
        results[grid_key] = simulate(ip3, N, c0, c1, v1, v2, v3, K3, dt_sim, T_obs, dt_out_seconds, n_replicates)

# Save results to npz file
np.savez("ca_traces.npz", **results)

# Extract summary statistics
summary = {
    "ip3_levels": ip3_levels,
    "channel_counts": channel_counts,
    "period_vs_ip3": None,
    "cv_isi_vs_N": None,
    "p_open_vs_ip3": None,
    "conservation_error_max": None,
    "cr_ip3_idx_used": None
}

# Compute period vs IP3
largest_N = max(channel_counts)
largest_N_idx = channel_counts.index(largest_N)
ip3_idx_period = np.argmin(np.abs(np.array(ip3_levels) - 0.28))
periods = []
for rep in range(n_replicates):
    c = results[f"ip3{ip3_idx_period}_N{largest_N_idx}"]["c"][rep]
    c_L = results[f"ip3{ip3_idx_period}_N{largest_N_idx}"]["c_L"][rep]
    peaks = []
    for t in range(len(c) - 1):
        if c[t + 1] > c[t] and c[t] > 0.1 and c[t + 1] > 0.2:
            peaks.append(t)
    if len(peaks) < 3:
        # Use autocorrelation
        c_filtered = c[100:]
        autocorr = np.correlate(c_filtered, c_filtered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        peaks = []
        for lag in range(2, len(autocorr)):
            if autocorr[lag] > 0.2:
                peaks.append(lag)
        if len(peaks) < 1:
            continue
        period = np.mean(peaks)
    else:
        # Use peak spacing
        periods.append(np.median([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]))
if periods:
    summary["period_vs_ip3"] = np.median(periods)
else:
    summary["period_vs_ip3"] = None

# Compute CV(ISI) vs N
ip3_idx_cv = np.argmin(np.abs(np.array(ip3_levels) - 0.28))
N_idx_cv = np.argmin(np.abs(np.array(channel_counts) - 80))
cv_isis = []
for rep in range(n_replicates):
    c = results[f"ip3{ip3_idx_cv}_N{N_idx_cv}"]["c"][rep]
    c_L = results[f"ip3{ip3_idx_cv}_N{N_idx_cv}"]["c_L"][rep]
    peaks = []
    for t in range(len(c) - 1):
        if c[t + 1] > c[t] and c[t] > 0.1 and c[t + 1] > 0.2:
            peaks.append(t)
    if len(peaks) < 3:
        continue
    isi = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    cv_isi = np.std(isi) / np.mean(isi)
    cv_isis.append(cv_isi)
if cv_isis:
    summary["cv_isi_vs_N"] = np.mean(cv_isis)
else:
    summary["cv_isi_vs_N"] = None

# Compute P_open vs IP3
largest_N_idx = channel_counts.index(max(channel_counts))
p_open = []
for ip3_idx in range(len(ip3_levels)):
    c = results[f"ip3{ip3_idx}_N{largest_N_idx}"]["avg_P_open"]
    p_open.append(np.mean(c[100:]))
summary["p_open_vs_ip3"] = p_open

# Compute conservation error max
conservation_errors = []
for ip3_idx in range(len(ip3_levels)):
    for N_idx in range(len(channel_counts)):
        avg_conservation_error = results[f"ip3{ip3_idx}_N{N_idx}"]["avg_conservation_error"]
        conservation_errors.append(np.max(avg_conservation_error))
summary["conservation_error_max"] = np.max(conservation_errors)
summary["cr_ip3_idx_used"] = ip3_idx_cv

# Write summary to JSON
with open("summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("IP3R Channel Dynamics and Calcium Oscillations")

# Plot a representative trace
rep_idx = 0
c = results[f"ip3{ip3_idx_cv}_N{N_idx_cv}"]["c"][rep_idx]
c_L = results[f"ip3{ip3_idx_cv}_N{N_idx_cv}"]["c_L"][rep_idx]
axs[0, 0].plot(c, label="Cytoplasmic Ca")
axs[0, 0].plot(c_L, label="ER Ca")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Ca (uM)")
axs[0, 0].legend()
axs[0, 0].set_title("Representative Trace")

# Plot period vs IP3
if summary["period_vs_ip3"] is not None:
    axs[0, 1].plot(ip3_levels, [summary["period_vs_ip3"]]*len(ip3_levels), "o-")
    axs[0, 1].set_xlabel("IP3 (uM)")
    axs[0, 1].set_ylabel("Period (s)")
    axs[0, 1].set_title("Period vs IP3")
else:
    axs[0, 1].set_visible(False)

# Plot CV(ISI) vs N
if summary["cv_isi_vs_N"] is not None:
    axs[1, 0].plot(channel_counts, [summary["cv_isi_vs_N"]]*len(channel_counts), "o-")
    axs[1, 0].set_xlabel("Channel Count")
    axs[1, 0].set_ylabel("CV(ISI)")
    axs[1, 0].set_title("CV(ISI) vs Channel Count")
else:
    axs[1, 0].set_visible(False)

# Plot P_open vs IP3
axs[1, 1].plot(ip3_levels, summary["p_open_vs_ip3"], "o-")
axs[1, 1].set_xlabel("IP3 (uM)")
axs[1, 1].set_ylabel("P_open")
axs[1, 1].set_title("P_open vs IP3")

plt.tight_layout()
plt.savefig("calcium_summary.png")
plt.close()