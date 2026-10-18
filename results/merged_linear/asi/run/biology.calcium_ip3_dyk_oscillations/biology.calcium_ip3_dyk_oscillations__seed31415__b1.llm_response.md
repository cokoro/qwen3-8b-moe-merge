```python
import os
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# Load data
DATA_DIR = "data"
system_info = np.load(os.path.join(DATA_DIR, "system_info.json"), allow_pickle=True).item()
scan_spec = np.load(os.path.join(DATA_DIR, "scan_spec.json"), allow_pickle=True).item()
n_samples = scan_spec.get("n_samples", 1200)

# Constants from system_info
a1, d1 = system_info["a1"], system_info["d1"]
a2, d2 = system_info["a2"], system_info["d2"]
a3, d3 = system_info["a3"], system_info["d3"]
a4, d4 = system_info["a4"], system_info["d4"]
a5, d5 = system_info["a5"], system_info["d5"]
c0, c1 = system_info["c0"], system_info["c1"]
v1, v2 = system_info["v1"], system_info["v2"]
v3, K3 = system_info["v3"], system_info["K3"]
dt_sim = 0.005  # macro step size

ip3_levels = scan_spec["ip3_levels_micromolar"]
channel_counts = scan_spec["channel_counts"]
n_replicates = scan_spec["n_replicates"]
T_obs_seconds = scan_spec["T_obs_seconds"]
dt_out_seconds = scan_spec["dt_out_seconds"]

# Precompute state transition rates
def get_rate_matrix(c, ip3):
    c_ca = c  # [Ca2+]_cyt
    c_er = (c0 - c) / c1  # [Ca2+]_ER
    rates = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            s_i = i
            s_j = j
            if (s_i & 0b100) != (s_j & 0b100):
                k = s_i & 0b100 >> 2
                if k == 0:
                    binding_rate = a1 * ip3
                    unbinding_rate = a1 * d1
                else:
                    binding_rate = a3 * ip3
                    unbinding_rate = a3 * d3
                rates[i, j] = binding_rate
                rates[j, i] = unbinding_rate
            elif (s_i & 0b010) != (s_j & 0b010):
                binding_rate = a5 * c_ca
                unbinding_rate = a5 * d5
                rates[i, j] = binding_rate
                rates[j, i] = unbinding_rate
            elif (s_i & 0b001) != (s_j & 0b001):
                i_val = s_i & 0b001
                if i_val == 0:
                    binding_rate = a4 * c_ca
                    unbinding_rate = a4 * d4
                else:
                    binding_rate = a2 * c_ca
                    unbinding_rate = a2 * d2
                rates[i, j] = binding_rate
                rates[j, i] = unbinding_rate
    return rates

# Simulate the system
ca_traces = {}
for ip3_idx, ip3 in enumerate(ip3_levels):
    for N in channel_counts:
        print(f"Simulating ip3={ip3:.3f} uM, N={N}")
        ca = []
        p_open = []
        conservation_error = []
        for rep in range(n_replicates):
            n_sub = 3 * N
            # Initialize subunit states
            c = 0.1
            c_er = (c0 - c) / c1
            states = np.random.randint(0, 8, size=n_sub)
            for t in range(int(T_obs_seconds / dt_sim)):
                # Build rate matrix
                rates = get_rate_matrix(c, ip3)
                Q = (rates - np.diag(np.sum(rates, axis=1))) / dt_sim
                P = scipy.linalg.expm(Q)
                # Sample next states
                new_states = []
                for s in states:
                    new_states.append(np.random.choice(8, p=P[s, :]))
                new_states = np.array(new_states)
                # Group into channels
                channel_states = new_states.reshape((N, 3))
                # Compute P_open
                open_channels = np.sum((channel_states == 3).all(axis=1))
                P_open = open_channels / N
                # Euler-step c and c_er
                J_ip3r = c1 * v1 * P_open * (c_er - c)
                J_leak = c1 * v2 * (c_er - c)
                J_serca = v3 * c**2 / (K3**2 + c**2)
                dc = J_ip3r + J_leak - J_serca
                dc_er = -(J_ip3r + J_leak - J_serca) / c1
                c += dc * dt_sim
                c_er += dc_er * dt_sim
                # Track conservation error
                conservation_error.append(abs(c + c1 * c_er - c0) / c0)
                # Save c and P_open every dt_out_seconds
                if t * dt_sim >= dt_out_seconds:
                    ca.append(c)
                    p_open.append(P_open)
            # Save traces
            key = f"ip3{ip3_idx}_N{N}"
            if key not in ca_traces:
                ca_traces[key] = []
            ca_traces[key].append(np.array(ca))
            if key not in p_open:
                p_open[key] = []
            p_open[key].append(np.array(p_open))

# Save traces
np.savez("ca_traces.npz", **{k: np.array(v) for k, v in ca_traces.items()})

# Compute summary statistics
period_vs_ip3 = []
cv_isi_vs_N = []
p_open_vs_ip3 = []
conservation_error_max = 0.0

for ip3_idx, ip3 in enumerate(ip3_levels):
    for N in channel_counts:
        key = f"ip3{ip3_idx}_N{N}"
        traces = ca_traces[key]
        for rep in range(n_replicates):
            c = traces[rep]
            # Detect peaks and compute period
            burn_in = int(0.15 * len(c))
            c_burn = c[burn_in:]
            smoothed = np.convolve(c_burn, np.ones(10), mode="same")
            peaks = []
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                    peaks.append(i)
            if len(peaks) >= 3:
                peak_times = np.array(peaks) * dt_out_seconds
                spacing = peak_times[1:] - peak_times[:-1]
                period = np.median(spacing[(spacing >= 2) & (spacing <= 50)])
                period_vs_ip3.append(period)
            else:
                # Autocorrelation
                c_mean = np.mean(c_burn)
                c_centered = c_burn - c_mean
                acf = np.correlate(c_centered, c_centered, mode="full")[len(c_centered) - 1:]
                lag = np.argmax(acf[(acf >= 0.2) & (acf < 1.0)])
                period = lag * dt_out_seconds
                period_vs_ip3.append(period)
        # CV(ISI) vs N
        if abs(ip3 - 0.28) < 1e-5:
            isi = []
            for rep in range(n_replicates):
                c = traces[rep]
                burn_in = int(0.15 * len(c))
                c_burn = c[burn_in:]
                baseline = np.percentile(c_burn, 30)
                peak_max = np.percentile(c_burn, 98)
                threshold = baseline + 0.5 * (peak_max - baseline)
                isis = []
                for i in range(1, len(c_burn) - 1):
                    if c_burn[i] > threshold:
                        isis.append(i)
                if len(isis) >= 3:
                    isi.extend(isis)
            if len(isi) >= 3:
                cv_isi = np.std(isi) / np.mean(isi)
                cv_isi_vs_N.append(cv_isi)
        # P_open vs IP3
        if N == max(channel_counts):
            p_open_avg = np.mean([traces[rep][-1] for rep in range(n_replicates)])
            p_open_vs_ip3.append(p_open_avg)
        # Conservation error
        conservation_error_max = max(conservation_error_max, max(conservation_error))

# Write summary stats
summary_stats = {
    "ip3_levels": ip3_levels,
    "channel_counts": channel_counts,
    "period_vs_ip3": period_vs_ip3,
    "cv_isi_vs_N": cv_isi_vs_N,
    "p_open_vs_ip3": p_open_vs_ip3,
    "conservation_error_max": conservation_error_max,
    "cr_ip3_idx_used": np.argmin(np.abs(np.array(ip3_levels) - 0.28)),
}

with open("summary_stats.json", "w") as f:
    json.dump(summary_stats, f)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
for ip3_idx, ip3 in enumerate(ip3_levels):
    for N in channel_counts:
        key = f"ip3{ip3_idx}_N{N}"
        traces = ca_traces[key]
        for rep in range(n_replicates):
            plt.plot(np.arange(len(traces[rep])) * dt_out_seconds, traces[rep], alpha=0.3)
plt.title("Calcium Concentration Traces")

plt.subplot(2, 2, 2)
plt.plot(ip3_levels, [np.median(p) for p in period_vs_ip3], "o-")
plt.title("Period vs IP3")

plt.subplot(2, 2, 3)
plt.plot(channel_counts, cv_isi_vs_N, "o-")
plt.title("CV(ISI) vs Channel Count")

plt.subplot(2, 2, 4)
plt.plot(ip3_levels, p_open_vs_ip3, "o-")
plt.title("P_open vs IP3")

plt.tight_layout()
plt.savefig("calcium_summary.png")
plt.close()
```