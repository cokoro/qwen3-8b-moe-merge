import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
import os

def read_params(path):
    with open(path, 'r') as f:
        return json.load(f)

def simulate():
    system = read_params('data/system_info.json')
    scan = read_params('data/scan_spec.json')
    ip3_levels = np.array(scan['ip3_levels_micromolar'])
    channel_counts = np.array(scan['channel_counts'])
    n_replicates = scan['n_replicates']
    T_obs = scan['T_obs_seconds']
    dt_out = scan['dt_out_seconds']
    n_samples = int(np.ceil(T_obs / dt_out))

    a1, d1 = system['a1'], system['d1']
    a2, d2 = system['a2'], system['d2']
    a3, d3 = system['a3'], system['d3']
    a4, d4 = system['a4'], system['d4']
    a5, d5 = system['a5'], system['d5']
    c0, c1, v1, v2, v3, K3 = system['c0'], system['c1'], system['v1'], system['v2'], system['v3'], system['K3']

    results = {}

    for ip3_idx, ip3 in enumerate(ip3_levels):
        ip3_val = ip3 * 1e-6  # convert to uM
        for N in channel_counts:
            if N == 0:
                continue
            ca_traces = []
            for rep in range(n_replicates):
                n_sub = 3 * N
                # Initialize subunits from stationary distribution at c=0.1 uM
                # Placeholder: use uniform distribution for now
                states = np.random.randint(0, 8, n_sub)
                c = 0.1  # uM
                c_L = (c0 - c) / c1

                t = 0.0
                samples = []
                conservation_error = []
                for _ in range(n_samples):
                    if t >= T_obs:
                        break
                    # Build rate matrix Q
                    Q = np.zeros((8, 8))
                    for i in range(8):
                        s = i
                        # Apply transitions
                        # Flip i
                        if s % 2 == 0:  # i=0
                            rate = a1 * ip3_val * (1 if (s & 4) else 0)  # k=0 or 1
                            Q[s, s ^ 1] += rate
                            Q[s, s] -= rate
                        else:  # i=1
                            rate = a3 * ip3_val * (1 if (s & 4) else 0)
                            Q[s, s ^ 1] += rate
                            Q[s, s] -= rate
                        # Flip j
                        if (s & 2) == 0:  # j=0
                            rate = a5 * c * (1 if (s & 4) else 0)  # j=0
                            Q[s, s ^ 2] += rate
                            Q[s, s] -= rate
                        else:  # j=1
                            rate = a5 * c * (1 if (s & 4) else 0)
                            Q[s, s ^ 2] += rate
                            Q[s, s] -= rate
                        # Flip k
                        if (s & 1) == 0:  # i=0
                            rate = a4 * c * (1 if (s & 2) else 0)  # i=0
                            Q[s, s ^ 4] += rate
                            Q[s, s] -= rate
                        else:  # i=1
                            rate = a2 * c * (1 if (s & 2) else 0)
                            Q[s, s ^ 4] += rate
                            Q[s, s] -= rate
                        # Add unbinding rates
                        # Flip i unbinding
                        if s % 2 == 0:  # i=0
                            rate = a1 * d1 * (1 if (s & 4) else 0)
                            Q[s, s ^ 1] += rate
                            Q[s, s] -= rate
                        else:  # i=1
                            rate = a3 * d3 * (1 if (s & 4) else 0)
                            Q[s, s ^ 1] += rate
                            Q[s, s] -= rate
                        # Flip j unbinding
                        if (s & 2) == 0:  # j=0
                            rate = a5 * d5 * (1 if (s & 4) else 0)
                            Q[s, s ^ 2] += rate
                            Q[s, s] -= rate
                        else:  # j=1
                            rate = a5 * d5 * (1 if (s & 4) else 0)
                            Q[s, s ^ 2] += rate
                            Q[s, s] -= rate
                        # Flip k unbinding
                        if (s & 1) == 0:  # i=0
                            rate = a4 * d4 * (1 if (s & 2) else 0)
                            Q[s, s ^ 4] += rate
                            Q[s, s] -= rate
                        else:  # i=1
                            rate = a2 * d2 * (1 if (s & 2) else 0)
                            Q[s, s ^ 4] += rate
                            Q[s, s] -= rate

                    # Propagate Q by dt_sim = 0.005 s
                    dt_sim = 0.005
                    P = la.expm(Q * dt_sim)
                    new_states = []
                    for s in states:
                        new_states.append(np.random.choice(8, p=P[s, :]))
                    states = np.array(new_states)

                    # Group into N channels
                    channels = np.reshape(states, (N, 3))
                    p_open = np.mean((channels == 3).all(axis=1))

                    # Update c and c_L
                    J_IP3R = c1 * v1 * p_open * (c_L - c)
                    J_leak = c1 * v2 * (c_L - c)
                    J_SERCA = v3 * c**2 / (K3**2 + c**2)
                    dc = J_IP3R + J_leak - J_SERCA
                    dc_L = -(J_IP3R + J_leak - J_SERCA) / c1

                    c += dc * dt_sim
                    c_L += dc_L * dt_sim

                    # Track conservation error
                    conservation_error.append(abs(c + c1 * c_L - (c0 - 0.1 * c1)) / (c0 - 0.1 * c1))

                    # Record c and p_open
                    samples.append((c, p_open))
                    t += dt_sim

                # Save samples
                ca_traces.append(np.array(samples))

            results[f'ip3{ip3_idx}_N{N}'] = ca_traces

    # Save results
    np.savez('ca_traces.npz', **results)

    # Compute statistics
    summary = {}

    # period_vs_ip3
    max_N = channel_counts[-1]
    ip3_idx_max = np.argmin(np.abs(ip3_levels - 0.28))
    ip3_val_max = ip3_levels[ip3_idx_max] * 1e-6
    n_samples_period = int(np.ceil(T_obs / dt_out))
    period = None
    for rep in range(n_replicates):
        data = results[f'ip3{ip3_idx_max}_N{max_N}'][rep]
        c_trace = np.array([d[0] for d in data])
        # Detect peaks
        peaks = []
        for i in range(1, len(c_trace) - 1):
            if c_trace[i] > c_trace[i-1] and c_trace[i] > c_trace[i+1]:
                peaks.append(i)
        if len(peaks) >= 3:
            spacing = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            period = np.median(spacing)
        else:
            # Use autocorrelation
            pass

    # cv_isi_vs_N
    ip3_val_cr = 0.28 * 1e-6
    ip3_idx_cr = np.argmin(np.abs(ip3_levels - 0.28))
    cv_isi = []
    for N in channel_counts:
        isis = []
        for rep in range(n_replicates):
            data = results[f'ip3{ip3_idx_cr}_N{N}'][rep]
            c_trace = np.array([d[0] for d in data])
            peaks = []
            for i in range(1, len(c_trace) - 1):
                if c_trace[i] > c_trace[i-1] and c_trace[i] > c_trace[i+1]:
                    peaks.append(i)
            if len(peaks) >= 3:
                isis.extend(peaks[1:] - peaks[:-1])
        if len(isis) >= 3:
            cv_isi.append(np.std(isis) / np.mean(isis))
        else:
            cv_isi.append(None)

    # p_open_vs_ip3
    max_N = channel_counts[-1]
    ip3_idx_popen = np.argmin(np.abs(ip3_levels - 0.28))
    p_open = []
    for rep in range(n_replicates):
        data = results[f'ip3{ip3_idx_popen}_N{max_N}'][rep]
        p_open.append(np.mean([d[1] for d in data[100:]]))  # burn-in
    p_open = np.mean(p_open)

    # conservation_error_max
    conservation_error_max = np.max(np.abs(np.concatenate([e for key in results for trace in results[key] for e in trace[1]])) / (c0 - 0.1 * c1))

    # Write summary
    summary['ip3_levels'] = ip3_levels.tolist()
    summary['channel_counts'] = channel_counts.tolist()
    summary['period_vs_ip3'] = period if period is not None else None
    summary['cv_isi_vs_N'] = [cv if cv is not None else None for cv in cv_isi]
    summary['p_open_vs_ip3'] = p_open
    summary['conservation_error_max'] = conservation_error_max
    summary['cr_ip3_idx_used'] = ip3_idx_cr

    with open('summary_stats.json', 'w') as f:
        json.dump(summary, f)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ip3_levels, [np.median([p[0] for p in results[f'ip3{i}_N{max_N}'][0]]) for i in range(len(ip3_levels))])
    plt.xlabel('IP3 [uM]')
    plt.ylabel('Period [s]')
    plt.savefig('calcium_summary.png')

if __name__ == '__main__':
    simulate()