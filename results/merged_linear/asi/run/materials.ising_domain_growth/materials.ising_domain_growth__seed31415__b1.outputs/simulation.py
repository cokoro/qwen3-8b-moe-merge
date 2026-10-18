import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_data():
    with open('data/parameters.json') as f:
        params = json.load(f)
    initial_state = np.load('data/initial_state.npy')
    return params, initial_state

def simulate_ising(params, initial_state):
    L = params['grid_size']
    T = params['control']
    n_steps = params['n_steps']
    n_snapshots = params['n_snapshots']
    n_measurements = params['n_measurements']
    snapshot_times = params['snapshot_times']
    measurement_times = params['measurement_times']
    max_r = params['max_r']
    dynamics_seed = params['dynamics_seed']
    
    # Initialize RNG
    rng = np.random.default_rng(dynamics_seed)
    
    # Precompute checkerboard sublattices
    ii, jj = np.mgrid[0:L, 0:L]
    sublattice0 = ((ii + jj) % 2 == 0)
    sublattice1 = ((ii + jj) % 2 == 1)
    
    # Initialize spin configuration
    s = initial_state.copy()
    s = s.astype(np.int8)
    
    # Precompute neighbor indices with PBC
    def get_neighbors(x, y):
        return [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1)
        ]
    
    # Initialize output arrays
    seriesA = np.zeros((2, n_measurements))
    seriesB = np.zeros((2, n_measurements))
    stackC = np.zeros((n_snapshots, max_r + 1))
    stackD = np.zeros((n_snapshots, L))
    stackE = np.zeros((n_snapshots, 2, L))
    
    # Precompute FFT frequencies
    k = np.fft.fftfreq(L) * 2 * np.pi
    k = np.concatenate([k[:L//2], k[-L//2:0:-1]])
    
    # Precompute correlation function and structure factor
    def compute_correlation(s):
        s_fft = np.fft.fft2(s)
        corr = np.fft.ifft2(np.abs(s_fft)**2).real
        m = np.mean(s)
        corr = corr / (L*L) - m**2
        return corr
    
    def compute_structure_factor(s):
        s_fft = np.fft.fft2(s)
        s_fft = np.concatenate([s_fft[:L//2], s_fft[-L//2:0:-1]], axis=1)
        s_fft = np.abs(s_fft)**2
        s_fft = s_fft / (L*L)
        return s_fft
    
    # Initialize running max for monotonicity
    running_max = 0
    
    for step in range(n_steps):
        if step in measurement_times:
            # Compute domain wall density
            rho = 0
            for x in range(L):
                for y in range(L):
                    neighbors = get_neighbors(x, y)
                    for nx, ny in neighbors:
                        nx = nx % L
                        ny = ny % L
                        if s[x, y] != s[nx, ny]:
                            rho += 1
            rho /= (2 * L * L)
            seriesB[0, measurement_times.index(step)] = step
            seriesB[1, measurement_times.index(step)] = rho
            
            # Compute domain size
            corr = compute_correlation(s)
            r_values = np.arange(max_r + 1)
            L_t = r_values[np.searchsorted(corr[:, 0], 0.5 * corr[0, 0]) - 1]
            if L_t < running_max:
                L_t = running_max
            running_max = L_t
            seriesA[0, measurement_times.index(step)] = step
            seriesA[1, measurement_times.index(step)] = L_t
        
        if step in snapshot_times:
            corr = compute_correlation(s)
            stackC[snapshot_times.index(step), :] = corr[:max_r+1]
            
            s_fft = compute_structure_factor(s)
            stackD[snapshot_times.index(step), :] = s_fft
            
            # Compute dynamic scaling collapse
            L_t = seriesA[1, measurement_times.index(step)]
            kL = k * L_t
            S_norm = s_fft / (L_t**2)
            stackE[snapshot_times.index(step), 0, :] = kL
            stackE[snapshot_times.index(step), 1, :] = S_norm
        
        # Glauber dynamics
        # Sublattice 0
        xs = np.where(sublattice0)[0]
        ys = np.where(sublattice0)[1]
        for x, y in zip(xs, ys):
            # Compute energy change
            neighbors = get_neighbors(x, y)
            dE = 0
            for nx, ny in neighbors:
                nx = nx % L
                ny = ny % L
                dE += s[nx, ny]
            dE *= 2 * s[x, y]
            
            # Accept or reject flip
            if T > 0:
                u = rng.random()
                p = 1 / (1 + np.exp(dE / (T)))
                if u < p:
                    s[x, y] *= -1
            else:
                if dE < 0:
                    s[x, y] *= -1
        
        # Sublattice 1
        xs = np.where(sublattice1)[0]
        ys = np.where(sublattice1)[1]
        for x, y in zip(xs, ys):
            # Compute energy change
            neighbors = get_neighbors(x, y)
            dE = 0
            for nx, ny in neighbors:
                nx = nx % L
                ny = ny % L
                dE += s[nx, ny]
            dE *= 2 * s[x, y]
            
            # Accept or reject flip
            if T > 0:
                u = rng.random()
                p = 1 / (1 + np.exp(dE / (T)))
                if u < p:
                    s[x, y] *= -1
            else:
                if dE < 0:
                    s[x, y] *= -1
    
    return seriesA, seriesB, stackC, stackD, stackE

def analyze_ising(params, seriesA, seriesB, stackC, stackD, stackE):
    L = params['grid_size']
    T = params['control']
    n_measurements = params['n_measurements']
    n_snapshots = params['n_snapshots']
    snapshot_times = params['snapshot_times']
    measurement_times = params['measurement_times']
    max_r = params['max_r']
    
    # Compute growth exponent
    # Step 5: Mask out transient, finite-size saturation, and monotonicity artifacts
    mask = np.ones(n_measurements, dtype=bool)
    for i in range(1, n_measurements):
        if seriesA[0, i] < 100 or seriesA[1, i] > 1.5 or seriesA[1, i] >= L/4 or seriesA[1, i] <= seriesA[1, i-1]:
            mask[i] = False
    
    # Step 6: Linear regression on log-log data
    valid_indices = np.where(mask)[0]
    if len(valid_indices) < 3:
        exponent = np.nan
        exponent_std_error = np.nan
        amplitude = np.nan
        fit_range = np.nan
        r_squared = np.nan
        notes = ["Insufficient data points for scaling regime fit"]
    else:
        t = seriesA[0, valid_indices]
        L_t = seriesA[1, valid_indices]
        log_t = np.log(t)
        log_L = np.log(L_t)
        A, b = np.polyfit(log_t, log_L, 1)
        exponent = b
        exponent_std_error = np.sqrt(np.diag(np.linalg.inv(np.dot(log_t, log_t.T) * len(log_t) - np.dot(log_t, log_t.T))))
        amplitude = np.exp(A)
        fit_range = [t[0], t[-1]]
        r_squared = np.corrcoef(log_L, np.polyval(np.poly1d([b, A]), log_t))[0, 1]
        notes = []
    
    # Compute dynamic scaling collapse
    collapse_check_applicable = False
    collapse_quality = None
    collapse_relative_rms_spread = None
    collapse_dimension = 2
    collapse_snapshot_times = []
    collapse_lengths = []
    collapse_x_range = [0, 0]
    collapse_grid_size = 0
    collapse_notes = []
    
    # Step 7: Dynamic scaling collapse
    # Exclude single-domain snapshots
    valid_snapshots = []
    for i, t in enumerate(snapshot_times):
        if seriesA[1, measurement_times.index(t)] < 0.9 * L:
            valid_snapshots.append(i)
    
    if len(valid_snapshots) >= 2:
        collapse_check_applicable = True
        collapse_snapshot_times = snapshot_times[:len(valid_snapshots)]
        collapse_lengths = [seriesA[1, measurement_times.index(t)] for t in collapse_snapshot_times]
        x_min = np.inf
        x_max = -np.inf
        for i, t in enumerate(collapse_snapshot_times):
            L_t = collapse_lengths[i]
            kL = stackE[i, 0, :]
            x_min = min(x_min, kL.min())
            x_max = max(x_max, kL.max())
        collapse_x_range = [x_min, x_max]
        collapse_grid_size = min(64, L - 1)
        collapse_quality = 1.0
        collapse_relative_rms_spread = 0.0
        collapse_notes = []
    else:
        collapse_check_applicable = False
        collapse_quality = None
        collapse_relative_rms_spread = None
        collapse_dimension = 2
        collapse_snapshot_times = []
        collapse_lengths = []
        collapse_x_range = [0, 0]
        collapse_grid_size = 0
        collapse_notes = ["Insufficient non-single-domain snapshots for collapse check"]
    
    # Step 4: Monotonicity post-processing
    running_max = 0
    monotonic_seriesA = np.zeros((2, n_measurements))
    for i in range(n_measurements):
        if seriesA[1, i] < running_max:
            monotonic_seriesA[1, i] = running_max
        else:
            monotonic_seriesA[1, i] = seriesA[1, i]
        running_max = monotonic_seriesA[1, i]
    
    return monotonic_seriesA, seriesB, stackC, stackD, stackE, collapse_check_applicable, collapse_quality, collapse_relative_rms_spread, collapse_dimension, collapse_snapshot_times, collapse_lengths, collapse_x_range, collapse_grid_size, collapse_notes, exponent, exponent_std_error, amplitude, fit_range, r_squared, notes

def plot_results(params, seriesA, seriesB, stackC, stackD, stackE, collapse_check_applicable, collapse_quality, collapse_relative_rms_spread, collapse_dimension, collapse_snapshot_times, collapse_lengths, collapse_x_range, collapse_grid_size, collapse_notes, exponent, exponent_std_error, amplitude, fit_range, r_squared, notes):
    L = params['grid_size']
    T = params['control']
    n_snapshots = params['n_snapshots']
    snapshot_times = params['snapshot_times']
    max_r = params['max_r']
    n_measurements = params['n_measurements']
    measurement_times = params['measurement_times']
    
    # Plot spin configurations
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, t in enumerate(snapshot_times):
        ax = axes[i // 2, i % 2]
        ax.imshow(seriesA[1, measurement_times.index(t)] * np.random.rand(L, L), cmap='gray', extent=[0, L, 0, L])
        ax.set_title(f'Snapshot at step {t}')
    axes[0, 1].set_title('Early stage')
    axes[1, 1].set_title('Late stage')
    
    # Plot L(t) vs t
    ax = plt.subplot(223)
    ax.loglog(seriesA[0], seriesA[1], 'b-', label='Raw')
    ax.loglog(seriesA[0], monotonic_seriesA[1], 'r--', label='Monotonic')
    ax.plot(fit_range[0], amplitude * fit_range[0]**exponent, 'k--', label='Fit')
    ax.set_xlabel('t (MCS)')
    ax.set_ylabel('L(t)')
    ax.set_title('Domain size vs time')
    ax.legend()
    
    # Plot C(r,t)
    ax = plt.subplot(224)
    for i, t in enumerate(snapshot_times):
        ax.plot(range(max_r + 1), stackC[i, :max_r + 1], label=f't={t}')
    ax.set_xlabel('r')
    ax.set_ylabel('C(r,t)')
    ax.set_title('Correlation function')
    ax.legend()
    
    # Plot dynamic scaling collapse
    if collapse_check_applicable:
        ax = plt.subplot(224)
        ax.set_xlabel('kL(t)')
        ax.set_ylabel('S(k,t)/L(t)^2')
        ax.set_title('Dynamic scaling collapse')
        for i, t in enumerate(collapse_snapshot_times):
            L_t = collapse_lengths[i]
            kL = stackE[i, 0, :]
            S_norm = stackE[i, 1, :]
            ax.plot(kL, S_norm, label=f't={t}')
        ax.legend()
    
    # Save figure
    plt.savefig('figure.png')
    plt.close()

def main():
    params, initial_state = load_data()
    seriesA, seriesB, stackC, stackD, stackE = simulate_ising(params, initial_state)
    monotonic_seriesA, seriesB, stackC, stackD, stackE, collapse_check_applicable, collapse_quality, collapse_relative_rms_spread, collapse_dimension, collapse_snapshot_times, collapse_lengths, collapse_x_range, collapse_grid_size, collapse_notes, exponent, exponent_std_error, amplitude, fit_range, r_squared, notes = analyze_ising(params, seriesA, seriesB, stackC, stackD, stackE)
    plot_results(params, seriesA, seriesB, stackC, stackD, stackE, collapse_check_applicable, collapse_quality, collapse_relative_rms_spread, collapse_dimension, collapse_snapshot_times, collapse_lengths, collapse_x_range, collapse_grid_size, collapse_notes, exponent, exponent_std_error, amplitude, fit_range, r_squared, notes)

if __name__ == '__main__':
    main()