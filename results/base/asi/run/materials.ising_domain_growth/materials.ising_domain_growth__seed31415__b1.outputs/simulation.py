import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.stats import linregress

def read_parameters(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

def read_initial_state(filename):
    return np.load(filename)

def simulate_ising_model(params):
    L = params['grid_size']
    T = params['control']
    n_steps = params['n_steps']
    n_snapshots = params['n_snapshots']
    n_measurements = params['n_measurements']
    snapshot_times = params['snapshot_times']
    measurement_times = params['measurement_times']
    max_r = params['max_r']
    dynamics_seed = params['dynamics_seed']
    
    # Initialize lattice
    initial_state = read_initial_state('data/initial_state.npy')
    lattice = initial_state.copy()
    
    # Initialize RNG
    rng = np.random.default_rng(dynamics_seed)
    
    # Precompute sublattices
    ii, jj = np.mgrid[0:L, 0:L]
    sublattice0 = (ii + jj) % 2 == 0
    sublattice1 = (ii + jj) % 2 == 1
    
    # Function to compute energy change
    def compute_delta_E(s, x, y):
        neighbors = []
        # Periodic boundary conditions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + dx) % L
            ny = (y + dy) % L
            neighbors.append(s[nx, ny])
        return 2 * s[x, y] * sum(neighbors)
    
    # Function to update lattice
    def update_lattice(s, sublattice_mask, T):
        xs, ys = np.where(sublattice_mask)
        delta_E = np.array([compute_delta_E(s, x, y) for x, y in zip(xs, ys)])
        if T == 0:
            # Zero temperature Glauber rule
            prob = np.zeros_like(delta_E)
            prob[delta_E < 0] = 1.0
            prob[delta_E == 0] = 0.5
        else:
            # Glauber probability
            prob = 1.0 / (1 + np.exp(delta_E / (T)))
        u = rng.random(len(prob))
        flip_mask = u < prob
        s[xs[flip_mask], ys[flip_mask]] *= -1
        return s
    
    # Simulation loop
    lattice_history = []
    correlation_history = []
    structure_factor_history = []
    domain_size_history = []
    interface_density_history = []
    
    for step in range(n_steps):
        if step in measurement_times:
            # Compute domain size
            m = np.mean(lattice)
            C = correlate(lattice, lattice, mode='full', method='fft') / (L ** 2)
            C = C[L//2 - max_r:L//2 + max_r + 1] - m ** 2
            r_values = np.arange(0, max_r + 1)
            # Interpolate to find L(t)
            L_t = np.interp(0.5 * C[0], C, r_values)
            # Apply running maximum
            if domain_size_history:
                L_t = max(L_t, domain_size_history[-1][1])
            domain_size_history.append((step, L_t))
            
            # Compute interface density
            interface_density = 0
            for i in range(L):
                for j in range(L):
                    s = lattice[i, j]
                    neighbors = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = (i + dx) % L
                        ny = (j + dy) % L
                        neighbors.append(lattice[nx, ny])
                    interface_density += (s != neighbors[0]) + (s != neighbors[1]) + (s != neighbors[2]) + (s != neighbors[3])
            interface_density /= (2 * L ** 2)
            interface_density_history.append((step, interface_density))
        
        if step in snapshot_times:
            # Compute correlation function
            s_fft = fft2(lattice)
            C_fft = np.abs(s_fft) ** 2
            C = ifft2(C_fft).real / (L ** 2)
            C = C[L//2 - max_r:L//2 + max_r + 1] - (np.mean(lattice) ** 2)
            correlation_history.append((step, C))
            
            # Compute structure factor
            k_values = np.fft.fftfreq(L) * 2 * np.pi
            k_values = np.concatenate((k_values[:L//2], k_values[L//2:][::-1]))
            s_fft = fft2(lattice)
            S = np.abs(s_fft) ** 2 / (L ** 2)
            S = S[L//2 - L:L//2 + L]
            structure_factor_history.append((step, S))
        
        # Update lattice
        lattice = update_lattice(lattice, sublattice0, T)
        lattice = update_lattice(lattice, sublattice1, T)
    
    # Save outputs
    np.save('seriesA.npy', np.array(domain_size_history))
    np.save('seriesB.npy', np.array(interface_density_history))
    np.save('stackC.npy', np.array(correlation_history))
    np.save('stackD.npy', np.array(structure_factor_history))
    
    # Compute dynamic scaling collapse
    collapse_check_applicable = False
    collapse_quality = None
    collapse_relative_rms_spread = None
    collapse_x_range = None
    collapse_grid_size = min(64, L - 1)
    collapse_snapshot_times = []
    collapse_lengths = []
    collapse_x_values = []
    collapse_y_values = []
    
    for i, (step, C) in enumerate(correlation_history):
        if step in snapshot_times:
            # Compute L(t)
            m = np.mean(lattice)
            C = C[L//2 - max_r:L//2 + max_r + 1] - m ** 2
            r_values = np.arange(0, max_r + 1)
            L_t = np.interp(0.5 * C[0], C, r_values)
            if L_t < 0.9 * L:
                collapse_snapshot_times.append(step)
                collapse_lengths.append(L_t)
                k_values = np.fft.fftfreq(L) * 2 * np.pi
                k_values = np.concatenate((k_values[:L//2], k_values[L//2:][::-1]))
                k_values = k_values[L//2 - L:L//2 + L]
                S = np.abs(fft2(lattice)) ** 2 / (L ** 2)
                S = S[L//2 - L:L//2 + L]
                x_values = k_values * L_t
                y_values = S / (L_t ** 2)
                collapse_x_values.append(x_values)
                collapse_y_values.append(y_values)
    
    if len(collapse_snapshot_times) >= 2:
        collapse_check_applicable = True
        x_overlap = np.intersect1d(collapse_x_values[0], collapse_x_values[1])
        x_overlap = x_overlap[np.isfinite(x_overlap)]
        if len(x_overlap) > 0:
            collapse_x_range = [np.min(x_overlap), np.max(x_overlap)]
            collapse_quality = 1.0
            collapse_relative_rms_spread = 0.0
            for i in range(1, len(collapse_x_values)):
                for x, y in zip(collapse_x_values[i], collapse_y_values[i]):
                    if x in collapse_x_range:
                        collapse_quality -= abs(y - np.mean(collapse_y_values[0][collapse_x_values[0] == x])) / np.std(collapse_y_values[0][collapse_x_values[0] == x])
                        collapse_quality /= len(collapse_x_values[i])
            collapse_quality /= len(collapse_x_values)
            collapse_relative_rms_spread = np.std(collapse_y_values) / np.mean(collapse_y_values)
    
    # Save metadata
    metadataA = {
        'collapse_check_applicable': collapse_check_applicable,
        'collapse_quality': collapse_quality,
        'collapse_relative_rms_spread': collapse_relative_rms_spread,
        'collapse_dimension': 2,
        'collapse_snapshot_times': collapse_snapshot_times,
        'collapse_lengths': collapse_lengths,
        'collapse_x_range': collapse_x_range,
        'collapse_grid_size': collapse_grid_size,
        'notes': []
    }
    with open('metadataA.json', 'w') as f:
        json.dump(metadataA, f)
    
    # Fit power law
    exponent = np.nan
    exponent_std_error = np.nan
    amplitude = np.nan
    fit_range = np.nan
    r_squared = np.nan
    notes = []
    
    # Extract valid data for fitting
    valid_indices = []
    for i, (step, L_t) in enumerate(domain_size_history):
        if step >= 100 and L_t > 1.5 and L_t < L / 4 and L_t > domain_size_history[i-1][1] if i > 0 else True:
            valid_indices.append(i)
    
    if len(valid_indices) >= 3:
        log_t = np.log([domain_size_history[i][0] for i in valid_indices])
        log_L = np.log([domain_size_history[i][1] for i in valid_indices])
        slope, intercept, r_value, p_value, std_err = linregress(log_t, log_L)
        exponent = slope
        exponent_std_error = std_err
        amplitude = np.exp(intercept)
        fit_range = [domain_size_history[valid_indices[0]][0], domain_size_history[valid_indices[-1]][0]]
        r_squared = r_value ** 2
    else:
        notes.append('Insufficient data for fitting')
    
    metadataB = {
        'exponent': exponent,
        'exponent_std_error': exponent_std_error,
        'amplitude': amplitude,
        'fit_range': fit_range,
        'r_squared': r_squared,
        'notes': notes
    }
    with open('metadataB.json', 'w') as f:
        json.dump(metadataB, f)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for i, (step, C) in enumerate(correlation_history):
        if step in snapshot_times:
            plt.plot(range(max_r + 1), C, label=f't={step}')
    plt.xlabel('r')
    plt.ylabel('C(r,t)')
    plt.title('Correlation Function')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for i, (step, S) in enumerate(structure_factor_history):
        if step in snapshot_times:
            plt.plot(range(L), S, label=f't={step}')
    plt.xlabel('k')
    plt.ylabel('S(k,t)')
    plt.title('Structure Factor')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    log_t = np.log([d[0] for d in domain_size_history])
    log_L = np.log([d[1] for d in domain_size_history])
    plt.plot(log_t, log_L, 'o', label='Data')
    if len(valid_indices) >= 3:
        plt.plot(log_t[valid_indices], log_L[valid_indices], 'r', label='Fit')
        plt.legend()
    plt.xlabel('log(t)')
    plt.ylabel('log(L(t))')
    plt.title('Domain Size vs Time')
    
    plt.subplot(2, 2, 4)
    for i, (step, x, y) in enumerate(zip(collapse_snapshot_times, collapse_x_values, collapse_y_values)):
        plt.plot(x, y, label=f't={step}')
    plt.xlabel('kL(t)')
    plt.ylabel('S(k,t)/L(t)^2')
    plt.title('Dynamic Scaling Collapse')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figure.png')
    plt.close()