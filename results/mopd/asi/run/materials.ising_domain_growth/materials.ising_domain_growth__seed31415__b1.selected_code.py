import numpy as np
import matplotlib.pyplot as plt
import json
import os

def simulate_ising_model():
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    L = params['grid_size']
    T = params['control']
    n_steps = params['n_steps']
    n_snapshots = params['n_snapshots']
    n_measurements = params['n_measurements']
    snapshot_times = params['snapshot_times']
    measurement_times = params['measurement_times']
    max_r = params['max_r']
    dynamics_seed = params['dynamics_seed']
    
    # Load initial state
    initial_state = np.load('data/initial_state.npy')
    
    # Precompute checkerboard sublattice indices
    ii, jj = np.mgrid[0:L, 0:L]
    sublattice0 = (ii + jj) % 2 == 0
    sublattice1 = (ii + jj) % 2 == 1
    
    # Initialize RNG
    rng = np.random.default_rng(dynamics_seed)
    
    # Initialize spin configuration
    s = initial_state.copy()
    
    # Precompute neighbor indices for periodic boundary conditions
    neighbor_indices = np.zeros((L, L, 4, 2), dtype=int)
    for i in range(L):
        for j in range(L):
            for k, (di, dj) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                ni, nj = (i + di) % L, (j + dj) % L
                neighbor_indices[i, j, k] = (ni, nj)
    
    # Initialize arrays for output
    seriesA = np.zeros((2, n_measurements))
    seriesB = np.zeros((2, n_measurements))
    stackC = np.zeros((n_snapshots, max_r))
    stackD = np.zeros((n_snapshots, L))
    stackE = np.zeros((n_snapshots, 2, L))
    
    # Store snapshots and measurement times
    snapshot_times = np.array(snapshot_times)
    measurement_times = np.array(measurement_times)
    
    # Precompute FFT frequencies
    k = np.fft.fftfreq(L) * 2 * np.pi
    k = np.concatenate((k[:L//2], k[L//2:][::-1]))  # Symmetric, non-negative frequencies
    
    # Initialize domain size array
    L_t = np.zeros(n_measurements)
    t = 0
    
    # Main simulation loop
    for step in range(n_steps):
        # Update sublattice 0 first
        for i, j in np.ndindex((L, L)):
            if not sublattice0[i, j]:
                continue
            # Compute energy change
            x, y = i, j
            dx, dy = neighbor_indices[x, y]
            s_neighbors = s[dx, y] + s[(x+1)%L, y] + s[x, (y+1)%L] + s[x, (y-1)%L]
            dE = 2 * s[x, y] * s_neighbors
            # Compute flip probability
            if T > 0:
                p = 1 / (1 + np.exp(dE / (T)))
            else:
                p = 0.5 if dE == 0 else 0 if dE > 0 else 1
            # Flip with probability p
            if rng.random() < p:
                s[x, y] *= -1
        
        # Update sublattice 1
        for i, j in np.ndindex((L, L)):
            if not sublattice1[i, j]:
                continue
            # Compute energy change
            x, y = i, j
            dx, dy = neighbor_indices[x, y]
            s_neighbors = s[dx, y] + s[(x+1)%L, y] + s[x, (y+1)%L] + s[x, (y-1)%L]
            dE = 2 * s[x, y] * s_neighbors
            # Compute flip probability
            if T > 0:
                p = 1 / (1 + np.exp(dE / (T)))
            else:
                p = 0.5 if dE == 0 else 0 if dE > 0 else 1
            # Flip with probability p
            if rng.random() < p:
                s[x, y] *= -1
        
        # Check if this step is a measurement time
        if step in measurement_times:
            # Compute domain size
            m = np.mean(s)
            L_t[measurement_times == step] = 2 * np.sqrt(np.sum((np.fft.fft2(s) ** 2) / (L ** 2)) / (2 * m ** 2))
            # Apply running maximum
            L_t[measurement_times == step] = max(L_t[measurement_times == step], L_t[measurement_times == step - 1])
        
        # Check if this step is a snapshot time
        if step in snapshot_times:
            # Compute correlation function
            s_fft = np.fft.fft2(s)
            C = np.fft.ifft2(np.abs(s_fft) ** 2) / (L ** 2) - (np.mean(s) ** 2)
            C = np.mean(C[np.abs(np.indices(C.shape) - np.array([L//2, L//2])) < max_r], axis=(1, 2))
            stackC[snapshot_times == step] = C
            
            # Compute structure factor
            S = np.fft.fftshift(np.abs(np.fft.fft2(s)) ** 2) / (L ** 2)
            S = np.mean(S[np.abs(np.indices(S.shape) - np.array([L//2, L//2])) < max_r], axis=(1, 2))
            stackD[snapshot_times == step] = S
            
            # Compute dynamic scaling
            L_t_snapshot = L_t[measurement_times == step]
            kL = k * L_t_snapshot
            S_rescaled = S / (L_t_snapshot ** 2)
            stackE[snapshot_times == step, 0] = kL
            stackE[snapshot_times == step, 1] = S_rescaled
        
        # Check if this step is a measurement time for domain wall density
        if step in measurement_times:
            # Compute domain wall density
            rho = np.sum(s[i, j] != s[(i+1)%L, j] for i, j in np.ndindex((L, L))) / (2 * L ** 2)
            seriesB[measurement_times == step] = rho
        
        # Check if this step is a measurement time for domain size
        if step in measurement_times:
            seriesA[measurement_times == step, 0] = step
            seriesA[measurement_times == step, 1] = L_t[measurement_times == step]
        
        t += 1
    
    # Compute growth exponent
    mask = (t >= 100) & (L_t < L/4) & (L_t > 1.5)
    t_masked = t[mask]
    L_masked = L_t[mask]
    if len(t_masked) < 3:
        exponent = np.nan
        exponent_std_error = np.nan
        amplitude = np.nan
        fit_range = np.nan
        r_squared = np.nan
        notes = ['Insufficient data to fit power law']
    else:
        log_t = np.log(t_masked)
        log_L = np.log(L_masked)
        slope, intercept, r_value, p_value, std_err = np.polyfit(log_t, log_L, 1, full=True)
        exponent = slope
        exponent_std_error = std_err
        amplitude = np.exp(intercept)
        fit_range = [t_masked[0], t_masked[-1]]
        r_squared = r_value ** 2
        notes = []
    
    # Compute dynamic scaling quality
    collapse_check_applicable = False
    collapse_quality = None
    collapse_relative_rms_spread = None
    collapse_dimension = 2
    collapse_snapshot_times = []
    collapse_lengths = []
    collapse_x_range = []
    collapse_grid_size = 0
    notes = []
    
    # Save results
    np.save('seriesA.npy', seriesA)
    np.save('seriesB.npy', seriesB)
    np.save('stackC.npy', stackC)
    np.save('stackD.npy', stackD)
    np.save('stackE.npy', stackE)
    
    metadataA = {
        'collapse_check_applicable': collapse_check_applicable,
        'collapse_quality': collapse_quality,
        'collapse_relative_rms_spread': collapse_relative_rms_spread,
        'collapse_dimension': collapse_dimension,
        'collapse_snapshot_times': collapse_snapshot_times,
        'collapse_lengths': collapse_lengths,
        'collapse_x_range': collapse_x_range,
        'collapse_grid_size': collapse_grid_size,
        'notes': notes
    }
    with open('metadataA.json', 'w') as f:
        json.dump(metadataA, f)
    
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
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(s, cmap='gray', interpolation='none')
    plt.title('Spin Configuration')
    plt.subplot(2, 2, 2)
    plt.plot(seriesA[1, :], label='L(t)')
    plt.plot(fit_range, amplitude * fit_range ** exponent, '--', label='Fit')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Domain Size')
    plt.subplot(2, 2, 3)
    for i in range(len(snapshot_times)):
        plt.plot(stackC[i], label=f't={snapshot_times[i]}')
    plt.legend()
    plt.xlabel('r')
    plt.ylabel('C(r,t)')
    plt.subplot(2, 2, 4)
    for i in range(len(snapshot_times)):
        plt.plot(stackE[i, 0], stackE[i, 1], label=f't={snapshot_times[i]}')
    plt.xlabel('kL(t)')
    plt.ylabel('S(k,t)/L(t)^2')
    plt.legend()
    plt.savefig('figure.png')
    plt.close()