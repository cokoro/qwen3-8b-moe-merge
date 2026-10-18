```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Read parameters
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

# Simulation setup
grid = initial_state.copy()
grid = grid.astype(np.int8)

# RNG setup
rng = np.random.default_rng(dynamics_seed)

# Precompute sublattice indices
ii, jj = np.mgrid[0:L, 0:L]
sublattice_mask = (ii + jj) % 2 == 0
sublattice_mask = sublattice_mask.astype(np.bool)

# Step 1: Glauber dynamics
def update_step(grid, T, sublattice_mask, rng):
    # Compute neighbor sums
    neighbors = np.zeros_like(grid)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = (np.roll(grid, dx, axis=0) + np.roll(grid, dy, axis=1)).astype(np.int8)
        neighbors += nx * ny
    delta_e = 2 * (neighbors * grid).astype(np.int8)

    # Compute probabilities
    if T == 0:
        p = (delta_e < 0).astype(np.float64)
        p[delta_e == 0] = 0.5
    else:
        p = 1.0 / (1.0 + np.exp(delta_e / T))

    # Flip spins
    flip = rng.random(len(sublattice_mask)).astype(np.float64) < p[sublattice_mask]
    grid[sublattice_mask] = np.where(flip, -grid[sublattice_mask], grid[sublattice_mask])

    return grid

# Step 2: Correlation function
def compute_correlation(grid, L, max_r):
    m = np.mean(grid)
    fft_grid = np.fft.fft2(grid)
    corr = np.fft.ifft2(np.abs(fft_grid)**2).real / (L**2)
    corr -= m**2
    # Circular average
    r = np.arange(0, max_r + 1)
    radial = np.zeros_like(r)
    for i in range(len(r)):
        radial[i] = np.mean(corr[r == i])
    return radial

# Step 3: Structure factor
def compute_structure_factor(grid, L, max_k):
    fft_grid = np.fft.fftshift(np.fft.fft2(grid))
    k = np.fft.fftshift(np.fft.fftfreq(L) * 2 * np.pi)
    k_vals = k[:L//2 + 1]
    k_squared = k_vals**2
    S = np.abs(fft_grid)**2 / (L**2)
    # Radial average
    radial = np.zeros_like(k_vals)
    for i in range(len(k_vals)):
        radial[i] = np.mean(S[np.sqrt(k_squared) == k_vals[i]])
    return k_vals, radial

# Step 4: Domain size
def estimate_L(t, corr, max_r):
    L_t = 0
    for r in range(1, max_r + 1):
        if corr[r] < 0.5 * corr[0]:
            L_t = r
            break
    return L_t

# Step 5: Fit exponent
def fit_exponent(L_t, t, max_t, max_L):
    mask = (t >= 100) & (L_t > 1.5) & (L_t < max_L / 4)
    valid = mask & (L_t > np.roll(L_t, 1))
    valid = valid[1:]
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan, []
    t_valid = t[valid]
    L_valid = L_t[valid]
    log_t = np.log(t_valid)
    log_L = np.log(L_valid)
    A, n = np.polyfit(log_t, log_L, 1)
    R_sq = np.corrcoef(log_t, log_L)[0, 1]**2
    return n, np.std(log_L - n*log_t), np.exp(A), t_valid.min(), t_valid.max(), R_sq

# Step 6: Domain interface density
def compute_rho(grid, L):
    rho = 0
    for i in range(L):
        for j in range(L):
            s = grid[i,j]
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni = (i + dx) % L
                nj = (j + dy) % L
                if grid[ni, nj] != s:
                    rho += 1
    return rho / (2 * L**2)

# Step 7: Dynamic scaling collapse
def collapse_data(k_vals, S_vals, L_t, max_k):
    x = k_vals * L_t
    y = S_vals / (L_t**2)
    return x, y

# Main simulation loop
seriesA = []
seriesB = []
stackC = []
stackD = []
stackE = []

for step in range(n_steps):
    if step in snapshot_times:
        # Compute correlation and structure factor
        corr = compute_correlation(grid, L, max_r)
        k_vals, S_vals = compute_structure_factor(grid, L, L)
        stackC.append(corr)
        stackD.append(S_vals)
        # Compute collapse data
        x, y = collapse_data(k_vals, S_vals, L, L)
        stackE.append((x, y))
    
    if step in measurement_times:
        # Compute domain size
        corr = compute_correlation(grid, L, max_r)
        L_t = estimate_L(step, corr, max_r)
        # Apply running maximum
        if seriesA and L_t < seriesA[-1][1]:
            L_t = seriesA[-1][1]
        seriesA.append((step, L_t))
        # Compute interface density
        rho = compute_rho(grid, L)
        seriesB.append((step, rho))
    
    # Update grid
    grid = update_step(grid, T, sublattice_mask, rng)

# Output arrays
np.save('seriesA.npy', np.array(seriesA))
np.save('seriesB.npy', np.array(seriesB))
np.save('stackC.npy', np.array(stackC))
np.save('stackD.npy', np.array(stackD))
np.save('stackE.npy', np.array(stackE))

# Metadata
metadata = {
    'collapse_check_applicable': False,
    'collapse_quality': None,
    'collapse_relative_rms_spread': None,
    'collapse_dimension': 2,
    'collapse_snapshot_times': [],
    'collapse_lengths': [],
    'collapse_x_range': [0, 0],
    'collapse_grid_size': 0,
    'notes': []
}

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# Spin snapshots
for t in snapshot_times:
    axes[0, 0].imshow(grid, cmap='gray', extent=[0, L, 0, L], origin='lower')
    axes[0, 0].set_title(f'Snapshot at t={t}')
# L(t) vs t
t = [s[0] for s in seriesA]
L_t = [s[1] for s in seriesA]
axes[0, 1].loglog(t, L_t, 'o-')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('L(t)')
# C(r,t)
for i in range(len(stackC)):
    r = np.arange(0, max_r + 1)
    axes[1, 0].plot(r, stackC[i], label=f't={snapshot_times[i]}')
axes[1, 0].set_xlabel('r')
axes[1, 0].set_ylabel('C(r,t)')
# Dynamic scaling collapse
x = stackE[0][0]
y = stackE[0][1]
axes[1, 1].plot(x, y, 'o-')
axes[1, 1].set_xlabel('kL(t)')
axes[1, 1].set_ylabel('S(k,t)/L(t)^2')
plt.tight_layout()
plt.savefig('figure.png')
```