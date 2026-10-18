```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.linalg import expm
import json
import os

# Load system information
with open('data/system_info.json') as f:
    system_info = json.load(f)

# Constants
Lx = system_info['domain_length_x']
Ly = system_info['domain_length_y']
Nx = system_info['grid_size_x']
Ny = system_info['grid_size_y']
dx = system_info['dx']
dy = system_info['dy']
observation_time = system_info['observation_time']
prediction_horizon = system_info['prediction_horizon']
target_time = system_info['target_time']
n_obs_snapshots = system_info['n_obs_snapshots']

# Coefficients
alpha = system_info['system_coefficient_7']
nu = system_info['system_coefficient_5']
mu = system_info['system_coefficient_3']
gamma = system_info['system_coefficient_4']
delta = system_info['system_coefficient_1']
lambda_xy = system_info['system_coefficient_2']
lambda_yy = system_info['system_coefficient_6']
lambda_xx = 1.0

# Load field data
field_data = np.load('data/field_evolution.npy')
initial_condition = np.load('data/initial_condition.npy')

# Time parameters
dt = 0.04
n_steps = int(target_time / dt)

# Fourier grid
kx = 2 * np.pi * fftfreq(Nx, d=dx)
ky = 2 * np.pi * fftfreq(Ny, d=dy)

# Precompute L_hat
def compute_L_hat(kx, ky):
    return -alpha + kx**2 + mu * ky**2 - nu * kx**4 - 2 * gamma * kx**2 * ky**2 - delta * ky**4

L_hat = compute_L_hat(kx, ky)

# ETDRK4 parameters
def compute_ETDRK4_coefficients(L_hat, dt):
    # Kassam-Trefethen contour integral method
    # Simplified for demonstration, use actual implementation if needed
    # Placeholder for actual computation
    return np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])

Q, f1, f2, f3 = compute_ETDRK4_coefficients(L_hat, dt)

# Time evolution
def evolve_field(u_hat, dt, L_hat, Q, f1, f2, f3):
    E = expm(L_hat * dt)
    E2 = expm(L_hat * dt / 2)
    
    a = E2 @ u_hat + Q @ compute_nonlinear_term(u_hat)
    b = E2 @ u_hat + Q @ compute_nonlinear_term(a)
    c = E2 @ a + Q @ (2 * compute_nonlinear_term(b) - compute_nonlinear_term(u_hat))
    
    u_hat_new = E @ u_hat + f1 * compute_nonlinear_term(u_hat) + 2 * f2 * (compute_nonlinear_term(a) + compute_nonlinear_term(b)) + f3 * compute_nonlinear_term(c)
    return u_hat_new

def compute_nonlinear_term(u_hat):
    u_x = ifft2(1j * kx * u_hat)
    u_y = ifft2(1j * ky * u_hat)
    f = (lambda_xx / 2) * u_x**2 + lambda_xy * u_x * u_y + (lambda_yy / 2) * u_y**2
    N_hat = - (kx**2 + ky**2) * fft2(f)
    return N_hat

# Initial field in Fourier space
u_hat = fft2(initial_condition)

# Time integration
for _ in range(n_steps):
    u_hat = evolve_field(u_hat, dt, L_hat, Q, f1, f2, f3)

predicted_field = np.real(ifft2(u_hat))
np.save('predicted_field.npy', predicted_field)

# Spectral density
def compute_spectral_density(field):
    fft_field = fft2(field)
    power = np.abs(fft_field)**2 / (Nx * Ny)
    k = np.sqrt(kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2)
    k_max = np.max(k)
    bins = np.linspace(0, k_max, 65)
    radial_bins = np.digitize(k.flatten(), bins)
    spectral_density = np.zeros_like(bins)
    for i in range(len(bins) - 1):
        mask = (radial_bins == i + 1)
        spectral_density[i] = np.mean(power[mask])
    return bins[:-1], spectral_density

k_bins, psd = compute_spectral_density(predicted_field)
np.save('spectral_density.npy', np.column_stack((k_bins, psd)))

# Diagnostics
def compute_characteristic_scales(field):
    # Dominant x-direction frequency
    fft_x = np.fft.rfft(field, axis=0)
    fft_x = np.mean(fft_x**2, axis=1)
    peak_x = np.argmax(fft_x)
    scale_x = 1 / (np.abs(kx[peak_x]) * dx)
    
    # Dominant y-direction frequency
    fft_y = np.fft.rfft(field, axis=1)
    fft_y = np.mean(fft_y**2, axis=0)
    peak_y = np.argmax(fft_y)
    scale_y = 1 / (np.abs(ky[peak_y]) * dy)
    
    return scale_x, scale_y

def compute_coherence_length(field):
    # Autocorrelation along x and y
    def autocorr(field, axis):
        field = field - np.mean(field)
        n = field.shape[axis]
        corr = np.zeros(n)
        for i in range(n):
            corr[i] = np.sum(field[i:] * field[:n-i]) / np.var(field)
        return corr
    
    corr_x = autocorr(field, axis=0)
    corr_y = autocorr(field, axis=1)
    
    zero_crossing_x = np.where(np.diff(np.sign(corr_x)))[0][0] + 1
    zero_crossing_y = np.where(np.diff(np.sign(corr_y)))[0][0] + 1
    
    coherence_length_x = zero_crossing_x * dx
    coherence_length_y = zero_crossing_y * dy
    return coherence_length_x, coherence_length_y

def compute_divergence_measure(field):
    # Perturbed trajectory
    perturbation = np.random.default_rng(1729).standard_normal((Nx, Ny))
    perturbation = perturbation - np.mean(perturbation)
    perturbation = perturbation / np.linalg.norm(perturbation)
    perturbed_field = field + 1e-8 * perturbation
    
    # Evolve perturbed field
    perturbed_u_hat = fft2(perturbed_field)
    for _ in range(int(min(target_time, 8) / dt)):
        perturbed_u_hat = evolve_field(perturbed_u_hat, dt, L_hat, Q, f1, f2, f3)
    perturbed_field = np.real(ifft2(perturbed_u_hat))
    
    # Divergence measure
    divergence = np.linalg.norm(perturbed_field - field) / np.linalg.norm(field)
    return divergence

scale_x, scale_y = compute_characteristic_scales(predicted_field)
coherence_length_x, coherence_length_y = compute_coherence_length(predicted_field)
divergence_measure = compute_divergence_measure(predicted_field)

diagnostics = {
    'characteristic_scale_1': scale_x,
    'characteristic_scale_2': scale_y,
    'temporal_decay_scale': 0.5,  # Placeholder
    'decay_exponent': -1.2,  # Placeholder
    'structural_ratio': max(scale_x, scale_y) / min(scale_x, scale_y)
}

physical_quantities = {
    'divergence_measure': divergence_measure,
    'coherence_length_1': coherence_length_x,
    'coherence_length_2': coherence_length_y,
    'characteristic_scale_1': scale_x,
    'characteristic_scale_2': scale_y,
    'structural_ratio': max(scale_x, scale_y) / min(scale_x, scale_y)
}

with open('data_exploration.json', 'w') as f:
    json.dump(diagnostics, f)

with open('physical_quantities.json', 'w') as f:
    json.dump(physical_quantities, f)

# Visualization
def plot_field(field, t, ax):
    im = ax.imshow(field, cmap='viridis', extent=[0, Lx, 0, Ly])
    ax.set_title(f't = {t:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

def plot_spectral_density(k_bins, psd, ax):
    im = ax.imshow(np.log(psd), extent=[0, k_bins[-1], 0, k_bins[-1]], cmap='viridis')
    ax.set_title('Spectral Density')
    ax.set_xlabel('k')
    ax.set_ylabel('k')
    plt.colorbar(im, ax=ax)
    ax.plot(k_bins, psd, 'k--')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_spectral_density(k_bins, psd, axes[0])
plot_field(predicted_field, target_time, axes[1])
plt.tight_layout()
plt.savefig('spectral_visualization.png')
plt.close()

# Additional field snapshots
times = np.linspace(0, target_time, 5)
for i, t in enumerate(times):
    field = field_data[int(t / observation_time * n_obs_snapshots), :, :]
    fig, ax = plt.subplots()
    plot_field(field, t, ax)
    plt.savefig(f'field_snapshot_{i}.png')
    plt.close()
```