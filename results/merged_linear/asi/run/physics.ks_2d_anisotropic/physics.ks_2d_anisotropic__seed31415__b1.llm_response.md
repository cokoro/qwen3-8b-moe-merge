```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq, rfft
from scipy.optimize import curve_fit
import json
import os

# Load system information
with open('data/system_info.json') as f:
    system_info = json.load(f)

# Set constants
Lx = system_info['domain_length_x']
Ly = system_info['domain_length_y']
Nx = system_info['grid_size_x']
Ny = system_info['grid_size_y']
dx = system_info['dx']
dy = system_info['dy']
dt = 0.05  # Stable time step
observation_time = system_info['observation_time']
prediction_horizon = system_info['prediction_horizon']
target_time = system_info['target_time']

# Set physical parameters
alpha = system_info['system_coefficient_7']
nu = system_info['system_coefficient_5']
mu = system_info['system_coefficient_3']
gamma = system_info['system_coefficient_4']
delta = system_info['system_coefficient_1']
lambda_xy = system_info['system_coefficient_2']
lambda_yy = system_info['system_coefficient_6']
lambda_xx = 1.0

# Load data
field_evolution = np.load('data/field_evolution.npy')
initial_condition = np.load('data/initial_condition.npy')

# Precompute wavenumbers
kx = 2 * np.pi * fftfreq(Nx, d=dx)
ky = 2 * np.pi * fftfreq(Ny, d=dy)

# Define linear operator
def L_hat(kx, ky):
    return -alpha + kx**2 + mu * ky**2 - nu * kx**4 - 2 * gamma * kx**2 * ky**2 - delta * ky**4

# Precompute exponential terms
kx_fft = fftfreq(Nx, d=dx)
ky_fft = fftfreq(Ny, d=dy)
kx_mesh, ky_mesh = np.meshgrid(kx_fft, ky_fft, indexing='ij')
kx_mesh = kx_mesh.flatten()
ky_mesh = ky_mesh.flatten()

# ETDRK4 parameters
Q = np.zeros((4, Nx, Ny), dtype=np.complex128)
f1 = np.zeros((4, Nx, Ny), dtype=np.complex128)
f2 = np.zeros((4, Nx, Ny), dtype=np.complex128)
f3 = np.zeros((4, Nx, Ny), dtype=np.complex128)
Q[0] = 1.0
Q[1] = 2.0
Q[2] = 1.0
Q[3] = 2.0
f1[0] = 0.0
f1[1] = 0.0
f1[2] = 0.0
f1[3] = 1.0
f2[0] = 1.0
f2[1] = 0.0
f2[2] = 0.0
f2[3] = 0.0
f3[0] = 0.0
f3[1] = 0.0
f3[2] = 0.0
f3[3] = 1.0

# Define ETDRK4 step
def etdrk4_step(u_hat, L, N_hat, dt):
    # Compute exponential terms
    E = np.exp(L * dt)
    E2 = np.exp(L * dt / 2)
    # Compute nonlinear terms
    u = ifft2(u_hat).real
    u_x = ifft2(1j * kx_mesh * u_hat).real
    u_y = ifft2(1j * ky_mesh * u_hat).real
    f = 0.5 * lambda_xx * u_x**2 + lambda_xy * u_x * u_y + 0.5 * lambda_yy * u_y**2
    N = - (kx_mesh**2 + ky_mesh**2) * fft2(f)
    # Apply 2/3 rule dealiasing
    N = N * (np.abs(kx_mesh) < 2*Nx/3)
    N = N * (np.abs(ky_mesh) < 2*Ny/3)
    # ETDRK4 step
    a = E2 * u_hat + Q[0] * N
    b = E2 * u_hat + Q[1] * N
    c = E2 * a + Q[2] * (2 * N - N)
    u_hat_new = E * u_hat + f1[0] * N + 2 * f2[0] * (N + N) + f3[0] * N
    return u_hat_new

# Initialize field
u_hat = fft2(initial_condition)

# Evolve field to target time
current_time = 0.0
while current_time < target_time:
    # Compute linear operator
    L = L_hat(kx_mesh, ky_mesh)
    # Compute nonlinear terms
    u = ifft2(u_hat).real
    u_x = ifft2(1j * kx_mesh * u_hat).real
    u_y = ifft2(1j * ky_mesh * u_hat).real
    f = 0.5 * lambda_xx * u_x**2 + lambda_xy * u_x * u_y + 0.5 * lambda_yy * u_y**2
    N = - (kx_mesh**2 + ky_mesh**2) * fft2(f)
    # Apply 2/3 rule dealiasing
    N = N * (np.abs(kx_mesh) < 2*Nx/3)
    N = N * (np.abs(ky_mesh) < 2*Ny/3)
    # ETDRK4 step
    a = np.exp(L * dt / 2) * u_hat + Q[0] * N
    b = np.exp(L * dt / 2) * u_hat + Q[1] * N
    c = np.exp(L * dt / 2) * a + Q[2] * (2 * N - N)
    u_hat = np.exp(L * dt) * u_hat + f1[0] * N + 2 * f2[0] * (N + N) + f3[0] * N
    current_time += dt

# Save predicted field
predicted_field = ifft2(u_hat).real
np.save('predicted_field.npy', predicted_field)

# Compute spectral density
k = np.sqrt(kx_mesh**2 + ky_mesh**2)
k_sorted = np.sort(k)
k_unique = np.unique(k_sorted)
k_bins = np.logspace(np.log10(k_unique[0]), np.log10(k_unique[-1]), 65)

# Compute power spectral density
power = np.abs(fft2(predicted_field))**2 / (Nx * Ny)

# Compute radial average
radial_avg = np.zeros(len(k_bins) - 1)
for i in range(len(k_bins) - 1):
    radial_avg[i] = np.mean(power[(k >= k_bins[i]) & (k < k_bins[i+1])])

np.save('spectral_density.npy', radial_avg)

# Compute diagnostics
# Characteristic scales
x_freq = np.abs(rfft(field_evolution, axis=0))**2
y_freq = np.abs(rfft(field_evolution, axis=1))**2
x_peak = np.argmax(np.mean(x_freq, axis=1))
y_peak = np.argmax(np.mean(y_freq, axis=0))
characteristic_scale_1 = np.argmin(x_freq[x_peak, :]) * dx
characteristic_scale_2 = np.argmin(y_freq[:, y_peak]) * dy

# Temporal decay
autocorr = np.correlate(predicted_field.ravel(), predicted_field.ravel(), mode='full')[Nx*Ny//2:]
decay_time = np.argmin(autocorr[1:]) + 1
temporal_decay_scale = decay_time * dt

# Spectral tail
k_tail = k[k >= np.argmax(radial_avg) + 1]
radial_avg_tail = radial_avg[k >= np.argmax(radial_avg) + 1]
slope, _ = np.polyfit(np.log(k_tail), np.log(radial_avg_tail), 1)
decay_exponent = -slope

# Structural ratio
structural_ratio = max(characteristic_scale_1, characteristic_scale_2) / min(characteristic_scale_1, characteristic_scale_2)

# Save diagnostics
with open('data_exploration.json', 'w') as f:
    json.dump({
        'characteristic_scale_1': characteristic_scale_1,
        'characteristic_scale_2': characteristic_scale_2,
        'temporal_decay_scale': temporal_decay_scale,
        'decay_exponent': decay_exponent,
        'structural_ratio': structural_ratio
    }, f)

# Compute physical quantities
# Divergence measure
perturbation = np.random.default_rng(1729).standard_normal((Nx, Ny))
perturbation = perturbation - np.mean(perturbation)
perturbation = perturbation / np.linalg.norm(perturbation)
perturbation_field = initial_condition + 1e-8 * perturbation
perturbation_hat = fft2(perturbation_field)
perturbation_energy = np.abs(perturbation_hat)**2
perturbation_energy = np.sum(perturbation_energy) / (Nx * Ny)

# Coherence lengths
def autocorr(field, axis):
    field = field.reshape(-1, 1)
    field = np.concatenate((field, np.zeros((1, field.shape[1]))), axis=0)
    field = np.concatenate((field, np.zeros((field.shape[0], 1))), axis=1)
    autocorr = np.zeros((field.shape[0], field.shape[1]))
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            autocorr[i, j] = np.sum(field[i:i+2, j:j+2] * field[i:i+2, j:j+2])
    return autocorr

x_autocorr = autocorr(predicted_field, axis=0)
y_autocorr = autocorr(predicted_field, axis=1)
x_zero_crossing = np.argmax(x_autocorr[0, :] < 1e-6)
y_zero_crossing = np.argmax(y_autocorr[:, 0] < 1e-6)
coherence_length_1 = x_zero_crossing * dx
coherence_length_2 = y_zero_crossing * dy

# Characteristic scales
x_freq = np.abs(rfft(predicted_field, axis=0))**2
y_freq = np.abs(rfft(predicted_field, axis=1))**2
x_peak = np.argmax(np.mean(x_freq, axis=1))
y_peak = np.argmax(np.mean(y_freq, axis=0))
characteristic_scale_1 = np.argmin(x_freq[x_peak, :]) * dx
characteristic_scale_2 = np.argmin(y_freq[:, y_peak]) * dy

# Structural ratio
structural_ratio = max(characteristic_scale_1, characteristic_scale_2) / min(characteristic_scale_1, characteristic_scale_2)

# Save physical quantities
with open('physical_quantities.json', 'w') as f:
    json.dump({
        'divergence_measure': np.log(perturbation_energy / (Nx * Ny)),
        'coherence_length_1': coherence_length_1,
        'coherence_length_2': coherence_length_2,
        'characteristic_scale_1': characteristic_scale_1,
        'characteristic_scale_2': characteristic_scale_2,
        'structural_ratio': structural_ratio
    }, f)

# Plot field visualization
plt.figure(figsize=(12, 8))
for i in range(0, 21, 5):
    plt.subplot(3, 4, i+1)
    plt.imshow(field_evolution[i], cmap='viridis', extent=[0, Lx, 0, Ly])
    plt.colorbar()
    plt.title(f't={i*dt:.1f}')
plt.savefig('field_visualization.png')

# Plot spectral visualization
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(fft2(predicted_field)), cmap='viridis', extent=[0, Lx, 0, Ly])
plt.colorbar()
plt.title('2D Spectral Heatmap')
plt.subplot(1, 2, 2)
plt.semilogy(k_unique, radial_avg, 'o-', label='Radial Average')
plt.xlabel('k')
plt.ylabel('PSD')
plt.title('Radial Spectral Density')
plt.legend()
plt.savefig('spectral_visualization.png')
```