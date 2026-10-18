import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import json
import os

# Load system information
with open('data/system_info.json', 'r') as f:
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
lambda_xx = 1.0  # Fixed

# Load field evolution and initial condition
field_evolution = np.load('data/field_evolution.npy')
initial_condition = np.load('data/initial_condition.npy')

# Precompute kx and ky
kx = 2 * np.pi * fftfreq(Nx, d=dx)
ky = 2 * np.pi * fftfreq(Ny, d=dy)

# ETDRK4 parameters
dt = 0.04
E = expm(dt * L_hat(kx, ky))
E2 = expm(dt * L_hat(kx, ky) / 2)

# ETDRK4 coefficient arrays
Q = np.zeros((4, Nx, Ny), dtype=np.complex128)
f1 = np.zeros((4, Nx, Ny), dtype=np.complex128)
f2 = np.zeros((4, Nx, Ny), dtype=np.complex128)
f3 = np.zeros((4, Nx, Ny), dtype=np.complex128)

# Define the linear operator
def L_hat(kx, ky):
    return -alpha + kx**2 + mu * ky**2 - nu * kx**4 - 2 * gamma * kx**2 * ky**2 - delta * ky**4

# Define the nonlinear term
def nonlinear_term(u_hat):
    u_x = ifft2(1j * kx * u_hat)
    u_y = ifft2(1j * ky * u_hat)
    f = (lambda_xx / 2) * u_x**2 + lambda_xy * u_x * u_y + (lambda_yy / 2) * u_y**2
    N_hat = - (kx**2 + ky**2) * fft2(f)
    return N_hat

# ETDRK4 time integration
def etdrk4_step(u_hat, dt):
    N_hat = nonlinear_term(u_hat)
    a = E2 * u_hat + Q[0] * N_hat
    b = E2 * u_hat + Q[...]
    c = E2 * a +1 * (2 * N_hat(b) - N_hat(u_hat))
    u_hat_new = E * u_hat
    u_hat_new += f1 * N_hat(u_hat)
    u_hat_new += 2 * f2 * (N_hat(a) + N_hat(b))
    u_hat_new += f3 * N_hat(c)
    return u_hat_new

# Time integration
u_hat = fft2(initial_condition)
t = 0.0
while t < target_time:
    u_hat = etdrk4_step(u_hat, dt)
    t += dt

# Save predicted field
predicted_field = np.real(ifft2(u_hat))
np.save('predicted_field.npy', predicted_field)

# Compute spectral density
k = np.sqrt(kx**2 + ky**2)
k_max = np.max(k)
k_bins = np.linspace(0, k_max, 65)
spectral_density = np.zeros(len(k_bins) - 1, dtype=np.float64)
for i in range(len(k_bins) - 1):
    mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
    power = np.abs(fft2(predicted_field))**2 / (Nx * Ny)
    spectral_density[i] = np.mean(power[mask])
np.save('spectral_density