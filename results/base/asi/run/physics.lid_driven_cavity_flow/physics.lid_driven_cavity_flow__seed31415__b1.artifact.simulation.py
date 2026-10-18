import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import os

# Load data
case_config = np.load('case_config.json', allow_pickle=True).item()
bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
domain_mask = np.load('domain_mask.npy')
sample_track_a = np.load('sample_track_a.npy')
sample_track_b = np.load('sample_track_b.npy')
sample_track_c = np.load('sample_track_c.npy')

# Configuration
h = 1.0 / (128)  # grid spacing
Re = 1000
nu = 0.000909
Ny, Nx = 129, 129
Ny_int, Nx_int = 127, 127
h2 = h * h
h4 = h * h * h * h
h6 = h * h * h * h * h * h
h8 = h * h * h * h * h * h * h * h

# Initialize vorticity field
omega = np.zeros((Ny, Nx), dtype=np.float32)
psi = np.zeros((Ny, Nx), dtype=np.float32)

# Set boundary conditions for vorticity
for y in range(Ny):
    for x in range(Nx):
        if domain_mask[y, x]:
            if y == 0 or y == Ny - 1 or x == 0 or x == Nx - 1:
                omega[y, x] = 0.0
            else:
                omega[y, x] = 0.0

# Set moving lid vorticity using Ghia-style closure
for x in range(Nx):
    y = Ny - 1
    psi_below = psi[y - 1, x]
    psi_two_below = psi[y - 2, x]
    omega_top = -(4 * psi_below - 0.5 * psi_two_below) / h2 - 3 * bc_primary[y, x] / h
    omega[y, x] = omega_top

# Define finite difference operators
def laplacian(psi):
    lap = np.zeros_like(psi, dtype=np.float32)
    for y in range(1, Ny - 1):
        for x in range(1, Nx - 1):
            lap[y, x] = (psi[y + 1, x] - 2 * psi[y, x] + psi[y - 1, x]) / h2 + \
                        (psi[y, x + 1] - 2 * psi[y, x] + psi[y, x - 1]) / h2
    return lap

def divergence(omega):
    div = np.zeros_like(omega, dtype=np.float32)
    for y in range(1, Ny - 1):
        for x in range(1, Nx - 1):
            div[y, x] = (omega[y, x + 1] - omega[y, x - 1]) / (2 * h) + \
                        (omega[y + 1, x] - omega[y - 1, x]) / (2 * h)
    return div

def vorticity_update(omega, psi, dt):
    lap_psi = laplacian(psi)
    domega_dt = -divergence(omega) + nu * laplacian(omega)
    omega_new = omega + dt * domega_dt
    return omega_new

# Solve for streamfunction
def solve_streamfunction(omega):
    A = lil_matrix((Ny * Nx, Ny * Nx), dtype=np.float32)
    b = np.zeros(Ny * Nx, dtype=np.float32)
    for y in range(Ny):
        for x in range(Nx):
            if not domain_mask[y, x]:
                continue
            idx = y * Nx + x
            b[idx] = -omega[y, x]
            # Laplacian terms
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < Ny and 0 <= nx < Nx and domain_mask[ny, nx]:
                    A[idx, ny * Nx + nx] -= 1.0 / h2
            A[idx, idx] += 2.0 / h2
    A = A.tocsc()
    psi = spsolve(A, b)
    psi = psi.reshape((Ny, Nx))
    return psi

# Time-stepping
dt = 0.001
for _ in range(28000):
    psi = solve_streamfunction(omega)
    omega = vorticity_update(omega, psi, dt)

# Extract interior fields
omega_int = omega[1:-1, 1:-1]
psi_int = psi[1:-1, 1:-1]

# Save fields
np.save('results/field_a.npy', omega_int.astype(np.float32))
np.save('results/field_b.npy', psi_int.astype(np.float32))

# Sample report_a
report_a = np.zeros(len(sample_track_a), dtype=np.float32)
for i, y in enumerate(sample_track_a):
    x = 64  # x=0.5
    report_a[i] = psi_int[y, x]

np.save('results/report_a.npy', report_a)

# Sample report_b
report_b = np.zeros(len(sample_track_b), dtype=np.float32)
for i, x in enumerate(sample_track_b):
    y = 64  # y=0.5
    report_b[i] = -psi_int[y, x]

np.save('results/report_b.npy', report_b)

# Sample report_c
report_c = np.zeros(len(sample_track_c), dtype=np.float32)
for i, x in enumerate(sample_track_c):
    y = 1  # first interior row below lid
    report_c[i] = omega_int[y, x]

np.save('results/report_c.npy', report_c)

# Plot overview
plt.figure(figsize=(8, 8))
plt.imshow(omega_int, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar()
plt.title('Vorticity Field')
plt.savefig('results/overview.png')
plt.close()