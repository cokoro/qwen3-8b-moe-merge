import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import os

# Read input data
case_config = np.load('case_config.json', allow_pickle=True).item()
domain_mask = np.load('domain_mask.npy')
bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
sample_track_a = np.load('sample_track_a.npy')
sample_track_b = np.load('sample_track_b.npy')
sample_track_c = np.load('sample_track_c.npy')

# Initialize fields
ny, nx = domain_mask.shape
h = 1.0 / (nx - 1)
omega = np.zeros((ny, nx), dtype=np.float32)
psi = np.zeros((ny, nx), dtype=np.float32)

# Set boundary conditions
omega[0, 1:-1] = bc_primary[0, 1:-1]  # top wall
omega[-1, 1:-1] = bc_secondary[-1, 1:-1]  # bottom wall
omega[1:-1, 0] = bc_secondary[1:-1, 0]  # left wall
omega[1:-1, -1] = bc_primary[1:-1, -1]  # right wall

# Set up the discretization matrices
diag = np.zeros((ny, nx), dtype=np.float32)
offdiag = np.zeros((ny, nx), dtype=np.float32)
for i in range(1, ny-1):
    diag[i, 1:-1] = 2.0 / (h**2)
    offdiag[i, 1:-1] = -1.0 / (h**2)

# Build the laplacian matrix
laplacian = csr_matrix((ny*nx, ny*nx))
for i in range(1, ny-1):
    for j in range(1, nx-1):
        laplacian[i*(nx)+j, i*(nx)+j] += diag[i, j]
        laplacian[i*(nx)+j, (i-1)*(nx)+j] += offdiag[i, j]
        laplacian[i*(nx)+j, (i+1)*(nx)+j] += offdiag[i, j]
        laplacian[i*(nx)+j, i*(nx)+j-1] += offdiag[i, j]
        laplacian[i*(nx)+j, i*(nx)+j+1] += offdiag[i, j]

# Solve the Poisson equation
psi[1:-1, 1:-1] = spsolve(laplacian[1:-1, 1:-1].tocsc(), -omega[1:-1, 1:-1].flatten()).reshape((ny-2, nx-2))

# Set up the vorticity equation
dt = 0.01
for _ in range(28000):
    # Compute convection terms
    omega_convection = np.zeros((ny, nx), dtype=np.float32)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            omega_convection[i, j] = -omega[i, j] * (omega[i, j+1] - omega[i, j-1]) / (2*h)
    # Compute diffusion terms
    omega_diffusion = np.zeros((ny, nx), dtype=np.float32)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            omega_diffusion[i, j] = (omega[i-1, j] - 2*omega[i, j] + omega[i+1, j]) / h**2
    # Update vorticity
    omega[1:-1, 1:-1] += dt * (omega_convection[1:-1, 1:-1] + omega_diffusion[1:-1, 1:-1])

# Compute vorticity at the lid
omega_lid = np.zeros((ny, nx), dtype=np.float32)
for j in range(1, nx-1):
    omega_lid[0, j] = -(4*omega[1, j] - 0.5*omega[2, j]) / h**2 - 3*1.0/h
omega[0, 1:-1] = omega_lid[0, 1:-1]

# Save the vorticity field
np.save('results/field_a.npy', omega[1:-1, 1:-1].astype(np.float32))

# Compute the streamfunction
psi[1:-1, 1:-1] = spsolve(laplacian[1:-1, 1:-1].tocsc(), -omega[1:-1, 1:-1].flatten()).reshape((ny-2, nx-2))

# Save the streamfunction field
np.save('results/field_b.npy', psi[1:-1, 1:-1].astype(np.float32))

# Compute the u profile
u_profile = np.zeros((len(sample_track_a),), dtype=np.float32)
for i, y in enumerate(sample_track_a):
    u_profile[i] = (psi[1:-1, 1:-1][y, 0] - psi[1:-1, 1:-1][y, 1]) / h

# Compute the v profile
v_profile = np.zeros((len(sample_track_b),), dtype=np.float32)
for i, x in enumerate(sample_track_b):
    v_profile[i] = (psi[1:-1, 1:-1][0, x] - psi[1:-1, 1:-1][1, x]) / h

# Compute the vorticity trace
vorticity_trace = np.zeros((len(sample_track_c),), dtype=np.float32)
for i, x in enumerate(sample_track_c):
    vorticity_trace[i] = omega[1, x]

# Save the u profile
np.save('results/report_a.npy', u_profile.astype(np.float32))

# Save the v profile
np.save('results/report_b.npy', v_profile.astype(np.float32))

# Save the vorticity trace
np.save('results/report_c.npy', vorticity_trace.astype(np.float32))

# Plot the vorticity field
plt.figure(figsize=(10, 10))
plt.imshow(omega[1:-1, 1:-1], cmap='viridis', extent=[0, 1, 0, 1])
plt.colorbar()
plt.title('Vorticity Field')
plt.savefig('results/overview.png')
plt.close()