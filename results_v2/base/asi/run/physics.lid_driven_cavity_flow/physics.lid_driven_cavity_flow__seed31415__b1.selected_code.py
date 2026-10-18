import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Load data
case_config = np.load('case_config.json', allow_pickle=True).item()
bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
domain_mask = np.load('domain_mask.npy')
sample_track_a = np.load('sample_track_a.npy')
sample_track_b = np.load('sample_track_b.npy')
sample_track_c = np.load('sample_track_c.npy')

# Configuration
ny, nx = 129, 129
ny_int, nx_int = 127, 127
h = 1.0 / (nx - 1)
Re = 1000
nu = 1.0 / Re
U_lid = 1.0
tolerance = 2.0e-05
max_steps = 28000
output_dir = 'results'

# Initialize vorticity field
omega = np.zeros((ny, nx), dtype=np.float32)
psi = np.zeros((ny, nx), dtype=np.float32)

# Set boundary conditions for vorticity
for y in range(ny):
    for x in range(nx):
        if not domain_mask[y, x]:
            if y == 0:
                # Bottom wall
                omega[y, x] = 0.0
            elif y == ny - 1:
                # Top wall
                if x == nx - 1:
                    omega[y, x] = 0.0
                else:
                    omega[y, x] = bc_primary[y, x]
            elif x == 0:
                # Left wall
                omega[y, x] = 0.0
            elif x == nx - 1:
                # Right wall
                omega[y, x] = 0.0

# Initialize streamfunction
psi = np.zeros((ny, nx), dtype=np.float32)

# Define Laplacian matrix for streamfunction
def laplacian_matrix(nx, ny, h):
    diagonals = np.zeros((3, ny, nx))
    diagonals[0, :, :] = -2.0 / h**2
    diagonals[1, :, :] = 1.0 / h**2
    diagonals[2, :, :] = 1.0 / h**2
    laplacian = diags(diagonals.flatten(), [0, -1, 1], shape=(ny * nx, ny * nx))
    laplacian = laplacian.tocsc()
    return laplacian

laplacian = laplacian_matrix(nx, ny, h)

# Time-stepping parameters
dt = 1.0
time = 0.0
residual = 1.0
steps = 0

# Vorticity update
def update_omega(omega, psi, dt, nu, h):
    omega_new = np.zeros_like(omega)
    for y in range(ny):
        for x in range(nx):
            if domain_mask[y, x]:
                # Interior points
                omega_new[y, x] = omega[y, x] - dt * (nu * laplacian.dot(omega.flatten()).reshape(ny, nx)[y, x] - omega[y, x] * (omega[y, x] * (omega[y, x] - omega[y, x])) / (nu * h**2))
    return omega_new

# Solve for streamfunction
def solve_streamfunction(omega, psi, h):
    for y in range(ny):
        for x in range(nx):
            if domain_mask[y, x]:
                psi[y, x] = (omega[y, x] - (omega[y, x] - omega[y, x]) * (omega[y, x] - omega[y, x])) / (nu * h**2)
    return psi

# Main solver loop
while residual > tolerance and steps < max_steps:
    omega = update_omega(omega, psi, dt, nu, h)
    psi = solve_streamfunction(omega, psi, h)
    residual = np.max(np.abs(omega - update_omega(omega, psi, dt, nu, h)))
    steps += 1

# Save interior fields
omega_int = omega[1:-1, 1:-1]
psi_int = psi[1:-1, 1:-1]

np.save(f'{output_dir}/field_a.npy', omega_int.astype(np.float32))
np.save(f'{output_dir}/field_b.npy', psi_int.astype(np.float32))

# Extract report_a: u(x=0.5, y)
x_center = nx // 2
y_coords = sample_track_a
u_profile = np.zeros_like(y_coords, dtype=np.float32)
for i, y in enumerate(y_coords):
    u_profile[i] = (psi_int[y, x_center] - psi_int[y, x_center - 1]) / h

np.save(f'{output_dir}/report_a.npy', u_profile.astype(np.float32))

# Extract report_b: v(x, y=0.5)
y_center = ny // 2
x_coords = sample_track_b
v_profile = np.zeros_like(x_coords, dtype=np.float32)
for i, x in enumerate(x_coords):
    v_profile[i] = -(psi_int[y_center, x] - psi_int[y_center - 1, x]) / h

np.save(f'{output_dir}/report_b.npy', v_profile.astype(np.float32))

# Extract report_c: vorticity near lid
y_lid = ny - 2
x_coords = sample_track_c
omega_profile = np.zeros_like(x_coords, dtype=np.float32)
for i, x in enumerate(x_coords):
    omega_profile[i] = omega[y_lid, x]

np.save(f'{output_dir}/report_c.npy', omega_profile.astype(np.float32))

# Plot overview
plt.figure(figsize=(8, 8))
plt.contourf(psi_int, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Streamfunction Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f'{output_dir}/overview.png')
plt.close()