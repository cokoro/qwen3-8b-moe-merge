```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import diags

# Load configuration
config = np.load('case_config.json', allow_pickle=True).item()

# Load boundary conditions
bc_primary = np.load('bc_primary.npy')  # [y, x]
bc_secondary = np.load('bc_secondary.npy')  # [y, x]
domain_mask = np.load('domain_mask.npy')  # [y, x]
sample_a_y = np.load('sample_track_a.npy')  # y coordinates for u profile
sample_b_x = np.load('sample_track_b.npy')  # x coordinates for v profile
sample_c_x = np.load('sample_track_c.npy')  # x coordinates for vorticity trace

# Setup grid
ny, nx = 129, 129
h = 1.0 / (nx - 1)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

# Convert to interior indices
interior_mask = domain_mask.copy()
interior_mask[0, :] = interior_mask[-1, :] = interior_mask[:, 0] = interior_mask[:, -1] = False
interior_mask = np.where(interior_mask)[0].astype(int)

# Initialize vorticity field
omega = np.zeros((ny, nx), dtype=np.float32)

# Step 1: Initial guess from lid-driven cavity closure
for iy in range(1, ny-1):
    for ix in range(1, nx-1):
        if interior_mask[iy, ix]:
            # Use Ghia-style wall closure
            pass

# Step 2: Iterative solve using vorticity equation
def solve_vorticity(omega, bc_primary, bc_secondary, h, ny, nx, n_steps=28000):
    for _ in range(n_steps):
        # Compute divergence
        div = np.zeros((ny, nx), dtype=np.float32)
        div[1:-1, 1:-1] = (omega[2:-1, 1:-1] - omega[0:-2, 1:-1]) / h \
                         + (omega[1:-1, 2:-1] - omega[1:-1, 0:-2]) / h
        # Update vorticity
        omega[1:-1, 1:-1] -= 0.5 * div * h ** 2
    return omega

omega = solve_vorticity(omega, bc_primary, bc_secondary, h, ny, nx)

# Step 3: Compute streamfunction via Poisson solve
A = diags([-1.0, 2.0, -1.0], [0, 1, -1], shape=(ny, nx), format='csr')
A = A @ A.T

# Apply Neumann boundary conditions
def apply_neumann(A, bc):
    for i in range(ny):
        A[i, :] = A[i, :] * (1 - (i == 0 or i == ny-1))
    return A

A = apply_neumann(A, bc_primary)

# Solve Poisson equation for streamfunction
psi = np.zeros((ny, nx), dtype=np.float32)
psi[1:-1, 1:-1] = spla.spsolve(A, -omega[1:-1, 1:-1].flatten()).reshape(ny-2, nx-2)

# Step 4: Compute u, v profiles
report_a = np.zeros(145, dtype=np.float32)
report_b = np.zeros(145, dtype=np.float32)
report_c = np.zeros(145, dtype=np.float32)

# Sample u(x=0.5, y=sample_a_y)
report_a[:] = np.gradient(psi, axis=0)[1:-10, 64]

# Sample v(x=sample_b_x, y=0.5)
report_b[:] = -np.gradient(psi, axis=1)[64, 1:-10]

# Sample vorticity on first interior row below lid
report_c[:] = omega[1, 1:-1]

# Step 5: Save outputs
np.save('results/report_a.npy', report_a.astype(np.float32))
np.save('results/report_b.npy', report_b.astype(np.float32))
np.save('results/report_c.npy', report_c.astype(np.float32))
np.save('results/field_a.npy', omega[interior_mask].astype(np.float32))
np.save('results/field_b.npy', psi[interior_mask].astype(np.float32))

# Step 6: Generate visualization
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(x, y, psi[interior_mask], 20, cmap='viridis')
plt.colorbar(contour)
plt.savefig('results/overview.png', dpi=300)
plt.close()
```