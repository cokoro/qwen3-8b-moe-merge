```python
import numpy as np
import matplotlib.pyplot as plt

# Load input data
domain_mask = np.load('domain_mask.npy')
bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
sample_a = np.load('sample_track_a.npy')
sample_b = np.load('sample_track_b.npy')
sample_c = np.load('sample_track_c.npy')

# Constants
N = 127
h = 1.0 / (N + 1)
Re = 1000.0
nu = 1.0 / Re

# Initialize vorticity
omega = np.zeros((N + 2, N + 2), dtype=np.float32)
omega[1:-1, 1:-1][domain_mask] = 0.0

# Compute boundary vorticity closure
psi = np.zeros((N + 2, N + 2), dtype=np.float32)

# Moving lid (top wall)
psi[0, 1:-1] = 0.5 * (h**2) * (bc_primary[0, 1:-1] / (1.0 - h**2 * bc_primary[0, 1:-1]))
omega[0, 1:-1] = -(4 * psi[1, 1:-1] - 0.5 * psi[2, 1:-1]) / h**2 - 3 * bc_primary[0, 1:-1] / h

# Stationary walls
psi[1:-1, 0] = 0.5 * (h**2) * (bc_secondary[1:-1, 0] / (1.0 - h**2 * bc_secondary[1:-1, 0]))
omega[1:-1, 0] = -(4 * psi[1:-1, 1] - 0.5 * psi[1:-1, 2]) / h**2

psi[1:-1, -1] = 0.5 * (h**2) * (bc_secondary[1:-1, -1] / (1.0 - h**2 * bc_secondary[1:-1, -1]))
omega[1:-1, -1] = -(4 * psi[1:-1, -2] - 0.5 * psi[1:-1, -3]) / h**2

psi[-1, 1:-1] = 0.5 * (h**2) * (bc_primary[-1, 1:-1] / (1.0 - h**2 * bc_primary[-1, 1:-1]))
omega[-1, 1:-1] = -(4 * psi[-2, 1:-1] - 0.5 * psi[-3, 1:-1]) / h**2 - 3 * bc_primary[-1, 1:-1] / h

# Time-stepping parameters
dt = 0.001
num_steps = 28000
residuals = []

# Time-stepping loop
for _ in range(num_steps):
    # Compute convective terms
    u = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
    v = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)

    convective = (u * (psi[1:-1, 2:] - psi[1:-1, :-2]) + v * (psi[2:, 1:-1] - psi[:-2, 1:-1])) / (2 * h)

    # Compute diffusive terms
    laplacian = ((psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / h**2 +
                 (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / h**2)

    # Update vorticity
    omega[1:-1, 1:-1] -= dt * (convective - nu * laplacian)

    # Compute residual
    residual = np.abs(omega[1:-1, 1:-1][domain_mask]).max()
    residuals.append(residual)

    if residual < 2.0e-05:
        break

# Save results
omega_interior = omega[1:-1, 1:-1][domain_mask]
np.save('results/report_a.npy', omega_interior[sample_a.astype(np.int32)])
np.save('results/report_b.npy', omega_interior[sample_b.astype(np.int32)])
np.save('results/report_c.npy', omega_interior[sample_c.astype(np.int32)])
np.save('results/field_a.npy', omega_interior)
np.save('results/field_b.npy', psi[1:-1, 1:-1][domain_mask])

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(omega_interior, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.title('Vorticity Field')
plt.savefig('results/overview.png')
plt.close()
```