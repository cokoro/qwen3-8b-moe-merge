```bash
cat > simulation.py << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

with open('case_config.json') as f:
    import json
    case = json.load(f)

h = case['grid']['h']
dy = h
dx = h

N = case['grid']['N']
Ny = N
Nx = N

# Read boundary data
bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
domain_mask = np.load('domain_mask.npy')

# Read sampling tracks
sample_track_a = np.load('sample_track_a.npy')
sample_track_b = np.load('sample_track_b.npy')
sample_track_c = np.load('sample_track_c.npy')

# Initialize vorticity field
omega = np.zeros((Ny, Nx), dtype=np.float32)

# Initialize streamfunction field
psi = np.zeros((Ny, Nx), dtype=np.float32)

# Initialize velocity field
u = np.zeros((Ny, Nx), dtype=np.float32)
v = np.zeros((Ny, Nx), dtype=np.float32)

# Read lid velocity from primary boundary array
u_lid = bc_primary[1, 1:Nx+1]

# Initialize interior wall-vorticity closure
psi_adjacent = psi[1, 0:Nx]
psi_next_adjacent = psi[2, 0:Nx]

# Initialize solution
for _ in range(1000):
    # Compute wall-vorticity closure
    omega_wall = -(4 * psi_adjacent - 0.5 * psi_next_adjacent) / (dy ** 2)
    omega_wall[1:Nx] = 0.0

    # Compute moving lid-vorticity closure
    psi_below = psi[2, 1:Nx+1]
    omega_top = -(4 * psi_below - 0.5 * psi[3, 1:Nx+1]) / (dy ** 2) - 3 * u_lid / (dy)

    # Apply wall-vorticity closure
    omega[1:Ny, 1:Nx] = omega_wall[1:Ny, 1:Nx]

    # Apply moving lid-vorticity closure
    omega[0, 1:Nx] = omega_top

    # Compute convection terms
    omega_conv = np.zeros((Ny, Nx), dtype=np.float32)
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            omega_conv[i, j] = -u[i, j] * omega[i, j+1] - v[i, j] * omega[i+1, j]

    # Compute diffusion terms
    omega_diff = np.zeros((Ny, Nx), dtype=np.float32)
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            omega_diff[i, j] = 0.5 * (omega[i+1, j] - 2 * omega[i, j] + omega[i-1, j]) / dy ** 2

    # Update vorticity
    omega[1:Ny, 1:Nx] += omega_conv[1:Ny, 1:Nx] + omega_diff[1:Ny, 1:Nx]

    # Update streamfunction
    psi[1:Ny, 1:Nx] += 0.5 * (omega[1:Ny, 1:Nx] + omega[1:Ny, 1:Nx])

    # Update velocity
    u[1:Ny, 1:Nx] = -psi[2:Ny+1, 1:Nx] + psi[0:Ny-1, 1:Nx]
    v[1:Ny, 1:Nx] = psi[1:Ny, 2:Nx+1] - psi[1:Ny, 0:Nx-1]

    # Apply boundary conditions
    u[0, 1:Nx] = u_lid
    v[1:Ny, 0] = 0.0
    v[1:Ny, Nx] = 0.0

    # Apply domain mask
    omega[~domain_mask] = 0.0

# Read interior field and save
field_a = omega[1:Ny, 1:Nx]
np.save('results/field_a.npy', field_a)

# Read interior field and save
field_b = psi[1:Ny, 1:Nx]
np.save('results/field_b.npy', field_b)

# Read required report_a profile
report_a = field_a[sample_track_a.astype(np.int64)]
np.save('results/report_a.npy', report_a)

# Read required report_b profile
report_b = field_b[sample_track_b.astype(np.int64)]
np.save('results/report_b.npy', report_b)

# Read required report_c profile
report_c = field_a[sample_track_c.astype(np.int64)]
np.save('results/report_c.npy', report_c)

# Plot interior field
plt.figure(figsize=(10, 10))
plt.imshow(field_a, cmap='viridis', extent=(0, 1, 0, 1))
plt.colorbar()
plt.title('Interior Vorticity Field')
plt.savefig('results/overview.png')
plt.close()
EOF
```