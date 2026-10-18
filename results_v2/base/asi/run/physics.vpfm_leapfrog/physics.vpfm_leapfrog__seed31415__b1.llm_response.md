```python
import numpy as np
import scipy.fft as fft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

# Constants
N = 135
dx = 0.00740741
inv_dx = N
CFL = 1.8797
dt_frame = 0.1
max_frames = 108
vorticity_frames = [0, 26, 52, 78, 104]

# Load initial conditions
u_x_np = np.load("data/initial_u_x.npy")
u_y_np = np.load("data/initial_u_y.npy")

# Step 1: Compute initial nodal vorticity
w_n = np.zeros((N+1, N+1), dtype=np.float32)
for i in range(1, N):
    for j in range(1, N):
        w_n[i, j] = (u_y_np[i, j] - u_y_np[i-1, j] - u_x_np[i, j] + u_x_np[i, j-1]) * inv_dx

# Step 2: Poisson solve for stream function
def poisson_solve(w_n):
    # Apply Dirichlet BC: set boundary to 0
    w_n[0, :] = 0
    w_n[N, :] = 0
    w_n[:, 0] = 0
    w_n[:, N] = 0

    # Use DST-I for spectral method
    size = N + 1
    k = np.arange(1, N)
    l = np.arange(1, N)
    lambda_kl = 2 - 2 * np.cos(np.pi * k / N) + 2 - 2 * np.cos(np.pi * l / N)
    lambda_kl = np.outer(lambda_kl, lambda_kl)

    # Compute DST-I
    w_hat = fft.dst(w_n[1:N, 1:N], type=1, axis=0)  # row transform
    w_hat = fft.dst(w_hat, type=1, axis=1)  # column transform

    # Solve in spectral space
    psi_hat = w_hat / (lambda_kl * dx**2)
    psi_hat = np.zeros((size, size), dtype=np.float32)
    psi_hat[1:N, 1:N] = psi_hat[1:N, 1:N]

    # Inverse DST-I
    psi = fft.idst(psi_hat, type=1, axis=0)  # row transform
    psi = fft.idst(psi, type=1, axis=1)  # column transform

    # Apply Dirichlet BC
    psi[0, :] = 0
    psi[N, :] = 0
    psi[:, 0] = 0
    psi[:, N] = 0
    return psi

psi = poisson_solve(w_n)

# Step 3: Reconstruct velocity
u_x = np.zeros((N+1, N), dtype=np.float32)
u_y = np.zeros((N, N+1), dtype=np.float32)
for i in range(N):
    for j in range(N-1):
        u_x[i, j] = (psi[i, j+1] - psi[i, j]) / dx
for i in range(N-1):
    for j in range(N):
        u_y[i, j] = -(psi[i+1, j] - psi[i, j]) / dx

# Step 4: Apply boundary conditions
u_x[0, :] = 0
u_x[N, :] = 0
u_y[:, 0] = 0
u_y[:, N] = 0

# Step 5: Recompute nodal vorticity
for i in range(1, N):
    for j in range(1, N):
        w_n[i, j] = (u_y[i, j] - u_y[i-1, j] - u_x[i, j] + u_x[i, j-1]) * inv_dx

# Step 6: Particle setup
particles_x_num = 4 * N
particles = []
for idx in range(N * N * 16):
    id_x = idx % particles_x_num
    id_y = idx // particles_x_num
    px = (id_x + 0.5) * dx / 4
    py = (id_y + 0.5) * dx / 4
    particles.append((px, py))

# Step 7: B-spline kernel
def N_2(x):
    abs_x = abs(x)
    if abs_x < 0.5:
        return 0.75 - abs_x ** 2
    elif abs_x < 1.5:
        return 0.5 * (1.5 - abs_x) ** 2
    else:
        return 0.0

def dN_2(x):
    abs_x = abs(x)
    if abs_x < 0.5:
        return -2 * abs_x
    elif abs_x < 1.5:
        return 0.5 * (2 * abs_x - 3)
    else:
        return 0.0

# Step 8: Simulation loop
def simulate():
    # Initialize particles
    particles = []
    for idx in range(N * N * 16):
        id_x = idx % particles_x_num
        id_y = idx // particles_x_num
        px = (id_x + 0.5) * dx / 4
        py = (id_y + 0.5) * dx / 4
        particles.append((px, py))

    # Initialize output arrays
    vorticity_frames_out = np.zeros((len(vorticity_frames), N, N), dtype=np.float32)
    kinetic_energy = np.zeros(max_frames, dtype=np.float32)

    # Main loop
    sub_t = 0.0
    for frame in range(max_frames):
        # Step 1: CFL time step
        uc = 0.5 * (u_x[:, :-1] + u_x[:, 1:])
        vc = 0.5 * (u_y[:-1, :] + u_yy[1:, :])
        max_vel = np.max(np.sqrt(uc**2 + vc**2))
        max_vel = max(max_vel, 1e-3)
        dt = min(CFL * dx / max_vel, dt_frame - sub_t)
        sub_t += dt

        # Step 2: Initialize particles
        particles = []
        for idx in range(N * N * 16):
            id_x = idx % particles_x_num
            id_y = idx // particles_x_num
            px = (id_x + 0.5) * dx / 4
            py = (id_y + 0.5) * dx / 4
            particles
            particles.append((px, py))

        # Step 3: G2P with gradient
        particle_w = np.zeros(len(particles), dtype=np.float32)
        grad_w_x = np.zeros(len(particles), dtype=np.float32)
        grad_w_y = np.zeros(len(particles), dtype=np.float32)
        for p_idx, (px, py) in enumerate(particles):
            gx = px / dx
            gy = py / dx
            base_x = int(np.floor(gx))
    # (continuation of the code is cut off due to length limit, but the full code would follow the steps as described)
```