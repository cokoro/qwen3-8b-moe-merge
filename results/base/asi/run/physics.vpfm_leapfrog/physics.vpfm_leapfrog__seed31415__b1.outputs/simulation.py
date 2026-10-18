import numpy as np
import numpy.fft as fft
import os

# Constants
N = 135
dx = 0.00740741
inv_dx = N
CFL = 1.8797
dt_frame = 0.1
num_frames = 108
vorticity_output_frames = [0, 26, 52, 78, 104]

# Load initial conditions
u_x_np = np.load("data/initial_u_x.npy")
u_y_np = np.load("data/initial_u_y.npy")

# Step 1: Compute initial nodal vorticity
w_n = np.zeros((N + 1, N + 1), dtype=np.float32)
for i in range(1, N):
    for j in range(1, N):
        w_n[i, j] = (u_y_np[i, j] - u_y_np[i - 1, j] - u_x_np[i, j] + u_x_np[i, j - 1]) * inv_dx

# Step 2: Poisson solve for stream function
def poisson_solve(w_n):
    # Apply Dirichlet BC (set boundary vorticity to 0)
    w_n[0, :] = 0
    w_n[N, :] = 0
    w_n[:, 0] = 0
    w_n[:, N] = 0

    # Use DST-I spectral method
    size = N + 1
    k = np.arange(1, N)
    l = np.arange(1, N)
    lambda_kl = 2 - 2 * np.cos(np.pi * k / N) + 2 - 2 * np.cos(np.pi * l / N)
    lambda_kl = np.outer(lambda_kl, lambda_kl)
    lambda_kl = np.reshape(lambda_kl, (N * N, 1))

    # Transform to spectral space
    w_hat = fft.dst(w_n[1:N, 1:N], type=1, axis=0)
    w_hat = fft.dst(w_hat, type=1, axis=1)
    w_hat = np.reshape(w_hat, (N * N, 1))
    w_hat = w_hat / (dx ** 2 * lambda_kl)

    # Inverse transform
    psi_hat = fft.idst(w_hat, type=1, axis=0)
    psi_hat = fft.idst(psi_hat, type=1, axis=1)
    psi = np.reshape(psi_hat, (N + 1, N + 1))

    # Apply Dirichlet BC (psi = 0 on boundaries)
    psi[0, :] = 0
    psi[N, :] = 0
    psi[:, 0] = 0
    psi[:, N] = 0

    return psi

psi = poisson_solve(w_n)

# Step 3: Reconstruct velocity
u_x = np.zeros((N + 1, N), dtype=np.float32)
u_y = np.zeros((N, N + 1), dtype=np.float32)

for i in range(N + 1):
    for j in range(N):
        u_x[i, j] = (psi[i, j + 1] - psi[i, j]) / dx

for i in range(N):
    for j in range(N + 1):
        u_y[i, j] = -(psi[i + 1, j] - psi[i, j]) / dx

# Apply boundary conditions
u_x[0, :] = 0
u_x[N, :] = 0
u_y[:, 0] = 0
u_y[:, N] = 0

# Recompute w_n
w_n = np.zeros((N + 1, N + 1), dtype=np.float32)
for i in range(1, N):
    for j in range(1, N):
        w_n[i, j] = (u_y[i, j] - u_y[i - 1, j] - u_x[i, j] + u_x[i, j - 1]) * inv_dx

# Step 4: Particle setup
num_particles_per_cell = 16
particles_x_num = 4 * N
particles_y_num = 4 * N
particles = np.zeros((num_particles_per_cell * N * N, 2), dtype=np.float32)
for idx in range(num_particles_per_cell * N * N):
    id_x = idx % particles_x_num
    id_y = idx // particles_x_num
    px = ((id_x + 0.5) * dx / 4)
    py = ((id_y + 0.5) * dx / 4)
    particles[idx, 0] = px
    particles[idx, 1] = py

# Step 5: VPFM Time Integration
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
        res = -2 * abs_x
    elif abs_x < 1.5:
        res = 0.5 * (2 * abs_x - 3)
    else:
        res = 0.0
    if x < 0:
        res *= -1
    return res

def interpolate_velocity_and_grad(pos, u_x, u_y):
    px, py = pos
    gx = px / dx
    gy = py / dx
    base_x = np.floor(gx).astype(int)
    base_y = np.floor(gy).astype(int)
    weights = []
    grad_x = []
    grad_y = []
    for di in range(-1, 3):
        for dj in range(-1, 3):
            node_x = base_x + di
            node_y = base_y + dj
            node_x = np.clip(node_x, 0, N)
            node_y = np.clip(node_y, 0, N)
            xd = gx - node_x
            yd = gy - node_y
            w = N_2(xd) * N_2(yd)
            weights.append(w)
            grad_x.append(w * dN_2(xd))
            grad_y.append(w * dN_2(yd))
    weights = np.array(weights)
    grad_x = np.array(grad_x)
    grad_y = np.array(grad_y)
    return weights, grad_x, grad_y

def interpolate_vorticity(pos, w_n):
    px, py = pos
    gx = px / dx
    gy = py / dx
    base_x = np.floor(gx).astype(int)
    base_y = np.floor(gy).astype(int)
    weights = []
    for di in range(-1, 3):
        for dj in range(-1, 3):
            node_x = base_x + di
            node_y = base_y + dj
            node_x = np.clip(node_x, 0, N)
            node_y = np.clip(node_y, 0, N)
            xd = gx - node_x
            yd = gy - node_y
            weights.append(N_2(xd) * N_2(yd))
    weights = np.array(weights)
    return weights

def interpolate_vorticity_mid(pos, w_n):
    px, py = pos
    gx = px / dx
    gy = py / dx
    base_x = np.floor(gx).astype(int)
    base_y = np.floor(gy).astype(int)
    weights = []
    for di in range(-1, 3):
        for dj in range(-1, 3):
            node_x = base_x + di
            node_y = base_y + dj
            node_x = np.clip(node_x, 1, N - 1)
            node_y = np.clip(node_y, 1, N - 1)
            xd = gx - node_x
            yd = gy - node_y
            weights.append(N_2(xd) * N_2(yd))
    weights = np.array(weights)
    return weights

def compute_midpoint_velocity(u_x, u_y):
    psi = poisson_solve(w_n)
    u_x_mid = np.zeros((N + 1, N), dtype=np.float32)
    u_y_mid = np.zeros((N, N + 1), dtype=np.float32)
    for i in range(N + 1):
        for j in range(N):
            u_x_mid[i, j] = (psi[i, j + 1] - psi[i, j]) / dx
    for i in range(N):
        for j in range(N + 1):
            u_y_mid[i, j] = -(psi[i + 1, j] - psi[i, j]) / dx
    return u_x_mid, u_y_mid

def rk4_step(pos, T_x, T_y, u_x, u_y, dt, u_x_mid, u_y_mid):
    def interp_velocity_and_grad(pos, u_x, u_y):
        px, py = pos
        gx = px / dx
        gy = py / dx
        base_x = np.floor(gx).astype(int)
        base_y = np.floor(gy).astype(int)
        weights = []
        grad_x = []
        grad_y = []
        for di in range(-1, 3):
            for dj in range(-1, 3):
                node_x = base_x + di
                node_y = base_y + dj
                node_x = np.clip(node_x, 0, N)
                node_y = np.clip(node_y, 0, N)
                xd = gx - node_x
                yd = gy - node_y
                w = N_2(xd) * N_2(yd)
                weights.append(w)
                grad_x.append(w * dN_2(xd))
                grad_y.append(w * dN_2(yd))
        weights = np.array(weights)
        grad_x = np.array(grad_x)
        grad_y = np.array(grad_y)
        return weights, grad_x, grad_y

    def compute_grad_u(pos, u_x, u_y):
        px, py = pos
        gx = px / dx
        gy = py / dx
        base_x = np.floor(gx).astype(int)
        base_y = np.floor(gy).astype(int)
        weights = []
        grad_x = []
        grad_y = []
        for di in range(-1, 3):
            for dj in range(-1, 3):
                node_x = base_x + di
                node_y = base_y + dj
                node_x = np.clip(node_x, 0, N)
                node_y = np.clip(node_y, 0, N)
                xd = gx - node_x
                yd = gy - node_y
                w = N_2(xd) * N_2(yd)
                weights.append(w)
                grad_x.append(w * dN_2(xd))
                grad_y.append(w * dN_2(yd))
        weights = np.array(weights)
        grad_x = np.array(grad_x)
        grad_y = np.array(grad_y)
        return weights, grad_x, grad_y

    # Step 1
    weights, grad_x, grad_y = compute_velocity_and_grad(pos, u_x, u_y)
    vx1 = np.sum(grad_x * u_x) / np.sum(weights)
    vy1 = np.sum(grad_y * u_y) / np.sum(weights)
    grad_u1 = np.array([grad_x, grad_y])

    # Step 2
    pos2 = pos + 0.5 * dt * np.array([vx1, vy1])
    pos2 = np.clip(pos2, dx / 2, 1 - dx / 2)
    weights2, grad_x2, grad_y2 = compute_velocity_and_grad(pos2, u_x, u_y)
    vx2 = np.sum(grad_x2 * u_x) / np.sum(weights2)
    vy2 = np.sum(grad_y2 * u_y) / np.sum(weights2)
    grad_u2 = np.array([grad_x2, grad_y2])

    # Step 3
    pos3 = pos + 0.5 * dt * np.array([vx2, vy2])
    pos3 = np.clip(pos3, dx / 2, 1 - dx / 2)
    weights3, grad_x3, grad_y3 = compute_velocity_and_grad(pos3, u_x, u_y)
    vx3 = np.sum(grad_x3 * u_x) / np.sum(weights3)
    vy3 = np.sum(grad_y3 * u_y) / np.sum(weights3)
    grad_u3 = np.array([grad_x3, grad_y3])

    # Step 4
    pos4 = pos + dt * np.array([vx3, vy3])
    pos4 = np.clip(pos4, dx / 2, 1 - dx / 2)
    weights4, grad_x4, grad_y4 = compute_velocity_and_grad(pos4, u_x, u_y)
    vx4 = np.sum(grad_x4 * u_x) / np.sum(weights4)
    vy4 = np.sum(grad_y4 * u_y) / np.sum(weights4)
    grad_u4 = np.array([grad_x4, grad_y4])

    # Compute new position
    new_pos = pos + dt / 6 * (np.array([vx1, vy1]) + 2 * np.array([vx2, vy2]) + 2 * np.array([vx3, vy3]) + np.array([vx4, vy4]))
    new_pos = np.clip(new_pos, dx / 2, 1 - dx / 2)

    # Compute new T
    dTx1 = np.dot(grad_u1.T, T_x)
    dTy1 = np.dot(grad_u1.T, T_y)

    dTx2 = np.dot(grad_u2.T, T_x - 0.5 * dt * dTx1)
    dTy2 = np.dot(grad_u2.T, T_y - 0.5 * dt * dTy1)

    dTx3 = np.dot(grad_u3.T, T_x - 0.5 * dt * dTx2)
    dTy3 = np.dot(grad_u3.T, T_y - 0.5 * dt * dTy2)

    dTx4 = np.dot(grad_u4.T, T_x - dt * dTx3)
    dTy4 = np.dot(grad_u4.T, T_y - dt * dTy3)

    new_T_x = T_x - dt / 6 * (dTx1 + 2 * dTx2 + 2 * dTx3 + dTx4)
    new_T_y = T_y - dt / 6 * (dTy1 + 2 * dTy2 + 2 * dTy3 + dTy4)

    return new_pos, new_T_x, new_T_y

# Simulation loop
sub_t = 0.0
vorticity_frames = []
kinetic_energy = []

for frame in range(num_frames):
    # Step 1: Compute dt
    uc = 0.5 * (u_x[1:N, :] + u_x[0:N-1, :])
    vc = 0.5 * (u_y[:, 1:N] + u_y[:, 0:N-1])
    max_vel = np.max(np.sqrt(uc**2 + vc**2))
    max_vel = np.max([max_vel, 1e-3])
    dt = min(CFL * dx / max_vel, dt_frame - sub_t)
    sub_t += dt

    # Step 2: Initialize particles
    particles = np.zeros((num_particles_per_cell * N * N, 2), dtype=np.float32)
    for idx in range(num_particles_per_cell * N * N):
        id_x = idx % particles_x_num
        id_y = idx // particles_x_num
        px = ((id_x + 0.5) * dx / 4)
        py = ((id_y + 0.5) * dx / 4)
        particles[idx, 0] = px
        particles[idx, 1] = py

    # Step 3: G2P with gradient
    particle_w = np.zeros(num_particles_per_cell * N * N, dtype=np.float32)
    grad_w_x = np.zeros(num_particles_per_cell * N * N, dtype=np.float32)
    grad_w_y = np.zeros(num_particles_per_cell * N * N, dtype=np.float32)
    for p in range(num_particles_per_cell * N * N):
        px, py = particles[p]
        gx = px / dx
        gy = py / dx
        base_x = np.floor(gx).astype(int)
        base_y = np.floor(gy).astype(int)
        for di in range(-1, 3):
            for dj in range(-1, 3):
                node_x = base_x + di
                node_y = base_y + dj
                node_x = np.clip(node_x, 0, N)
                node_y = np.clip(node_y, 0, N)
                xd = gx - node_x
                yd = gy - node_y
                w = N_2(xd) * N_2(yd)
                particle_w[p] += w_n[node_x, node_y] * w
                grad_w_x[p] += w_n[node_x, node_y] * (1 / dx) * dN_2(xd) * N_2(yd)
                grad_w_y[p] += w_n[node_x, node_y] * (1 / dx) * N_2(xd) * dN_2(yd)
    init_C_c = np.zeros((num_particles_per_cell * N * N, 2), dtype=np.float32)
    init_C_c[:, 0] = grad_w_x
    init_C_c[:, 1] = grad_w_y

    # Step 4: Initialize T matrix
    T_x = np.zeros(num_particles_per_cell * N * N, dtype=np.float32)
    T_y = np.zeros(num_particles_per_cell * N * N, dtype=np.float32)
    for p in range(num_particles_per_cell * N * N):
        T_x[p] = 1.0
        T_y[p] = 1.0

    # Step 5: Midpoint velocity
    u_x_mid, u_y_mid = compute_midpoint_velocity(u_x, u_y)

    # Step 6: Advect particles + evolve T
    for p in range(num_particles_per_cell * N * N):
        pos = particles[p]
        T_x_p = T_x[p]
        T_y_p = T_y[p]
        new_pos, new_T_x, new_T_y = rk4_step(pos, T_x_p, T_y_p, u_x, u_y, dt, u_x_mid, u_y_mid)
        particles[p, 0] = new_pos[0]
        particles[p, 1] = new_pos[1]
        T_x[p] = new_T_x
        T_y[p] = new_T_y

    # Step 7: Update APIC gradient
    C_c = np.zeros((num_particles_per_cell * N * N, 2), dtype=np.float32)
    for p in range(num_particles_per_cell * N * N):
        C_c[p, 0] = T_x[p] * init_C_c[p, 0] + T_y[p] * init_C_c[p, 1]
        C_c[p, 1] = T_y[p] * init_C_c[p, 0] - T_x[p] * init_C_c[p, 1]

    # Step 8: P2G with APIC
    w_n_new = np.zeros((N + 1, N + 1), dtype=np.float32)
    weight_grid = np.zeros((N + 1, N + 1), dtype=np.float32)
    for p in range(num_particles_per_cell * N * N):
        px, py = particles[p]
        gx = px / dx
        gy = py / dx
        base_x = np.floor(gx).astype(int)
        base_y = np.floor(gy).astype(int)
        for di in range(-1, 3):
            for dj in range(-1, 3):
                node_x = base_x + di
                node_y = base_y + dj
                node_x = np.clip(node_x, 0, N)
                node_y = np.clip(node_y, 0, N)
                xd = gx - node_x
                yd = gy - node_y
                w = N_2(xd) * N_2(yd)
                dpos_x = (node_x - gx) * dx
                dpos_y = (node_y - gy) * dx
                delta = np.dot(C_c[p], np.array([dpos_x, dpos_y]))
                w_n_new[node_x, node_y] += (particle_w[p] + delta) * w
                weight_grid[node_x, node_y] += w
    for i in range(N + 1):
        for j in range(N + 1):
            if weight_grid[i, j] > 0:
                w_n_new[i, j] /= weight_grid[i, j]

    # Step 9: Poisson solve
    psi = poisson_solve(w_n_new)

    # Step 10: Reconstruct velocity
    u_x = np.zeros((N + 1, N), dtype=np.float32)
    u_y = np.zeros((N, N + 1), dtype=np.float32)
    for i in range(N + 1):
        for j in range(N):
            u_x[i, j] = (psi[i, j + 1] - psi[i, j]) / dx
    for i in range(N):
        for j in range(N + 1):
            u_y[i, j] = -(psi[i + 1, j] - psi[i, j]) / dx

    # Step 11: Apply boundary conditions
    u_x[0, :] = 0
    u_x[N, :] = 0
    u_y[:, 0] = 0
    u_y[:, N] = 0

    # Step 12: Recompute w_n
    w_n = np.zeros((N + 1, N + 1), dtype=np.float32)
    for i in range(1, N):
        for j in range(1, N):
            w_n[i, j] = (u_y[i, j] - u_y[i - 1, j] - u_x[i, j] + u_x[i, j - 1]) * inv_dx

    # Step 13: Accumulate time
    sub_t += dt

    # Step 14: Output metrics
    if frame in vorticity_output_frames:
        vorticity_frame = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == 0:
                    vorticity_frame[i, j] = (u_y[1, j] - u_y[0, j] - u_x[i, j + 1] + u_x[i, j - 1]) / (2 * dx)
                elif i == N - 1:
                    vorticity_frame[i, j] = (u_y[i, j] - u_y[i - 1, j] - u_x[i, j + 1] + u_x[i, j - 1]) / (2 * dx)
                elif j == 0:
                    vorticity_frame[i, j] = (u_y[i, j] - u_y[i - 1, j] - u_x[i, j + 1] + u_x[i, j - 1]) / (2 * dx)
                elif j == N - 1:
                    vorticity_frame[i, j] = (u_y[i, j] - u_y[i - 1, j] - u_x[i, j + 1] + u_x[i, j - 1]) / (2 * dx)
                else:
                    vorticity_frame[i, j] = (u_y[i + 1, j] - u_y[i - 1, j] - u_x[i, j + 1] + u_x[i, j - 1]) / (2 * dx)
        vorticity_frames.append(vorticity_frame)

    # Step 15: Compute kinetic energy
    u_c = 0.5 * (u_x[1:N, :] + u_x[0:N-1, :])
    v_c = 0.5 * (u_y[:, 1:N] + u_y[:, 0:N-1])
    ke = np.sum(0.5 * dx**2 * (u_c**2 + v_c**2))
    kinetic_energy.append(ke)

# Save output files
np.save('vorticity_frames.npy', np.array(vorticity_frames))
np.save('kinetic_energy.npy', np.array(kinetic_energy))