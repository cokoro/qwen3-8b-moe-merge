import numpy as np
import scipy.fft as fft

# -----------------------------
# Constants
# -----------------------------
N = 135
dx = 0.00740741
inv_dx = N
CFL = 1.8797
dt_frame = 0.1

# -----------------------------
# Helper functions
# -----------------------------
def N2(x):
    abs_x = abs(x)
    if abs_x < 0.5:
        return 0.75 - abs_x ** 2
    elif abs_x < 1.5:
        return 0.5 * (1.5 - abs_x) ** 2
    else:
        return 0.0

def dN2(x):
    abs_x = abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    else:
        return 0.0
    if x < 0:
        result *= -1
    return result

# -----------------------------
# Poisson solver (DST-I spectral method)
# -----------------------------
def solve_poisson(w_n):
    N = w_n.shape[0] - 1
    w = np.zeros((N+1, N+1), dtype=np.float32)
    w[1:N, 1:N] = w_n[1:N, 1:N]
    
    # Apply DST-I
    w_hat = fft.dst(w, type=1, axis=0, overwrite_x=True)
    w_hat = fft.dst(w_hat, type=1, axis=1, overwrite_x=True)
    
    # Compute eigenvalues
    k = np.arange(1, N+1)
    λ = 2 - 2 * np.cos(np.pi * k / N)
    λ = np.reshape(λ, (N, 1)) + np.reshape(λ, (1, N))
    λ = λ - 2 * np.cos(np.pi * k / N) * np.cos(np.pi * k / N)
    
    # Solve in spectral space
    ψ_hat = w_hat * dx**2 / λ
    ψ = fft.idst(fft.idst(ψ_hat, type=1, axis=1, overwrite_x=True), type=1, axis=0, overwrite_x=True)
    
    # Apply Dirichlet BC
    ψ[0, :] = 0
    ψ[-1, :] = 0
    ψ[:, 0] = 0
    ψ[:, -1] = 0
    
    return ψ

# -----------------------------
# Main simulation
# -----------------------------
if __name__ == "__main__":
    # Load initial conditions
    u_x = np.load("data/initial_u_x.npy", dtype=np.float32)
    u_y = np.load("data/initial_u_y.npy", dtype=np.float32)
    
    # Step 1: Compute initial nodal vorticity
    w_n = np.zeros((N+1, N+1), dtype=np.float32)
    for i in range(1, N):
        for j in range(1, N):
            w_n[i, j] = (u_y[i, j] - u_y[i-1, j] - u_x[i, j] + u_x[i, j-1]) * inv_dx
    
    # Step 2: Poisson solve for stream function
    ψ = solve_poisson(w_n)
    
    # Step 3: Reconstruct velocity from stream function
    u_x_new = np.zeros((N+1, N), dtype=np.float32)
    u_y_new = np.zeros((N, N+1), dtype=np.float32)
    for i in range(N):
        for j in range(N-1):
            u_x_new[i, j] = (ψ[i, j+1] - ψ[i, j]) / dx
    for i in range(N-1):
        for j in range(N):
            u_y_new[i, j] = -(ψ[i+1, j] - ψ[i, j]) / dx
    
    # Step 4: Apply boundary conditions
    u_x_new[0, :] = 0
    u_x_new[N, :] = 0
    u_y_new[:, 0] = 0
    u_y_new[:, N] = 0
    
    # Step 5: Recompute nodal vorticity
    w_n_new = np.zeros((N+1, N+1), dtype=np.float32)
    for i in range(1, N):
        for j in range(1, N):
            w_n_new[i, j] = (u_y_new[i, j] - u_y_new[i-1, j] - u_x_new[i, j] + u_x_new[i, j-1]) * inv_dx
    
    # Step 6: Initialize particles
    particles = []
    for idx in range(N * N * 16):
        id_x = idx % (4 * N)
        id_y = idx // (4 * N)
        px = ((id_x + 0.5) * dx / 4)
        py = ((id_y + 0.5) * dx / 4)
        particles.append((px, py))
    
    # Step 7: Initialize T matrix
    T_x = np.zeros((N * N * 16, 2), dtype=np.float32)
    T_y = np.zeros((N * N * 16, 2), dtype=np.float32)
    for p in range(N * N * 16):
        T_x[p] = [1.0, 0.0]
        T_y[p] = [0.0, 1.0]
    
    # Step 8: Initialize C_c
    C_c = np.zeros((N * N * 16, 2), dtype=np.float32)
    for p in range(N * N * 16):
        gx = particles[p][0] / dx
        gy = particles[p][1] / dx
        base = np.floor([gx, gy]).astype(int)
        weight = 0
        grad_w_x = 0
        grad_w_y = 0
        for di in [-1, 0, 1, 2]:
            for dj in [-1, 0, 1, 2]:
                node = (base[0] + di, base[1] + dj)
                if node[0] < 0 or node[0] >= N or node[1] < 0 or node[1] >= N:
                    continue
                xd = gx - node[0]
                yd = gy - node[1]
                w = N2(xd) * N2(yd)
                grad_w_x += w_n_new[node[0], node[1]] * (1 / dx) * dN2(xd) * N2(yd)
                grad_w_y += w_n_new[node[0], node[1]] * (1 / dx) * N2(xd) * dN2(yd)
                weight += w
        C_c[p] = [grad_w_x, grad_w_y]
    
    # Step 9: Initialize w_n
    w_n_particles = np.zeros(N * N * 16, dtype=np.float32)
    for p in range(N * N * 16):
        w_n_particles[p] = 0
    
    # Step 10: Main loop
    frames = 108
    t = 0
    KE = np.zeros(frames, dtype=np.float32)
    vorticity_frames = np.zeros((5, N, N), dtype=np.float32)
    
    for frame in range(frames):
        # Step 1: CFL time step
        max_vel = 0
        for i in range(N):
            for j in range(N):
                uc = 0.5 * (u_x_new[i, j] + u_x_new[i+1, j])
                vc = 0.5 * (u_y_new[i, j] + u_y_new[i, j+1])
                max_vel = max(max_vel, np.sqrt(uc**2 + vc**2))
        max_vel = max(max_vel, 1e-3)
        dt = min(CFL * dx / max_vel, dt_frame)
        dt_remaining = dt_frame - dt
        
        # Step 2: Initialize particles
        particles = []
        for idx in range(N * N * 16):
            id_x = idx % (4 * N)
            id_y = idx // (4 * N)
            px = ((id_x + 0.5) * dx / 4)
            py = ((id_y + 0.5) * dx / 4)
            particles.append((px, py))
        
        # Step 3: G2P with gradient
        w_n_particles = np.zeros(N * N * 16, dtype=np.float32)
        grad_w_x = np.zeros(N * N * 16, dtype=np.float32)
        grad_w_y = np.zeros(N * N * 16, dtype=np.float32)
        for p in range(N * N * 16):
            px, py = particles[p]
            gx = px / dx
            gy = py / dx
            base = np.floor([gx, gy]).astype(int)
            weight = 0
            for di in [-1, 0, 1, 2]:
                for dj in [-1, 0, 1, 2]:
                    node = (base[0] + di, base[1] + dj)
                    if node[0] < 0 or node[0] >= N or node[1] < 0 or node[1] >= N:
                        continue
                    xd = gx - node[0]
                    yd = gy - node[1]
                    w = N2(xd) * N2(yd)
                    w_n_particles[p] += w_n_new[node[0], node[1]] * w
                    grad_w_x[p] += w_n_new[node[0], node[1]] * (1 / dx) * dN2(xd) * N2(yd)
                    grad_w_y[p] += w_n_new[node[0], node[1]] * (1 / dx) * N2(xd) * dN2(yd)
                    weight += w
            C_c[p] = [grad_w_x[p], grad_w_y[p]]
        
        # Step 4: Initialize T matrix
        T_x_new = np.zeros((N * N * 16, 2), dtype=np.float32)
        T_y_new = np.zeros((N * N * 16, 2), dtype=np.float32)
        for p in range(N * N * 16):
            T_x_new[p] = [1.0, 0.0]
            T_y_new[p] = [0.0, 1.0]
        
        # Step 5: Midpoint velocity
        u_mid = np.zeros((N+1, N), dtype=np.float32)
        u_mid_y = np.zeros((N, N+1), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                u_mid[i, j] = 0.5 * (u_x_new[i, j] + u_x_new[i+1, j])
                u_mid_y[i, j] = 0.5 * (u_y_new[i, j] + u_y_new[i, j+1])
        
        # Step 6: Advect particles + evolve T
        for p in range(N * N * 16):
            px, py = particles[p]
            gx = px / dx
            gy = py / dx
            base = np.floor([gx, gy]).astype(int)
            for di in [-1, 0, 1, 2]:
                for dj in [-1, 0, 1, 2]:
                    node = (base[0] + di, base[1] + dj)
                    if node[0] < 0 or node[0] >= N or node[1] < 0 or node[1] >= N:
                        continue
                    xd = gx - node[0]
                    yd = gy - node[1]
                    weight = N2(xd) * N2(yd)
                    grad_w_x[p] += w_n_new[node[0], node[1]] * (1 / dx) * dN2(xd) * N2(yd)
                    grad_w_y[p] += w_n_new[node[0], node[1]] * (1 / dx) * N2(xd) * dN2(yd)
                    weight += w
        
        # Step 7: Update APIC gradient
        C_c_new = np.zeros((N * N * 16, 2), dtype=np.float32)
        for p in range(N * N * 16):
            T = np.array([[T_x_new[p][0], T_x_new[p][1]], [T_y_new[p][0], T_y_new[p][1]]])
            C_c_new[p] = np.dot(T.T, C_c[p])
        
        # Step 8: P2G with APIC
        w_n_new = np.zeros((N+1, N+1), dtype=np.float32)
        weight_grid = np.zeros((N+1, N+1), dtype=np.float32)
        for p in range(N * N * 16):
            px, py = particles[p]
            gx = px / dx
            gy = py / dx
            base = np.floor([gx, gy]).astype(int)
            for di in [-1, 0, 1, 2]:
                for dj in [-1, 0, 1, 2]:
                    node = (base[0] + di, base[1] + dj)
                    if node[0] < 0 or node[0] >= N or node[1] < 0 or node[1] >= N:
                        continue
                    xd = gx - node[0]
                    yd = gy - node[1]
                    w = N2(xd) * N2(yd)
                    dpos = ((node[0] - gx) * dx, (node[1] - gy) * dx)
                    delta = np.dot(C_c_new[p], dpos)
                    w_n_new[node[0], node[1]] += (w_n_particles[p] + delta) * w
                    weight_grid[node[0], node[1]] += w
        w_n_new /= weight_grid
        
        # Step 9: Poisson solve
        ψ_new = solve_poisson(w_n_new)
        
        # Step 10: Reconstruct velocity
        u_x_new = np.zeros((N+1, N), dtype=np.float32)
        u_y_new = np.zeros((N, N+1), dtype=np.float32)
        for i in range(N):
            for j in range(N-1):
                u_x_new[i, j] = (ψ_new[i, j+1] - ψ_new[i, j]) / dx
        for i in range(N-1):
            for j in range(N):
                u_y_new[i, j] = -(ψ_new[i+1, j] - ψ_new[i, j]) / dx
        
        # Step 11: Apply boundary conditions
        u_x_new[0, :] = 0
        u_x_new[N, :] = 0
        u_y_new[:, 0] = 0
        u_y_new[:, N] = 0
        
        # Step 12: Recompute nodal vorticity
        w_n_new = np.zeros((N+1, N+1), dtype=np.float32)
        for i in range(1, N):
            for j in range(1, N):
                w_n_new[i, j] = (u_y_new[i, j] - u_y_new[i-1, j] - u_x_new[i, j] + u_x_new[i, j-1]) * inv_dx
        
        # Step 13: Accumulate time
        t += dt
    
    # Save output files
    np.save("vorticity_frames.npy", vorticity_frames)
    np.save("kinetic_energy.npy", KE)