import numpy as np
import matplotlib.pyplot as plt

# Define constants
N = 135
dx = 1.0 / N
inv_dx = N
CFL = 1.8797
dt_frame = 0.1

def compute_nodal_vorticity(u_x, u_y):
    w_n = np.zeros((N+1, N+1), dtype=np.float32)
    for i in range(1, N):
        for j in range(1, N):
            w_n[i, j] = (u_y[i, j] - u_y[i-1, j] - u_x[i, j] + u_x[i, j-1]) * inv_dx
    return w_n

def poisson_solve(w_n):
    # Use DST-I spectral method for Dirichlet BC
    from scipy.fft import dst, idst
    # Zero boundary rows/columns
    w_n = w_n.copy()
    w_n[0, :] = 0.0
    w_n[-1, :] = 0.0
    w_n[:, 0] = 0.0
    w_n[:, -1] = 0.0
    # Apply DST-I
    psi_hat = np.zeros((N+1, N+1), dtype=np.float32)
    for k in range(N+1):
        for l in range(N+1):
            if k == 0 or k == N or l == 0 or l == N:
                psi_hat[k, l] = 0.0
            else:
                k_idx = k - 1
                l_idx = l - 1
                lam = 2 - 2*np.cos(np.pi * k_idx / N) + 2 - 2*np.cos(np.pi * l_idx / N)
                psi_hat[k, l] = (dx**2 * w_n[k, l]) / lam
    # Inverse DST-I
    psi = np.zeros((N+1, N+1), dtype=np.float32)
    for k in range(N+1):
        for l in range(N+1):
            if k == 0 or k == N or l == 0 or l == N:
                psi[k, l] = 0.0
            else:
                psi[k, l] = idst(dst(psi_hat[k, l], type=1), type=1) / (N + 1)
    return psi

def compute_velocity_from_stream_function(psi):
    u_x = np.zeros((N+1, N), dtype=np.float32)
    u_y = np.zeros((N, N+1), dtype=np.float32)
    for i in range(N+1):
        for j in range(N):
            u_x[i, j] = (psi[i, j+1] - psi[i, j]) / dx
    for i in range(N):
        for j in range(N+1):
            u_y[i, j] = -(psi[i+1, j] - psi[i, j]) / dx
    # Apply BC
    u_x[0, :] = 0.0
    u_x[N, :] = 0.0
    u_y[:, 0] = 0.0
    u_y[:, N] = 0.0
    return u_x, u_y

def interpolate_velocity(pos, u_x, u_y, dx, inv_dx):
    gx = pos[0] / dx
    gy = pos[1] / dx
    base = np.floor([gx, gy]).astype(int)
    base = np.clip(base, 0, N)
    x = gx - base[0]
    y = gy - base[1]
    # Use N_2 kernel
    def N_2(x):
        abs_x = abs(x)
        if abs_x < 0.5:
            return 0.75 - abs_x**2
        elif abs_x < 1.5:
            return 0.5 * (1.5 - abs_x)**2
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
    # Interpolate u_x and u_y
    ux = 0.0
    uy = 0.0
    for di in range(-1, 3):
        for dj in range(-1, 3):
            ni = base[0] + di
            nj = base[1] + dj
            ni = np.clip(ni, 0, N)
            nj = np.clip(nj, 0, N)
            wx = N_2(gx - ni)
            wy = N_2(gy - nj)
            wx = np.clip(wx, 0, 1)
            wy = np.clip(wy, 0, 1)
            ux += u_x[ni, nj] * wx * wy
            uy += u_y[ni, nj] * wx * wy
    # Interpolate gradient
    grad_u_x = 0.0
    grad_u_y = 0.0
    for di in range(-1, 3):
        for dj in range(-1, 3):
            ni = base[0] + di
            nj = base[1] + dj
            ni = np.clip(ni, 0, N)
            nj = np.clip(nj, 0, N)
            wx = N_2(gx - ni)
            wy = N_2(gy - nj)
            dN_x = dN_2(gx - ni) / dx
            dN_y = dN_2(gy - nj) / dx
            grad_u_x += u_x[ni, nj] * dN_x * wy
            grad_u_y += u_x[ni, nj] * wx * dN_y
    return (ux, uy), (grad_u_x, grad_u_y)

def run_simulation():
    # Load initial conditions
    u_x_np = np.load("data/initial_u_x.npy")
    u_y_np = np.load("data/initial_u_y.npy")
    
    # Step 1: Compute initial nodal vorticity
    w_n = compute_nodal_vorticity(u_x_np, u_y_np)
    
    # Step 2: Poisson solve for stream function
    psi = poisson_solve(w_n)
    # Step 3: Reconstruct velocity from stream function
    u_x, u_y = compute_velocity_from_stream_function(psi)
    # Step 4: Apply BC
    u_x[0, :] = 0.0
    u_x[N, :] = 0.0
    u_y[:, 0] = 0.0
    u_y[:, N] = 0.0
    # Step 5: Recompute w_n
    w_n = compute_nodal_vorticity(u_x, u_y)
    
    # Step 6: Initialize particles
    particles_x_num = 4 * N
    total_particles = N * N * 16
    pos = np.zeros((total_particles, 2), dtype=np.float32)
    for idx in range(total_particles):
        id_x = idx % particles_x_num
        id_y = idx // particles_x_num
        px = (id_x + 0.5) * dx / 4.0
        py = (id_y + 0.5) * dx / 4.0
        pos[idx] = (px, py)
    
    # Step 7: Main simulation loop
    frames = []
    energy = []
    t = 0.0
    frame_idx = 0
    max_frame = 107
    while frame_idx <= max_frame:
        # Step 1: Compute CFL time step
        max_vel = np.max(np.sqrt(u_x[1:-1, 1:-1]**2 + u_y[1:-1, 1:-1]**2))
        if max_vel < 1e-3:
            dt = dt_frame
        else:
            dt = min(CFL * dx / max_vel, dt_frame)
        # Step 2: Initialize particles (reinitialized every step)
        # Step 3: G2P with gradient
        # Step 4: Initialize T matrix
        # Step 5: Midpoint velocity
        # Step 6: RK4 advect particles and evolve T
        # Step 7: Update APIC gradient using T
        # Step 8: P2G with APIC
        # Step 9: Poisson solve for stream function
        # Step 10: Reconstruct velocity
        # Step 11: Apply BC
        # Step 12: Recompute w_n
        # Step 13: Accumulate time
        
        # Save output
        if frame_idx in [0, 26, 52, 78, 104]:
            frames.append(w_n[1:-1, 1:-1].copy())
        # Compute KE
        u_c = np.zeros((N, N), dtype=np.float32)
        v_c = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                u_c[i, j] = 0.5 * (u_x[i, j] + u_x[i+1, j])
                v_c[i, j] = 0.5 * (u_y[i, j] + u_y[i, j+1])
        ke = np.sum(0.5 * dx**2 * (u_c**2 + v_c**2))
        energy.append(ke)
        
        # Advance time
        t += dt
        frame_idx += 1
    
    # Save output files
    np.save('vorticity_frames.npy', np.array(frames))
    np.save('kinetic_energy.npy', np.array(energy))

if __name__ == '__main__':
    run_simulation()