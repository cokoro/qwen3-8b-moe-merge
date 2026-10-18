import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def simulation():
    # Load data
    bc_primary = np.load('bc_primary.npy')
    bc_secondary = np.load('bc_secondary.npy')
    domain_mask = np.load('domain_mask.npy')
    sample_track_a = np.load('sample_track_a.npy')
    sample_track_b = np.load('sample_track_b.npy')
    sample_track_c = np.load('sample_track_c.npy')
    
    # Grid parameters
    ny, nx = 129, 129
    h = 1.0 / (nx - 1)
    h2 = h * h
    h2inv = 1.0 / h2
    h4inv = 1.0 / (h2 * h2)
    
    # Initialization
    omega = np.zeros((ny, nx), dtype=np.float32)
    psi = np.zeros((ny, nx), dtype=np.float32)
    
    # Boundary conditions
    # Set boundary velocities
    bc_primary = bc_primary.astype(np.float32)
    bc_secondary = bc_secondary.astype(np.float32)
    
    # Initialize vorticity from boundary conditions
    for y in range(ny):
        for x in range(nx):
            if domain_mask[y, x]:
                if y == 0:
                    # Top lid
                    if x > 0 and x < nx - 1:
                        psi[y, x] = 0.0
                        omega[y, x] = -(4.0 * psi[y+1, x] - 0.5 * psi[y+2, x]) / h2 - 3.0 * bc_primary[y, x] / h
                elif y == ny - 1:
                    # Bottom lid
                    if x > 0 and x < nx - 1:
                        psi[y, x] = 0.0
                        omega[y, x] = -(4.0 * psi[y-1, x] - 0.5 * psi[y-2, x]) / h2
                elif x == 0:
                    # Left wall
                    if y > 0 and y < ny - 1:
                        psi[y, x] = 0.0
                        omega[y, x] = -(4.0 * psi[y, x+1] - 0.5 * psi[y, x+2]) / h2
                elif x == nx - 1:
                    # Right wall
                    if y > 0 and y < ny - 1:
                        psi[y, x] = 0.0
                        omega[y, x] = -(4.0 * psi[y, x-1] - 0.5 * psi[y, x-2]) / h2
                else:
                    # Interior
                    pass
    
    # Time-stepping
    for _ in range(28000):
        # Update vorticity
        omega_new = np.zeros((ny, nx), dtype=np.float32)
        for y in range(ny):
            for x in range(nx):
                if domain_mask[y, x]:
                    # Convection
                    if x > 0:
                        omega_new[y, x] += -bc_secondary[y, x] * (omega[y, x-1] - omega[y, x]) / h
                    if x < nx - 1:
                        omega_new[y, x] += bc_secondary[y, x] * (omega[y, x+1] - omega[y, x]) / h
                    if y > 0:
                        omega_new[y, x] += bc_primary[y, x] * (omega[y-1, x] - omega[y, x]) / h
                    if y < ny - 1:
                        omega_new[y, x] += -bc_primary[y, x] * (omega[y+1, x] - omega[y, x]) / h
                    # Diffusion
                    if x > 0:
                        omega_new[y, x] += h2inv * (omega[y, x-1] - 2 * omega[y, x] + omega[y, x+1])
                    if y > 0:
                        omega_new[y, x] += h2inv * (omega[y-1, x] - 2 * omega[y, x] + omega[y+1, x])
        omega = omega_new
    
    # Save results
    # Report a
    report_a = omega[1:128, 64]
    np.save('results/report_a.npy', report_a.astype(np.float32))
    # Report b
    report_b = omega[64, 1:128]
    np.save('results/report_b.npy', report_b.astype(np.float32))
    # Report c
    report_c = omega[1, 1:128]
    np.save('results/report_c.npy', report_c.astype(np.float32))
    # Field a
    np.save('results/field_a.npy', omega[1:128, 1:128].astype(np.float32))
    # Field b
    np.save('results/field_b.npy', psi[1:128, 1:128].astype(np.float32))
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(omega[1:128, 1:128], cmap='viridis', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.savefig('results/overview.png')
    plt.close()

if __name__ == '__main__':
    simulation()