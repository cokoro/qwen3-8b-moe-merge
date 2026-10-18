import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def read_data(file_path):
    return np.load(file_path, allow_pickle=True)

def write_data(file_path, data):
    np.save(file_path, data.astype(np.float32))

def compute_vorticity_field(omega, psi, dy, dx):
    omega[1:-1, 1:-1] = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dy**2 - \
                        (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dx**2
    return omega

def compute_streamfunction_field(psi, omega, dy, dx):
    psi[1:-1, 1:-1] = (omega[2:, 1:-1] + 2 * omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / (2 * dy**2) + \
                      (omega[1:-1, 2:] + 2 * omega[1:-1, 1:-1] + omega[1:-1, :-2]) / (2 * dx**2)
    return psi

def compute_velocity_field(psi, dy, dx):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    v[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)
    return u, v

def main():
    # Load input data
    case_config = read_data('case_config.json')
    bc_primary = read_data('bc_primary.npy')
    bc_secondary = read_data('bc_secondary.npy')
    domain_mask = read_data('domain_mask.npy')
    sample_track_a = read_data('sample_track_a.npy')
    sample_track_b = read_data('sample_track_b.npy')
    sample_track_c = read_data('sample_track_c.npy')

    # Configuration
    domain = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    interior_grid = np.array([127, 127], dtype=np.int32)
    full_grid = np.array([129, 129], dtype=np.int32)
    reynolds = case_config['reynolds']
    kinematic_viscosity = case_config['kinematic_viscosity']
    probe_count = case_config['probe_count']
    h = domain[1] - domain[0]

    # Grid spacing
    dy = h / full_grid[0]
    dx = h / full_grid[1]

    # Initialize vorticity and streamfunction
    omega = np.zeros((full_grid[0], full_grid[1]), dtype=np.float32)
    psi = np.zeros((full_grid[0], full_grid[1]), dtype=np.float32)

    # Apply boundary conditions
    omega[0, :] = 0.0
    omega[-1, :] = 0.0
    omega[:, 0] = 0.0
    omega[:, -1] = 0.0

    # Apply moving lid condition
    lid_velocity = bc_primary[0, 128]
    omega[1, 1:129] = -4 * (psi[2, 1:129] - psi[0, 1:129]) / dy**2 - 3 * lid_velocity / dy

    # Solve vorticity equation
    for _ in range(28000):
        omega_new = omega.copy()
        omega_new[1:-1, 1:-1] = 2 * omega[1:-1, 1:-1] - omega[2, 1:-1] - omega[0, 1:-1] - \
                               omega[1:-1, 2] - omega[1:-1, 0] + 2 * kinematic_viscosity * (omega[2, 1:-1] - 2 * omega[1:-1, 1:-1] + omega[0, 1:-1]) / dy**2 + \
                               2 * kinematic_viscosity * (omega[1:-1, 2] - 2 * omega[1:-1, 1:-1] + omega[1:-1, 0]) / dx**2 + \
                               2 * kinematic_viscosity * (omega[2, 1:-1] - 2 * omega[1:-1, 1:-1] + omega[0, 1:-1]) / dy**2 + \
                               2 * kinematic_viscosity * (omega[1:-1, 2] - 2 * omega[1:-1, 1:-1] + omega[1:-1, 0]) / dx**2
        omega = omega_new

    # Compute streamfunction
    psi = compute_streamfunction_field(psi, omega, dy, dx)

    # Compute velocity
    u, v = compute_velocity_field(psi, dy, dx)

    # Extract interior fields
    interior_omega = omega[1:-1, 1:-1]
    interior_psi = psi[1:-1, 1:-1]

    # Save fields
    write_data('results/field_a.npy', interior_omega)
    write_data('results/field_b.npy', interior_psi)

    # Extract reports
    report_a = u[1:-1, int(sample_track_a[0]) - 1]
    report_b = v[int(sample_track_b[0]) - 1, 1:-1]
    report_c = omega[1, int(sample_track_c[0]) - 1]

    # Save reports
    write_data('results/report_a.npy', report_a)
    write_data('results/report_b.npy', report_b)
    write_data('results/report_c.npy', report_c)

    # Plot overview
    plt.figure(figsize=(10, 10))
    plt.imshow(interior_omega, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
    plt.colorbar(label='Vorticity')
    plt.title('Steady Solution: Interior Vorticity Field')
    plt.savefig('results/overview.png')

if __name__ == '__main__':
    main()