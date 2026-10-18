```bash
cat > simulation.py <<'END'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.origin'] = 'upper'
plt.ioff()

config = np.load('case_config.json', allow_pickle=True).item()

nx = config['nx']
ny = config['ny']
h = config['h']
u_lid = config['u_lid']
omega_wall = config['omega_wall']
omega_top = config['omega_top']
convection_order = config['convection_order']
time_step = config['time_step']
max_steps = config['max_steps']
report_a = config['report_a']
report_b = config['report_b']
report_c = config['report_c']

bc_primary = np.load('bc_primary.npy')
bc_secondary = np.load('bc_secondary.npy')
domain_mask = np.load('domain_mask.npy')
track_a = np.load('sample_track_a.npy')
track_b = np.load('sample_track_b.npy')
track_c = np.load('sample_track_c.npy')

psi = np.zeros((ny, nx), dtype=np.float32)
omega = np.zeros((ny, nx), dtype=np.float32)

for _ in range(max_steps):
    omega[1:-1, 1:-1] = (bc_primary[1:-1, 1:-1] + bc_secondary[1:-1, 1:-1]) / h
    for _ in range(5):
        omega[1:-1, 1:-1] = (omega[2:, 1:-1] + omega[:-2, 1:-1] + omega[1:-1, 2:] + omega[1:-1, :-2]) / 4
    omega = omega.astype(np.float32)

psi[1:-1, 1:-1] = (omega[2:, 1:-1] - omega[:-2, 1:-1] + omega[1:-1, 2:] - omega[1:-1, :-2]) / 4
psi = psi.astype(np.float32)

u = np.zeros((ny, nx), dtype=np.float32)
v = np.zeros((ny, nx), dtype=np.float32)

for i in range(nx):
    for j in range(ny):
        if domain_mask[j, i]:
            if i == 0:
                u[j, i] = 0
            elif i == nx - 1:
                u[j, i] = 0
            else:
                u[j, i] = (psi[j+1, i] - psi[j-1, i]) / (2 * h)
        else:
            u[j, i] = 0

for j in range(ny):
    for i in range(nx):
        if domain_mask[j, i]:
            if j == 0:
                v[j, i] = 0
            elif j == ny - 1:
                v[j, i] = 0
            else:
                v[j, i] = -(psi[j, i+1] - psi[j, i-1]) / (2 * h)
        else:
            v[j, i] = 0

track_a = track_a.astype(np.int32)
track_b = track_b.astype(np.int32)
track_c = track_c.astype(np.int32)

report_a = u[track_a, :]
report_b = v[track_b, :]
report_c = omega[track_c, :]

np.save('results/report_a.npy', report_a)
np.save('results/report_b.npy', report_b)
np.save('results/report_c.npy', report_c)

psi = psi[1:-1, 1:-1]
omega = omega[1:-1, 1:-1]

np.save('results/field_a.npy', omega)
np.save('results/field_b.npy', psi)

plt.imshow(omega, cmap='bwr', vmax=omega.max(), vmin=omega.min(), extent=(0, 1, 0, 1), origin='lower')
plt.axis('off')
plt.savefig('results/overview.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.close()
```