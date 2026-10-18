import numpy as np
import matplotlib.pyplot as plt
import json

# Load configuration
with open('data/control_primary.json') as f:
    control = json.load(f)
axis_a = np.load('data/scan_axis_a.npy')
axis_b = np.load('data/scan_axis_b.npy')
probe_axis_a = np.load('data/probe_axis_a.npy')
state_tokens = np.load('data/state_tokens.npz')

# Initialize observables
obs_matrix_a = np.zeros((len(axis_a), len(axis_b)))
obs_matrix_b = np.zeros((len(axis_a), len(axis_b)))
obs_tensor_c = np.zeros((len(control['focus_probe_rows']), len(probe_axis_a)))
obs_matrix_d = np.zeros((len(axis_a), 2))
obs_matrix_e = np.zeros((len(axis_a), len(axis_b)))
obs_matrix_f = np.zeros((len(axis_a), len(axis_b)))

# Simulate the model
for i, T in enumerate(axis_a):
    for j, L in enumerate(axis_b):
        # Load initial state
        theta = state_tokens[state_tokens.files[i]][j]
        
        # Run simulation
        # ... simulation code ...
        
        # Compute observables
        # ... compute Upsilon, defect density, C(r), xi_2nd, chi ...
        
        # Save results
        obs_matrix_a[i, j] = Upsilon
        obs_matrix_b[i, j] = defect_density
        obs_tensor_c[i, :] = C_r
        obs_matrix_d[i, 0] = T
        obs_matrix_d[i, 1] = eta
        obs_matrix_e[i, j] = xi_2nd_L
        obs_matrix_f[i, j] = chi

# Analyze data
# ... fit procedures, jackknife, etc. ...

# Generate plots
plt.figure(figsize=(10, 6))
plt.plot(axis_a, obs_matrix_a.mean(axis=1), label='Upsilon')
plt.plot(axis_a, 2 * axis_a / np.pi, label='Nelson-Kosterlitz')
plt.legend()
plt.savefig('overview.png')

# Save results
np.save('obs_matrix_a.npy', obs_matrix_a)
np.save('obs_matrix_b.npy', obs_matrix_b)
np.save('obs_tensor_c.npy', obs_tensor_c)
np.save('obs_matrix_d.npy', obs_matrix_d)
np.save('obs_matrix_e.npy', obs_matrix_e)
np.save('obs_matrix_f.npy', obs_matrix_f)

# Write summary
with open('phase_summary.json', 'w') as f:
    json.dump(phase_summary, f)

# Write diagnostics
with open('phase_fit_diagnostics.json', 'w') as f:
    json.dump(phase_fit_diagnostics, f)