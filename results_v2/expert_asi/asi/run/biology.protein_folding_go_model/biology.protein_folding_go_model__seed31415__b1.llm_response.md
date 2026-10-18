```python
import numpy as np
import json
import os
from scipy.interpolate import CubicSpline

# Read parameters
with open('data/parameters.json') as f:
    params = json.load(f)

N = params['n_beads']
T_list = params['temperature_list']
friction = params['friction']
dt = params['dt_reduced']
n_steps_per_T = params['n_steps_per_T']
save_interval = params['save_interval']
contact_cutoff = params['contact_cutoff']
min_seq_sep = params['min_seq_separation']

# Load reference coordinates
ref_coords = np.load('data/target_coords.npy')
ref_distances = np.linalg.norm(ref_coords[np.newaxis, :, :] - ref_coords[:, np.newaxis, :], axis=2)
s_min = min_seq_sep
r_c = contact_cutoff
C = np.where((np.abs(np.arange(N)[:, np.newaxis] - np.arange(N)[np.newaxis, :]) >= s_min) &
             (ref_distances <= r_c))
C = np.array(C).T
n_contacts = len(C)

# Precompute reference distances and contact parameters
r_ref = ref_distances[C[:, 0], C[:, 1]]
K_bond = 200.0
epsilon_native = 1.5
K_rep = 0.8
sigma = 3.5

# Initialize output arrays
K = len(T_list)
order_traj = np.zeros((K, n_steps_per_T // save_interval))
histograms = np.zeros((K, 50))
heat_cap = np.zeros((2, 100))
q_energy_corr = np.zeros(K)
q_energy_joint = np.zeros((K, 20, 20))
native_probs = np.zeros((K, n_contacts))

# Simulate each temperature
for idx, T in enumerate(T_list):
    # Initialize positions
    r = ref_coords.copy()
    # Maxwell velocities
    v = np.sqrt(2.0 * T / (friction * dt)) * np.random.randn(N, 3)
    # Equilibrate
    for _ in range(20000):
        # BAOAB step
        # ... (full BAOAB implementation here)
    # Run production
    for step in range(n_steps_per_T):
        # BAOAB step
        # ... (full BAOAB implementation here)
        # Compute Q
        r_ij = np.linalg.norm(r[np.newaxis, :, :] - r[:, np.newaxis, :], axis=2)
        q = np.mean(np.sum(np.where(r_ij <= 1.2 * r_ref, 1, 0), axis=1) <= n_contacts)
        order_traj[idx, step // save_interval] = q
        # Save energy, etc.
    # Compute histograms and other diagnostics
    # ... (histogram computation, heat capacity, etc.)

# Interpolate Q(T) to find T_f
T_f = find_transition_temperature(T_list, np.mean(order_traj, axis=1))
# Reweight to find F(Q)
# ... (multi-histogram reweighting)

# Save outputs
np.save('final_coords.npy', r)
np.save('order_parameter_trajectory.npy', order_traj)
np.save('order_parameter_histograms.npy', histograms)
np.save('heat_capacity_curve.npy', heat_cap)
with open('temperature_observables.json', 'w') as f:
    json.dump({'temperatures': T_list,
               'mean_order_parameter_by_T': np.mean(order_traj, axis=1),
               'var_order_parameter_by_T': np.var(order_traj, axis=1),
               'mean_energy_by_T': ...,
               'var_energy_by_T': ...,
               'heat_capacity_direct_by_T': ...,
               'q_energy_correlation_by_T': q_energy_corr},
              f)
np.save('transition_free_energy_profile.npy', ...)
np.save('q_energy_joint_histograms.npy', q_energy_joint)
np.save('native_contact_probability.npy', native_probs)
with open('transition_analysis.json', 'w') as f:
    json.dump({'transition_temperature': T_f,
               'peak_heat_capacity': ...,
               'order_parameter_at_transition_mean': ...,
               'order_parameter_at_transition_modes': ...,
               'mode_separation': ...,
               'free_energy_barrier_kT': ...,
               'reference_contact_jaccard': ...,
               'warnings': ...},
              f)
```