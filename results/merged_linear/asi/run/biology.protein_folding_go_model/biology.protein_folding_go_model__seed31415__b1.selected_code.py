import numpy as np
import json
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import entropy

# Load parameters
with open('data/parameters.json') as f:
    params = json.load(f)
n_beads = params['n_beads']
friction = params['friction']
dt_reduced = params['dt_reduced']
temperature_list = np.array(params['temperature_list'])
n_steps_per_T = params['n_steps_per_T']
contact_cutoff = params['contact_cutoff']
min_seq_separation = params['min_seq_separation']
save_interval = params['save_interval']

# Load reference coordinates
target_coords = np.load('data/target_coords.npy')

# Compute native contact map
r_ref = np.linalg.norm(target_coords[1:] - target_coords[:-1], axis=1)
contact_mask = np.zeros((n_beads, n_beads), dtype=bool)
for i in range(n_beads):
    for j in range(n_beads):
        if abs(i - j) >= min_seq_separation and np.linalg.norm(target_coords[i] - target_coords[j]) <= contact_cutoff:
            contact_mask[i, j] = True

# Compute reference distances for native contacts
r_ref_contacts = r_ref[contact_mask]

# Simulation
K_bond = 200.0
epsilon_native = 1.5
K_rep = 0.8
sigma = 3.5

def compute_forces(coords, temp):
    # Bond forces
    r_bond = coords[1:] - coords[:-1]
    r_bond_ref = target_coords[1:] - target_coords[:-1]
    bond_forces = K_bond * (r_bond - r_bond_ref) / np.linalg.norm(r_bond, axis=1, keepdims=True)

    # Native contact forces
    r_native = coords[contact_mask]
    r_native_ref = target_coords[contact_mask]
    native_forces = np.zeros_like(coords)
    for i in range(n_beads):
        for j in range(n_beads):
            if contact_mask[i, j]:
                dr = r_native[i] - r_native_ref
                r = np.linalg.norm(dr)
                if r < 0.6 * r_ref_contacts[i]:
                    r = 0.6 * r_ref_contacts[i]
                force = epsilon_native * (5 * (r_ref_contacts[i] / r)**12 - 6 * (r_ref_contacts[i] / r)**10)
                native_forces[i] += force * (dr / r)
                native_forces[j] -= force * (dr / r)

    # Non-native repulsion forces
    r_nonnative = coords[~contact_mask]
    r_nonnative_ref = target_coords[~contact_mask]
    nonnative_forces = np.zeros_like(coords)
    for i in range(n_beads):
        for j in range(n_beads):
            if not contact_mask[i, j] and abs(i - j) >= min_seq_separation:
                dr = r_nonnative[i] - r_nonnative_ref
                r = np.linalg.norm(dr)
                if r < 0.7 * sigma:
                    r = 0.7 * sigma
                force = K_rep * (sigma / r)**12
                nonnative_forces[i] += force * (dr / r)
                nonnative_forces[j] -= force * (dr / r)

    return bond_forces + native_forces + nonnative_forces

def langevin_step(coords, velocities, temp):
    c1 = np.exp(-friction * dt_reduced)
    c2 = np.sqrt((1 - c1**2) * temp / n_beads)
    velocities += 0.5 * dt_reduced * compute_forces(coords, temp) / n_beads
    coords += 0.5 * dt_reduced * velocities
    velocities = c1 * velocities + c2 * np.random.randn(n_beads, 3)
    coords += 0.5 * dt_reduced * velocities
    return coords, velocities

def compute_order_parameter(coords):
    r_ij = np.linalg.norm(coords[contact_mask], axis=1)
    r_ref_contacts = np.linalg.norm(target_coords[contact_mask], axis=1)
    q = np.sum(r_ij <= 1.2 * r_ref_contacts) / len(r_ref_contacts)
    return q

def run_simulation(temp):
    coords = target_coords.copy()
    velocities = np.random.randn(n_beads, 3)
    velocities *= np.sqrt(temp)
    equil_steps = 20000
    prod_steps = n_steps_per_T
    save_steps = save_interval
    q_trajectory = []
    energy_trajectory = []
    for step in range(equil_steps + prod_steps):
        coords, velocities = langevin_step(coords, velocities, temp)
        if step >= equil_steps:
            q = compute_order_parameter(coords)
            q_trajectory.append(q)
            energy = np.sum(compute_forces(coords, temp) * velocities) / n_beads
            energy_trajectory.append(energy)
    return np.array(q_trajectory), np.array(energy_trajectory)

# Run simulations
K = len(temperature_list)
order_parameter_trajectory = []
energy_trajectory = []
for temp in temperature_list:
    q, e = run_simulation(temp)
    order_parameter_trajectory.append(q)
    energy_trajectory.append(e)

order_parameter_trajectory = np.array(order_parameter_trajectory)
energy_trajectory = np.array(energy_trajectory)

# Save outputs
np.save('final_coords.npy', target_coords)
np.save('order_parameter_trajectory.npy', order_parameter_trajectory)
np.save('order_parameter_histograms.npy', [np.histogram(q, bins=50, range=(0, 1))[0] for q in order_parameter_trajectory])
np.save('heat_capacity_curve.npy', np.array([[temp for temp in temperature_list], [np.var(e) / temp**2 for e in energy_trajectory]]))
np.save('q_energy_joint_histograms.npy', [np.histogram2d(q, e, bins=(20, 20), range=((0, 1), (-10, 10)))[0] for q, e in zip(order_parameter_trajectory, energy_trajectory)])
np.save('native_contact_probability.npy', np.array([np.sum(q * contact_mask) / len(contact_mask) for q in order_parameter_trajectory]))

# Transition temperature
def compute_avg_q(temp):
    q = np.mean(order_parameter_trajectory[np.where(temperature_list == temp)])
    return q

avg_q = np.array([compute_avg_q(t) for t in temperature_list])
interpolator = interp1d(temperature_list, avg_q, kind='cubic')
T_f = minimize_scalar(lambda t: np.abs(interpolator(t) - 0.5), bounds=(temperature_list[0], temperature_list[-1]), method='bounded').x

# Reweighting
def reweight_q(temp, q):
    return np.exp(-temp * (np.mean(energy_trajectory[np.where(temperature_list == temp)] - np.mean(energy_trajectory[np.where(temperature_list == T_f)]))))

P_q = np.array([reweight_q(t, q) for t in temperature_list for q in order_parameter_trajectory])
P_q /= np.sum(P_q)

# Free-energy profile
n_bins = 50
q_centers = np.linspace(0, 1, n_bins)
F_q = -np.log(P_q.reshape(len(temperature_list), -1).sum(axis=0) / len(temperature_list))
F_q -= F_q.min()

# Save free-energy profile
np.save('transition_free_energy_profile.npy', np.array([q_centers, F_q]))

# Analyze modes
peaks = np.where(np.gradient(F_q) > 0)[0]
if len(peaks) < 2:
    mode_separation = 0.0
    free_energy_barrier_kT = 0.0
    warnings = ['Single mode detected']
else:
    mode_separation = np.min(np.diff(peaks))
    free_energy_barrier_kT = np.max(F_q[peaks])
    warnings = []

# Reference contact Jaccard index
ref_contacts = np.zeros((n_beads, n_beads), dtype=bool)
for i in range(n_beads):
    for j in range(n_beads):
        if abs(i - j) >= min_seq_separation and np.linalg.norm(target_coords[i] - target_coords[j]) <= contact_cutoff:
            ref_contacts[i, j] = True
sim_contacts = np.zeros((n_beads, n_beads), dtype=bool)
for i in range(n_beads):
    for j in range(n_beads):
        if abs(i - j) >= min_seq_separation and np.linalg.norm(order_parameter_trajectory[-1][i] - order_parameter_trajectory[-1][j]) <= contact_cutoff:
            sim_contacts[i, j] = True
jaccard = np.sum(ref_contacts & sim_contacts) / np.sum(ref_contacts | sim_contacts)

# Save analysis
with open('transition_analysis.json', 'w') as f:
    json.dump({
        'transition_temperature': T_f,
        'peak_heat_capacity': np.max(np.var(energy_trajectory, axis=1) / temperature_list**2),
        'order_parameter_at_transition_mean': np.mean(order_parameter_trajectory[np.where(temperature_list == T_f)]),
        'order_parameter_at_transition_modes': peaks,
        'mode_separation': mode_separation,
        'free_energy_barrier_kT': free_energy_barrier_kT,
        'reference_contact_jaccard': jaccard,
        'warnings': warnings
    }, f)

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(target_coords[:, 0], target_coords[:, 1], 'o')
plt.title('Final Configuration')
plt.subplot(2, 2, 2)
for i in range(K):
    plt.hist(order_parameter_trajectory[i], bins=50, range=(0, 1), alpha=0.5)
plt.title('Q Histograms')
plt.subplot(2, 2, 3)
plt.plot(temperature_list, np.var(energy_trajectory, axis=1) / temperature_list**2)
plt.title('Heat Capacity')
plt.subplot(2, 2, 4)
plt.plot(q_centers, F_q)
plt.title('Free-Energy Profile')
plt.savefig('visualization.png')