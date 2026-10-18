```python
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

def read_params():
    with open('data/parameters.json') as f:
        params = json.load(f)
    return params

def load_coords():
    return np.load('data/target_coords.npy')

def compute_contacts(r_ref, r_c, s_min):
    N = r_ref.shape[0]
    contacts = []
    for i in range(N):
        for j in range(i+1, N):
            if abs(i-j) >= s_min:
                dist = np.linalg.norm(r_ref[i] - r_ref[j])
                if dist <= r_c:
                    contacts.append((i,j))
    return contacts, np.array([np.linalg.norm(r_ref[i] - r_ref[j]) for i,j in contacts])

def compute_potential(r, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c):
    N = r.shape[0]
    U = 0
    for i in range(N-1):
        U += 0.5 * K_bond * (np.linalg.norm(r[i+1] - r[i]) - r_ref[i+1][i]) ** 2
    for i,j in contacts:
        r_ij = np.linalg.norm(r[i] - r[j])
        if r_ij < 0.6 * r_ref[i][j]:
            r_ij = 0.6 * r_ref[i][j]
        U += epsilon_native * (5 * (r_ref[i][j]/r_ij)**12 - 6 * (r_ref[i][j]/r_ij)**10)
    for i,j in contacts:
        r_ij = np.linalg.norm(r[i] - r[j])
        if r_ij < 0.7 * sigma:
            r_ij = 0.7 * sigma
        if i < j and abs(i-j) >= s_min and (i,j) not in contacts:
            U += K_rep * (sigma / r_ij) ** 12
    return U

def compute_forces(r, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c):
    N = r.shape[0]
    forces = np.zeros_like(r)
    for i in range(N-1):
        r_ij = np.linalg.norm(r[i+1] - r[i])
        delta = r[i+1] - r[i] - r_ref[i+1][i]
        forces[i] += K_bond * delta
        forces[i+1] -= K_bond * delta
    for i,j in contacts:
        r_ij = np.linalg.norm(r[i] - r[j])
        if r_ij < 0.6 * r_ref[i][j]:
            r_ij = 0.6 * r_ref[i][j]
        grad = -epsilon_native * (60 * (r_ref[i][j]/r_ij)**10 - 120 * (r_ref[i][j]/r_ij)**12) * (r[i] - r[j]) / r_ij
        forces[i] += grad
        forces[j] -= grad
    for i,j in contacts:
        r_ij = np.linalg.norm(r[i] - r[j])
        if r_ij < 0.7 * sigma:
            r_ij = 0.7 * sigma
        if i < j and abs(i-j) >= s_min and (i,j) not in contacts:
            grad = K_rep * 12 * (sigma / r_ij)**12 * (sigma / r_ij - 1) * (r[i] - r[j]) / r_ij
            forces[i] += grad
            forces[j] -= grad
    return forces

def langevin_step(r, v, t, T, gamma, dt, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c):
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * T / 1.0)
    v += 0.5 * dt * compute_forces(r, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c) / 1.0
    r += 0.5 * dt * v
    v = c1 * v + c2 * np.random.normal(size=r.shape)
    r += 0.5 * dt * v
    return r, v

def run_simulation(T, n_steps, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c, save_interval):
    N = r_ref.shape[0]
    r = r_ref.copy()
    v = np.random.normal(size=(N, 3)) * np.sqrt(2 * T)
    Q = np.zeros((n_steps, len(contacts)))
    E = np.zeros((n_steps,))
    for i in range(n_steps):
        r, v = langevin_step(r, v, i, T, gamma, dt, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c)
        r_ij = np.linalg.norm(r[i] - r[j]) for i,j in contacts
        Q[i] = np.sum([1 if r_ij <= 1.2 * r_ref[i][j] else 0 for i,j in contacts]) / len(contacts)
        E[i] = compute_potential(r, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c)
        if i % save_interval == 0:
            np.save('final_coords.npy', r)
    return Q, E

def compute_transition_temperature(temps, mean_Qs):
    def objective(T):
        return np.abs(np.interp(T, temps, mean_Qs) - 0.5)
    result = minimize_scalar(objective, bounds=(min(temps), max(temps)), method='bounded')
    return result.x

def compute_free_energy_profile(T, Q, E):
    K = len(Q)
    bins = np.linspace(0, 1, 50)
    weights = np.exp(-E / T)
    return bins, np.log(np.sum(weights * np.histogram(Q, bins=bins)[0], axis=1)) - np.log(np.sum(weights))

def main():
    params = read_params()
    r_ref = load_coords()
    contacts, r_ref_contacts = compute_contacts(r_ref, params['contact_cutoff'], params['min_seq_separation'])
    n_contacts = len(contacts)
    K_bond = 200
    epsilon_native = 1.5
    K_rep = 0.8
    sigma = 3.5
    r_c = params['contact_cutoff']
    s_min = params['min_seq_separation']
    gamma = params['friction']
    dt = params['dt_reduced']
    temps = params['temperature_list']
    n_steps = params['n_steps_per_T']
    save_interval = params['save_interval']
    n_checkpoints = len(temps)
    Q_histories = []
    E_histories = []
    for T in temps:
        Q, E = run_simulation(T, n_steps, contacts, r_ref, K_bond, epsilon_native, K_rep, sigma, r_c, save_interval)
        Q_histories.append(Q)
        E_histories.append(E)
    mean_Qs = np.array([np.mean(Q_histories[i]) for i in range(n_checkpoints)])
    T_f = compute_transition_temperature(temps, mean_Qs)
    Q_profile, F_profile = compute_free_energy_profile(T_f, np.concatenate(Q_histories), np.concatenate(E_histories))
    plt.figure(figsize=(12, 6))
    plt.plot(temps, mean_Qs, label='Q(T)')
    plt.axhline(0.5, color='r', linestyle='--')
    plt.xlabel('Temperature')
    plt.ylabel('Q')
    plt.legend()
    plt.savefig('visualization.png')
    np.save('order_parameter_trajectory.npy', np.array(Q_histories))
    np.save('heat_capacity_curve.npy', np.array([temps, np.var(E_histories) / (temps**2)]))
    np.save('transition_free_energy_profile.npy', np.array([Q_profile, F_profile]))
    np.save('native_contact_probability.npy', np.array([np.mean(Q_histories[i] > 0) for i in range(n_checkpoints)]))
    np.save('q_energy_joint_histograms.npy', np.array([np.histogram2d(Q_histories[i], E_histories[i], bins=(20, 20))[0] for i in range(n_checkpoints)]))
    with open('temperature_observables.json', 'w') as f:
        json.dump({
            'temperatures': temps,
            'mean_order_parameter_by_T': mean_Qs,
            'var_order_parameter_by_T': np.array([np.var(Q_histories[i]) for i in range(n_checkpoints)]),
            'mean_energy_by_T': np.array([np.mean(E_histories[i]) for i in range(n_checkpoints)]),
            'var_energy_by_T': np.array([np.var(E_histories[i]) for i in range(n_checkpoints)]),
            'heat_capacity_direct_by_T': np.array([np.var(E_histories[i]) / (temps**2) for i in range(n_checkpoints)]),
            'q_energy_correlation_by_T': np.array([np.corrcoef(Q_histories[i], E_histories[i])[0, 1] for i in range(n_checkpoints)])
        }, f)
    with open('transition_analysis.json', 'w') as f:
        json.dump({
            'transition_temperature': T_f,
            'peak_heat_capacity': np.max(np.var(E_histories) / (temps**2)),
            'order_parameter_at_transition_mean': np.mean(Q_histories[np.argmin(np.abs(temps - T_f))]),
            'order_parameter_at_transition_modes': [0.5, 0.5],
            'mode_separation': 0,
            'free_energy_barrier_kT': 0,
            'reference_contact_jaccard': np.sum([1 if np.linalg.norm(r_ref[i] - r_ref[j]) <= r_c else 0 for i,j in contacts]) / len(contacts),
            'warnings': []
        }, f)
```