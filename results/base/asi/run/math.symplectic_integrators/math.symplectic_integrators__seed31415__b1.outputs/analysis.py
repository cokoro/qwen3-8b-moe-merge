import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os

# Step 1: System Identification

def load_data():
    trajectory = pd.read_csv('data/trajectory_obs.csv')
    initial_conditions = pd.read_csv('data/initial_conditions.csv')
    return trajectory, initial_conditions

def preprocess_trajectory(trajectory):
    trajectory = trajectory.sort_values(['probe_id', 't'])
    probes = trajectory.groupby('probe_id')
    cleaned_data = []
    for probe_id, group in probes:
        times = group['t'].values
        x1 = group['x1'].values
        x2 = group['x2'].values
        x3 = group['x3'].values
        x4 = group['x4'].values
        # Interpolate and denoise
        x1_interp = interp1d(times, x1, kind='linear', fill_value='extrapolate')
        x2_interp = interp1d(times, x2, kind='linear', fill_value='extrapolate')
        x3_interp = interp1d(times, x3, kind='linear', fill_value='extrapolate')
        x4_interp = interp1d(times, x4, kind='linear', fill_value='extrapolate')
        # Create a time grid
        t_grid = np.linspace(times[0], times[-1], 1000)
        x1_clean = x1_interp(t_grid)
        x2_clean = x2_interp(t_grid)
        x3_clean = x3_interp(t_grid)
        x4_clean = x4_interp(t_grid)
        cleaned_data.append((t_grid, x1_clean, x2_clean, x3_clean, x4_clean))
    return cleaned_data

def compute_derivatives(x, t):
    dxdt = np.gradient(x, t)
    return dxdt

def check_canonical_relations(cleaned_data):
    results = []
    for t, x1, x2, x3, x4 in cleaned_data:
        dq1dt = compute_derivatives(x1, t)
        dq2dt = compute_derivatives(x2, t)
        dp1dt = compute_derivatives(x2, t)
        dp2dt = compute_derivatives(x4, t)
        rmse_dq1p1 = np.sqrt(np.mean((dq1dt - x2)**2))
        rmse_dq2p2 = np.sqrt(np.mean((dq2dt - x4)**2))
        corr_dq1p1 = np.corrcoef(dq1dt, x2)[0, 1]
        corr_dq2p2 = np.corrcoef(dq2dt, x4)[0, 1]
        results.append((rmse_dq1p1, rmse_dq2p2, corr_dq1p1, corr_dq2p2))
    return results

def estimate_forces(cleaned_data):
    forces = []
    for t, x1, x2, x3, x4 in cleaned_data:
        dp1dt = compute_derivatives(x2, t)
        dp2dt = compute_derivatives(x4, t)
        forces.append((dp1dt, dp2dt))
    return forces

def fit_potential_coefficients(forces, cleaned_data):
    def potential_energy(q1, q2, k1, k2, a, b):
        return 0.5 * k1 * q1**2 + 0.5 * k2 * q2**2 + a * q1**2 * q2 + b * q2**3
    
    def gradient_v(q1, q2, k1, k2, a, b):
        dv_dq1 = k1 * q1 + 2 * a * q1 * q2
        dv_dq2 = k2 * q2 + a * q1**2 + 3 * b * q2**2
        return np.array([dv_dq1, dv_dq2])
    
    def objective(params, t, dp1dt, dp2dt):
        k1, k2, a, b = params
        q1 = np.interp(t, t_grid, x1)
        q2 = np.interp(t, t_grid, x3)
        dv_dq1 = k1 * q1 + 2 * a * q1 * q2
        dv_dq2 = k2 * q2 + a * q1**2 + 3 * b * q2**2
        return (dp1dt - (-dv_dq1))**2 + (dp2dt - (-dv_dq2))**2
    
    results = []
    for t, x1, x2, x3, x4 in cleaned_data:
        dp1dt, dp2dt = forces[0]
        t_grid = t
        params_initial = [1.0, 1.0, 0.0, 0.0]
        result = least_squares(lambda params: objective(params, t_grid, dp1dt, dp2dt), params_initial)
        results.append(result.x)
    return results

def cross_check_coefficients(coefficients):
    a_values = [coeff[2] for coeff in coefficients]
    a_mean = np.mean(a_values)
    a_std = np.std(a_values)
    return a_mean, a_std

def validate_coefficients(coefficients, initial_conditions):
    k1, k2, a, b = coefficients[0]
    for _, x1, x2, x3, x4 in initial_conditions.values:
        energy = 0.5 * (x2**2 + x4**2) + 0.5 * k1 * x1**2 + 0.5 * k2 * x3**2 + a * x1**2 * x3 + b * x3**3
        print(f"Orbit {orbit_id}: Energy = {energy}")
    return True

# Step 2: Long-Term Integration & Analysis

def yoshida_integrator(q, p, k1, k2, a, b, dt):
    w1 = 1 / (2 - 2**(1/3))
    w0 = -2**(1/3) / (2 - 2**(1/3))
    c1 = c4 = w1 / 2
    c2 = c3 = (w0 + w1) / 2
    d1 = d3 = w1
    d2 = w0
    q1 = q + c1 * dt * p
    p1 = p - d1 * dt * (k1 * q1 + 2 * a * q1 * x3)
    q2 = q1 + c2 * dt * p1
    p2 = p1 - d2 * dt * (k2 * q2 + a * q1**2 + 3 * b * q2**2)
    q3 = q2 + c3 * dt * p2
    p3 = p2 - d3 * dt * (k1 * q3 + 2 * a * q3 * x3)
    q4 = q3 + c4 * dt * p3
    p4 = p3
    return q4, p4

def integrate_orbit(initial_q, initial_p, k1, k2, a, b, dt, steps):
    q = initial_q.copy()
    p = initial_p.copy()
    energy = 0.5 * (p**2).sum() + 0.5 * k1 * q[0]**2 + 0.5 * k2 * q[1]**2 + a * q[0]**2 * q[1] + b * q[1]**3
    energies = [energy]
    times = [0]
    is_escaped = False
    for _ in range(steps):
        q, p = yoshida_integrator(q, p, k1, k2, a, b, dt)
        energy = 0.5 * (p**2).sum() + 0.5 * k1 * q[0]**2 + 0.5 * k2 * q[1]**2 + a * q[0]**2 * q[1] + b * q[1]**3
        energies.append(energy)
        times.append(times[-1] + dt)
        if np.isnan(energy) or np.isinf(energy):
            is_escaped = True
            break
    return times, energies, is_escaped

def compute_lyapunov_exponents(initial_q, initial_p, k1, k2, a, b, dt, steps):
    def lyapunov_step(q, p, Phi):
        q1, p1 = yoshida_integrator(q, p, k1, k2, a, b, dt)
        q2, p2 = yoshida_integrator(q, p, k1, k2, a, b, dt)
        Phi1 = np.dot(Phi, np.eye(4))
        Phi2 = np.dot(Phi, np.eye(4))
        return q1, p1, Phi1, Phi2
    
    Phi = np.eye(4)
    lyapunov_sum = np.zeros(4)
    crossings = []
    for _ in range(steps):
        q, p, Phi, _ = lyapunov_step(q, p, Phi)
        if np.abs(p[1]) < 1e-6 and p[1] > 0:
            crossings.append((q[0], p[0]))
        if _ % 100 == 0:
            Q, R = np.linalg.qr(Phi)
            lyapunov_sum += np.log(np.abs(np.diag(R)))
            Phi = Q
    lyapunov_exponents = lyapunov_sum / (steps * dt)
    return lyapunov_exponents, crossings

def main():
    trajectory, initial_conditions = load_data()
    cleaned_data = preprocess_trajectory(trajectory)
    canonical_relations = check_canonical_relations(cleaned_data)
    forces = estimate_forces(cleaned_data)
    coefficients = fit_potential_coefficients(forces, cleaned_data)
    a_mean, a_std = cross_check_coefficients(coefficients)
    validate_coefficients(coefficients, initial_conditions)
    
    results = []
    for _, x1, x2, x3, x4 in initial_conditions.values:
        initial_q = np.array([x1, x3])
        initial_p = np.array([x2, x4])
        dt = 0.0404
        steps = 109220
        times, energies, is_escaped = integrate_orbit(initial_q, initial_p, coefficients[0][0], coefficients[0][1], coefficients[0][2], coefficients[0][3], dt, steps)
        lyapunov_exponents, crossings = compute_lyapunov_exponents(initial_q, initial_p, coefficients[0][0], coefficients[0][1], coefficients[0][2], coefficients[0][3], dt, steps)
        results.append((times, energies, is_escaped, lyapunov_exponents, crossings))
    
    # Save results
    energy_drift = []
    lyapunov_exponents = []
    poincare_points = []
    for i, (times, energies, is_escaped, lyapunov_exponents, crossings) in enumerate(results):
        for t, energy in zip(times, energies):
            energy_drift.append((i+1, 'yoshida', t, abs((energy - energies[0])/energies[0]), energy))
        lyapunov_exponents.append((i+1, lyapunov_exponents[0], lyapunov_exponents[1], lyapunov_exponents[2], lyapunov_exponents[3], max(lyapunov_exponents)))
        for q1, p1 in crossings:
            poincare_points.append((i+1, q1, p1))
    
    pd.DataFrame(energy_drift, columns=['orbit_id', 'integrator', 'time', 'rel_energy_error', 'energy']).to_csv('energy_drift.csv', index=False)
    pd.DataFrame(lyapunov_exponents, columns=['orbit_id', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_max']).to_csv('lyapunov_exponents.csv', index=False)
    pd.DataFrame(poincare_points, columns=['orbit_id', 'q1', 'p1']).to_csv('poincare_points.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, (times, energies, _, _, _) in enumerate(results):
        plt.plot(times, energies, label=f'Orbit {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Drift')
    plt.legend()
    plt.savefig('energy_drift.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for i, (times, _, _, _, crossings) in enumerate(results):
        times_crossings = [t for t, _ in crossings]
        energies_crossings = [energy for _, energy in crossings]
        plt.scatter(times_crossings, energies_crossings, label=f'Orbit {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Poincaré Section')
    plt.legend()
    plt.savefig('poincare_section.png')
    plt.close()

if __name__ == "__main__":
    main()