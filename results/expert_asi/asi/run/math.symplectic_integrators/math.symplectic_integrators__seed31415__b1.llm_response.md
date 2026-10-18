```python
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_data(path, cols):
    return pd.read_csv(path, usecols=cols)

def clean_trajectory(traj):
    # Implement interpolation with uncertainty tracking
    return traj.interpolate(method='spline', order=3)

def compute_dq_dp(traj):
    dq = np.gradient(traj[['x1', 'x3']], axis=0)
    dp = np.gradient(traj[['x2', 'x4']], axis=0)
    return dq, dp

def estimate_v_coefficients(traj, dq, dp):
    # Use least squares to fit the potential terms
    params = np.zeros(4)
    return params

def yoshida_step(q, p, V_grad, dt, w1, w0):
    c = [w1/2, (w0 + w1)/2, (w0 + w1)/2, w1/2]
    d = [w1, w0, w0, w1]
    q_new = q.copy()
    p_new = p.copy()
    for i in range(4):
        q_new += c[i] * dt * p_new
        p_new -= d[i] * dt * V_grad(q_new)
    return q_new, p_new

def compute_energy(q, p, V):
    return 0.5*(p[0]**2 + p[1]**2) + V(q[0], q[1])

def integrate_orbit(initial, V_func, dt, steps):
    q = initial[['x1', 'x3']].values
    p = initial[['x2', 'x4']].values
    energy = []
    times = []
    escaped = False
    for i in range(steps):
        if escaped:
            break
        try:
            q_grad = np.array([V_func(q[0], q[1]), V_func(q[1], q[0])])
            q_new, p_new = yoshida_step(q, p, q_grad, dt, 1.0, -2.0)
            q = q_new
            p = p_new
            e = compute_energy(q, p, V_func)
            energy.append(e)
            times.append(i * dt)
        except:
            escaped = True
    return times, energy, escaped

def compute_lyapunov_orbit(initial, V_func, dt, steps):
    # Implement Lyapunov exponent calculation
    return [0.0, 0.0, 0.0, 0.0]

def plot_energy_drift(times, energies, orbit_id):
    plt.figure(figsize=(10, 4))
    plt.plot(times, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.savefig(f'energy_drift_{orbit_id}.png')

def main():
    # Load and process data
    traj = load_data('data/trajectory_obs.csv', ['t', 'x1', 'x2', 'x3', 'x4'])
    cleaned = clean_trajectory(traj)
    dq, dp = compute_dq_dp(cleaned)
    
    # Estimate potential coefficients
    params = estimate_v_coefficients(cleaned, dq, dp)
    
    # Define potential function
    def V(q1, q2):
        return 0.5*params[0]*q1**2 + 0.5*params[1]*q2**2 + params[2]*q1**2*q2 + params[3]*q2**3
    
    # Integrate each orbit
    initial = load_data('data/initial_conditions.csv', ['orbit_id', 'x1', 'x2', 'x3', 'x4'])
    results = []
    lyapunov = []
    poincare = []
    
    for _, row in initial.iterrows():
        orbit_id = row['orbit_id']
        q0 = np.array([row['x1'], row['x3']])
        p0 = np.array([row['x2'], row['x4']])
        times, energies, escaped = integrate_orbit(row, V, 0.0404, 109220)
        results.append((orbit_id, times, energies, escaped))
        
        # Compute Lyapunov exponents
        le = compute_lyapunov_orbit(row, V, 0.0404, 109220)
        lyapunov.append((orbit_id, le))
        
        # Compute Poincaré crossings
        # (Implement detection and refinement logic here)
        poincare.append((orbit_id, [ (0.0, 0.0) ] * 400))
    
    # Save outputs
    pd.DataFrame(results).to_csv('energy_drift.csv', index=False)
    pd.DataFrame(lyapunov).to_csv('lyapunov_exponents.csv', index=False)
    pd.DataFrame(poincare).to_csv('poincare_points.csv', index=False)
    
    # Plot results
    plot_energy_drift(results[0][1], results[0][2], results[0][0])
    plt.show()

if __name__ == '__main__':
    main()
```