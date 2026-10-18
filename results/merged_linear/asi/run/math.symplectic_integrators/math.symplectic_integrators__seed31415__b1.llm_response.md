```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Step 1: System Identification

def load_and_clean_trajectory(data_path):
    df = pd.read_csv(data_path)
    probe_trajectories = df.groupby('probe_id').apply(lambda x: x.sort_values('t').reset_index(drop=True))
    cleaned = pd.DataFrame()
    for idx, traj in probe_trajectories.iterrows():
        # Simple linear interpolation
        traj = traj.interpolate(method='linear', limit_direction='both')
        # Remove points with large missingness
        if (traj[['x1', 'x2', 'x3', 'x4']].isna().sum(axis=1) > 1).any():
            continue
        cleaned = pd.concat([cleaned, traj])
    return cleaned

cleaned_data = load_and_clean_trajectory('data/trajectory_obs.csv')
print(f"Cleaned {len(cleaned_data)} points from {len(probe_trajectories)} probes")

def compute_canonical_relations(df, q_cols=['x1', 'x2', 'x3', 'x4']):
    dq = df[q_cols[::2]].diff() / df['t'].diff()
    dp = df[q_cols[1::2]].diff() / df['t'].diff()
    correlation = dq.corr(dp)
    rmse = np.sqrt(((dq - dp) ** 2).mean())
    return correlation, rmse

q1_col, p1_col, q2_col, p2_col = 'x1', 'x2', 'x3', 'x4'
q1_dq, p1_dp = df[q1_col].diff() / df['t'].diff(), df[p1_col].diff() / df['t'].diff()
q2_dq, p2_dp = df[q2_col].diff() / df['t'].diff(), df[p2_col].diff() / df['t'].diff()
correlation_q1, rmse_q1 = q1_dq.corr(p1_dp), np.sqrt(((q1_dq - p1_dp) ** 2).mean())
correlation_q2, rmse_q2 = q2_dq.corr(p2_dp), np.sqrt(((q2_dq - p2_dp) ** 2).mean())

print(f"Canonical correlation for q1-p1: {correlation_q1:.4f}, RMSE: {rmse_q1:.4f}")
print(f"Canonical correlation for q2-p2: {correlation_q2:.4f}, RMSE: {rmse_q2:.4f}")

def compute_force_signals(df, q_cols=['x1', 'x2', 'x3', 'x4']):
    dq = df[q_cols[::2]].diff() / df['t'].diff()
    dp = df[q_cols[1::2]].diff() / df['t'].diff()
    return dq, dp

q1_dq, p1_dp = compute_force_signals(cleaned_data)
q2_dq, p2_dp = compute_force_signals(cleaned_data)

def fit_henon_heiles_coefficients(df, q_cols=['x1', 'x2', 'x3', 'x4']):
    # Extract data
    t = df['t'].values
    q1 = df[q_cols[0]].values
    q2 = df[q_cols[2]].values
    p1 = df[q_cols[1]].values
    p2 = df[q_cols[3]].values
    
    # Compute dp/dt
    dp1 = (p1[1:] - p1[:-1]) / (t[1:] - t[:-1])
    dp2 = (p2[1:] - p2[:-1]) / (t[1:] - t[:-1])
    
    # Build Vandermonde matrix
    V = np.column_stack([
        q1[:-1]**2,  # k1*q1^2
        q2[:-1]**2,  # k2*q2^2
        q1[:-1]**2 * q2[:-1],  # a*q1^2*q2
        q2[:-1]**3  # b*q2^3
    ])
    
    # Solve for coefficients
    coeffs, _, _, _ = np.linalg.lstsq(V, dp1, rcond=None)
    return coeffs

k1, k2, a, b = fit_henon_heiles_coefficients(cleaned_data)
print(f"Fitted coefficients: k1={k1:.4f}, k2={k2:.4f}, a={a:.4f}, b={b:.4f}")

def check_dp_consistency(coeffs, df, q_cols=['x1', 'x2', 'x3', 'x4']):
    k1, k2, a, b = coeffs
    q1 = df[q_cols[0]].values
    q2 = df[q_cols[2]].values
    dp1_pred = -k1 * q1 - 2 * a * q1 * q2
    dp2_pred = -k2 * q2 - a * q1**2 - 3 * b * q2**2
    return np.corrcoef(dp1_pred, dp2_pred)[0, 1]

corr = check_dp_consistency([k1, k2, a, b], cleaned_data)
print(f"Consistency between dp1 and dp2: {corr:.4f}")

# Step 2: Long-Term Integration

def yoshida_step(q, p, V, dt, w1, w0):
    c = [w1/2, (w0 + w1)/2, (w0 + w1)/2, w1/2]
    d = [w1, w0, w0, w1]
    q_new = q.copy()
    p_new = p.copy()
    for i in range(4):
        q_new += c[i] * dt * p_new
        p_new -= d[i] * dt * V(q_new)
    q_new += c[3] * dt * p_new
    return q_new, p_new

def integrate_orbit(initial_q, initial_p, V, dt, steps, w1=1/(2-2**1/3), w0=-2**1/3/(2-2**1/3)):
    q, p = initial_q.copy(), initial_p.copy()
    energy = []
    time = []
    for i in range(steps):
        q, p = yoshida_step(q, p, V, dt, w1, w0)
        # Compute energy
        H = 0.5 * (p[0]**2 + p[1]**2) + 0.5 * k1 * q[0]**2 + 0.5 * k2 * q[2]**2 + a * q[0]**2 * q[2] + b * q[2]**3
        energy.append(H)
        time.append(i * dt)
    return np.array(time), np.array(energy), q, p

# Load initial conditions
initial_data = pd.read_csv('data/initial_conditions.csv')

def compute_lyapunov_exponents(q0, p0, V, dt, steps):
    # Initialize variational matrix
    phi = np.eye(4)
    lyapunov = np.zeros(4)
    time = 0
    for i in range(steps):
        q, p = yoshida_step(q0, p0, V, dt, w1, w0)
        # Update variational matrix
        J = np.zeros((4, 4))
        # Jacobian for q1
        J[0, 0] = 1
        J[0, 1] = dt * p0[0]
        J[0, 2] = 0
        J[0, 3] = 0
        # Jacobian for p1
        J[1, 0] = -dt * V(q)[0]
        J[1, 1] = 1
        J[1, 2] = 0
        J[1, 3] = 0
        # Jacobian for q2
        J[2, 0] = 0
        J[2, 1] = 0
        J[2, 2] = 1
        J[2, 3] = dt * p0[1]
        # Jacobian for p2
        J[3, 0] = -dt * V(q)[1]
        J[3, 1] = 0
        J[3, 2] = 1
        J[3, 3] = 1
        # Update variational matrix
        phi = np.dot(phi, J)
        # QR decomposition
        Q, R = np.linalg.qr(phi)
        lyapunov += np.log(np.abs(R.diagonal()))
        phi = Q
        time += dt
    return lyapunov / time

def find_poincare_crossings(q0, p0, V, dt, steps):
    crossings = []
    time = 0
    for i in range(steps):
        q, p = yoshida_step(q0, p0, V, dt, w1, w0)
        # Check if crossing q2=0 with p2>0
        if abs(q[2]) < 1e-8 and p[1] > 0:
            crossings.append((q[0], p[0]))
        time += dt
    return crossings

# Main execution
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Write energy drift
with open(os.path.join(output_dir, 'energy_drift.csv'), 'w') as f:
    f.write('orbit_id,integrator,time,rel_energy_error,energy\n')
    for idx, row in initial_data.iterrows():
        orbit_id = row['orbit_id']
        x1, x2, x3, x4 = row['x1'], row['x2'], row['x3'], row['x4']
        t, energy, _, _ = integrate_orbit([x1, x2, x3, x4], [x1, x2, x3, x4], lambda q: [k1 * q[0], k2 * q[2]], dt=0.0404, steps=109220)
        rel_energy_error = np.abs(energy - energy[0]) / np.abs(energy[0])
        for i in range(len(t)):
            f.write(f"{orbit_id},yoshida,{t[i]:.4f},{rel_energy_error[i]:.4f},{energy[i]:.4f}\n")

# Write Lyapunov exponents
with open(os.path.join(output_dir, 'lyapunov_exponents.csv'), 'w') as f:
    f.write('orbit_id,lambda_1,lambda_2,lambda_3,lambda_4,lambda_max\n')
    for idx, row in initial_data.iterrows():
        orbit_id = row['orbit_id']
        x1, x2, x3, x4 = row['x1'], row['x2'], row['x3'], row['x4']
        lyapunov = compute_lyapunov_exponents([x1, x2, x3, x4], [x1, x2, x3, x4], lambda q: [k1 * q[0], k2 * q[2]], dt=0.0404, steps=109220)
        f.write(f"{orbit_id},{lyapunov[0]:.4f},{lyapunov[1]:.4f},{lyapunov[2]:.4f},{lyapunov[3]:.4f},{np.max(lyapunov):.4f}\n")

# Write Poincaré section
with open(os.path.join(output_dir, 'poincare_points.csv'), 'w') as f:
    f.write('orbit_id,q1,p1\n')
    for idx, row in initial_data.iterrows():
        orbit_id = row['orbit_id']
        x1, x2, x3, x4 = row['x1'], row['x2'], row['x3'], row['x4']
        crossings = find_poincare_crossings([x1, x2, x3, x4], [x1, x2, x3, x4], lambda q: [k1 * q[0], k2 * q[2]], dt=0.0404, steps=109220)
        for q1, p1 in crossings[:400]:
            f.write(f"{orbit_id},{q1:.4f},{p1:.4f}\n")

# Plot energy drift
plt.figure(figsize=(10, 6))
sns.lineplot(x='time', y='rel_energy_error', hue='orbit_id', data=pd.read_csv(os.path.join(output_dir, 'energy_drift.csv')))
plt.xlabel('Time')
plt.ylabel('Relative Energy Error')
plt.title('Energy Drift')
plt.savefig(os.path.join(output_dir, 'energy_drift.png'))
plt.close()

# Plot Poincaré section
poincare_data = pd.read_csv(os.path.join(output_dir, 'poincare_points.csv'))
plt.figure(figsize=(10, 6))
sns.scatterplot(x='q1', y='p1', hue='orbit_id', data=poincare_data, palette='viridis')
plt.xlabel('q1')
plt.ylabel('p1')
plt.title('Poincaré Section')
plt.savefig(os.path.join(output_dir, 'poincare_section.png'))
plt.close()
```