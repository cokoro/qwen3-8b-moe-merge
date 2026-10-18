```bash
cat > analysis.py << 'EOF'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_initial_conditions():
    df = pd.read_csv('data/initial_conditions.csv', header=0)
    return df

def read_trajectory():
    df = pd.read_csv('data/trajectory_obs.csv', header=0)
    return df

def preprocess_trajectory(df, t_col='t', q1_col='x1', q2_col='x3', p1_col='x2', p2_col='x4'):
    # Filter out rows with missing data
    df_clean = df.dropna()
    df_clean[t_col] = pd.to_numeric(df_clean[t_col], errors='coerce')
    df_clean = df_clean.sort_values(t_col)

    # Interpolate missing values in each column
    df_clean = df_clean.interpolate(method='linear', limit_direction='both')

    # Compute the time step
    dt = df_clean[t_col].iloc[1] - df_clean[t_col].iloc[0]

    return df_clean, dt

def verify_canonical_relations(df, q1_col='x1', p1_col='x2', q2_col='x3', p2_col='x4', t_col='t'):
    df['dq1_dt'] = df[q1_col].diff() / df[t_col].diff()
    df['dp1_dt'] = df[p1_col].diff() / df[t_col].diff()
    df['dq2_dt'] = df[q2_col].diff() / df[t_col].diff()
    df['dp2_dt'] = df[p2_col].diff() / df[t_col].diff()

    corr1 = df['dq1_dt'].corr(df['p1'])
    corr2 = df['dq2_dt'].corr(df['p2'])

    rmse1 = np.sqrt(np.mean((df['dq1_dt'] - df['p1'])**2))
    rmse2 = np.sqrt(np.mean((df['dq2_dt'] - df['p2'])**2))

    return corr1, rmse1, corr2, rmse2

def compute_forces(df, q1_col='x1', p1_col='x2', q2_col='x3', p2_col='x4', t_col='t'):
    df['dp1_dt'] = df[p1_col].diff() / df[t_col].diff()
    df['dp2_dt'] = df[p2_col].diff() / df[t_col].diff()
    return df[['q1', 'p1', 'dp1_dt', 'q2', 'p2', 'dp2_dt']]

def fit_potential_coefficients(df, q1_col='q1', q2_col='q2', dp1_col='dp1_dt', dp2_col='dp2_dt'):
    X = df[[q1_col, q2_col]]
    y = df[[dp1_col, dp2_col]]
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeff

def validate_coefficients(df, coeff, q1_col='q1', q2_col='q2', dp1_col='dp1_dt', dp2_col='dp2_dt'):
    X = df[[q1_col, q2_col]]
    y = df[[dp1_col, dp2_col]]
    y_pred = X @ coeff
    rmse1 = np.sqrt(np.mean((y_pred[:, 0] - y[:, 0])**2))
    rmse2 = np.sqrt(np.mean((y_pred[:, 1] - y[:, 1])**2))
    return rmse1, rmse2

def simulate_initial_conditions(df_initial, coeff, dt=0.0404, steps=109220):
    orbits = []
    for _, row in df_initial.iterrows():
        orbit_id = row['orbit_id']
        q1 = row['x1']
        p1 = row['x2']
        q2 = row['x3']
        p2 = row['x4']

        # Yoshida's 4th-order symplectic integrator step
        c = np.array([1.0/(2-2**(1/3)), (2**(1/3)+1.0)/(2-2**(1/3)), (2**(1/3)+1.0)/(2-2**(1/3)), 1.0/(2-2**(1/3))])
        d = np.array([2**(1/3)/(2-2**(1/3)), -2**(1/3)/(2-2**(1/3)), -2**(1/3)/(2-2**(1/3)), 2**(1/3)/(2-2**(1/3))])

        q1_vals = np.zeros(steps+1)
        p1_vals = np.zeros(steps+1)
        q2_vals = np.zeros(steps+1)
        p2_vals = np.zeros(steps+1)

        q1_vals[0] = q1
        p1_vals[0] = p1
        q2_vals[0] = q2
        p2_vals[0] = p2

        for i in range(1, steps+1):
            q1_vals[i] = q1_vals[i-1] + c[0]*dt*p1_vals[i-1]
            p1_vals[i] = p1_vals[i-1] - d[0]*dt*(coeff[0]*q1_vals[i-1] + coeff[2]*q1_vals[i-1]*q2_vals[i-1])
            q2_vals[i] = q2_vals[i-1] + c[1]*dt*p2_vals[i-1]
            p2_vals[i] = p2_vals[i-1] - d[1]*dt*(coeff[1]*q2_vals[i-1] + coeff[3]*q1_vals[i-1]**2 + coeff[4]*q2_vals[i-1]**2)
            q1_vals[i] = q1_vals[i] + c[2]*dt*p1_vals[i]
            q2_vals[i] = q2_vals[i] + c[3]*dt*p2_vals[i]

        orbits.append((orbit_id, q1_vals, p1_vals, q2_vals, p2_vals))
    return orbits

def compute_energy(q1, p1, q2, p2, coeff):
    return 0.5*(p1**2 + p2**2) + coeff[0]*q1**2 + coeff[1]*q2**2 + coeff[2]*q1**2*q2 + coeff[3]*q2**3

def compute_lyapunov_exponents(orbits, coeff, dt=0.0404, steps=109220):
    lyapunovs = []
    for orbit_id, q1_vals, p1_vals, q2_vals, p2_vals in orbits:
        # Initialize variational matrix
        phi = np.eye(4)
        lyapunov = np.zeros(4)

        for i in range(1, steps+1):
            # Compute Jacobian
            dq1_dt = p1_vals[i-1] + coeff[2]*q1_vals[i-1]*q2_vals[i-1] + coeff[2]*q1_vals[i-1]*q2_vals[i-1]
            dp1_dt = -coeff[0]*2*q1_vals[i-1] - coeff[2]*2*q1_vals[i-1]*q2_vals[i-1] - coeff[2]*q1_vals[i-1]**2
            dq2_dt = p2_vals[i-1] + coeff[3]*q2_vals[i-1]**2 + coeff[2]*q1_vals[i-1]**2
            dp2_dt = -coeff[1]*2*q2_vals[i-1] - coeff[3]*3*q2_vals[i-1]**2 - coeff[2]*2*q1_vals[i-1]*q2_vals[i-1]

            # Apply Yoshida step
            phi = np.array([
                [1, dt*dq1_dt, 0, 0],
                [-dt*dp1_dt, 1, 0, 0],
                [0, 0, 1, dt*dq2_dt],
                [0, 0, -dt*dp2_dt, 1]
            ]) @ phi

            # QR decomposition
            q, r = np.linalg.qr(phi)
            lyapunov += np.log(np.abs(np.diag(r)))
            phi = q

        lyapunov /= (steps * dt)
        lyapunovs.append((orbit_id, lyapunov[0], lyapunov[1], lyapunov[2], lyapunov[3], max(lyapunov)))
    return lyapunovs

def find_poincare_crossings(orbits, coeff, dt=0.0404, steps=109220, t_start=1000, t_end=9999999):
    crossings = []
    for orbit_id, q1_vals, p1_vals, q2_vals, p2_vals in orbits:
        q1 = q1_vals[t_start:t_end]
        p1 = p1_vals[t_start:t_end]
        q2 = q2_vals[t_start:t_end]
        p2 = p2_vals[t_start:t_end]

        crossings_i = []
        for i in range(len(q1)-1):
            if q2[i] > 0 and q2[i+1] < 0:
                # Use Newton-Raphson to find the exact crossing time
                def f(t):
                    return q2_vals[int(t*dt)] - q2_vals[int((t-1)*dt)]
                t_cross = (t_start + i) + (f(t_start + i + 1) / (f(t_start + i + 1) - f(t_start + i))) * dt
                q1_cross = q1_vals[int(t_cross*dt)]
                p1_cross = p1_vals[int(t_cross*dt)]
                crossings_i.append((orbit_id, q1_cross, p1_cross))
        crossings.extend(crossings_i)
    return crossings

def main():
    df_initial = read_initial_conditions()
    df_traj = read_trajectory()
    df_clean, dt = preprocess_trajectory(df_traj)
    corr1, rmse1, corr2, rmse2 = verify_canonical_relations(df_clean)
    df_forces = compute_forces(df_clean)
    coeff = fit_potential_coefficients(df_forces)
    rmse1, rmse2 = validate_coefficients(df_forces, coeff)
    orbits = simulate_initial_conditions(df_initial, coeff, dt=dt, steps=109220)
    lyapunovs = compute_lyapunov_exponents(orbits, coeff, dt=dt, steps=109220)
    crossings = find_poincare_crossings(orbits, coeff, dt=dt, steps=109220, t_start=1000, t_end=9999999)

    # Output energy drift
    energy_drift = []
    for orbit_id, q1_vals, p1_vals, q2_vals, p2_vals in orbits:
        energy = compute_energy(q1_vals, p1_vals, q2_vals, p2_vals, coeff)
        energy_drift.append((orbit_id, energy))
    df_energy = pd.DataFrame(energy_drift, columns=['orbit_id', 'energy'])
    df_energy.to_csv('energy_drift.csv', index=False)

    # Output Lyapunov exponents
    df_lyapunov = pd.DataFrame(lyapunovs, columns=['orbit_id', 'lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'lambda_max'])
    df_lyapunov.to_csv('lyapunov_exponents.csv', index=False)

    # Output Poincaré section points
    df_crossings = pd.DataFrame(crossings, columns=['orbit_id', 'q1', 'p1'])
    df_crossings.to_csv('poincare_points.csv', index=False)

    # Plot energy drift
    plt.figure(figsize=(12, 6))
    for _, row in df_energy.iterrows():
        plt.plot(row['orbit_id'], row['energy'], marker='o', linestyle='-', label=f'orbit_{row["orbit_id"]}')
    plt.title('Energy Drift Over Time')
    plt.xlabel('Orbit ID')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.legend()
    plt.savefig('energy_drift.png')

    # Plot Poincaré section
    plt.figure(figsize=(12, 6))
    for _, row in df_crossings.iterrows():
        plt.plot(row['q1'], row['p1'], marker='.', linestyle='None', color='blue')
    plt.title('Poincaré Section')
    plt.xlabel('q1')
    plt.ylabel('p1')
    plt.grid(True)
    plt.savefig('poincare_section.png')

if __name__ == '__main__':
    main()
```