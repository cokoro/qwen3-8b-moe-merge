```bash
cat << 'END' > analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import interp1d

import os
import json

from os.path import join

# --- Load the manifest ---
with open('data/physics_manifest.json') as f:
    manifest = json.load(f)

# --- Load the observations ---
mr_observations = pd.read_csv('data/mr_observations.csv')

# --- Load the fragments ---
profile_fragments = pd.read_csv('data/profile_fragments.csv')

# --- Load the query targets ---
with open('data/query_targets.json') as f:
    query_targets = json.load(f)

# --- Define the structure ODEs ---
def structure_ode(r, state, mu_e):
    rho, m_enc = state
    drho_dr = -mu_e * rho * (m_enc + 4 * np.pi * r**3 / 3) / r
    dm_dr = 4 * np.pi * r**2 * rho
    return np.array([drho_dr, dm_dr])

# --- Solve the structure ODE for a given central density ---
def solve_structure(mu_e, mass_fraction):
    r = 0.0
    rho = 1.0
    m_enc = 0.0
    t = 0.0
    dt = 0.0001
    while True:
        next_rho = rho - dt * (mu_e * rho * (m_enc + 4 * np.pi * r**3 / 3) / r)
        next_m = m_enc + 4 * np.pi * r**2 * rho * dt
        next_r = r + dt
        t += dt
        if next_rho <= 0:
            break
        r = next_r
        rho = next_rho
        m_enc = next_m
    return r, rho, m_enc

# --- Build a dense dimensionless stable branch ---
def build_stable_branch(mu_e, max_mass_fraction=0.999):
    mass_fraction = np.linspace(0.0, max_mass_fraction, 500)
    r_branch = []
    rho_branch = []
    m_enc_branch = []
    for mass_frac in mass_fraction:
        r, rho, m_enc = solve_structure(mu_e, mass_frac)
        r_branch.append(r)
        rho_branch.append(rho)
        m_enc_branch.append(m_enc)
    return np.array(r_branch), np.array(rho_branch), np.array(m_enc_branch)

# --- Scale the branch into physical units ---
def scale_branch(mu_e, r_branch, rho_branch, m_enc_branch, mass_limit_solar):
    surface_density = 1e-4
    surface_radius = 1e4
    surface_mass = 1.0
    surface_radius_scaled = surface_radius * mu_e**(-1)
    surface_density_scaled = surface_density * mu_e**(-2)
    surface_mass_scaled = surface_mass * mu_e**(-2)
    radius_scaled = r_branch * surface_radius_scaled / surface_radius
    density_scaled = rho_branch * surface_density_scaled / surface_density
    mass_scaled = m_enc_branch * surface_mass_scaled / surface_mass
    return radius_scaled, density_scaled, mass_scaled

# --- Evaluate the weighted residual ---
def evaluate_residual(mu_e, mr_observations, mass_limit_solar):
    r_branch, rho_branch, m_enc_branch = build_stable_branch(mu_e)
    radius_scaled, density_scaled, mass_scaled = scale_branch(mu_e, r_branch, rho_branch, m_enc_branch, mass_limit_solar)
    # Interpolate the branch and compute the residuals
    mass_grid = np.linspace(0.0, mass_limit_solar, 500)
    radius_grid = np.interp(mass_grid, mass_scaled, radius_scaled)
    rho_grid = np.interp(mass_grid, mass_scaled, density_scaled)
    m_enc_grid = np.interp(mass_grid, mass_scaled, m_enc_branch)
    # Compute the residuals for each observation
    residuals = []
    for _, row in mr_observations.iterrows():
        mass_obs = row['mass_solar']
        radius_obs = row['radius_km']
        sigma_obs = row['radius_km_uncertainty']
        radius_pred = np.interp(mass_obs, mass_grid, radius_grid)
        residual = (radius_pred - radius_obs) / sigma_obs
        residuals.append(residual)
    return np.sum(residuals**2)

# --- Fit the hidden parameter ---
mu_e_initial = 1.0
residuals = evaluate_residual(mu_e_initial, mr_observations, 0.6)
mu_e_opt = minimize(lambda x: evaluate_residual(x, mr_observations, 0.6), mu_e_initial, method='L-BFGS-B').x[0]

# --- Build the physical branch ---
r_branch, rho_branch, m_enc_branch = build_stable_branch(mu_e_opt)
radius_scaled, density_scaled, mass_scaled = scale_branch(mu_e_opt, r_branch, rho_branch, m_enc_branch, 0.6)

# --- Interpolate the branch and generate the reconstruction ---
mass_grid = np.linspace(0.0, 0.6, 500)
radius_grid = np.interp(mass_grid, mass_scaled, radius_scaled)
rho_grid = np.interp(mass_grid, mass_scaled, density_scaled)

# --- Generate the profile predictions ---
profile_masses = [float(m) for m in query_targets['profiles']['mass_solar']]
profile_radius_fractions = np.array([float(r) for r in query_targets['profiles']['radius_fraction_grid']])
profile_radius_fractions = np.reshape(profile_radius_fractions, (-1, 1))
profile_mass_fractions = np.array([float(m) for m in query_targets['profiles']['mass_fraction_grid']])
profile_mass_fractions = np.reshape(profile_mass_fractions, (-1, 1))

# --- Write the results ---
os.makedirs('results', exist_ok=True)
np.savetxt(join('results', 'mr_reconstruction.csv'), np.vstack((mass_grid, radius_grid)).T, delimiter=',', header='mass_solar,radius_km')

# --- Write the profiles ---
profiles = {
    'density_fraction': np.zeros((len(profile_masses), len(profile_radius_fractions))),
    'enclosed_mass_fraction': np.zeros((len(profile_masses), len(profile_radius_fractions))),
}
for i, mass in enumerate(profile_masses):
    mass_grid = np.linspace(0.0, mass, 1000)
    radius_grid = np.interp(mass_grid, mass_scaled, radius_scaled)
    rho_grid = np.interp(mass_grid, mass_scaled, density_scaled)
    m_enc_grid = np.interp(mass_grid, mass_scaled, m_enc_branch)
    # Interpolate the profiles
    profiles['density_fraction'][i] = np.interp(profile_radius_fractions, radius_grid, rho_grid)
    profiles['enclosed_mass_fraction'][i] = np.interp(profile_radius_fractions, radius_grid, m_enc_grid)

np.savez(join('results', 'profiles_reconstruction.npz'), **profiles)

# --- Write the summary ---
summary = {
    'mass_limit_solar': 0.6,
    'radius_at_0p6Msun_km': float(radius_grid[-1]),
    'radius_at_1p0Msun_km': float(np.interp(1.0, mass_grid, radius_grid)),
    'radius_at_1p2Msun_km': float(np.interp(1.2, mass_grid, radius_grid)),
    'radius_at_1p3Msun_km': float(np.interp(1.3, mass_grid, radius_grid)),
}
with open(join('results', 'summary.json'), 'w') as f:
    json.dump(summary, f)

# --- Plot the reconstruction ---
plt.figure(figsize=(10, 6))
plt.plot(mass_grid, radius_grid, label='Reconstructed')
plt.scatter(mr_observations['mass_solar'], mr_observations['radius_km'], label='Observations', alpha=0.5)
plt.xlabel('Mass (solar masses)')
plt.ylabel('Radius (km)')
plt.legend()
plt.title('White-Dwarf Mass-Radius Reconstruction')
plt.savefig(join('results', 'reconstruction_plot.png'))
END
```