import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os

# Load data
mr_obs = pd.read_csv('data/mr_observations.csv')
profile_fragments = pd.read_csv('data/profile_fragments.csv')
query_targets = json.load(open('data/query_targets.json'))
physics_manifest = json.load(open('data/physics_manifest.json'))

# Parse physics manifest
mass_unit = physics_manifest['mass_unit']
radius_unit = physics_manifest['radius_unit']
surface_rule = physics_manifest['surface_rule']
density_unit = physics_manifest['density_unit']
enclosed_mass_unit = physics_manifest['enclosed_mass_unit']
density_fraction_unit = physics_manifest['density_fraction_unit']
enclosed_mass_fraction_unit = physics_manifest['enclosed_mass_fraction_unit']

# Convert observed mass and radius to dimensionless units
mr_obs['mass_solar'] = mr_obs['mass_solar'] * (1 / physics_manifest['solar_mass'])
mr_obs['radius_km'] = mr_obs['radius_km'] * (1 / physics_manifest['km'])

# Convert profile fragments to dimensionless units
profile_fragments['mass_solar'] = profile_fragments['mass_solar'] * (1 / physics_manifest['solar_mass'])
profile_fragments['radius_km'] = profile_fragments['radius_km'] * (1 / physics_manifest['km'])
profile_fragments['density'] = profile_fragments['density'] * (1 / physics_manifest['density'])
profile_fragments['enclosed_mass'] = profile_fragments['enclosed_mass'] * (1 / physics_manifest['enclosed_mass'])

# Define the white-dwarf structure equations
def ode_system(r, y, mu_e):
    rho, M_enc = y
    c_sq = (mu_e ** 2) / (4 * np.pi * (1 + mu_e ** 2)) * (1 + 3 * (1 - 2 * M_enc / (r ** 2 * mu_e ** 2)) ** 0.5)
    d_rho_dr = - (c_sq * rho * M_enc) / (r * (1 + mu_e ** 2))
    d_M_enc_dr = 4 * np.pi * r ** 2 * rho
    return [d_rho_dr, d_M_enc_dr]

# Build a dense dimensionless stable branch by scanning central state values
def build_branch(mu_e):
    central_density = np.linspace(1e-10, 100, 1000)
    branch = []
    for rho_c in central_density:
        sol = solve_ivp(lambda r, y: ode_system(r, y, mu_e), [0, 1e5], [rho_c, 0], dense_output=True, events=lambda r, y: y[1] - 1e-10)
        r_sol, y_sol = sol.sol([0, 1e5])
        r_surface = sol.t_events[0][0]
        r_sol = r_sol[r_sol < r_surface]
        y_sol = y_sol[:, r_sol < r_surface]
        rho_sol = y_sol[0]
        M_enc_sol = y_sol[1]
        M_enc_surface = M_enc_sol[-1]
        R = r_sol[-1]
        branch.append((M_enc_surface, R, rho_sol, M_enc_sol))
    branch.sort()
    return np.array(branch)

# Fit the hidden parameter mu_e
def objective_function(mu_e):
    branch = build_branch(mu_e)
    mass_solar = branch[:, 0] * physics_manifest['solar_mass']
    radius_km = branch[:, 1] * physics_manifest['km']
    weights = 1 / (mr_obs['uncertainty_mass_solar'] ** 2 + mr_obs['uncertainty_radius_km'] ** 2)
    residual_mass = (mass_solar - mr_obs['mass_solar']) * weights
    residual_radius = (radius_km - mr_obs['radius_km']) * weights
    return np.sum(residual_mass ** 2 + residual_radius ** 2)

mu_e_opt = minimize(objective_function, x0=1.0).x[0]

# Reconstruct the stable branch
branch = build_branch(mu_e_opt)

# Interpolate the branch quantities
mass_solar_interp = interp1d(branch[:, 0], branch[:, 1], kind='linear', fill_value='extrapolate')
radius_km_interp = interp1d(branch[:, 0], branch[:, 1], kind='linear', fill_value='extrapolate')

# Predict holdout masses and profiles
holdout_masses = [float(m) for m in query_targets['holdout_masses_solar']]
holdout_profiles = query_targets['profile_targets']

holdout_masses_solar = np.array(holdout_masses)
holdout_radius_km = radius_km_interp(holdout_masses_solar)

holdout_profile_masses = [float(m) for m in holdout_profiles]
holdout_profile_indices = np.searchsorted(branch[:, 0], holdout_profile_masses)
holdout_profile_rho = branch[holdout_profile_indices, 2]
holdout_profile_M_enc = branch[holdout_profile_indices, 3]

# Generate the results
results = {
    'mass_limit_solar': branch[-1, 0],
    'radius_at_0p6Msun_km': radius_km_interp(0.6),
    'radius_at_1p0Msun_km': radius_km_interp(1.0),
    'radius_at_1p2Msun_km': radius_km_interp(1.2),
    'radius_at_1p3Msun_km': radius_km_interp(1.3)
}

# Save the results
with open('results/summary.json', 'w') as f:
    json.dump(results, f)

mr_reconstruction = pd.DataFrame({
    'mass_solar': branch[:, 0],
    'radius_km': branch[:, 1]
})
mr_reconstruction.to_csv('results/mr_reconstruction.csv', index=False)

np.savez('results/profiles_reconstruction.npz',
         density_fraction=holdout_profile_rho,
         enclosed_mass_fraction=holdout_profile_M_enc)

# Plot the reconstruction
plt.figure(figsize=(10, 6))
plt.plot(mr_obs['mass_solar'], mr_obs['radius_km'], 'ko', label='Observations')
plt.plot(branch[:, 0], branch[:, 1], 'r-', label='Reconstruction')
plt.xlabel('Mass (solar mass)')
plt.ylabel('Radius (km)')
plt.legend()
plt.savefig('results/reconstruction_plot.png')
plt.close()