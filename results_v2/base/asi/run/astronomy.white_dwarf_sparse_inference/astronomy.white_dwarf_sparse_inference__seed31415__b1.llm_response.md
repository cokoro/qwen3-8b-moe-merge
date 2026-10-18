```python
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os

# Load data
mr_obs = pd.read_csv('data/mr_observations.csv')
profile_fragments = pd.read_csv('data/profile_fragments.csv')
query_targets = json.load(open('data/query_targets.json'))
physics_manifest = json.load(open('data/physics_manifest.json'))

# Constants from physics manifest
surface_density = physics_manifest['surface_density']
surface_pressure = physics_manifest['surface_pressure']
surface_rule = physics_manifest['surface_rule']
units = physics_manifest['units']
mass_limit_solar = physics_manifest['mass_limit_solar']

# Define the white-dwarf structure equations
def structure_equations(r, y, mu_e):
    rho_c, M_c, R_c = y
    # Convert to dimensionless variables
    r_dim = r * R_c
    rho_dim = rho_c / rho_c
    M_dim = M_c / M_c
    R_dim = R_c / R_c
    # Solve the Lane-Emden equation for degenerate gas
    # This is a simplified version; actual implementation would require more detailed equations
    # For demonstration, we assume a polytropic solution
    # In practice, this should be replaced with the full structure equations
    # Here, we simulate a polytropic solution with n = 3
    # This is a placeholder and should be replaced with actual physics
    # The actual implementation would involve solving the Lane-Emden equation
    # with the appropriate polytropic index for a white dwarf
    # For simplicity, we assume a linear relation here
    return [0.0, 0.0, 0.0]

# Function to solve the structure equations for a given mu_e
def solve_structure(mu_e):
    # Initial guess for central density and mass
    # This should be based on the physics of white dwarfs
    # For demonstration, we use a simple initial guess
    # In practice, this would require a more sophisticated approach
    y0 = [1.0, 1.0, 1.0]  # rho_c, M_c, R_c
    sol = solve_ivp(structure_equations, [0, 1], y0, args=(mu_e,), dense_output=True)
    # Extract the solution
    r = sol.t
    rho = sol.y[0]
    M = sol.y[1]
    R = sol.y[2]
    # Find the surface where the density or pressure meets the surface rule
    surface_idx = np.argmin(np.abs(rho - surface_density))
    # Extract the physical quantities
    mass = M[surface_idx]
    radius = R[surface_idx]
    # Return the dimensionless quantities
    return r, rho, M, R, mass, radius

# Function to compute the residuals for the observations
def compute_residuals(mu_e, mr_obs):
    r, rho, M, R, mass, radius = solve_structure(mu_e)
    # Compute the residuals for the observations
    residuals = []
    for _, row in mr_obs.iterrows():
        mass_obs = row['mass_solar']
        radius_obs = row['radius_km']
        # Find the closest mass in the solution
        mass_idx = np.argmin(np.abs(M - mass_obs))
        radius_sol = R[mass_idx]
        residual = (radius_sol - radius_obs) / radius_obs
        residuals.append(residual)
    return np.array(residuals)

# Function to compute the residuals for the profile fragments
def compute_profile_residuals(mu_e, profile_fragments):
    r, rho, M, R, mass, radius = solve_structure(mu_e)
    residuals = []
    for _, row in profile_fragments.iterrows():
        mass_frag = row['mass_solar']
        # Find the closest mass in the solution
        mass_idx = np.argmin(np.abs(M - mass_frag))
        # Extract the density and enclosed mass profiles
        density = rho[mass_idx]
        enclosed_mass = M[mass_idx]
        # Compute the residuals for the fragment
        # This is a simplified version; actual implementation would require more detailed comparison
        residual = (density - row['density_fraction']) / row['density_fraction']
        residuals.append(residual)
    return np.array(residuals)

# Function to optimize mu_e
def optimize_mu_e(mr_obs):
    # Initial guess for mu_e
    mu_e_initial = 1.0
    # Define the objective function
    def objective(mu_e):
        # Compute the residuals for the observations
        residuals = compute_residuals(mu_e, mr_obs)
        # Compute the weighted sum of residuals
        weight = 1.0 / (mr_obs['uncertainty_solar'] + 1e-6)
        weighted_residuals = residuals * weight
        return np.sum(weighted_residuals ** 2)
    # Perform optimization
    result = minimize(objective, mu_e_initial, method='L-BFGS-B')
    return result.x[0]

# Optimize mu_e
mu_e_opt = optimize_mu_e(mr_obs)

# Solve the structure equations for the optimized mu_e
r, rho, M, R, mass, radius = solve_structure(mu_e_opt)

# Interpolate the mass-radius relation
mass_interpolated = np.linspace(0.1, mass_limit_solar, 1000)
radius_interpolated = interp1d(M, R, kind='linear', fill_value='extrapolate')(mass_interpolated)

# Compute the summary values
summary = {
    'mass_limit_solar': mass_limit_solar,
    'radius_at_0p6Msun_km': radius_interpolated[np.argmin(np.abs(mass_interpolated - 0.6))],
    'radius_at_1p0Msun_km': radius_interpolated[np.argmin(np.abs(mass_interpolated - 1.0))],
    'radius_at_1p2Msun_km': radius_interpolated[np.argmin(np.abs(mass_interpolated - 1.2))],
    'radius_at_1p3Msun_km': radius_interpolated[np.argmin(np.abs(mass_interpolated - 1.3))]
}

# Save the results
os.makedirs('results', exist_ok=True)
np.savetxt('results/mr_reconstruction.csv', np.column_stack((mass_interpolated, radius_interpolated)), delimiter=',', header='mass_solar,radius_km')
np.savez('results/profiles_reconstruction.npz', density_fraction=rho, enclosed_mass_fraction=M)
with open('results/summary.json', 'w') as f:
    json.dump(summary, f)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(mass_interpolated, radius_interpolated, label='Reconstructed M-R')
plt.scatter(mr_obs['mass_solar'], mr_obs['radius_km'], label='Observations')
plt.xlabel('Mass (solar masses)')
plt.ylabel('Radius (km)')
plt.legend()
plt.title('White-Dwarf Mass-Radius Reconstruction')
plt.savefig('results/reconstruction_plot.png')
plt.close()
```