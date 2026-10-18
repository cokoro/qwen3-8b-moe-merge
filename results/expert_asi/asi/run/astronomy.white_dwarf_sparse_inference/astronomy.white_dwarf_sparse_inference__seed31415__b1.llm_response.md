```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Load data
mr_obs = pd.read_csv('data/mr_observations.csv')
profile_frags = pd.read_csv('data/profile_fragments.csv')
query = pd.read_json('data/query_targets.json')
physics = pd.read_json('data/physics_manifest.json')

# Define constants
G = physics['G_cgs']
c = physics['c']
hbar = physics['hbar']
m_e = physics['m_e']
m_p = physics['m_p']
k_B = physics['k_B']
mu_e = physics['mu_e']  # Placeholder for inferred parameter

# Convert observed data to dimensionless form
def normalize_obs(mass, radius, mu_e):
    # Convert to solar mass and km
    mass_solar = mass / (1.98847e33)  # kg/solar mass
    radius_km = radius / 1000.0  # m -> km
    # Apply scaling
    mass_scaled = mass_solar * mu_e**(-2)
    radius_scaled = radius_km * mu_e**(-1)
    return mass_scaled, radius_scaled

# Build the white-dwarf branch
def build_branch(mu_e):
    # Solve ODEs for each central state
    # Use a regular-center start with small radius
    # Return mass, radius, profiles for each model
    pass

# Objective function for optimization
def objective(mu_e):
    # Compute residuals for each observation
    # Return sum of squared residuals
    pass

# Main execution
mu_e_opt = minimize(objective, x0=1.0).x[0]
branch = build_branch(mu_e_opt)

# Generate holdout predictions
holdout_masses = query['holdout_masses']
holdout_profiles = query['profile_masses']

# Save results
np.savetxt('results/mr_reconstruction.csv', branch, delimiter=',')
np.savez('results/profiles_reconstruction.npz', density_fraction=..., enclosed_mass_fraction=...)
with open('results/summary.json', 'w') as f:
    json.dump({'mass_limit_solar': ..., 'radius_at_0p6Msun_km': ..., ...}, f)
plt.plot(branch['mass'], branch['radius'])
plt.savefig('results/reconstruction_plot.png')
```